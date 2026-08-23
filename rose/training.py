"""Full-batch fitting and label-free checkpoint diagnostics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

from .correction import BoundedResidualCorrector, correction_loss
from .metrics import kmeans_partition, normalize_spectral_scores
from .model import S2HierarchicalSpectralMapper, joint_spectral_loss
from .selector import CheckpointDiagnostics, select_checkpoint
from .spectral import SpectralTarget


@dataclass(frozen=True)
class MapperTrainingSettings:
    """Frozen mapper architecture, optimizer, and diagnostic schedule."""

    epochs: int
    evaluation_interval: int
    batch_mode: str
    learning_rate: float
    weight_decay: float
    stem_width: int
    latent_width: int
    residual_blocks: int
    dropout: float
    spectral_weight: float
    reconstruction_weight: float
    cross_view_weight: float


@dataclass(frozen=True)
class SelectorSettings:
    """Frozen label-free selection thresholds."""

    minimum_epoch: int
    minimum_cluster_entropy: float
    maximum_largest_cluster_fraction: float
    minimum_effective_rank_ratio: float
    minimum_coordinate_variance: float
    topology_ceiling_threshold: float


@dataclass(frozen=True)
class CorrectionTrainingSettings:
    """Frozen bounded-correction training settings."""

    epochs: int
    evaluation_interval: int
    hidden_width: int
    learning_rate: float
    weight_decay: float
    relative_norm_weight: float
    rho: float


@dataclass(frozen=True)
class MapperEpochRecord:
    """One scheduled mapper diagnostic without ground-truth inputs."""

    epoch: int
    loss_total: float
    loss_spec: float
    loss_rec: float
    loss_cv: float
    diagnostics: CheckpointDiagnostics


@dataclass(frozen=True)
class MapperTrainingResult:
    """Selected frozen mapper and its label-free training record."""

    mapper: S2HierarchicalSpectralMapper
    history: tuple[MapperEpochRecord, ...]
    selected: CheckpointDiagnostics
    base_scores: np.ndarray
    base_fused: np.ndarray
    base_prediction: np.ndarray


@dataclass(frozen=True)
class CorrectionEpochRecord:
    """One scheduled correction-training diagnostic."""

    epoch: int
    loss_total: float
    loss_fit: float
    loss_relative_norm: float
    maximum_residual_ratio: float


@dataclass(frozen=True)
class CorrectionTrainingResult:
    """Final frozen corrector and its training record."""

    corrector: BoundedResidualCorrector
    history: tuple[CorrectionEpochRecord, ...]


def set_deterministic_seed(seed: int) -> None:
    """Set the random generators used by the reference implementation."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _effective_rank(values: np.ndarray) -> float:
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )
    energy = singular_values * singular_values
    probabilities = energy / max(float(np.sum(energy)), 1.0e-12)
    probabilities = probabilities[probabilities > 0.0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _checkpoint_diagnostics(
    epoch: int,
    losses: dict[str, torch.Tensor],
    scores: np.ndarray,
    spectral_target: SpectralTarget,
    n_clusters: int,
    seed: int,
    strict_observed_invariant: bool,
) -> tuple[CheckpointDiagnostics, np.ndarray]:
    normalized = normalize_spectral_scores(scores)
    prediction = kmeans_partition(scores, n_clusters, seed)
    counts = np.bincount(prediction, minlength=int(n_clusters)).astype(np.float64)
    proportions = counts / max(float(np.sum(counts)), 1.0)
    positive = proportions[proportions > 0.0]
    entropy = (
        float(-np.sum(positive * np.log(positive)) / math.log(int(n_clusters)))
        if int(n_clusters) > 1
        else 0.0
    )
    target_rank = _effective_rank(np.asarray(spectral_target.target))
    score_rank = _effective_rank(normalized)
    numeric_losses = [float(value.detach().cpu()) for value in losses.values()]
    diagnostics = CheckpointDiagnostics(
        epoch=int(epoch),
        loss_spec=float(losses["loss_spec"].detach().cpu()),
        graph_partition_nmi=float(
            normalized_mutual_info_score(
                spectral_target.graph_partition,
                prediction,
                average_method="arithmetic",
            )
        ),
        active_clusters=int(np.sum(counts > 0.0)),
        cluster_entropy=entropy,
        largest_cluster_fraction=float(np.max(proportions)),
        effective_rank_ratio=float(score_rank / max(target_rank, 1.0e-12)),
        feature_variance_mean=float(np.var(normalized, axis=0).mean()),
        strict_observed_invariant=bool(strict_observed_invariant),
        finite=bool(
            np.all(np.isfinite(numeric_losses))
            and np.all(np.isfinite(normalized))
        ),
    )
    return diagnostics, prediction.astype(np.int64)


def train_mapper(
    views: list[np.ndarray],
    mask: np.ndarray,
    spectral_target: SpectralTarget,
    n_clusters: int,
    seed: int,
    model_seed: int,
    settings: MapperTrainingSettings,
    selector_settings: SelectorSettings,
    topology_ceiling: bool,
    strict_observed_invariant: bool,
    device: torch.device,
) -> MapperTrainingResult:
    """Fit S2 with full-batch AdamW and select a label-free checkpoint."""

    if settings.batch_mode != "full":
        raise ValueError("The frozen mapper requires full-batch fitting.")
    if settings.epochs != 120 or settings.evaluation_interval != 5:
        raise ValueError("The frozen mapper schedule is 120 epochs with interval 5.")
    if selector_settings.minimum_epoch != 20:
        raise ValueError("The effective checkpoint minimum is epoch 20.")
    if settings.epochs % settings.evaluation_interval != 0:
        raise ValueError("The evaluation interval must divide the epoch count.")
    if len(views) != mask.shape[1]:
        raise ValueError("The number of views must match the mask width.")
    if spectral_target.target.shape != (mask.shape[0], int(n_clusters)):
        raise ValueError("The spectral target shape does not match the run.")

    set_deterministic_seed(model_seed)
    view_tensors = [
        torch.as_tensor(value, dtype=torch.float32, device=device)
        for value in views
    ]
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(
        spectral_target.target,
        dtype=torch.float32,
        device=device,
    )
    mapper = S2HierarchicalSpectralMapper(
        view_dims=[int(value.shape[1]) for value in views],
        n_clusters=int(n_clusters),
        width=int(settings.stem_width),
        latent_dim=int(settings.latent_width),
        blocks=int(settings.residual_blocks),
        dropout=float(settings.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        mapper.parameters(),
        lr=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
    )

    records: list[MapperEpochRecord] = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    endpoints: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for epoch in range(1, int(settings.epochs) + 1):
        mapper.train()
        optimizer.zero_grad()
        _outputs, losses = joint_spectral_loss(
            mapper,
            view_tensors,
            mask_tensor,
            target_tensor,
            lambda_spec=float(settings.spectral_weight),
            lambda_rec=float(settings.reconstruction_weight),
            lambda_cv=float(settings.cross_view_weight),
        )
        if not bool(torch.isfinite(losses["loss_total"])):
            raise RuntimeError("Mapper training produced a non-finite loss.")
        losses["loss_total"].backward()
        optimizer.step()

        if epoch % int(settings.evaluation_interval) != 0:
            continue
        mapper.eval()
        with torch.no_grad():
            outputs, losses = joint_spectral_loss(
                mapper,
                view_tensors,
                mask_tensor,
                target_tensor,
                lambda_spec=float(settings.spectral_weight),
                lambda_rec=float(settings.reconstruction_weight),
                lambda_cv=float(settings.cross_view_weight),
            )
            scores = outputs["spectral"].detach().cpu().numpy()
        fused = outputs["fused"].detach().cpu().numpy()
        diagnostics, prediction = _checkpoint_diagnostics(
            epoch,
            losses,
            scores,
            spectral_target,
            int(n_clusters),
            int(seed),
            bool(strict_observed_invariant),
        )
        endpoints[int(epoch)] = (
            scores.copy(),
            fused.copy(),
            prediction.copy(),
        )
        records.append(
            MapperEpochRecord(
                epoch=int(epoch),
                loss_total=float(losses["loss_total"].detach().cpu()),
                loss_spec=float(losses["loss_spec"].detach().cpu()),
                loss_rec=float(losses["loss_rec"].detach().cpu()),
                loss_cv=float(losses["loss_cv"].detach().cpu()),
                diagnostics=diagnostics,
            )
        )
        states[int(epoch)] = _clone_state(mapper)

    selected = select_checkpoint(
        [record.diagnostics for record in records],
        int(n_clusters),
        topology_ceiling=bool(topology_ceiling),
        minimum_epoch=int(selector_settings.minimum_epoch),
        minimum_entropy=float(selector_settings.minimum_cluster_entropy),
        maximum_largest_cluster_fraction=float(
            selector_settings.maximum_largest_cluster_fraction
        ),
        minimum_effective_rank_ratio=float(
            selector_settings.minimum_effective_rank_ratio
        ),
        minimum_feature_variance=float(
            selector_settings.minimum_coordinate_variance
        ),
    )
    mapper.load_state_dict(states[int(selected.epoch)])
    mapper.eval()
    for parameter in mapper.parameters():
        parameter.requires_grad_(False)
    base_scores, base_fused, prediction = endpoints[int(selected.epoch)]
    return MapperTrainingResult(
        mapper=mapper,
        history=tuple(records),
        selected=selected,
        base_scores=base_scores.astype(np.float32),
        base_fused=base_fused.astype(np.float32),
        base_prediction=prediction.astype(np.int64),
    )


def train_corrector(
    batch: dict[str, torch.Tensor],
    latent_dim: int,
    n_views: int,
    n_clusters: int,
    corrector_seed: int,
    settings: CorrectionTrainingSettings,
) -> CorrectionTrainingResult:
    """Fit the optional corrector after the selected mapper has been frozen."""

    if settings.epochs != 40:
        raise ValueError("The frozen corrector schedule is 40 epochs.")
    if settings.evaluation_interval < 1:
        raise ValueError("The correction evaluation interval must be positive.")
    set_deterministic_seed(corrector_seed)
    corrector = BoundedResidualCorrector(
        latent_dim=int(latent_dim),
        n_views=int(n_views),
        n_clusters=int(n_clusters),
        width=int(settings.hidden_width),
    ).to(batch["fused"].device)
    optimizer = torch.optim.AdamW(
        corrector.parameters(),
        lr=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
    )
    records: list[CorrectionEpochRecord] = []
    for epoch in range(0, int(settings.epochs) + 1):
        if epoch > 0:
            corrector.train()
            optimizer.zero_grad()
            total, details = correction_loss(
                corrector,
                batch,
                rho=float(settings.rho),
                lambda_relative_norm=float(settings.relative_norm_weight),
            )
            if not bool(torch.isfinite(total)):
                raise RuntimeError("Correction training produced a non-finite loss.")
            total.backward()
            optimizer.step()
        if epoch % int(settings.evaluation_interval) != 0:
            continue
        corrector.eval()
        with torch.no_grad():
            total, details = correction_loss(
                corrector,
                batch,
                rho=float(settings.rho),
                lambda_relative_norm=float(settings.relative_norm_weight),
            )
        records.append(
            CorrectionEpochRecord(
                epoch=int(epoch),
                loss_total=float(total.detach().cpu()),
                loss_fit=float(details["fit"].detach().cpu()),
                loss_relative_norm=float(
                    details["relative_norm"].detach().cpu()
                ),
                maximum_residual_ratio=float(details["ratio"].max().detach().cpu()),
            )
        )
    corrector.eval()
    for parameter in corrector.parameters():
        parameter.requires_grad_(False)
    return CorrectionTrainingResult(corrector=corrector, history=tuple(records))
