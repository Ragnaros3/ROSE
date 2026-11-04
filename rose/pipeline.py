"""Auditable label-free ROSE execution followed by optional evaluation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .correction import NeedDecision, bootstrap_need_decision, bounded_delta, make_counterfactual_batch
from .graph import FixedSupportGraph, build_fixed_support_graph, graph_edge_set
from .metrics import (
    ClusteringScores,
    clustering_scores,
    kmeans_partition,
    normalize_spectral_scores,
)
from .protocol import (
    co_observed_giant_component_ratio,
    count_observed_support_violations,
    standardize_observed_views,
    summarize_one_view_drop_mask,
)
from .spectral import SpectralTarget, build_spectral_target
from .training import (
    CorrectionTrainingResult,
    CorrectionTrainingSettings,
    MapperTrainingResult,
    MapperTrainingSettings,
    SelectorSettings,
    train_corrector,
    train_mapper,
)


@dataclass(frozen=True)
class CorrectionAdmissionSettings:
    """Frozen label-free rule that decides whether correction is used."""

    optional: bool
    bootstrap_replicates: int
    bootstrap_alpha: float
    bootstrap_seed: int
    response_budget_factor: float
    admission_threshold: float


@dataclass(frozen=True)
class FrozenMethodSettings:
    """Validated settings loaded from the released configuration."""

    mapper: MapperTrainingSettings
    selector: SelectorSettings
    correction: CorrectionTrainingSettings
    admission: CorrectionAdmissionSettings
    graph_arguments: Mapping[str, int | float]
    spectral_tolerance: float
    kmeans_restarts: int
    runtime_device: str
    torch_threads: int
    cluster_counts: Mapping[str, int]
    incomplete_sample_ratios: tuple[float, ...]
    protocol_seeds: tuple[int, ...]
    seed_derivation: Mapping[str, str]


@dataclass(frozen=True)
class RoseRunResult:
    """Label-free run artifacts and the final partition."""

    graph: FixedSupportGraph
    spectral_target: SpectralTarget
    mapper_training: MapperTrainingResult
    need: NeedDecision | None
    correction_training: CorrectionTrainingResult | None
    topology_ceiling: bool
    co_observed_giant_ratio: float
    strict_observed_invariant: bool
    base_scores: np.ndarray
    final_scores: np.ndarray
    normalized_scores: np.ndarray
    correction_delta: np.ndarray
    residual_ratio: np.ndarray
    prediction: np.ndarray
    correction_skip_reason: str | None = None


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section {name!r} must be an object.")
    return value


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Configuration is missing {key!r}.")
    return mapping[key]


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"Frozen setting {name} must equal {expected!r}.")


def load_frozen_method(config_path: str | Path) -> FrozenMethodSettings:
    """Load and validate the complete released execution contract."""

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    mapper_raw = _mapping(_require(payload, "mapper"), "mapper")
    loss_raw = _mapping(_require(mapper_raw, "loss_weights"), "loss_weights")
    selector_raw = _mapping(
        _require(payload, "checkpoint_selection"),
        "checkpoint_selection",
    )
    correction_raw = _mapping(_require(payload, "correction"), "correction")
    support_raw = _mapping(_require(payload, "observed_support"), "observed_support")
    reliability_raw = _mapping(_require(payload, "reliability"), "reliability")
    numerical_raw = _mapping(_require(payload, "numerical"), "numerical")
    spectral_raw = _mapping(_require(payload, "spectral"), "spectral")
    protocol_raw = _mapping(_require(payload, "protocol"), "protocol")
    runtime_raw = _mapping(_require(payload, "runtime"), "runtime")
    seed_raw = _mapping(_require(payload, "seed_derivation"), "seed_derivation")

    _require_equal(_require(mapper_raw, "epochs"), 120, "mapper.epochs")
    _require_equal(
        _require(mapper_raw, "evaluation_interval"),
        5,
        "mapper.evaluation_interval",
    )
    _require_equal(_require(mapper_raw, "optimizer"), "AdamW", "mapper.optimizer")
    _require_equal(_require(mapper_raw, "batch_mode"), "full", "mapper.batch_mode")
    _require_equal(
        _require(selector_raw, "minimum_epoch"),
        20,
        "checkpoint_selection.minimum_epoch",
    )
    _require_equal(
        _require(selector_raw, "tie_break"),
        "earlier_epoch",
        "checkpoint_selection.tie_break",
    )
    _require_equal(
        _require(reliability_raw, "view_quality_support"),
        "adaptive_admitted_per_view_support",
        "reliability.view_quality_support",
    )
    _require_equal(
        _require(reliability_raw, "reconstruction_neighbors"),
        5,
        "reliability.reconstruction_neighbors",
    )
    _require_equal(_require(spectral_raw, "eigenvectors"), "K+1", "spectral.eigenvectors")
    _require_equal(_require(runtime_raw, "device"), "cpu", "runtime.device")

    rho = float(_require(correction_raw, "rho"))
    response_factor = float(_require(correction_raw, "response_budget_factor"))
    threshold = float(_require(correction_raw, "admission_threshold"))
    if not math.isclose(threshold, response_factor * rho, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("The correction threshold must equal response_budget_factor * rho.")
    confidence = float(_require(correction_raw, "bootstrap_confidence"))
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence must lie in (0, 1).")

    mapper = MapperTrainingSettings(
        epochs=int(_require(mapper_raw, "epochs")),
        evaluation_interval=int(_require(mapper_raw, "evaluation_interval")),
        batch_mode=str(_require(mapper_raw, "batch_mode")),
        learning_rate=float(_require(mapper_raw, "learning_rate")),
        weight_decay=float(_require(mapper_raw, "weight_decay")),
        stem_width=int(_require(mapper_raw, "stem_width")),
        latent_width=int(_require(mapper_raw, "latent_width")),
        residual_blocks=int(_require(mapper_raw, "residual_blocks")),
        dropout=float(_require(mapper_raw, "dropout")),
        spectral_weight=float(_require(loss_raw, "spectral")),
        reconstruction_weight=float(_require(loss_raw, "reconstruction")),
        cross_view_weight=float(_require(loss_raw, "cross_view")),
    )
    selector = SelectorSettings(
        minimum_epoch=int(_require(selector_raw, "minimum_epoch")),
        minimum_cluster_entropy=float(
            _require(selector_raw, "minimum_cluster_entropy")
        ),
        maximum_largest_cluster_fraction=float(
            _require(selector_raw, "maximum_largest_cluster_fraction")
        ),
        minimum_effective_rank_ratio=float(
            _require(selector_raw, "minimum_effective_rank_ratio")
        ),
        minimum_coordinate_variance=float(
            _require(selector_raw, "minimum_coordinate_variance")
        ),
        topology_ceiling_threshold=float(
            _require(selector_raw, "topology_ceiling_threshold")
        ),
    )
    correction = CorrectionTrainingSettings(
        epochs=int(_require(correction_raw, "epochs")),
        evaluation_interval=int(_require(correction_raw, "evaluation_interval")),
        hidden_width=int(_require(correction_raw, "hidden_width")),
        learning_rate=float(_require(correction_raw, "learning_rate")),
        weight_decay=float(_require(correction_raw, "weight_decay")),
        relative_norm_weight=float(
            _require(correction_raw, "relative_norm_weight")
        ),
        rho=rho,
    )
    admission = CorrectionAdmissionSettings(
        optional=bool(_require(correction_raw, "optional")),
        bootstrap_replicates=int(
            _require(correction_raw, "bootstrap_replicates")
        ),
        bootstrap_alpha=float(1.0 - confidence),
        bootstrap_seed=int(_require(correction_raw, "bootstrap_seed")),
        response_budget_factor=response_factor,
        admission_threshold=threshold,
    )
    graph_arguments: dict[str, int | float] = {
        "k_pool": int(_require(support_raw, "neighbor_pool")),
        "k_sigma": int(_require(support_raw, "local_scale_neighbor")),
        "k_boundary": int(_require(support_raw, "boundary_neighbor")),
        "adaptive_k0": int(_require(support_raw, "adaptive_k0")),
        "adaptive_k_min": int(_require(support_raw, "adaptive_k_min")),
        "adaptive_k_max": int(_require(support_raw, "adaptive_k_max")),
        "similarity_floor": float(_require(support_raw, "similarity_floor")),
        "eta_one": float(_require(reliability_raw, "single_direction_factor")),
        "lambda_a": float(_require(reliability_raw, "lambda_A")),
        "rank_outer_factor": float(
            _require(reliability_raw, "rank_outer_factor")
        ),
        "view_quality_neighbors": int(
            _require(reliability_raw, "reconstruction_neighbors")
        ),
        "gate_min": float(_require(reliability_raw, "positive_weight_min")),
        "gate_max": float(_require(reliability_raw, "positive_weight_max")),
        "tolerance": float(_require(spectral_raw, "solver_tolerance")),
        "epsilon": float(_require(numerical_raw, "default_epsilon")),
    }
    cluster_counts_raw = _mapping(
        _require(protocol_raw, "cluster_counts"),
        "protocol.cluster_counts",
    )
    incomplete_sample_ratios = tuple(
        float(value) for value in _require(protocol_raw, "incomplete_sample_ratios")
    )
    protocol_seeds = tuple(int(value) for value in _require(protocol_raw, "seeds"))
    if not incomplete_sample_ratios or not protocol_seeds:
        raise ValueError("The protocol eta and seed sets must be nonempty.")
    return FrozenMethodSettings(
        mapper=mapper,
        selector=selector,
        correction=correction,
        admission=admission,
        graph_arguments=graph_arguments,
        spectral_tolerance=float(_require(spectral_raw, "solver_tolerance")),
        kmeans_restarts=int(_require(spectral_raw, "kmeans_restarts")),
        runtime_device=str(_require(runtime_raw, "device")),
        torch_threads=int(_require(runtime_raw, "torch_threads")),
        cluster_counts={
            str(name): int(value) for name, value in cluster_counts_raw.items()
        },
        incomplete_sample_ratios=incomplete_sample_ratios,
        protocol_seeds=protocol_seeds,
        seed_derivation={str(name): str(value) for name, value in seed_raw.items()},
    )


def _linear_seed(expression: str, seed: int) -> int:
    compact = expression.replace(" ", "")
    match = re.fullmatch(r"(\d+)\*seed\+(\d+)", compact)
    if match:
        return int(match.group(1)) * int(seed) + int(match.group(2))
    match = re.fullmatch(r"(\d+)\+(\d+)\*seed", compact)
    if match:
        return int(match.group(1)) + int(match.group(2)) * int(seed)
    if compact == "seed":
        return int(seed)
    raise ValueError(f"Unsupported deterministic seed rule: {expression!r}.")


def run_rose(
    views: list[np.ndarray],
    mask: np.ndarray,
    n_clusters: int,
    seed: int,
    settings: FrozenMethodSettings,
) -> RoseRunResult:
    """Run graph fusion, one-way transfer, and the optional correction branch."""

    mask_array = np.asarray(mask, dtype=np.float32)
    summary = summarize_one_view_drop_mask(mask_array)
    if len(views) != summary.n_views:
        raise ValueError("The number of views must match the released mask.")
    if any(np.asarray(view).shape[0] != summary.n_samples for view in views):
        raise ValueError("Every view must contain one row per mask row.")
    if int(n_clusters) < 2 or int(n_clusters) > summary.n_samples - 2:
        raise ValueError("n_clusters must satisfy 2 <= K <= n - 2.")

    torch.set_num_threads(int(settings.torch_threads))
    device = torch.device(settings.runtime_device)
    graph = build_fixed_support_graph(
        views,
        mask_array,
        n_clusters=int(n_clusters),
        seed=int(seed),
        **dict(settings.graph_arguments),
    )
    strict_observed_invariant = bool(
        graph_edge_set(graph.adjacency) == set(graph.support)
        and count_observed_support_violations(graph.adjacency, mask_array) == 0
    )
    if not strict_observed_invariant:
        raise RuntimeError("The observed-support invariant did not pass.")
    spectral_target = build_spectral_target(
        graph.adjacency,
        n_clusters=int(n_clusters),
        seed=int(seed),
        tolerance=float(settings.spectral_tolerance),
        kmeans_restarts=int(settings.kmeans_restarts),
    )
    scaled_views = standardize_observed_views(views, mask_array)
    giant_ratio = co_observed_giant_component_ratio(mask_array)
    topology_ceiling = bool(
        giant_ratio < float(settings.selector.topology_ceiling_threshold)
    )
    mapper_seed = _linear_seed(settings.seed_derivation["mapper"], int(seed))
    mapper_training = train_mapper(
        scaled_views,
        mask_array,
        spectral_target,
        int(n_clusters),
        int(seed),
        mapper_seed,
        settings.mapper,
        settings.selector,
        topology_ceiling,
        strict_observed_invariant,
        device,
    )

    view_tensors = [
        torch.as_tensor(value, dtype=torch.float32, device=device)
        for value in scaled_views
    ]
    mask_tensor = torch.as_tensor(mask_array, dtype=torch.float32, device=device)
    need: NeedDecision | None = None
    correction_skip_reason: str | None = None
    if not settings.admission.optional:
        correction_skip_reason = "disabled"
    elif not bool(torch.any(torch.sum(mask_tensor, dim=1) == float(summary.n_views))):
        correction_skip_reason = "no_complete_samples"
    else:
        counterfactual = make_counterfactual_batch(
            mapper_training.mapper,
            view_tensors,
            mask_tensor,
        )
        need = bootstrap_need_decision(
            counterfactual,
            seed=int(seed),
            rho=float(settings.correction.rho),
            response_budget_factor=float(settings.admission.response_budget_factor),
            bootstrap_replicates=int(settings.admission.bootstrap_replicates),
            bootstrap_alpha=float(settings.admission.bootstrap_alpha),
            bootstrap_seed=int(settings.admission.bootstrap_seed),
        )
        if not math.isclose(
            need.threshold,
            settings.admission.admission_threshold,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("The computed Need threshold differs from the configuration.")
        if not need.need_admitted:
            correction_skip_reason = "not_admitted"

    correction_training: CorrectionTrainingResult | None = None
    base_scores = torch.as_tensor(
        mapper_training.base_scores,
        dtype=torch.float32,
        device=device,
    )
    if need is not None and need.need_admitted:
        corrector_seed = _linear_seed(
            settings.seed_derivation["corrector"],
            int(seed),
        )
        correction_training = train_corrector(
            counterfactual,
            latent_dim=int(settings.mapper.latent_width),
            n_views=summary.n_views,
            n_clusters=int(n_clusters),
            corrector_seed=corrector_seed,
            settings=settings.correction,
        )
        fused = torch.as_tensor(
            mapper_training.base_fused,
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            raw = correction_training.corrector(fused, mask_tensor)
            delta, residual_ratio = bounded_delta(
                raw,
                base_scores,
                mask_tensor,
                rho=float(settings.correction.rho),
                force_complete_identity=True,
            )
    else:
        delta = torch.zeros_like(base_scores)
        residual_ratio = torch.zeros(
            (base_scores.shape[0], 1),
            dtype=base_scores.dtype,
            device=base_scores.device,
        )
    final_scores = (base_scores + delta).detach().cpu().numpy().astype(np.float32)
    normalized_scores = normalize_spectral_scores(final_scores)
    prediction = kmeans_partition(final_scores, int(n_clusters), int(seed))
    return RoseRunResult(
        graph=graph,
        spectral_target=spectral_target,
        mapper_training=mapper_training,
        need=need,
        correction_training=correction_training,
        topology_ceiling=topology_ceiling,
        co_observed_giant_ratio=giant_ratio,
        strict_observed_invariant=strict_observed_invariant,
        base_scores=mapper_training.base_scores.copy(),
        final_scores=final_scores,
        normalized_scores=normalized_scores,
        correction_delta=delta.detach().cpu().numpy().astype(np.float32),
        residual_ratio=residual_ratio.detach().cpu().numpy().astype(np.float32),
        prediction=prediction.astype(np.int64),
        correction_skip_reason=correction_skip_reason,
    )


def evaluate_partition(result: RoseRunResult, labels: np.ndarray) -> ClusteringScores:
    """Evaluate a completed run after all label-free decisions are final."""

    return clustering_scores(labels, result.prediction)
