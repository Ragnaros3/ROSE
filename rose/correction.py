"""Optional pointwise-bounded score correction and admission rule."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BoundedResidualCorrector(nn.Module):
    """Small residual network conditioned on fused features and availability."""

    def __init__(
        self,
        latent_dim: int,
        n_views: int,
        n_clusters: int,
        width: int = 32,
    ) -> None:
        super().__init__()
        if min(latent_dim, n_views, n_clusters, width) < 1:
            raise ValueError("Corrector dimensions must be positive.")
        self.network = nn.Sequential(
            nn.Linear(int(latent_dim) + int(n_views), int(width)),
            nn.LayerNorm(int(width)),
            nn.ReLU(inplace=True),
            nn.Linear(int(width), int(n_clusters)),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("The final corrector layer must be linear.")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, fused: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if fused.ndim != 2 or mask.ndim != 2 or fused.shape[0] != mask.shape[0]:
            raise ValueError("fused and mask must contain the same sample rows.")
        return self.network(torch.cat([fused, mask], dim=1))


def bounded_delta(
    raw_residual: torch.Tensor,
    base_scores: torch.Tensor,
    mask: torch.Tensor,
    rho: float = 0.10,
    force_complete_identity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project, clip, and optionally suppress the residual on complete rows.

    The returned correction is orthogonal to each nonzero base-score row and
    has norm at most ``rho`` times the corresponding base-score norm. A zero
    base row maps to an exact zero correction.
    """

    if raw_residual.shape != base_scores.shape:
        raise ValueError("raw_residual and base_scores must have the same shape.")
    if raw_residual.ndim != 2:
        raise ValueError("Score tensors must be two-dimensional.")
    if mask.ndim != 2 or mask.shape[0] != base_scores.shape[0]:
        raise ValueError("mask must contain one row per score row.")
    if float(rho) < 0.0:
        raise ValueError("rho must be nonnegative.")

    base_sq = torch.sum(base_scores * base_scores, dim=1, keepdim=True)
    projection = torch.sum(raw_residual * base_scores, dim=1, keepdim=True)
    nonzero_base = base_sq > 0.0
    safe_base_sq = torch.where(nonzero_base, base_sq, torch.ones_like(base_sq))

    tangent = raw_residual - projection / safe_base_sq * base_scores
    tangent = torch.where(nonzero_base, tangent, torch.zeros_like(tangent))

    base_norm = torch.sqrt(base_sq)
    tangent_norm = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
    nonzero_tangent = tangent_norm > 0.0
    safe_tangent_norm = torch.where(
        nonzero_tangent,
        tangent_norm,
        torch.ones_like(tangent_norm),
    )
    clipped_scale = torch.clamp(
        float(rho) * base_norm / safe_tangent_norm,
        max=1.0,
    )
    scale = torch.where(
        nonzero_tangent,
        clipped_scale,
        torch.ones_like(clipped_scale),
    )
    delta = tangent * scale

    delta_projection = torch.sum(delta * base_scores, dim=1, keepdim=True)
    delta = delta - delta_projection / safe_base_sq * base_scores
    delta = torch.where(nonzero_base, delta, torch.zeros_like(delta))

    if force_complete_identity:
        incomplete = (
            torch.sum(mask, dim=1, keepdim=True) < mask.shape[1] - 0.5
        ).to(delta.dtype)
        delta = delta * incomplete

    safe_base_norm = torch.where(
        nonzero_base,
        base_norm,
        torch.ones_like(base_norm),
    )
    ratio = torch.linalg.vector_norm(delta, dim=1, keepdim=True) / safe_base_norm
    return delta, ratio


def make_counterfactual_batch(
    mapper: nn.Module,
    views: list[torch.Tensor],
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Delete each view once from every actually complete parent sample."""

    if not views:
        raise ValueError("At least one view is required.")
    n_views = int(mask.shape[1])
    if len(views) != n_views:
        raise ValueError("The number of views must match the mask width.")
    complete = torch.sum(mask, dim=1) == float(n_views)
    complete_indices = torch.nonzero(complete, as_tuple=False).squeeze(1)
    if int(complete_indices.numel()) == 0:
        raise ValueError("At least one complete parent sample is required.")

    parent_views = [
        view.index_select(0, complete_indices)
        for view in views
    ]
    parent_count = int(complete_indices.numel())
    parent_mask = torch.ones(
        (parent_count, n_views),
        dtype=mask.dtype,
        device=mask.device,
    )
    child_fused: list[torch.Tensor] = []
    child_scores: list[torch.Tensor] = []
    child_masks: list[torch.Tensor] = []
    parent_targets: list[torch.Tensor] = []

    mapper.eval()
    with torch.no_grad():
        parent_output = mapper(parent_views, parent_mask)
        parent_scores = parent_output["spectral"].detach()
        for view_id in range(n_views):
            child_mask = parent_mask.clone()
            child_mask[:, view_id] = 0.0
            child_output = mapper(parent_views, child_mask)
            child_fused.append(child_output["fused"].detach())
            child_scores.append(child_output["spectral"].detach())
            child_masks.append(child_mask)
            parent_targets.append(parent_scores)

    return {
        "fused": torch.cat(child_fused, dim=0),
        "spectral": torch.cat(child_scores, dim=0),
        "mask": torch.cat(child_masks, dim=0),
        "parent": torch.cat(parent_targets, dim=0),
        "complete_indices": complete_indices,
        "parent_local_index": torch.arange(
            parent_count, device=mask.device, dtype=torch.long
        ).repeat(n_views),
        "dropped_view_id": torch.arange(
            n_views, device=mask.device, dtype=torch.long
        ).repeat_interleave(parent_count),
    }


def correction_loss(
    corrector: nn.Module,
    batch: dict[str, torch.Tensor],
    rho: float = 0.10,
    lambda_relative_norm: float = 0.01,
    denominator_floor: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the counterfactual fitting objective for one training step."""

    raw = corrector(batch["fused"], batch["mask"])
    delta, ratio = bounded_delta(
        raw,
        batch["spectral"],
        batch["mask"],
        rho=float(rho),
        force_complete_identity=False,
    )
    corrected = batch["spectral"] + delta
    fit = F.mse_loss(corrected, batch["parent"])
    relative_norm = torch.mean(
        torch.sum(delta * delta, dim=1)
        / (
            torch.sum(batch["spectral"] * batch["spectral"], dim=1)
            + float(denominator_floor)
        )
    )
    total = fit + float(lambda_relative_norm) * relative_norm
    return total, {
        "fit": fit,
        "relative_norm": relative_norm,
        "delta": delta,
        "ratio": ratio,
    }


def parent_relative_scores(
    batch: dict[str, torch.Tensor],
    denominator_floor: float = 1.0e-8,
) -> np.ndarray:
    """Aggregate symmetric relative response energy by complete parent."""

    required = {"spectral", "parent", "parent_local_index", "complete_indices"}
    missing = required - set(batch)
    if missing:
        raise KeyError(f"Counterfactual batch is missing fields: {sorted(missing)}")
    with torch.no_grad():
        difference = batch["spectral"] - batch["parent"]
        difference_sq = torch.sum(difference * difference, dim=1)
        parent_sq = torch.sum(batch["parent"] * batch["parent"], dim=1)
        child_sq = torch.sum(batch["spectral"] * batch["spectral"], dim=1)
        denominator = 0.5 * (parent_sq + child_sq) + float(denominator_floor)
        relative_sq = (difference_sq / denominator).detach().cpu().numpy()
        parent_ids = batch["parent_local_index"].detach().cpu().numpy()

    parent_count = int(batch["complete_indices"].numel())
    totals = np.bincount(parent_ids, weights=relative_sq, minlength=parent_count)
    counts = np.bincount(parent_ids, minlength=parent_count)
    if np.any(counts <= 0):
        raise ValueError("Every complete parent must retain all deletion children.")
    return totals / counts


@dataclass(frozen=True)
class NeedDecision:
    """Bootstrap decision for enabling the optional correction."""

    response_rms: float
    threshold: float
    ci_lower: float
    ci_upper: float
    point_estimate_above_threshold: bool
    need_admitted: bool
    bootstrap_replicates: int
    confidence_level: float
    parent_count: int


def bootstrap_need_decision(
    batch: dict[str, torch.Tensor],
    seed: int,
    rho: float = 0.10,
    response_budget_factor: float = 2.0,
    bootstrap_replicates: int = 1000,
    bootstrap_alpha: float = 0.05,
    bootstrap_seed: int = 60013,
    denominator_floor: float = 1.0e-8,
) -> NeedDecision:
    """Apply the fixed lower-confidence-bound admission rule."""

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    if not 0.0 < float(bootstrap_alpha) < 1.0:
        raise ValueError("bootstrap_alpha must lie in (0, 1).")
    parent_scores = parent_relative_scores(batch, denominator_floor)
    if parent_scores.size == 0:
        raise ValueError("At least one complete parent is required.")

    response_rms = float(math.sqrt(float(np.mean(parent_scores))))
    threshold = float(response_budget_factor) * float(rho)
    rng = np.random.default_rng(int(bootstrap_seed) + int(seed) * 1009)
    values = np.empty(int(bootstrap_replicates), dtype=np.float64)
    parent_count = int(parent_scores.size)
    chunk_size = min(100, int(bootstrap_replicates))
    for start in range(0, int(bootstrap_replicates), chunk_size):
        size = min(chunk_size, int(bootstrap_replicates) - start)
        indices = rng.integers(
            0,
            parent_count,
            size=(size, parent_count),
        )
        values[start : start + size] = np.sqrt(
            np.mean(parent_scores[indices], axis=1)
        )

    lower = float(np.quantile(values, float(bootstrap_alpha) / 2.0))
    upper = float(np.quantile(values, 1.0 - float(bootstrap_alpha) / 2.0))
    return NeedDecision(
        response_rms=response_rms,
        threshold=threshold,
        ci_lower=lower,
        ci_upper=upper,
        point_estimate_above_threshold=bool(response_rms >= threshold),
        need_admitted=bool(lower >= threshold),
        bootstrap_replicates=int(bootstrap_replicates),
        confidence_level=float(1.0 - float(bootstrap_alpha)),
        parent_count=parent_count,
    )
