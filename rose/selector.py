"""Label-free checkpoint selection for the spectral mapper."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointDiagnostics:
    """Selection inputs computed without ground-truth labels."""

    epoch: int
    loss_spec: float
    graph_partition_nmi: float
    active_clusters: int
    cluster_entropy: float
    largest_cluster_fraction: float
    effective_rank_ratio: float
    feature_variance_mean: float
    strict_observed_invariant: bool
    finite: bool = True


def is_valid_checkpoint(
    row: CheckpointDiagnostics,
    n_clusters: int,
    *,
    minimum_epoch: int = 20,
    minimum_entropy: float = 0.60,
    maximum_largest_cluster_fraction: float = 0.60,
    minimum_effective_rank_ratio: float = 0.65,
    minimum_feature_variance: float = 1.0e-5,
) -> bool:
    """Apply the fixed non-collapse criteria."""

    numeric = (
        row.loss_spec,
        row.graph_partition_nmi,
        row.cluster_entropy,
        row.largest_cluster_fraction,
        row.effective_rank_ratio,
        row.feature_variance_mean,
    )
    return bool(
        row.finite
        and row.strict_observed_invariant
        and all(math.isfinite(float(value)) for value in numeric)
        and int(row.epoch) >= int(minimum_epoch)
        and int(row.active_clusters) == int(n_clusters)
        and float(row.cluster_entropy) >= float(minimum_entropy)
        and float(row.largest_cluster_fraction)
        <= float(maximum_largest_cluster_fraction)
        and float(row.effective_rank_ratio) >= float(minimum_effective_rank_ratio)
        and float(row.feature_variance_mean) >= float(minimum_feature_variance)
    )


def _minimum_spectral_loss(rows: list[CheckpointDiagnostics]) -> CheckpointDiagnostics:
    return min(rows, key=lambda row: (float(row.loss_spec), int(row.epoch)))


def _current_selector(
    rows: list[CheckpointDiagnostics],
    candidates: list[CheckpointDiagnostics],
    topology_ceiling: bool,
) -> CheckpointDiagnostics:
    if not candidates:
        return _minimum_spectral_loss(rows)
    if not topology_ceiling:
        return _minimum_spectral_loss(candidates)
    best_agreement = max(float(row.graph_partition_nmi) for row in candidates)
    guarded = [
        row
        for row in candidates
        if float(row.graph_partition_nmi) >= 0.95 * best_agreement - 1.0e-12
    ]
    if not guarded:
        return _minimum_spectral_loss(candidates)
    maximum_entropy = max(float(row.cluster_entropy) for row in candidates)
    minimum_largest = min(float(row.largest_cluster_fraction) for row in candidates)
    balanced = [
        row
        for row in guarded
        if float(row.cluster_entropy) >= maximum_entropy - 0.010
        and float(row.largest_cluster_fraction) <= minimum_largest + 0.030
    ]
    return min(balanced or guarded, key=lambda row: int(row.epoch))


def select_checkpoint(
    rows: list[CheckpointDiagnostics],
    n_clusters: int,
    *,
    topology_ceiling: bool,
    minimum_epoch: int = 20,
    minimum_entropy: float = 0.60,
    maximum_largest_cluster_fraction: float = 0.60,
    minimum_effective_rank_ratio: float = 0.65,
    minimum_feature_variance: float = 1.0e-5,
) -> CheckpointDiagnostics:
    """Apply the frozen validity, selection and balance-drift rules.

    The minimum epoch restricts valid candidates only. If none is valid,
    choose the minimum spectral loss among all recorded checkpoints,
    including epoch zero when supplied; loss ties choose the earlier epoch.
    A fallback does not certify that a checkpoint meets validity criteria.
    """

    if not rows:
        raise ValueError("At least one checkpoint is required.")
    candidates = [
        row
        for row in rows
        if is_valid_checkpoint(
            row,
            n_clusters,
            minimum_epoch=minimum_epoch,
            minimum_entropy=minimum_entropy,
            maximum_largest_cluster_fraction=maximum_largest_cluster_fraction,
            minimum_effective_rank_ratio=minimum_effective_rank_ratio,
            minimum_feature_variance=minimum_feature_variance,
        )
    ]
    base = _current_selector(rows, candidates, topology_ceiling)
    if not candidates or int(base.epoch) < 100:
        return base
    warmup = min(candidates, key=lambda row: int(row.epoch))
    best_loss = min(float(row.loss_spec) for row in candidates)
    loss_threshold = best_loss + 0.25 * max(
        float(warmup.loss_spec) - best_loss, 0.0
    )
    best_agreement = max(float(row.graph_partition_nmi) for row in candidates)
    earlier = [
        row
        for row in candidates
        if int(row.epoch) < int(base.epoch)
        and float(row.loss_spec) <= loss_threshold + 1.0e-12
        and float(row.graph_partition_nmi) >= 0.88 * best_agreement - 1.0e-12
    ]
    risky = [
        row
        for row in earlier
        if float(base.largest_cluster_fraction)
        - float(row.largest_cluster_fraction)
        >= 0.05 - 1.0e-12
        and float(row.cluster_entropy) - float(base.cluster_entropy)
        >= 0.008 - 1.0e-12
    ]
    return min(risky, key=lambda row: int(row.epoch)) if risky else base
