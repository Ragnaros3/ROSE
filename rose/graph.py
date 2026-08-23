"""Observed-support construction and reliability weighting for ROSE."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from .protocol import count_observed_support_violations, standardize_observed_views
from .spectral import auxiliary_spectral_cluster, leading_eigenspace


Edge = tuple[int, int]


@dataclass(frozen=True)
class RankViewGraph:
    """Directed neighbor information and symmetric kernel weights for one view."""

    view_id: int
    observed_indices: np.ndarray
    rank: list[dict[int, int]]
    directed_distance: list[dict[int, float]]
    sigma: np.ndarray
    edge_weight: dict[Edge, float]


@dataclass(frozen=True)
class FixedSupportGraph:
    """Final graph and the same-support factors used to construct it."""

    adjacency: sp.csr_matrix
    support: dict[Edge, dict[int, float]]
    adaptive_limits: dict[int, int]
    context_reliability: dict[Edge, float]
    context_weights: dict[Edge, float]
    rank_factors: dict[Edge, float]
    view_errors: np.ndarray
    view_gates: np.ndarray
    view_consensus: np.ndarray
    utilization_alpha: float
    utilization_reason: str


def edge_key(left: int, right: int) -> Edge:
    return (left, right) if left < right else (right, left)


def build_weighted_graph(
    edge_scores: dict[Edge, float],
    n_samples: int,
) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (left, right), value in edge_scores.items():
        if left == right or float(value) <= 0.0:
            continue
        rows.extend([int(left), int(right)])
        cols.extend([int(right), int(left)])
        data.extend([float(value), float(value)])
    graph = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(int(n_samples), int(n_samples)),
        dtype=np.float32,
    ).tocsr()
    graph.sum_duplicates()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def graph_edge_set(adjacency: sp.spmatrix) -> set[Edge]:
    upper = sp.triu(adjacency.tocsr(), k=1).tocoo()
    return {(int(left), int(right)) for left, right in zip(upper.row, upper.col)}


def build_rank_view_graphs(
    views: list[np.ndarray],
    mask: np.ndarray,
    k_pool: int = 45,
    k_sigma: int = 7,
    k_boundary: int = 30,
    similarity_floor: float = 0.01,
) -> tuple[list[RankViewGraph], list[np.ndarray]]:
    """Build observed-view neighbor ranks without using missing entries."""

    scaled_views = standardize_observed_views(views, mask)
    n_samples, n_views = np.asarray(mask).shape
    if len(scaled_views) != n_views:
        raise ValueError("The number of views must match the mask width.")
    graphs: list[RankViewGraph] = []
    for view_id, values in enumerate(scaled_views):
        observed = np.flatnonzero(mask[:, view_id] > 0.5).astype(np.int64)
        rank = [{} for _ in range(n_samples)]
        directed_distance = [{} for _ in range(n_samples)]
        sigma = np.full(n_samples, np.nan, dtype=np.float32)
        edge_weight: dict[Edge, float] = {}
        if observed.size <= 1:
            graphs.append(
                RankViewGraph(
                    view_id=view_id,
                    observed_indices=observed,
                    rank=rank,
                    directed_distance=directed_distance,
                    sigma=sigma,
                    edge_weight=edge_weight,
                )
            )
            continue
        maximum = min(
            max(int(k_pool), int(k_sigma), int(k_boundary)),
            int(observed.size) - 1,
        )
        neighbors = NearestNeighbors(n_neighbors=maximum + 1, metric="euclidean")
        neighbors.fit(values[observed])
        distances, indices = neighbors.kneighbors(values[observed])
        scale_position = min(max(1, int(k_sigma)), distances.shape[1] - 1)
        observed_sigma = np.maximum(distances[:, scale_position], 1.0e-6).astype(
            np.float32
        )
        sigma[observed] = observed_sigma
        for local_left in range(indices.shape[0]):
            left = int(observed[local_left])
            sigma_left = float(sigma[left])
            for position in range(1, indices.shape[1]):
                right = int(observed[int(indices[local_left, position])])
                if left == right:
                    continue
                distance = float(distances[local_left, position])
                rank[left][right] = int(position)
                directed_distance[left][right] = distance
                if position <= int(k_pool):
                    sigma_right = (
                        float(sigma[right]) if np.isfinite(sigma[right]) else sigma_left
                    )
                    denominator = max(sigma_left * sigma_right, 1.0e-12)
                    weight = max(
                        float(similarity_floor),
                        math.exp(-(distance * distance) / denominator),
                    )
                    edge = edge_key(left, right)
                    edge_weight[edge] = max(edge_weight.get(edge, 0.0), weight)
        graphs.append(
            RankViewGraph(
                view_id=view_id,
                observed_indices=observed,
                rank=rank,
                directed_distance=directed_distance,
                sigma=sigma,
                edge_weight=edge_weight,
            )
        )
    return graphs, scaled_views


def adaptive_view_limits(
    mask: np.ndarray,
    k0: int = 25,
    k_min: int = 10,
    k_max: int = 35,
) -> dict[int, int]:
    """Compute the observed-count-adaptive neighbor limit for each view."""

    array = np.asarray(mask)
    n_samples = int(array.shape[0])
    limits: dict[int, int] = {}
    for view_id in range(array.shape[1]):
        observed = int(np.sum(array[:, view_id] > 0.5))
        raw = int(round(float(k0) * math.sqrt(observed / max(n_samples, 1))))
        clipped = int(np.clip(raw, int(k_min), int(k_max)))
        limits[view_id] = min(clipped, max(observed - 1, 0))
    return limits


def support_from_view_limits(
    view_graphs: list[RankViewGraph],
    limits: dict[int, int],
) -> dict[Edge, dict[int, float]]:
    """Admit the union of directed in-limit observed-view neighborhoods."""

    support: dict[Edge, dict[int, float]] = defaultdict(dict)
    for graph in view_graphs:
        limit = int(limits.get(int(graph.view_id), 0))
        if limit <= 0:
            continue
        for left, rank_map in enumerate(graph.rank):
            for right, position in rank_map.items():
                if int(position) > limit:
                    continue
                edge = edge_key(left, int(right))
                value = float(graph.edge_weight.get(edge, 1.0))
                support[edge][graph.view_id] = max(
                    support[edge].get(graph.view_id, 0.0), value
                )
    return dict(support)


def _directed_similarity(
    graph: RankViewGraph,
    left: int,
    right: int,
    floor: float,
) -> float | None:
    distance = graph.directed_distance[int(left)].get(int(right))
    if distance is None:
        return None
    sigma_left = float(graph.sigma[int(left)]) if np.isfinite(graph.sigma[int(left)]) else 1.0
    sigma_right = (
        float(graph.sigma[int(right)])
        if np.isfinite(graph.sigma[int(right)])
        else sigma_left
    )
    denominator = max(sigma_left * sigma_right, 1.0e-12)
    return max(float(floor), math.exp(-(float(distance) ** 2) / denominator))


def context_confidence(
    support: dict[Edge, dict[int, float]],
    view_graphs: list[RankViewGraph],
    limits: dict[int, int],
    eta_one: float = 0.70,
    similarity_floor: float = 0.01,
    epsilon: float = 1.0e-8,
) -> dict[Edge, float]:
    """Compute pairwise context confidence on the admitted support."""

    directed_amplitude: dict[tuple[int, int, int], float] = {}
    for graph in view_graphs:
        limit = int(limits.get(graph.view_id, 0))
        for left, rank_map in enumerate(graph.rank):
            candidates = [
                int(right)
                for right, position in rank_map.items()
                if int(position) <= limit
            ]
            similarities = [
                (right, _directed_similarity(graph, left, right, similarity_floor))
                for right in candidates
            ]
            similarities = [
                (right, float(value))
                for right, value in similarities
                if value is not None
            ]
            if not similarities:
                continue
            values = np.asarray([value for _right, value in similarities], dtype=np.float64)
            q50 = float(np.quantile(values, 0.50))
            q90 = float(np.quantile(values, 0.90))
            denominator = max(q90 - q50, float(epsilon))
            for right, value in similarities:
                directed_amplitude[(left, right, graph.view_id)] = float(
                    np.clip((value - q50) / denominator, 0.0, 1.0)
                )
    output: dict[Edge, float] = {}
    one_direction = float(np.clip(eta_one, 0.0, 1.0))
    for edge, sources in support.items():
        left, right = edge
        values = []
        for view_id in sources:
            forward = directed_amplitude.get((left, right, int(view_id)))
            backward = directed_amplitude.get((right, left, int(view_id)))
            if forward is not None and backward is not None:
                values.append(math.sqrt(max(forward, 0.0) * max(backward, 0.0)))
            elif forward is not None or backward is not None:
                values.append(one_direction * max(forward or 0.0, backward or 0.0))
            else:
                values.append(0.0)
        output[edge] = float(np.mean(values)) if values else 0.0
    return output


def _pattern_ids(mask: np.ndarray) -> np.ndarray:
    mapping: dict[tuple[int, ...], int] = {}
    identifiers = np.zeros(mask.shape[0], dtype=np.int64)
    for sample_id in range(mask.shape[0]):
        pattern = tuple((mask[sample_id] > 0.5).astype(int).tolist())
        if pattern not in mapping:
            mapping[pattern] = len(mapping)
        identifiers[sample_id] = mapping[pattern]
    return identifiers


def _quantile_ranks(values: dict[Edge, float]) -> dict[Edge, float]:
    edges = sorted(values)
    if not edges:
        return {}
    ranks = stats.rankdata(
        np.asarray([float(values[edge]) for edge in edges], dtype=np.float64),
        method="average",
    )
    denominator = max(len(edges) - 1, 1)
    return {
        edge: float((rank - 1.0) / denominator)
        for edge, rank in zip(edges, ranks)
    }


def calibrate_context_confidence(
    confidence: dict[Edge, float],
    support: dict[Edge, dict[int, float]],
    mask: np.ndarray,
    minimum_group_size: int | None = None,
) -> dict[Edge, float]:
    """Convert context scores to within-stratum quantile ranks."""

    if not confidence:
        return {}
    n_samples = mask.shape[0]
    degree = np.zeros(n_samples, dtype=np.float64)
    for left, right in support:
        degree[left] += 1.0
        degree[right] += 1.0
    q33, q66 = np.quantile(degree, [0.33, 0.66])
    degree_bin = np.zeros(n_samples, dtype=np.int8)
    degree_bin[degree > q33] = 1
    degree_bin[degree > q66] = 2
    patterns = _pattern_ids(mask)
    edges = sorted(confidence)
    minimum = (
        int(minimum_group_size)
        if minimum_group_size is not None
        else (50 if len(edges) < 5000 else 100)
    )

    def keys(edge: Edge) -> list[tuple[object, ...]]:
        left, right = edge
        coverage = len(support.get(edge, {}))
        degrees = tuple(sorted((int(degree_bin[left]), int(degree_bin[right]))))
        same_pattern = int(patterns[left] == patterns[right])
        return [
            (coverage, degrees, same_pattern),
            (coverage, degrees),
            (coverage,),
            ("global",),
        ]

    grouped: list[dict[tuple[object, ...], list[Edge]]] = [
        defaultdict(list) for _ in range(4)
    ]
    for edge in edges:
        for level, key in enumerate(keys(edge)):
            grouped[level][key].append(edge)
    rank_maps: list[dict[tuple[object, ...], dict[Edge, float]]] = []
    for level_groups in grouped:
        rank_maps.append(
            {
                key: _quantile_ranks(
                    {edge: confidence[edge] for edge in group_edges}
                )
                for key, group_edges in level_groups.items()
            }
        )
    calibrated: dict[Edge, float] = {}
    for edge in edges:
        edge_keys = keys(edge)
        selected = 3
        for level in range(3):
            if len(grouped[level][edge_keys[level]]) >= minimum:
                selected = level
                break
        calibrated[edge] = rank_maps[selected][edge_keys[selected]][edge]
    return calibrated


def context_weights(
    support: dict[Edge, dict[int, float]],
    reliability: dict[Edge, float],
    lambda_a: float = 0.50,
    lower: float = 0.50,
    upper: float = 1.50,
    epsilon: float = 1.0e-8,
) -> dict[Edge, float]:
    """Map calibrated reliability to positive context weights."""

    output: dict[Edge, float] = {}
    for edge in support:
        value = float(np.clip(reliability.get(edge, 0.0), 0.0, 1.0))
        factor = float(
            np.clip(1.0 + float(lambda_a) * (2.0 * value - 1.0), lower, upper)
        )
        output[edge] = max(float(epsilon), factor)
    return output


def rank_taper_factors(
    support: dict[Edge, dict[int, float]],
    view_graphs: list[RankViewGraph],
    limits: dict[int, int],
    outer_factor: float = 0.50,
) -> dict[Edge, float]:
    """Taper outer-neighbor influence without removing admitted edges."""

    if not 0.0 < float(outer_factor) <= 1.0:
        raise ValueError("outer_factor must lie in (0, 1].")
    graphs = {graph.view_id: graph for graph in view_graphs}
    decay = -math.log(float(outer_factor))
    output: dict[Edge, float] = {}
    for edge, sources in support.items():
        left, right = edge
        per_view = []
        for view_id in sources:
            graph = graphs[int(view_id)]
            limit = int(limits[int(view_id)])
            directional = []
            for origin, destination in ((left, right), (right, left)):
                rank = graph.rank[origin].get(destination)
                if rank is None or int(rank) > limit:
                    continue
                denominator = max(limit - 1, 1)
                directional.append(
                    math.exp(-decay * (float(rank) - 1.0) / denominator)
                )
            if len(directional) == 2:
                per_view.append(math.sqrt(directional[0] * directional[1]))
            elif directional:
                per_view.append(directional[0])
        if not per_view:
            raise RuntimeError(f"Admitted edge {edge} has no in-limit rank.")
        output[edge] = float(np.mean(per_view))
    return output


def observed_view_quality(
    scaled_views: list[np.ndarray],
    mask: np.ndarray,
    view_graphs: list[RankViewGraph],
    neighbors: int = 5,
    lower: float = 0.50,
    upper: float = 1.50,
    epsilon: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed-only reconstruction errors and bounded relative gates."""

    errors = []
    for view_id, (values, graph) in enumerate(zip(scaled_views, view_graphs)):
        residual_energy = 0.0
        signal_energy = 0.0
        for sample_id in np.flatnonzero(mask[:, view_id] > 0.5):
            ranked = sorted(
                (
                    (int(position), int(other))
                    for other, position in graph.rank[int(sample_id)].items()
                    if int(position) <= int(neighbors)
                ),
                key=lambda item: item[0],
            )
            if not ranked:
                continue
            local_neighbors = [other for _position, other in ranked]
            weights = []
            for other in local_neighbors:
                distance = graph.directed_distance[int(sample_id)].get(other)
                if distance is None:
                    weights.append(0.0)
                    continue
                sigma_left = (
                    float(graph.sigma[int(sample_id)])
                    if np.isfinite(graph.sigma[int(sample_id)])
                    else 1.0
                )
                sigma_right = (
                    float(graph.sigma[other])
                    if np.isfinite(graph.sigma[other])
                    else sigma_left
                )
                weights.append(
                    math.exp(
                        -(float(distance) ** 2)
                        / max(sigma_left * sigma_right, float(epsilon))
                    )
                )
            weight_array = np.asarray(weights, dtype=np.float64)
            if float(weight_array.sum()) <= float(epsilon):
                weight_array = np.ones(len(local_neighbors), dtype=np.float64)
            weight_array /= float(weight_array.sum())
            reconstruction = weight_array @ np.asarray(
                values[local_neighbors], dtype=np.float64
            )
            target = np.asarray(values[int(sample_id)], dtype=np.float64)
            residual_energy += float(
                np.dot(target - reconstruction, target - reconstruction)
            )
            signal_energy += float(np.dot(target, target))
        errors.append(max(residual_energy / max(signal_energy, epsilon), epsilon))
    error_array = np.asarray(errors, dtype=np.float64)
    median_error = float(np.median(error_array))
    raw = median_error / np.clip(error_array, epsilon, None)
    raw /= math.exp(float(np.mean(np.log(np.clip(raw, epsilon, None)))))
    gates = np.clip(raw, float(lower), float(upper))
    gates /= math.exp(float(np.mean(np.log(np.clip(gates, epsilon, None)))))
    gates = np.clip(gates, float(lower), float(upper))
    return error_array, gates.astype(np.float64)


def observed_view_partition_consensus(
    support: dict[Edge, dict[int, float]],
    mask: np.ndarray,
    n_clusters: int,
    seed: int,
    tolerance: float = 1.0e-4,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    """Measure observed-only agreement among view-specific graph partitions."""

    n_samples, n_views = mask.shape
    observed_by_view: list[np.ndarray] = []
    partitions: list[np.ndarray] = []
    for view_id in range(n_views):
        scores = {
            edge: float(sources[view_id])
            for edge, sources in support.items()
            if view_id in sources
        }
        graph = build_weighted_graph(scores, n_samples)
        observed = np.flatnonzero(mask[:, view_id] > 0.5).astype(np.int64)
        if observed.size < int(n_clusters):
            raise ValueError("Each observed view must contain at least K samples.")
        subgraph = graph[observed][:, observed]
        view_seed = int(seed) + 101 * int(view_id)
        leading_eigenspace(
            subgraph,
            n_components=int(n_clusters) + 1,
            seed=view_seed,
            tolerance=float(tolerance),
        )
        partition = auxiliary_spectral_cluster(
            subgraph,
            n_clusters=int(n_clusters),
            seed=view_seed,
            tolerance=float(tolerance),
        )
        observed_by_view.append(observed)
        partitions.append(partition.astype(np.int64))
    consensus = []
    for view_id in range(n_views):
        agreements = []
        weights = []
        for other in range(n_views):
            if other == view_id:
                continue
            common = np.intersect1d(observed_by_view[view_id], observed_by_view[other])
            if common.size < max(2, int(n_clusters)):
                continue
            left_positions = np.searchsorted(observed_by_view[view_id], common)
            right_positions = np.searchsorted(observed_by_view[other], common)
            agreements.append(
                float(
                    normalized_mutual_info_score(
                        partitions[view_id][left_positions],
                        partitions[other][right_positions],
                        average_method="arithmetic",
                    )
                )
            )
            weights.append(int(common.size))
        value = float(np.average(agreements, weights=weights)) if agreements else 1.0
        consensus.append(max(value, float(epsilon)))
    return np.asarray(consensus, dtype=np.float64)


def selective_utilization(
    feature_dimensions: list[int],
    reconstruction_errors: np.ndarray,
    partition_consensus: np.ndarray,
    gate_min: float = 0.50,
    gate_max: float = 1.50,
    epsilon: float = 1.0e-8,
) -> tuple[float, str]:
    """Return the label-free interpolation exponent for view-quality gates."""

    dimensions = np.asarray(feature_dimensions, dtype=np.int64)
    errors = np.asarray(reconstruction_errors, dtype=np.float64)
    consensus = np.asarray(partition_consensus, dtype=np.float64)
    if not (len(dimensions) == len(errors) == len(consensus)):
        raise ValueError("View-quality inputs must have matching lengths.")
    n_views = len(dimensions)
    comparable = bool(np.all(dimensions == dimensions[0]))
    quality_ratio = float(np.max(errors) / max(float(np.min(errors)), epsilon))
    dynamic_range = float(gate_max) / float(gate_min)
    decisive = bool(quality_ratio >= dynamic_range)
    eligible = bool(comparable or decisive)
    contrast = float(
        np.clip(
            math.log(max(quality_ratio, 1.0)) / math.log(dynamic_range),
            0.0,
            1.0,
        )
    )
    if n_views <= 2:
        if eligible:
            reason = "two_view_equal_dimension" if comparable else "two_view_decisive_quality_ratio"
            return 1.0, reason
        return 0.0, "two_view_heterogeneous_abstention"
    quality = 1.0 / np.clip(errors, epsilon, None)
    quality /= max(float(np.sum(quality)), epsilon)
    structural = np.clip(consensus, epsilon, None)
    structural /= max(float(np.sum(structural)), epsilon)
    uniform = np.full(n_views, 1.0 / n_views, dtype=np.float64)
    quality_centered = quality - uniform
    structural_centered = structural - uniform
    denominator = float(
        np.linalg.norm(quality_centered) * np.linalg.norm(structural_centered)
    )
    cosine = (
        float(np.dot(quality_centered, structural_centered) / max(denominator, epsilon))
        if denominator > epsilon
        else 0.0
    )
    alignment = max(cosine, 0.0)
    kl = float(
        np.sum(structural * np.log(np.clip(structural / uniform, epsilon, None)))
    )
    concentration = math.sqrt(max(kl / max(math.log(float(n_views)), epsilon), 0.0))
    alpha = float(np.clip(float(eligible) * contrast * alignment * concentration, 0.0, 1.0))
    if not eligible:
        reason = "multi_view_ineligible_abstention"
    elif alignment <= epsilon:
        reason = "multi_view_nonpositive_alignment_abstention"
    elif alpha <= 0.05:
        reason = "multi_view_low_evidence_shrinkage"
    else:
        reason = "multi_view_continuous_utilization"
    return alpha, reason


def edge_gate_factors(
    support: dict[Edge, dict[int, float]],
    gates: np.ndarray,
) -> dict[Edge, float]:
    return {
        edge: float(np.mean([float(gates[int(view_id)]) for view_id in sources]))
        for edge, sources in support.items()
    }


def multiply_scores(
    base: dict[Edge, float],
    *factors: dict[Edge, float],
) -> dict[Edge, float]:
    return {
        edge: float(value * math.prod(float(factor[edge]) for factor in factors))
        for edge, value in base.items()
    }


def log_geodesic_graph(
    rank_graph: sp.csr_matrix,
    raw_gate_graph: sp.csr_matrix,
    alpha: float,
) -> sp.csr_matrix:
    """Interpolate positive weights while preserving the sparse support."""

    left = rank_graph.tocsr(copy=True)
    right = raw_gate_graph.tocsr(copy=True)
    left.sort_indices()
    right.sort_indices()
    if not np.array_equal(left.indptr, right.indptr) or not np.array_equal(
        left.indices, right.indices
    ):
        raise ValueError("Graphs must have identical support.")
    if np.any(left.data <= 0.0) or np.any(right.data <= 0.0):
        raise ValueError("Interpolation requires positive edge weights.")
    if float(alpha) <= 1.0e-15:
        return left
    if float(alpha) >= 1.0 - 1.0e-15:
        return right
    output = left.copy()
    output.data = left.data * np.exp(
        float(alpha) * np.log(right.data / left.data)
    )
    output.eliminate_zeros()
    return output


def build_fixed_support_graph(
    views: list[np.ndarray],
    mask: np.ndarray,
    n_clusters: int,
    seed: int,
    *,
    k_pool: int = 45,
    k_sigma: int = 7,
    k_boundary: int = 30,
    adaptive_k0: int = 25,
    adaptive_k_min: int = 10,
    adaptive_k_max: int = 35,
    similarity_floor: float = 0.01,
    eta_one: float = 0.70,
    lambda_a: float = 0.50,
    rank_outer_factor: float = 0.50,
    view_quality_neighbors: int = 5,
    gate_min: float = 0.50,
    gate_max: float = 1.50,
    tolerance: float = 1.0e-4,
    epsilon: float = 1.0e-8,
) -> FixedSupportGraph:
    """Construct the final reliability-weighted graph on admitted support."""

    view_graphs, scaled = build_rank_view_graphs(
        views,
        mask,
        k_pool=k_pool,
        k_sigma=k_sigma,
        k_boundary=k_boundary,
        similarity_floor=similarity_floor,
    )
    limits = adaptive_view_limits(
        mask,
        k0=adaptive_k0,
        k_min=adaptive_k_min,
        k_max=adaptive_k_max,
    )
    support = support_from_view_limits(view_graphs, limits)
    if not support:
        raise ValueError("Observed-view neighborhoods produced an empty support.")
    raw_context = context_confidence(
        support,
        view_graphs,
        limits,
        eta_one=eta_one,
        similarity_floor=similarity_floor,
        epsilon=epsilon,
    )
    reliability = calibrate_context_confidence(raw_context, support, mask)
    context = context_weights(
        support,
        reliability,
        lambda_a=lambda_a,
        lower=0.50,
        upper=1.50,
        epsilon=epsilon,
    )
    rank = rank_taper_factors(
        support,
        view_graphs,
        limits,
        outer_factor=rank_outer_factor,
    )
    errors, gates = observed_view_quality(
        scaled,
        mask,
        view_graphs,
        neighbors=view_quality_neighbors,
        lower=gate_min,
        upper=gate_max,
        epsilon=epsilon,
    )
    consensus = observed_view_partition_consensus(
        support,
        mask,
        n_clusters=n_clusters,
        seed=seed,
        tolerance=tolerance,
        epsilon=epsilon,
    )
    alpha, reason = selective_utilization(
        [int(values.shape[1]) for values in scaled],
        errors,
        consensus,
        gate_min=gate_min,
        gate_max=gate_max,
        epsilon=epsilon,
    )
    rank_scores = multiply_scores(context, rank)
    gate_factors = edge_gate_factors(support, gates)
    raw_gate_scores = multiply_scores(context, rank, gate_factors)
    rank_graph = build_weighted_graph(rank_scores, mask.shape[0])
    raw_gate_graph = build_weighted_graph(raw_gate_scores, mask.shape[0])
    graph = log_geodesic_graph(rank_graph, raw_gate_graph, alpha)
    if graph_edge_set(graph) != set(support):
        raise RuntimeError("Reliability weighting changed the admitted support.")
    if count_observed_support_violations(graph, mask) != 0:
        raise RuntimeError("Graph contains a relation without co-observed evidence.")
    return FixedSupportGraph(
        adjacency=graph,
        support=support,
        adaptive_limits=limits,
        context_reliability=reliability,
        context_weights=context,
        rank_factors=rank,
        view_errors=errors,
        view_gates=gates,
        view_consensus=consensus,
        utilization_alpha=alpha,
        utilization_reason=reason,
    )
