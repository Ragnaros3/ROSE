"""Post-hoc clustering evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def normalize_spectral_scores(values: np.ndarray) -> np.ndarray:
    """Column-center scores and normalize each sample row."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional array.")
    centered = array - np.mean(array, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return (centered / np.clip(norms, 1.0e-12, None)).astype(np.float32)


def kmeans_partition(
    scores: np.ndarray,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    """Normalize score rows and compute the fixed K-means partition."""

    if n_clusters < 1:
        raise ValueError("n_clusters must be positive.")
    normalized = normalize_spectral_scores(scores)
    return KMeans(
        n_clusters=int(n_clusters),
        n_init=20,
        random_state=int(seed),
    ).fit_predict(normalized)


def clustering_accuracy(labels: np.ndarray, prediction: np.ndarray) -> float:
    """Compute permutation-invariant clustering accuracy."""

    truth, estimated = _validated_label_vectors(labels, prediction)
    dimension = int(max(int(truth.max()), int(estimated.max())) + 1)
    weights = np.zeros((dimension, dimension), dtype=np.int64)
    for cluster_id, label_id in zip(estimated, truth):
        weights[int(cluster_id), int(label_id)] += 1
    rows, columns = linear_sum_assignment(weights.max() - weights)
    return float(weights[rows, columns].sum() / estimated.size)


def clustering_purity(labels: np.ndarray, prediction: np.ndarray) -> float:
    """Compute cluster purity."""

    truth, estimated = _validated_label_vectors(labels, prediction)
    total = 0
    for cluster_id in np.unique(estimated):
        members = estimated == cluster_id
        total += int(np.bincount(truth[members]).max())
    return float(total / estimated.size)


@dataclass(frozen=True)
class ClusteringScores:
    """External clustering metrics."""

    acc: float
    nmi: float
    ari: float
    purity: float


def clustering_scores(
    labels: np.ndarray,
    prediction: np.ndarray,
) -> ClusteringScores:
    """Compute ACC, NMI, ARI, and purity after training and routing."""

    truth, estimated = _validated_label_vectors(labels, prediction)
    return ClusteringScores(
        acc=clustering_accuracy(truth, estimated),
        nmi=float(
            normalized_mutual_info_score(
                truth,
                estimated,
                average_method="arithmetic",
            )
        ),
        ari=float(adjusted_rand_score(truth, estimated)),
        purity=clustering_purity(truth, estimated),
    )


def _validated_label_vectors(
    labels: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(labels).reshape(-1).astype(np.int64)
    estimated = np.asarray(prediction).reshape(-1).astype(np.int64)
    if truth.size == 0 or truth.shape != estimated.shape:
        raise ValueError("labels and prediction must be nonempty vectors of equal length.")
    if int(truth.min()) < 0 or int(estimated.min()) < 0:
        raise ValueError("labels and prediction must use nonnegative integer identifiers.")
    return truth, estimated
