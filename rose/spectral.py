"""Fixed spectral target construction and coordinate normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as sklearn_normalize
from torch.nn import functional as torch_functional


@dataclass(frozen=True)
class SpectralTarget:
    """Graph coordinates and the frozen target used by the mapper."""

    full_coordinates: np.ndarray
    graph_coordinates: np.ndarray
    first_normalization: np.ndarray
    target: np.ndarray
    eigenvalues: np.ndarray
    graph_partition: np.ndarray


def row_map_eigenvectors(
    values: np.ndarray,
    epsilon_row: float | None = None,
) -> np.ndarray:
    """Apply the zero-safe row map used before selecting coordinates."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Eigenvectors must form a two-dimensional array.")
    if epsilon_row is None:
        return sklearn_normalize(array, norm="l2", axis=1).astype(np.float32)
    threshold = float(epsilon_row)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    denominator = np.where(norms < threshold, 1.0, norms)
    return (array / denominator).astype(np.float32)


def normalize_coordinates(
    values: np.ndarray,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    """Column-center coordinates and normalize each nonzero row."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Coordinates must form a two-dimensional array.")
    centered = array - np.mean(array, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    denominator = np.maximum(norms, float(epsilon))
    return (centered / denominator).astype(np.float32)


def finalize_spectral_target(values: np.ndarray) -> np.ndarray:
    """Apply the mapper-loss normalization used for the frozen target."""

    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    with torch.no_grad():
        centered = tensor - tensor.mean(dim=0, keepdim=True)
        normalized = torch_functional.normalize(centered, dim=1)
    return normalized.numpy().copy()


def normalized_adjacency(adjacency: sp.spmatrix) -> sp.csr_matrix:
    """Return the symmetric degree-normalized adjacency used by ROSE."""

    graph = adjacency.tocsr().astype(np.float64)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be square.")
    graph = graph.maximum(graph.T)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    if np.any(graph.data < 0.0):
        raise ValueError("adjacency must have nonnegative weights.")
    degree = np.asarray(graph.sum(axis=1)).ravel()
    isolated = degree <= 1.0e-12
    if np.any(isolated):
        graph = graph.tolil()
        graph[isolated, isolated] = 1.0
        graph = graph.tocsr()
        degree = np.asarray(graph.sum(axis=1)).ravel()
    inverse_sqrt = np.power(np.clip(degree, 1.0e-12, None), -0.5)
    return (sp.diags(inverse_sqrt) @ graph @ sp.diags(inverse_sqrt)).tocsr()


def leading_eigenspace(
    adjacency: sp.spmatrix,
    n_components: int,
    seed: int,
    tolerance: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and sign-canonicalize the leading normalized-adjacency modes."""

    normalized = normalized_adjacency(adjacency)
    n_samples = normalized.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required.")
    requested = int(n_components)
    if requested < 1:
        raise ValueError("n_components must be positive.")
    k = min(requested, n_samples - 1)
    try:
        rng = np.random.default_rng(int(seed))
        initial = rng.normal(size=n_samples)
        initial /= max(float(np.linalg.norm(initial)), 1.0e-12)
        eigenvalues, eigenvectors = eigsh(
            normalized,
            k=k,
            which="LA",
            tol=float(tolerance),
            v0=initial,
        )
    except Exception:
        if n_samples > 5000:
            raise
        eigenvalues, eigenvectors = np.linalg.eigh(normalized.toarray())
        eigenvalues = eigenvalues[-k:]
        eigenvectors = eigenvectors[:, -k:]
    order = np.argsort(eigenvalues)[::-1]
    coordinates = eigenvectors[:, order].astype(np.float32)
    values = np.asarray(eigenvalues[order], dtype=np.float64)
    if coordinates.shape[1] < requested:
        padding = np.zeros(
            (n_samples, requested - coordinates.shape[1]),
            dtype=np.float32,
        )
        coordinates = np.concatenate([coordinates, padding], axis=1)
    for column in range(coordinates.shape[1]):
        pivot = int(np.argmax(np.abs(coordinates[:, column])))
        if coordinates[pivot, column] < 0.0:
            coordinates[:, column] *= -1.0
    return row_map_eigenvectors(coordinates), values


def build_spectral_target(
    adjacency: sp.spmatrix,
    n_clusters: int,
    seed: int,
    tolerance: float = 1.0e-4,
    kmeans_restarts: int = 20,
) -> SpectralTarget:
    """Construct the graph partition coordinates and twice-normalized target."""

    clusters = int(n_clusters)
    if clusters < 2 or clusters > adjacency.shape[0] - 2:
        raise ValueError("n_clusters must satisfy 2 <= K <= n - 2.")
    full, eigenvalues = leading_eigenspace(
        adjacency,
        n_components=clusters + 1,
        seed=int(seed),
        tolerance=float(tolerance),
    )
    graph_coordinates = full[:, :clusters].copy()
    graph_partition = KMeans(
        n_clusters=clusters,
        n_init=int(kmeans_restarts),
        random_state=int(seed),
    ).fit_predict(graph_coordinates)
    first = normalize_coordinates(graph_coordinates)
    target = finalize_spectral_target(first)
    return SpectralTarget(
        full_coordinates=full,
        graph_coordinates=graph_coordinates,
        first_normalization=first,
        target=target,
        eigenvalues=eigenvalues,
        graph_partition=graph_partition.astype(np.int64),
    )


def spectral_cluster(
    adjacency: sp.spmatrix,
    n_clusters: int,
    seed: int,
    tolerance: float = 1.0e-4,
    kmeans_restarts: int = 20,
) -> np.ndarray:
    """Cluster the first K coordinates after the required K+1 extraction."""

    return build_spectral_target(
        adjacency,
        n_clusters=int(n_clusters),
        seed=int(seed),
        tolerance=float(tolerance),
        kmeans_restarts=int(kmeans_restarts),
    ).graph_partition.copy()


def auxiliary_spectral_cluster(
    adjacency: sp.spmatrix,
    n_clusters: int,
    seed: int,
    tolerance: float = 1.0e-4,
    kmeans_restarts: int = 20,
) -> np.ndarray:
    """Partition a view-specific auxiliary graph from its first K modes."""

    coordinates, _eigenvalues = leading_eigenspace(
        adjacency,
        n_components=int(n_clusters),
        seed=int(seed),
        tolerance=float(tolerance),
    )
    return KMeans(
        n_clusters=int(n_clusters),
        n_init=int(kmeans_restarts),
        random_state=int(seed),
    ).fit_predict(coordinates)
