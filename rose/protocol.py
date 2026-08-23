"""Incomplete-sample protocol and observed-data preprocessing."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import io as scipy_io
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class OneViewDropSummary:
    """Summary of an incomplete-sample one-view-drop mask."""

    n_samples: int
    n_views: int
    incomplete_samples: int
    missing_entries: int
    incomplete_sample_ratio: float
    entry_missing_ratio: float
    complete_sample_ratio: float


@dataclass(frozen=True)
class MaskManifestEntry:
    """Identity and checksum of one released one-view-drop mask."""

    dataset: str
    eta: float
    seed: int
    relative_file: str
    shape: tuple[int, int]
    dtype: str
    array_sha256: str


def _mat_view_names(path: Path) -> list[str]:
    inventory = {name for name, _shape, _kind in scipy_io.whosmat(path)}
    if "X" in inventory:
        return ["X"]
    if "data" in inventory:
        return ["data"]
    names: list[str] = []
    view_id = 1
    while f"X{view_id}" in inventory:
        names.append(f"X{view_id}")
        view_id += 1
    if not names:
        raise ValueError(f"No supported view fields were found in {path.name}.")
    return names


def _raw_mat_views(path: Path) -> list[np.ndarray]:
    names = _mat_view_names(path)
    values = scipy_io.loadmat(path, variable_names=names)
    if names == ["X"] and getattr(values["X"], "dtype", None) == np.object_:
        raw = values["X"].ravel()
    elif names == ["data"] and getattr(values["data"], "dtype", None) == np.object_:
        raw = values["data"].ravel()
    else:
        raw = [values[name] for name in names]
    views = [np.asarray(value, dtype=np.float32) for value in raw]
    if not views:
        raise ValueError(f"View fields in {path.name} could not be parsed.")
    return views


def load_multi_view_views(
    data_path: str | Path,
    n_samples: int | None = None,
) -> list[np.ndarray]:
    """Load only feature views from a supported MATLAB matrix.

    Supplying the released mask row count removes any need to inspect a label
    field while orienting the feature matrices.
    """

    path = Path(data_path)
    raw_views = _raw_mat_views(path)
    expected = int(n_samples) if n_samples is not None else None
    if expected is None:
        candidate_sets = [
            {int(value) for value in view.shape if int(value) > 1}
            for view in raw_views
        ]
        common = set.intersection(*candidate_sets)
        if not common:
            raise ValueError(
                "A common sample dimension could not be inferred from the views."
            )
        expected = max(common)
    if expected < 1:
        raise ValueError("n_samples must be positive.")

    views: list[np.ndarray] = []
    for raw_view in raw_views:
        view = (
            raw_view.reshape(raw_view.shape[0], -1)
            if raw_view.ndim > 2
            else raw_view
        )
        if view.ndim != 2:
            raise ValueError(f"View shape {view.shape} is not two-dimensional.")
        if view.shape[0] != expected and view.shape[1] == expected:
            view = view.T
        if view.shape[0] != expected:
            raise ValueError(
                f"View shape {view.shape} is incompatible with n_samples={expected}."
            )
        views.append(np.asarray(view, dtype=np.float32))
    return views


def load_multi_view_labels(
    data_path: str | Path,
    n_samples: int | None = None,
) -> np.ndarray:
    """Load the ground-truth vector for an explicitly requested evaluation."""

    path = Path(data_path)
    inventory = {name for name, _shape, _kind in scipy_io.whosmat(path)}
    label_key = next(
        (
            key
            for key in ("Y", "y", "gt", "labels", "truth", "gnd", "truelabel")
            if key in inventory
        ),
        None,
    )
    if label_key is None:
        raise ValueError(f"No supported label field was found in {path.name}.")
    raw_labels = scipy_io.loadmat(path, variable_names=[label_key])[label_key]
    candidates: list[np.ndarray] = []
    if getattr(raw_labels, "dtype", None) == np.object_:
        for cell in raw_labels.ravel():
            candidate = np.asarray(cell).ravel()
            if candidate.size and np.issubdtype(candidate.dtype, np.number):
                candidates.append(candidate.astype(np.int64))
    else:
        candidates.append(np.asarray(raw_labels, dtype=np.int64).ravel())
    if not candidates:
        raise ValueError(f"Labels in {path.name} could not be parsed.")
    if n_samples is not None:
        matching = [item for item in candidates if item.size == int(n_samples)]
        if not matching:
            raise ValueError(
                f"No label vector in {path.name} has {int(n_samples)} entries."
            )
        labels = matching[0]
    else:
        labels = max(candidates, key=lambda item: int(item.size))
    labels = labels.astype(np.int64)
    if labels.size and int(labels.min()) == 1:
        labels = labels - 1
    return labels


def load_multi_view_mat(data_path: str | Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Load views and labels together for explicit evaluation utilities."""

    labels = load_multi_view_labels(data_path)
    views = load_multi_view_views(data_path, n_samples=int(labels.size))
    return views, labels


def _read_mask_archive(path: str | Path) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as bundle:
        if bundle.files != ["mask"]:
            raise ValueError("A released mask archive must contain only mask.npy.")
        mask = np.asarray(bundle["mask"])
    return mask


def load_mask(path: str | Path) -> np.ndarray:
    """Load one released mask without permitting object arrays."""

    mask = np.asarray(_read_mask_archive(path), dtype=np.float32)
    summarize_one_view_drop_mask(mask)
    return mask


def load_verified_manifest_mask(
    mask_path: str | Path,
    manifest_path: str | Path,
    dataset: str,
    eta: float,
    seed: int,
) -> tuple[np.ndarray, MaskManifestEntry]:
    """Load a mask only after its complete manifest identity is verified."""

    manifest = Path(manifest_path).resolve()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset",
            "eta",
            "seed",
            "file",
            "shape",
            "dtype",
            "array_sha256",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("The mask manifest is missing required columns.")
        matches = [
            row
            for row in reader
            if row["dataset"].lower() == str(dataset).lower()
            and int(row["seed"]) == int(seed)
            and math.isclose(
                float(row["eta"]),
                float(eta),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ]
    if len(matches) != 1:
        raise ValueError("The dataset, eta, and seed do not identify one mask record.")

    row = matches[0]
    relative = Path(row["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("The mask manifest contains an unsafe file path.")
    manifest_root = manifest.parent
    expected_path = (manifest_root / relative).resolve()
    if not expected_path.is_relative_to(manifest_root):
        raise ValueError("The mask manifest file resolves outside the mask directory.")
    supplied_path = Path(mask_path).resolve()
    if supplied_path != expected_path:
        raise ValueError("The supplied mask path does not match the manifest record.")

    shape_parts = row["shape"].lower().split("x")
    if len(shape_parts) != 2:
        raise ValueError("The manifest mask shape is invalid.")
    expected_shape = (int(shape_parts[0]), int(shape_parts[1]))
    raw_mask = _read_mask_archive(supplied_path)
    if raw_mask.shape != expected_shape:
        raise ValueError("The supplied mask shape differs from the manifest.")
    if str(raw_mask.dtype) != row["dtype"]:
        raise ValueError("The supplied mask dtype differs from the manifest.")
    digest = hashlib.sha256(
        np.ascontiguousarray(raw_mask).tobytes(order="C")
    ).hexdigest()
    if digest.lower() != row["array_sha256"].lower():
        raise ValueError("The supplied mask checksum differs from the manifest.")

    mask = np.asarray(raw_mask, dtype=np.float32)
    summary = summarize_one_view_drop_mask(mask)
    if not math.isclose(
        summary.incomplete_sample_ratio,
        float(eta),
        rel_tol=0.0,
        abs_tol=1.0 / max(summary.n_samples, 1),
    ):
        raise ValueError("The supplied mask does not realize the requested eta.")
    entry = MaskManifestEntry(
        dataset=row["dataset"],
        eta=float(row["eta"]),
        seed=int(row["seed"]),
        relative_file=row["file"],
        shape=expected_shape,
        dtype=row["dtype"],
        array_sha256=row["array_sha256"].lower(),
    )
    return mask, entry


def make_one_view_drop_mask(
    n_samples: int,
    n_views: int,
    eta: float,
    seed: int,
) -> np.ndarray:
    """Generate the fixed incomplete-sample protocol used in the study.

    Exactly ``floor(eta * n_samples)`` samples lose one uniformly selected
    view. All remaining samples retain every view.
    """

    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative.")
    if n_views < 2:
        raise ValueError("The one-view-drop protocol requires at least two views.")
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError("eta must lie in [0, 1].")

    rng = np.random.default_rng(int(seed))
    mask = np.ones((int(n_samples), int(n_views)), dtype=np.float32)
    n_incomplete = int(math.floor(float(eta) * int(n_samples)))
    n_incomplete = max(0, min(n_incomplete, int(n_samples)))
    if n_incomplete == 0:
        return mask

    selected = rng.choice(int(n_samples), size=n_incomplete, replace=False)
    for sample_id in selected.astype(int):
        dropped_view = int(rng.integers(0, int(n_views)))
        mask[sample_id, dropped_view] = 0.0
    return mask


def summarize_one_view_drop_mask(mask: np.ndarray) -> OneViewDropSummary:
    """Validate and summarize a binary one-view-drop mask."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a two-dimensional array.")
    if array.shape[1] < 2:
        raise ValueError("mask must contain at least two views.")
    if not np.all(np.isin(array, (0, 1))):
        raise ValueError("mask entries must be binary.")

    n_samples, n_views = array.shape
    observed_per_sample = np.sum(array > 0.5, axis=1)
    missing_per_sample = n_views - observed_per_sample
    if np.any(missing_per_sample > 1):
        raise ValueError("Each incomplete sample must lose exactly one view.")
    if np.any(observed_per_sample < 1):
        raise ValueError("Every sample must retain at least one view.")

    incomplete = missing_per_sample == 1
    incomplete_count = int(np.sum(incomplete))
    missing_entries = int(np.sum(array <= 0.5))
    denominator = max(n_samples * n_views, 1)
    sample_denominator = max(n_samples, 1)
    return OneViewDropSummary(
        n_samples=int(n_samples),
        n_views=int(n_views),
        incomplete_samples=incomplete_count,
        missing_entries=missing_entries,
        incomplete_sample_ratio=float(incomplete_count / sample_denominator),
        entry_missing_ratio=float(missing_entries / denominator),
        complete_sample_ratio=float((n_samples - incomplete_count) / sample_denominator),
    )


def co_observation_adjacency(mask: np.ndarray) -> np.ndarray:
    """Return off-diagonal co-observation eligibility."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a two-dimensional array.")
    observed = array > 0.5
    adjacency = (observed.astype(np.int32) @ observed.astype(np.int32).T) > 0
    np.fill_diagonal(adjacency, False)
    return adjacency


def co_observed_giant_component_ratio(mask: np.ndarray) -> float:
    """Return the largest co-observation component fraction without labels."""

    array = np.asarray(mask)
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError("mask must contain at least one sample row.")
    parent = np.arange(array.shape[0], dtype=np.int64)
    size = np.ones(array.shape[0], dtype=np.int64)

    def find(node: int) -> int:
        while int(parent[node]) != node:
            parent[node] = parent[int(parent[node])]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    for view_id in range(array.shape[1]):
        observed = np.flatnonzero(array[:, view_id] > 0.5).astype(np.int64)
        if observed.size > 1:
            anchor = int(observed[0])
            for sample_id in observed[1:]:
                union(anchor, int(sample_id))
    roots = np.asarray([find(index) for index in range(array.shape[0])])
    counts = np.unique(roots, return_counts=True)[1]
    return float(int(counts.max()) / int(array.shape[0]))


def count_observed_support_violations(
    adjacency: np.ndarray | sp.spmatrix,
    mask: np.ndarray,
) -> int:
    """Count positive off-diagonal edges without co-observed evidence."""

    eligible = co_observation_adjacency(mask)
    if sp.issparse(adjacency):
        if adjacency.shape != eligible.shape:
            raise ValueError("adjacency and mask sample dimensions do not match.")
        upper = sp.triu(adjacency.tocsr(), k=1).tocoo()
        positive = upper.data > 0.0
        if not np.any(positive):
            return 0
        rows = upper.row[positive]
        cols = upper.col[positive]
        return int(np.sum(~eligible[rows, cols]))

    dense = np.asarray(adjacency)
    if dense.shape != eligible.shape:
        raise ValueError("adjacency and mask sample dimensions do not match.")
    positive = np.triu(dense > 0.0, k=1)
    return int(np.sum(positive & ~eligible))


def standardize_observed_views(
    views: list[np.ndarray],
    mask: np.ndarray,
) -> list[np.ndarray]:
    """Fit each standardizer on its observed rows and transform that view."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a two-dimensional array.")
    if len(views) != array.shape[1]:
        raise ValueError("The number of views must match the mask width.")

    scaled: list[np.ndarray] = []
    for view_id, values in enumerate(views):
        view = np.asarray(values, dtype=np.float32)
        if view.ndim != 2 or view.shape[0] != array.shape[0]:
            raise ValueError(f"Invalid shape for view {view_id}: {view.shape}.")
        observed = array[:, view_id] > 0.5
        if int(np.sum(observed)) < 2:
            scaled.append(view.copy())
            continue
        standardizer = StandardScaler().fit(view[observed])
        scaled.append(standardizer.transform(view).astype(np.float32))
    return scaled
