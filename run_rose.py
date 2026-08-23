"""Run one frozen ROSE setting from the minimal package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rose.pipeline import evaluate_partition, load_frozen_method, run_rose
from rose.protocol import (
    load_multi_view_labels,
    load_multi_view_views,
    load_verified_manifest_mask,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "frozen_method.json"
MASK_MANIFEST_PATH = ROOT / "masks" / "mask_manifest.csv"
DATA_SOURCES_PATH = ROOT / "data_sources.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("bdgp", "fashion", "handwritten", "hdigit", "mnist_usps"),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Load labels only after the partition and routing decisions are fixed.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data_record(dataset: str) -> dict[str, str]:
    with DATA_SOURCES_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset_id",
            "expected_filename",
            "sha256",
            "byte_size",
            "n_samples",
            "n_views",
            "n_clusters",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("data_sources.csv is missing required columns.")
        matches = [row for row in reader if row["dataset_id"] == dataset]
    if len(matches) != 1:
        raise ValueError("The dataset must identify one data-source record.")
    return matches[0]


def verify_data_file(path: Path, record: dict[str, str]) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")
    if resolved.name != record["expected_filename"]:
        raise ValueError("The dataset filename differs from data_sources.csv.")
    if resolved.stat().st_size != int(record["byte_size"]):
        raise ValueError("The dataset byte size differs from data_sources.csv.")
    digest = sha256_file(resolved)
    if digest.lower() != record["sha256"].lower():
        raise ValueError("The dataset SHA-256 differs from data_sources.csv.")
    return digest.lower()


def matching_eta(requested: float, allowed: tuple[float, ...]) -> float:
    matches = [value for value in allowed if abs(value - requested) <= 1.0e-12]
    if len(matches) != 1:
        raise ValueError("eta is not in the frozen protocol configuration.")
    return float(matches[0])


def mask_path(dataset: str, eta: float, seed: int) -> Path:
    filename = f"{dataset}_eta{eta:.2f}_seed{seed}.npz"
    return ROOT / "masks" / "one_view_drop" / filename


def main() -> None:
    args = parse_args()
    settings = load_frozen_method(CONFIG_PATH)
    dataset = str(args.dataset).lower()
    if dataset not in settings.cluster_counts:
        raise ValueError("The dataset is not in the frozen configuration.")
    if int(args.seed) not in settings.protocol_seeds:
        raise ValueError("The seed is not in the frozen protocol configuration.")
    eta = matching_eta(float(args.eta), settings.incomplete_sample_ratios)

    data_record = load_data_record(dataset)
    if int(data_record["n_clusters"]) != int(settings.cluster_counts[dataset]):
        raise ValueError("The cluster count differs between the data and method records.")
    data_digest = verify_data_file(args.data, data_record)

    released_mask_path = mask_path(dataset, eta, int(args.seed))
    mask, mask_record = load_verified_manifest_mask(
        released_mask_path,
        MASK_MANIFEST_PATH,
        dataset,
        eta,
        int(args.seed),
    )
    if mask.shape != (int(data_record["n_samples"]), int(data_record["n_views"])):
        raise ValueError("The released mask shape differs from data_sources.csv.")

    views = load_multi_view_views(args.data, n_samples=int(mask.shape[0]))
    result = run_rose(
        views,
        mask,
        n_clusters=int(settings.cluster_counts[dataset]),
        seed=int(args.seed),
        settings=settings,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "run_outputs.npz",
        prediction=result.prediction,
        base_scores=result.base_scores,
        final_scores=result.final_scores,
        normalized_scores=result.normalized_scores,
        correction_delta=result.correction_delta,
        residual_ratio=result.residual_ratio,
        spectral_target=result.spectral_target.target,
    )

    summary: dict[str, Any] = {
        "dataset": dataset,
        "eta": eta,
        "seed": int(args.seed),
        "data_sha256": data_digest,
        "mask_file": mask_record.relative_file,
        "mask_array_sha256": mask_record.array_sha256,
        "n_samples": int(mask.shape[0]),
        "n_views": int(mask.shape[1]),
        "n_clusters": int(settings.cluster_counts[dataset]),
        "selected_epoch": int(result.mapper_training.selected.epoch),
        "strict_observed_invariant": bool(result.strict_observed_invariant),
        "support_edges": int(len(result.graph.support)),
        "co_observed_giant_ratio": float(result.co_observed_giant_ratio),
        "topology_ceiling": bool(result.topology_ceiling),
        "utilization_alpha": float(result.graph.utilization_alpha),
        "utilization_reason": str(result.graph.utilization_reason),
        "need": asdict(result.need),
        "correction_applied": result.correction_training is not None,
    }
    if args.evaluate:
        labels = load_multi_view_labels(args.data, n_samples=int(mask.shape[0]))
        summary["evaluation"] = asdict(evaluate_partition(result, labels))

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
