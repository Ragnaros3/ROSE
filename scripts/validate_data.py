from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from scipy.io import whosmat


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark matrices against the release manifest.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "data_sources.csv")
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    failures = []
    for row in rows:
        path = args.data_dir / row["expected_filename"]
        if not path.is_file():
            failures.append(f"{row['dataset_id']}: missing {path.name}")
            continue
        if path.stat().st_size != int(row["byte_size"]):
            failures.append(f"{row['dataset_id']}: byte-size mismatch")
        if sha256_file(path).lower() != row["sha256"].lower():
            failures.append(f"{row['dataset_id']}: SHA-256 mismatch")
        fields = {name for name, _shape, _kind in whosmat(path)}
        required = set(row["feature_fields"].split("|")) | {row["label_field"]}
        missing = sorted(required - fields)
        if missing:
            failures.append(f"{row['dataset_id']}: missing fields {missing}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {len(rows)} benchmark matrices.")


if __name__ == "__main__":
    main()
