"""Verify a generated AST-Face perturbation manifest and its NPZ outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def verify(root: Path, sample_count: int = 20) -> dict:
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [
        (r["subject"], r["state"], r["type"], float(r["strength"]), int(r["seed"]))
        for r in records
    ]
    missing = [r["output"] for r in records if not (root / r["output"]).is_file()]
    sample_records = records[::max(1, len(records) // max(1, sample_count))][:sample_count]
    sample_errors = []
    for record in sample_records:
        with np.load(root / record["output"], allow_pickle=False) as archive:
            if tuple(archive["vertices"].shape) != tuple(record["vertices"]):
                sample_errors.append(record["output"] + ":vertices")
            if tuple(archive["faces"].shape) != tuple(record["faces"]):
                sample_errors.append(record["output"] + ":faces")
            if int(archive["valid_mask"].sum()) != int(record["valid_vertices"]):
                sample_errors.append(record["output"] + ":valid_mask")
    result = {
        "manifest_rows": len(records),
        "unique_keys": len(set(keys)),
        "duplicate_keys": len(keys) - len(set(keys)),
        "missing_outputs": len(missing),
        "npz_files": len(list(root.rglob("*.npz"))),
        "sampled_npz": len(sample_records),
        "sample_errors": len(sample_errors),
    }
    if result["duplicate_keys"] or missing or sample_errors:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify perturbation NPZ files and manifest")
    parser.add_argument("--root", default="code/reports/perturbations")
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(verify(Path(args.root), args.sample_count), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
