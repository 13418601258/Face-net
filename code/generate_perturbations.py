"""Generate the reproducible AST-Face perturbation dataset and manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from astface_loader import ASTFaceReader, parse_subjects
from perturb import (
    baseline,
    coordinate_noise,
    random_missing,
    rigid_transform,
    save_npz,
    shuffle_indices,
    smooth_local_deformation,
)


FUNCTIONS = {
    "baseline": baseline,
    "shuffle_indices": shuffle_indices,
    "coordinate_noise": coordinate_noise,
    "smooth_local_deformation": smooth_local_deformation,
    "random_missing": random_missing,
    "rigid_transform": rigid_transform,
}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("subjects", "states", "seeds", "operations"):
        if key not in config:
            raise ValueError(f"missing config field: {key}")
    return config


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, float, int]:
    """Return the stable identity of one planned perturbation sample."""

    return (
        str(record["subject"]),
        str(record["state"]),
        str(record["type"]),
        float(record["strength"]),
        int(record["seed"]),
    )


def _planned_key(subject: str, state: str, kind: str, strength: float, seed: int) -> tuple[str, str, str, float, int]:
    return subject, state, kind, float(strength), int(seed)


def _load_manifest(path: Path) -> tuple[dict[tuple[str, str, str, float, int], dict[str, Any]], int]:
    """Load valid rows and count malformed rows left by an interrupted run."""

    records: dict[tuple[str, str, str, float, int], dict[str, Any]] = {}
    malformed = 0
    if not path.exists():
        return records, malformed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = _record_key(record)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            malformed += 1
            continue
        records[key] = record
    return records, malformed


def _record_from_npz(
    path: Path,
    subject: str,
    state: str,
    kind: str,
    strength: float,
    seed: int,
    input_mesh: str,
    relative: Path,
) -> dict[str, Any]:
    """Rebuild a manifest row when a previous run saved NPZ before its row."""

    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        vertices = archive["vertices"]
        faces = archive["faces"]
        valid_mask = archive["valid_mask"]
        metadata_raw = archive["metadata"].item()
    metadata = json.loads(str(metadata_raw))
    return {
        "subject": subject,
        "state": state,
        "type": kind,
        "strength": float(strength),
        "seed": int(seed),
        "input_mesh": input_mesh,
        "output": str(relative).replace("\\", "/"),
        "vertices": list(vertices.shape),
        "faces": list(faces.shape),
        "valid_vertices": int(valid_mask.sum()),
        "metadata": metadata,
    }


def _write_manifest(path: Path, records: dict[tuple[str, str, str, float, int], dict[str, Any]]) -> None:
    """Atomically compact the manifest and remove duplicate rows."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for key in sorted(records):
            handle.write(json.dumps(records[key], ensure_ascii=False) + "\n")
    temporary.replace(path)


def generate(
    data_dir: Path,
    config_path: Path,
    output_dir: Path,
    max_samples: int | None = None,
    resume: bool = True,
) -> tuple[int, Path]:
    config = load_config(config_path)
    subjects = parse_subjects(config["subjects"])
    states = [str(state) for state in config["states"]]
    seeds = [int(seed) for seed in config["seeds"]]
    operations = config["operations"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    records, malformed = _load_manifest(manifest_path) if resume else ({}, 0)
    generated = 0
    skipped = 0
    repaired = 0

    with ASTFaceReader(data_dir) as reader, manifest_path.open("a", encoding="utf-8") as manifest:
        for subject in subjects:
            for state in states:
                sample = reader.read_sample(subject, state, modalities=("mesh",))
                assert sample.mesh is not None
                for operation in operations:
                    kind = str(operation["type"])
                    if kind not in FUNCTIONS:
                        raise ValueError(f"unknown perturbation type: {kind}")
                    strengths = operation.get("strengths", [0.0])
                    extra = {key: value for key, value in operation.items() if key not in {"type", "strengths"}}
                    for strength_value in strengths:
                        strength = float(strength_value)
                        for seed in seeds:
                            key = _planned_key(subject, state, kind, strength, seed)
                            relative = Path(subject) / state / kind / f"strength_{strength:.6g}" / f"seed_{seed}.npz"
                            destination = output_dir / relative
                            if resume and destination.exists():
                                if key not in records:
                                    record = _record_from_npz(
                                        destination,
                                        subject,
                                        state,
                                        kind,
                                        strength,
                                        seed,
                                        sample.mesh.path,
                                        relative,
                                    )
                                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                                    manifest.flush()
                                    records[key] = record
                                    repaired += 1
                                else:
                                    skipped += 1
                                continue
                            if max_samples is not None and generated >= max_samples:
                                _write_manifest(manifest_path, records)
                                return generated, manifest_path
                            if kind == "baseline":
                                result = baseline(sample.mesh.vertices, sample.mesh.faces)
                            else:
                                result = FUNCTIONS[kind](sample.mesh.vertices, sample.mesh.faces, strength, seed=seed, **extra)
                            save_npz(result, destination)
                            record = {
                                "subject": subject,
                                "state": state,
                                "type": kind,
                                "strength": strength,
                                "seed": seed,
                                "input_mesh": sample.mesh.path,
                                "output": str(relative).replace("\\", "/"),
                                "vertices": list(result.vertices.shape),
                                "faces": list(result.faces.shape),
                                "valid_vertices": int(result.valid_mask.sum()),
                                "metadata": result.metadata,
                            }
                            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                            manifest.flush()
                            records[key] = record
                            generated += 1
                            if generated % 100 == 0:
                                print(json.dumps({"generated": generated, "skipped": skipped, "repaired": repaired}, ensure_ascii=False), flush=True)

    _write_manifest(manifest_path, records)
    print(json.dumps({
        "generated": generated,
        "skipped": skipped,
        "repaired": repaired,
        "malformed_manifest_rows": malformed,
        "total_manifest_rows": len(records),
    }, ensure_ascii=False), flush=True)
    return generated, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-generate AST-Face perturbation samples")
    parser.add_argument("--data-dir", required=True, help="extracted ASTFace_Public directory")
    parser.add_argument("--config", default="code/perturb_config.json")
    parser.add_argument("--output-dir", default="code/reports/perturbations")
    parser.add_argument("--max-samples", type=int, default=None, help="generate at most N new samples")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing manifest and NPZ files")
    args = parser.parse_args()
    count, manifest = generate(
        Path(args.data_dir),
        Path(args.config),
        Path(args.output_dir),
        args.max_samples,
        resume=not args.no_resume,
    )
    print(json.dumps({"generated": count, "manifest": str(manifest)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
