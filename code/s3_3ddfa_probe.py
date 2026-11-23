"""Run S3 probes on the extracted 3DDFA_V2 outputs.

The 3DDFA meshes use a fixed BFM topology.  We use v00 within each identity
as a reference and fit a similarity pose from the 68 exported landmarks before
computing geometric probes.  The labels produced by this script are features
only; human quality labels are written as a separate blind template.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from audit_io import read_obj, sha256_file, topology_hash
from s1_probes import probe
from s2_point_probes import semantic_probe


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT / "data/3ddfa_outputs"
DEFAULT_OUTPUT = PROJECT / "code/reports/s3_3ddfa"
STATES = ("v00", "v05", "v10")
# Standard 68-point layout: four eye corners, two mouth corners, nose tip,
# chin.  Names deliberately avoid left/right claims because image-side
# conventions differ across 68-point implementations.
ANCHORS_68 = ((36, "eye_corner_36"), (39, "eye_corner_39"),
              (42, "eye_corner_42"), (45, "eye_corner_45"),
              (48, "mouth_corner_48"), (54, "mouth_corner_54"),
              (30, "nose_tip_30"), (8, "chin_8"))
ANCHOR_INDEX_68 = np.asarray([idx - 1 for idx, _ in ANCHORS_68], dtype=np.int64)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        if not rows:
            return
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_similarity(source, target):
    """Fit source @ (scale*R) + t to target, without reflection."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    x, y = source - source_mean, target - target_mean
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    aligned_unscaled = x @ rotation
    scale = float(np.sum(aligned_unscaled * y) / max(np.sum(x * x), 1e-30))
    scale = max(scale, 1e-12)
    transform = scale * rotation
    translation = target_mean - source_mean @ transform
    return transform, translation, scale


def apply_transform(points, transform, translation):
    return np.asarray(points, dtype=np.float64) @ transform + translation


def read_sample(root, sample_id):
    folder = root / sample_id
    required = [folder / "vertices.npy", folder / "faces.npy", folder / "landmarks.npy", folder / "metadata.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{sample_id}: missing {missing}")
    vertices = np.load(folder / "vertices.npy", allow_pickle=False).astype(np.float64)
    faces = np.load(folder / "faces.npy", allow_pickle=False).astype(np.int64)
    landmarks = np.load(folder / "landmarks.npy", allow_pickle=False).astype(np.float64)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return vertices, faces, landmarks, metadata


def scalar_summary(metrics, target_vertices=None, prefix=""):
    area = metrics["reference_area"]
    weights = area / max(float(np.sum(area)), 1e-30)
    result = {}
    for name in ("sigma_min", "sigma_max", "area_ratio", "normal_dot", "normal_angle_rad",
                 "edge_ratio_min", "edge_ratio_max"):
        values = metrics[name]
        finite = np.isfinite(values)
        if not np.any(finite):
            result[prefix + name + "_area_weighted_mean"] = None
            continue
        result[prefix + name + "_area_weighted_mean"] = float(np.sum(weights[finite] * values[finite]) /
                                                             max(float(np.sum(weights[finite])), 1e-30))
    result[prefix + "invalid_jacobian_faces"] = int(np.sum(~metrics["valid"]))
    diagonal = float(np.linalg.norm(np.ptp(target_vertices, axis=0))) if target_vertices is not None else 1.0
    result[prefix + "degenerate_target_faces"] = int(np.sum(metrics["target_area"] <=
                                                           1e-12 * max(1., diagonal) ** 2))
    result[prefix + "normal_reversal_faces"] = int(np.sum(metrics["normal_dot"] < 0))
    result[prefix + "normal_angle_over_45_faces"] = int(np.sum(metrics["normal_angle_rad"] > np.pi / 4))
    result[prefix + "strain_alert_faces"] = int(np.sum((metrics["sigma_min"] < .5) |
                                                        (metrics["sigma_max"] > 2.) |
                                                        (metrics["area_ratio"] < .5) |
                                                        (metrics["area_ratio"] > 2.)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "vertices.npy").exists())
    if not sample_dirs:
        raise SystemExit(f"no 3DDFA sample directories found under {root}")
    grouped = {}
    for folder in sample_dirs:
        stem = folder.name
        if "_" not in stem:
            continue
        identity, state = stem.rsplit("_", 1)
        grouped.setdefault(identity, {})[state] = stem
    if not grouped:
        raise SystemExit("no identity/state folders found")

    topology_groups = {}
    validation_rows = []
    samples = {}
    for identity, states in sorted(grouped.items()):
        for state, sample_id in sorted(states.items()):
            vertices, faces, landmarks, metadata = read_sample(root, sample_id)
            if vertices.shape[1] != 3 or faces.shape[1] != 3 or landmarks.shape != (68, 3):
                raise ValueError(f"{sample_id}: unexpected shapes {vertices.shape}, {faces.shape}, {landmarks.shape}")
            if not np.isfinite(vertices).all() or not np.isfinite(landmarks).all():
                raise ValueError(f"{sample_id}: non-finite vertices or landmarks")
            topology = topology_hash(type("Mesh", (), {"vertices": vertices, "faces": faces})())
            topology_groups.setdefault(topology, []).append(sample_id)
            samples[sample_id] = (vertices, faces, landmarks, metadata)
            validation_rows.append({"sample_id": sample_id, "identity_id": identity, "state": state,
                                   "n_vertices": len(vertices), "n_faces": len(faces),
                                   "n_landmarks": len(landmarks), "topology_hash": topology,
                                   "metadata_status": metadata.get("status"),
                                   "det_score": metadata.get("det_score")})

    feature_rows = []
    label_rows = []
    anchor_rows = []
    for identity, states in sorted(grouped.items()):
        if "v00" not in states:
            continue
        source_id = states["v00"]
        source_v, source_f, source_l, source_meta = samples[source_id]
        for state in ("v05", "v10"):
            if state not in states:
                continue
            target_id = states[state]
            target_v, target_f, target_l, target_meta = samples[target_id]
            if not np.array_equal(source_f, target_f):
                raise ValueError(f"{source_id}/{target_id}: face topology differs")
            transform, translation, scale = fit_similarity(target_l, source_l)
            aligned_v = apply_transform(target_v, transform, translation)
            aligned_l = apply_transform(target_l, transform, translation)
            metrics, geom_summary = probe(source_v, aligned_v, source_f, run_collision=False)
            semantic = semantic_probe(source_l, aligned_l, ANCHOR_INDEX_68)
            row = {"pair_id": f"{source_id}_to_{target_id}", "identity_id": identity,
                   "reference_sample_id": source_id, "target_sample_id": target_id,
                   "target_state": state, "model_name": target_meta.get("model_name"),
                   "alignment_scale_target_to_reference": scale,
                   "alignment_translation_norm": float(np.linalg.norm(translation)),
                   "metadata_det_score": target_meta.get("det_score"),
                   **scalar_summary(metrics, aligned_v), **semantic}
            feature_rows.append(row)
            label_base = {"pair_id": row["pair_id"], "identity_id": identity,
                          "reference_sample_id": source_id, "target_sample_id": target_id,
                          "target_state": state, "input_path": target_meta.get("input_path"),
                          "output_obj": str((root / target_id).with_suffix(".obj")),
                          "reference_quality": "", "reference_note": "",
                          "sample_label": "", "error_type": "", "region": "",
                          "confidence_1_to_5": "", "review_note": ""}
            label_rows.append(label_base)
            for one_based, name in ANCHORS_68:
                source_xyz = " ".join(f"{value:.6f}" for value in source_l[one_based - 1])
                target_xyz = " ".join(f"{value:.6f}" for value in target_l[one_based - 1])
                anchor_rows.append({"pair_id": row["pair_id"], "identity_id": identity,
                                    "reference_sample_id": source_id, "target_sample_id": target_id,
                                    "target_state": state, "anchor_id_68": one_based,
                                    "anchor_name": name, "source_xyz_3ddfa": source_xyz,
                                    "target_xyz_3ddfa": target_xyz, "anchor_consistent": "",
                                    "confidence_1_to_5": "", "review_note": ""})

    write_csv(output / "topology_validation.csv", validation_rows)
    write_csv(output / "s3_3ddfa_probe_features.csv", feature_rows)
    write_csv(output / "manual_pair_labels_template.csv", label_rows)
    write_csv(output / "manual_anchor_labels_template.csv", anchor_rows)
    write_json(output / "manual_label_instructions.json", {
        "blind_protocol": ["Do not inspect s3_3ddfa_probe_features.csv while labeling.",
                           "Review source and target mesh/rendering independently.",
                           "Mark uncertain instead of guessing."],
        "sample_label_values": ["acceptable", "review", "clear_failure", "uncertain"],
        "reference_quality_values": ["acceptable", "review", "clear_failure", "uncertain"],
        "error_type_values": ["correspondence_mismatch", "local_fold", "smooth_drift",
                              "missing_or_coverage", "self_intersection", "other"],
        "region_values": ["eye", "nose", "mouth", "chin", "cheek", "other"],
        "anchor_policy": "The source_xyz_3ddfa and target_xyz_3ddfa columns are filled automatically from landmarks.npy for traceability; only anchor_consistent, confidence and review_note are human fields. Anchor IDs are 36,39,42,45,48,54,30,8.",
        "note": "3DDFA has fixed BFM topology, so this pilot evaluates real reconstruction outputs and implicit template consistency; it is not an explicit learned forward/backward-map benchmark.",
    })
    report = {
        "schema_version": 1, "stage": "S3_3DDFA_probe_pilot", "status": "RUN_COMPLETE",
        "input_dir": str(root), "sample_count": len(samples), "identity_count": len(grouped),
        "pair_count": len(feature_rows), "topology_group_count": len(topology_groups),
        "topology_group_sizes": {key: len(value) for key, value in topology_groups.items()},
        "states": sorted({r["state"] for r in validation_rows}),
        "reference_policy": "within-identity v00; target v05/v10; similarity fit from exported 68 landmarks",
        "semantic_anchor_ids_68": [idx for idx, _ in ANCHORS_68],
        "label_status": "templates_generated; independent human labels not filled",
        "limitations": ["Similarity alignment removes global camera scale/pose and is not a correctness label.",
                        "Fixed BFM vertex numbering is implicit template correspondence; no learned explicit f/g is present.",
                        "Probe features are not quality labels until blind human review is completed."],
    }
    write_json(output / "s3_3ddfa_validation.json", report)
    sources = [Path(__file__), PROJECT / "code/s1_probes.py", PROJECT / "code/s2_point_probes.py", PROJECT / "code/audit_io.py"]
    write_json(output / "run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "sources": {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in sources},
        "outputs": {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "run_manifest.json"},
    })
    print(json.dumps({"status": report["status"], "samples": len(samples),
                      "identities": len(grouped), "pairs": len(feature_rows),
                      "topology_groups": len(topology_groups), "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
