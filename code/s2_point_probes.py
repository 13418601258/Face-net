"""S2 sparse semantic probes built from the supplied AST-Face landmarks.

The public landmark files are used in their own coordinate frame.  We do not
pretend that their coordinates are already vertex coordinates of the final
standardized meshes.  The probe therefore compares distances and rigidly
aligned residuals among selected anchors, which does not require a unit
conversion or a landmark-to-vertex assignment.  A mesh-face assignment is
reported only as a diagnostic in the output and is never used as a label.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from audit_io import ASTFaceReader, read_obj, sha256_file


PROJECT = Path(__file__).resolve().parents[1]
ANCHORS = (
    (1, "left_inner_eye", "En_L"),
    (5, "left_outer_eye", "Ex_L"),
    (9, "right_inner_eye", "En_R"),
    (13, "right_outer_eye", "Ex_R"),
    (49, "left_mouth_corner", "Ch_L"),
    (55, "right_mouth_corner", "Ch_R"),
    (76, "chin", "Po"),
    (84, "nose_tip", "Nt"),
)
ANCHOR_INDEX = np.asarray([item[0] - 1 for item in ANCHORS], dtype=np.int64)
ERRORS = ("anchor_swap", "anchor_drift", "anchor_missing")
LEVELS = (1, 2, 3)


def protocol_from_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        if not rows:
            return
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def proper_rigid_align(source, target):
    """Return source aligned to target using a no-scale, no-reflection fit."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    translation = target_mean - source_mean @ rotation
    return source @ rotation + translation, rotation, translation


def pairwise_distances(points):
    delta = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(delta, axis=2)


def semantic_probe(source_landmarks, target_landmarks, anchor_index=ANCHOR_INDEX):
    """Compute scale-free sparse semantic consistency features.

    The values are descriptive features, not a correctness label.  Missing or
    non-finite anchors are excluded from every statistic and their count is
    retained explicitly.
    """
    source = np.asarray(source_landmarks, dtype=np.float64)[anchor_index]
    target = np.asarray(target_landmarks, dtype=np.float64)[anchor_index]
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    valid = np.flatnonzero(finite)
    row = {
        "anchor_count": int(len(anchor_index)),
        "valid_anchor_count": int(len(valid)),
        "missing_anchor_count": int(len(anchor_index) - len(valid)),
        "semantic_probe_applicable": bool(len(valid) >= 3),
    }
    for key in ("pairwise_distance_rmse_norm", "pairwise_distance_q95_norm",
                "pairwise_distance_max_norm", "pairwise_relative_rmse",
                "rigid_residual_rmse_norm", "rigid_residual_q95_norm",
                "rigid_residual_max_norm", "anchor_displacement_q95_norm",
                "anchor_displacement_max_norm", "source_anchor_scale_native"):
        row[key] = None
    if len(valid) < 3:
        return row
    source, target = source[valid], target[valid]
    source_pairs = pairwise_distances(source)
    target_pairs = pairwise_distances(target)
    upper = np.triu_indices(len(valid), 1)
    source_d = source_pairs[upper]
    target_d = target_pairs[upper]
    scale = float(np.median(source_d[source_d > 1e-12])) if np.any(source_d > 1e-12) else 0.0
    if scale <= 0:
        row["semantic_probe_applicable"] = False
        return row
    diff = target_d - source_d
    relative = diff / np.maximum(source_d, 0.25 * scale)
    aligned, _, _ = proper_rigid_align(target, source)
    residual = np.linalg.norm(aligned - source, axis=1)
    row.update({
        "pairwise_distance_rmse_norm": float(np.sqrt(np.mean(diff ** 2)) / scale),
        "pairwise_distance_q95_norm": float(np.quantile(np.abs(diff), .95) / scale),
        "pairwise_distance_max_norm": float(np.max(np.abs(diff)) / scale),
        "pairwise_relative_rmse": float(np.sqrt(np.mean(relative ** 2))),
        "rigid_residual_rmse_norm": float(np.sqrt(np.mean(residual ** 2)) / scale),
        "rigid_residual_q95_norm": float(np.quantile(residual, .95) / scale),
        "rigid_residual_max_norm": float(np.max(residual) / scale),
        "anchor_displacement_q95_norm": float(np.quantile(np.linalg.norm(target - source, axis=1), .95) / scale),
        "anchor_displacement_max_norm": float(np.max(np.linalg.norm(target - source, axis=1)) / scale),
        "source_anchor_scale_native": scale,
    })
    return row


def deterministic_rng(seed, subject, state, family, level):
    token = f"{seed}:{subject}:{state}:{family}:{level}".encode("ascii")
    digest = hashlib.sha256(token).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def apply_landmark_corruption(target, subject, state, family, level, seed):
    """Create labelled point-level controls without changing the mesh."""
    result = np.asarray(target, dtype=np.float64).copy()
    changed = np.zeros(len(ANCHOR_INDEX), dtype=bool)
    rng = deterministic_rng(seed, subject, state, family, level)
    anchor_points = result[ANCHOR_INDEX]
    scale = float(np.median(pairwise_distances(anchor_points)[np.triu_indices(len(anchor_points), 1)]))
    if family == "anchor_swap":
        permutation = np.arange(len(ANCHOR_INDEX))
        # One, two, then three disjoint semantic swaps.
        all_pairs = [(0, 1), (4, 5), (2, 3)]
        for left, right in all_pairs[:level]:
            permutation[left], permutation[right] = permutation[right], permutation[left]
        result[ANCHOR_INDEX] = anchor_points[permutation]
        changed[:] = permutation != np.arange(len(ANCHOR_INDEX))
    elif family == "anchor_drift":
        count = level
        chosen = np.arange(count)
        amplitude = (0.05, 0.15, 0.30)[level - 1] * scale
        directions = rng.normal(size=(count, 3))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-30)
        result[ANCHOR_INDEX[chosen]] += amplitude * directions
        changed[chosen] = True
    elif family == "anchor_missing":
        count = level
        chosen = np.arange(count)
        result[ANCHOR_INDEX[chosen]] = np.nan
        changed[chosen] = True
    else:
        raise ValueError(f"unknown landmark corruption: {family}")
    return result, changed


def mesh_projection_diagnostic(mesh, landmarks):
    """Report why raw landmarks are not silently treated as mesh vertices.

    A simple per-axis bounding-box calibration is only a diagnostic.  The
    result is intentionally not consumed by the semantic probe.
    """
    vertices = mesh.vertices
    points = np.asarray(landmarks, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    if not np.any(finite):
        return {"projection_applicable": False, "finite_landmarks": 0, "median_nearest_vertex_distance_native": None}
    p = points[finite]
    p_range = np.ptp(p, axis=0)
    v_range = np.ptp(vertices, axis=0)
    scale = np.divide(v_range, p_range, out=np.ones(3), where=p_range > 1e-12)
    calibrated = (p - p.min(axis=0)) * scale + vertices.min(axis=0)
    # Chunked nearest-vertex distance avoids constructing a large 3-D array.
    nearest = []
    for start in range(0, len(calibrated), 32):
        block = calibrated[start:start + 32]
        nearest.extend(np.sqrt(np.min(np.sum((block[:, None, :] - vertices[None, :, :]) ** 2, axis=2), axis=1)))
    edge = np.concatenate((vertices[mesh.faces[:, 1]] - vertices[mesh.faces[:, 0]],
                           vertices[mesh.faces[:, 2]] - vertices[mesh.faces[:, 1]],
                           vertices[mesh.faces[:, 0]] - vertices[mesh.faces[:, 2]]))
    median_edge = float(np.median(np.linalg.norm(edge, axis=1)))
    median_distance = float(np.median(nearest))
    return {
        "projection_applicable": bool(median_distance <= 2.5 * median_edge),
        "finite_landmarks": int(np.sum(finite)),
        "median_nearest_vertex_distance_native": median_distance,
        "median_mesh_edge_native": median_edge,
        "distance_to_edge_ratio": median_distance / max(median_edge, 1e-30),
        "bbox_scale_x": float(scale[0]), "bbox_scale_y": float(scale[1]), "bbox_scale_z": float(scale[2]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/ASTFace_Public")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "code/reports/quality_audit_s2_points")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()
    protocol = protocol_from_json(PROJECT / "code/s0_protocol.json")
    subjects = args.subjects or protocol["subjects"]
    states = [state for state in protocol["pilot_states"] if state != protocol["neutral_state"]]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reader = ASTFaceReader(args.data_dir)
    inventory, _ = reader.inventory(subjects)
    clean_rows, corruption_rows, stability_rows = [], [], []
    projection_rows = []
    for subject in subjects:
        split = "development" if subject in protocol["development_subjects"] else "held_out"
        neutral_lm = np.loadtxt(inventory[(subject, protocol["neutral_state"])]["landmarks"], dtype=np.float64, ndmin=2)
        neutral_mesh = read_obj(inventory[(subject, protocol["neutral_state"])]["mesh"])
        projection = mesh_projection_diagnostic(neutral_mesh, neutral_lm)
        projection["subject"] = subject
        projection["split"] = split
        projection_rows.append(projection)
        for state in states:
            target_lm = np.loadtxt(inventory[(subject, state)]["landmarks"], dtype=np.float64, ndmin=2)
            features = semantic_probe(neutral_lm, target_lm)
            features.update({"sample_id": f"{subject}_au0_to_{state}", "subject": subject,
                             "state": state, "split": split, "case_type": "clean_expression"})
            clean_rows.append(features)
            for family in ERRORS:
                for level in LEVELS:
                    for seed in (args.seed, args.seed + 1, args.seed + 2):
                        corrupted, changed = apply_landmark_corruption(target_lm, subject, state, family, level, seed)
                        row = semantic_probe(neutral_lm, corrupted)
                        row.update({"sample_id": f"{subject}_au0_to_{state}", "subject": subject,
                                    "state": state, "split": split, "case_type": "corruption",
                                    "error_family": family, "severity": level, "seed": seed,
                                    "changed_anchors": int(np.sum(changed))})
                        corruption_rows.append(row)
    # Controls: exact self comparison and common rigid transformations.
    controls = []
    for subject in subjects:
        split = "development" if subject in protocol["development_subjects"] else "held_out"
        neutral = np.loadtxt(inventory[(subject, "au0")]["landmarks"], dtype=np.float64, ndmin=2)
        controls.append({**semantic_probe(neutral, neutral), "subject": subject, "split": split,
                         "control": "neutral_self"})
        theta = 0.37
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
        moved = neutral @ rotation + np.array([13.0, -7.0, 5.0])
        controls.append({**semantic_probe(neutral, moved), "subject": subject, "split": split,
                         "control": "common_rigid_transform"})
    # Seed and identity summaries are descriptive; no held-out threshold tuning.
    for family in ERRORS:
        for level in LEVELS:
            group = [r for r in corruption_rows if r["error_family"] == family and r["severity"] == level]
            for split in ("development", "held_out"):
                selected = [r for r in group if r["split"] == split and r["semantic_probe_applicable"]]
                for metric in ("pairwise_distance_rmse_norm", "rigid_residual_q95_norm", "pairwise_relative_rmse"):
                    values = np.asarray([r[metric] for r in selected], dtype=np.float64)
                    stability_rows.append({"analysis": "seed_identity_stability", "error_family": family,
                                           "severity": level, "split": split, "metric": metric,
                                           "count": int(len(values)),
                                           "median": float(np.median(values)) if len(values) else None,
                                           "q05": float(np.quantile(values, .05)) if len(values) else None,
                                           "q95": float(np.quantile(values, .95)) if len(values) else None,
                                           "std": float(np.std(values)) if len(values) else None})
    write_json(output / "anchor_definition.json", {
        "schema_version": 1, "anchors": [{"id_1based": i, "name": n, "official_name": o} for i, n, o in ANCHORS],
        "source": "AST-Face supplied 84-point landmark files", "coordinate_policy": "raw landmark frame; no mesh unit conversion",
        "semantic_features": "pairwise distance change and no-scale proper-rigid residual among eight anchors",
        "mesh_projection_policy": "bbox calibration is diagnostic only; projection is not used as semantic truth or a face label",
    })
    write_csv(output / "clean_point_probe.csv", clean_rows)
    write_csv(output / "point_corruption_controls.csv", corruption_rows)
    write_csv(output / "point_controls.csv", controls)
    write_csv(output / "projection_diagnostics.csv", projection_rows)
    write_csv(output / "stability_summary.csv", stability_rows)
    normal = [r for r in controls if r["control"] == "neutral_self"]
    rigid = [r for r in controls if r["control"] == "common_rigid_transform"]
    metrics = ("pairwise_distance_rmse_norm", "rigid_residual_q95_norm", "pairwise_relative_rmse")
    invariance = {m: float(max(abs((a[m] or 0.) - (b[m] or 0.)) for a, b in zip(normal, rigid))) for m in metrics}
    clean_dev = [r["pairwise_distance_rmse_norm"] for r in clean_rows if r["split"] == "development"]
    clean_q95 = float(np.quantile(clean_dev, .95)) if clean_dev else None
    response = {}
    for family in ERRORS:
        response[family] = {}
        for level in LEVELS:
            held = [r["pairwise_distance_rmse_norm"] for r in corruption_rows
                    if r["error_family"] == family and r["severity"] == level
                    and r["split"] == "held_out" and r["semantic_probe_applicable"]]
            response[family][str(level)] = {
                "held_out_count": len(held),
                "held_out_median": float(np.median(held)) if held else None,
                "above_development_clean_q95": float(np.mean(np.asarray(held) > clean_q95))
                if held and clean_q95 is not None else None,
            }
    summary = {
        "schema_version": 1, "stage": "S2_sparse_semantic_probe", "status": "RUN_COMPLETE",
        "subjects": subjects, "states": states, "clean_pair_count": len(clean_rows),
        "corruption_row_count": len(corruption_rows), "control_row_count": len(controls),
        "corruption_families": {family: sum(r["error_family"] == family for r in corruption_rows) for family in ERRORS},
        "normal_control_max_rigid_delta": invariance,
        "normal_control_max_self_metric": {m: float(max(abs(r[m] or 0.) for r in normal)) for m in metrics},
        "projection_applicable_subjects": int(sum(bool(r["projection_applicable"]) for r in projection_rows)),
        "projection_median_distance_edge_ratio": float(np.median([r["distance_to_edge_ratio"] for r in projection_rows])),
        "four_tasks": {
            "T1_anchor_ingestion": {"anchor_count": len(ANCHORS), "clean_expression_pairs": len(clean_rows),
                                     "all_rows_have_eight_anchor_slots": bool(all(r["anchor_count"] == 8 for r in clean_rows))},
            "T2_normal_controls": {"self_controls": len(normal), "rigid_controls": len(rigid),
                                    "max_rigid_delta": invariance},
            "T3_controlled_point_errors": {"threshold_metric": "pairwise_distance_rmse_norm",
                                             "development_clean_q95": clean_q95, "response": response},
            "T4_stability_and_boundary": {"seeds": [args.seed, args.seed + 1, args.seed + 2],
                                           "held_out_subjects": protocol["held_out_subjects"],
                                           "projection_applicable_subjects": int(sum(bool(r["projection_applicable"]) for r in projection_rows)),
                                           "missing_points_are_not_imputed": True},
        },
        "limitations": [
            "Official landmarks are not assumed to be independent semantic ground truth.",
            "Pairwise features preserve landmark-frame shape and do not provide a mesh-face localization without a validated coordinate transform.",
            "Synthetic point corruptions validate probe sensitivity, not performance on real model output.",
            "Missing anchors are excluded and reported as not applicable; no imputation is performed.",
        ],
    }
    write_json(output / "s2_point_validation.json", summary)
    sources = [Path(__file__), PROJECT / "code/s0_protocol.json", PROJECT / "code/audit_io.py"]
    write_json(output / "run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "sources": {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in sources},
        "outputs": {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "run_manifest.json"},
    })
    print(json.dumps({"status": summary["status"], "clean_pairs": len(clean_rows),
                      "corruption_rows": len(corruption_rows), "controls": len(controls),
                      "projection_applicable_subjects": summary["projection_applicable_subjects"],
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
