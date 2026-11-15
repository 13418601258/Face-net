"""Run controlled corruption tests for the S1 deterministic geometry probes."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from audit_io import ASTFaceReader, read_obj, sha256_file
from s1_probes import face_geometry, local_jacobian
from self_intersection import detect_self_intersections


PROJECT = Path(__file__).resolve().parents[1]
ERRORS = ("local_collapse", "local_fold", "local_permutation", "smooth_drift",
          "duplicate_mapping", "coverage_missing")
LEVELS = (1, 2, 3)
GEOMETRIC_ALERTS = ("degenerate", "normal_reversal", "large_normal_change", "strain")
ALERTS = GEOMETRIC_ALERTS + ("duplicate_mapping", "coverage_missing")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fast_local_jacobian(reference, target, faces):
    """S1-equivalent triangle probes using closed-form 2x2 singular values."""
    _, re1, re2, rarea, rn = face_geometry(reference, faces)
    _, te1, te2, tarea, tn = face_geometry(target, faces)
    source_len = np.linalg.norm(re1, axis=1)
    target_len = np.linalg.norm(te1, axis=1)
    source_u = np.divide(re1, source_len[:, None], out=np.zeros_like(re1), where=source_len[:, None] > 0)
    target_u = np.divide(te1, target_len[:, None], out=np.zeros_like(te1), where=target_len[:, None] > 0)
    source_v = np.cross(rn, source_u)
    target_v = np.cross(tn, target_u)
    s01 = np.einsum("ij,ij->i", re2, source_u)
    s11 = np.einsum("ij,ij->i", re2, source_v)
    t01 = np.einsum("ij,ij->i", te2, target_u)
    t11 = np.einsum("ij,ij->i", te2, target_v)
    valid = (np.abs(source_len * s11) > 1e-30) & (rarea > 0) & (tarea > 0)
    j00 = np.divide(target_len, source_len, out=np.zeros_like(target_len), where=source_len > 0)
    j01 = np.divide(t01 - j00 * s01, s11, out=np.zeros_like(t01), where=np.abs(s11) > 1e-30)
    j11 = np.divide(t11, s11, out=np.zeros_like(t11), where=np.abs(s11) > 1e-30)
    gram00 = j00 * j00
    gram01 = j00 * j01
    gram11 = j01 * j01 + j11 * j11
    discriminant = np.sqrt(np.maximum((gram00 - gram11) ** 2 + 4 * gram01 ** 2, 0))
    sigma_max = np.sqrt(np.maximum((gram00 + gram11 + discriminant) / 2, 0))
    sigma_min = np.sqrt(np.maximum((gram00 + gram11 - discriminant) / 2, 0))
    area_ratio = np.divide(tarea, rarea, out=np.full(len(faces), np.nan), where=rarea > 0)
    normal_dot = np.clip(np.einsum("ij,ij->i", rn, tn), -1.0, 1.0)
    normal_angle = np.arccos(normal_dot)
    normal_angle[normal_dot > 1.0 - 1e-12] = 0
    edge_ratios = np.column_stack((target_len / np.maximum(source_len, 1e-30),
                                   np.linalg.norm(te2, axis=1) / np.maximum(np.linalg.norm(re2, axis=1), 1e-30),
                                   np.linalg.norm(te2 - te1, axis=1) /
                                   np.maximum(np.linalg.norm(re2 - re1, axis=1), 1e-30)))
    return {"sigma_min": sigma_min, "sigma_max": sigma_max, "area_ratio": area_ratio,
            "normal_dot": normal_dot, "normal_angle_rad": normal_angle,
            "edge_ratio_min": np.min(edge_ratios, axis=1), "edge_ratio_max": np.max(edge_ratios, axis=1),
            "reference_area": rarea, "target_area": tarea, "target_normals": tn,
            "reference_normals": rn, "valid": valid}


def make_neighbors(vertex_count, faces):
    neighbors = [set() for _ in range(vertex_count)]
    for tri in faces:
        a, b, c = map(int, tri)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [np.asarray(sorted(items), dtype=np.int32) for items in neighbors]


def choose_center(subject, family, neighbors, seed):
    digest = hashlib.sha256(f"{seed}:{subject}:{family}".encode("ascii")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    candidates = np.flatnonzero(np.fromiter((len(x) >= 5 for x in neighbors), dtype=bool, count=len(neighbors)))
    return int(rng.choice(candidates))


def geodesic_patch(center, rings, neighbors):
    distance = {int(center): 0}
    frontier = [int(center)]
    for depth in range(1, rings + 1):
        next_frontier = []
        for vertex in frontier:
            for neighbor in neighbors[vertex]:
                neighbor = int(neighbor)
                if neighbor not in distance:
                    distance[neighbor] = depth
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    vertices = np.fromiter(distance.keys(), dtype=np.int32)
    depths = np.fromiter(distance.values(), dtype=np.int16)
    return vertices, depths


def local_frame(vertices, faces, center, patch_vertices):
    used = np.isin(faces, patch_vertices).any(axis=1)
    _, _, _, _, normals = face_geometry(vertices, faces)
    local_normals = normals[used]
    normal = np.sum(local_normals, axis=0)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        normal = normals[np.flatnonzero(used)[0]]
        norm = np.linalg.norm(normal)
    normal = normal / max(norm, 1e-30)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    tangent = np.cross(normal, axis)
    tangent /= np.linalg.norm(tangent)
    tangent2 = np.cross(normal, tangent)
    tangent2 /= np.linalg.norm(tangent2)
    return normal, tangent, tangent2


def apply_corruption(target, faces, neighbors, subject, family, level, seed):
    result = target.copy()
    rings = {"local_collapse": (1, 2, 3), "local_fold": (2, 3, 4),
             "local_permutation": (1, 2, 3), "smooth_drift": (7, 7, 7),
             "duplicate_mapping": (1, 2, 3), "coverage_missing": (1, 2, 3)}[family][level - 1]
    center = choose_center(subject, family, neighbors, seed)
    patch, depths = geodesic_patch(center, rings, neighbors)
    changed = np.zeros(len(target), dtype=bool)
    changed[patch] = True
    mapped_index = np.arange(len(target), dtype=np.int64)
    mapping_valid = np.ones(len(target), dtype=bool)
    normal, tangent, tangent2 = local_frame(target, faces, center, patch)
    median_edge = np.median(np.concatenate((
        np.linalg.norm(target[faces[:, 1]] - target[faces[:, 0]], axis=1),
        np.linalg.norm(target[faces[:, 2]] - target[faces[:, 1]], axis=1),
        np.linalg.norm(target[faces[:, 0]] - target[faces[:, 2]], axis=1),
    )))
    if family == "local_collapse":
        result[patch] = np.mean(target[patch], axis=0)
    elif family == "local_fold":
        # Reflect one tangent coordinate, reversing local parameter direction.
        factor = (0.25, 0.5, 1.0)[level - 1]
        origin = target[center]
        coordinate = (target[patch] - origin) @ tangent2
        result[patch] = target[patch] - (2.0 * factor * coordinate)[:, None] * tangent2
    elif family == "local_permutation":
        digest = hashlib.sha256(f"{seed}:{subject}:{family}:{level}".encode("ascii")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        permutation = rng.permutation(len(patch))
        if np.array_equal(permutation, np.arange(len(patch))) and len(patch) > 1:
            permutation[[0, 1]] = permutation[[1, 0]]
        result[patch] = target[patch[permutation]]
    elif family == "smooth_drift":
        amplitude = (0.5, 1.0, 2.0)[level - 1] * median_edge
        sigma = max(2.0, rings * 0.45)
        weights = np.exp(-0.5 * (depths.astype(np.float64) / sigma) ** 2)
        result[patch] += (amplitude * weights)[:, None] * tangent
    elif family == "duplicate_mapping":
        mapped_index[patch] = mapped_index[center]
        result = target[mapped_index]
    elif family == "coverage_missing":
        mapping_valid[patch] = False
    affected_faces = np.any(changed[faces], axis=1)
    return result, changed, affected_faces, {"center_vertex": center, "patch_vertices": int(len(patch)),
                                             "patch_faces": int(np.sum(affected_faces)),
                                             "patch_rings": rings, "median_edge_native": float(median_edge)}, \
        mapped_index, mapping_valid


def face_alerts(metrics, vertices, faces, mapping_valid=None, mapped_index=None):
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    target_twice_area = metrics["target_area"] * 2
    degenerate = target_twice_area <= 1e-12 * max(1.0, diagonal) ** 2
    alerts = {
        "degenerate": degenerate,
        "normal_reversal": metrics["normal_dot"] < 0,
        "large_normal_change": metrics["normal_angle_rad"] > np.pi / 4,
        "strain": (metrics["sigma_min"] < 0.5) | (metrics["sigma_max"] > 2.0) |
                  (metrics["area_ratio"] < 0.5) | (metrics["area_ratio"] > 2.0),
    }
    if mapping_valid is not None:
        alerts["coverage_missing"] = np.any(~mapping_valid[faces], axis=1)
    # Repeated target indices indicate a many-to-one vertex correspondence.
    if mapped_index is not None:
        valid_indices = mapped_index[mapping_valid]
        counts = np.bincount(valid_indices, minlength=len(mapped_index))
        duplicate_vertices = mapping_valid & (counts[mapped_index] > 1)
        alerts["duplicate_mapping"] = np.any(duplicate_vertices[faces], axis=1)
    return alerts


def evaluate_mapping(subject, state, split, reference, target, mapped, faces, changed_vertices, affected_faces,
                     case_type, family="", level=0, metadata=None, baseline_alerts=None,
                     mapped_index=None, mapping_valid=None):
    metrics = fast_local_jacobian(reference, mapped, faces)
    mapping_valid = np.ones(len(mapped), dtype=bool) if mapping_valid is None else mapping_valid
    mapped_index = np.arange(len(mapped), dtype=np.int64) if mapped_index is None else mapped_index
    alerts = face_alerts(metrics, mapped, faces, mapping_valid, mapped_index)
    incremental_alerts = {name: (alert & ~baseline_alerts[name]) if baseline_alerts else alert
                          for name, alert in alerts.items()}
    if family == "coverage_missing":
        for name in ("degenerate", "normal_reversal", "large_normal_change", "strain"):
            incremental_alerts[name] = np.zeros(len(faces), dtype=bool)
    vertex_error = np.linalg.norm(mapped - target, axis=1)
    vertex_error[~mapping_valid] = np.nan
    face_error = np.max(vertex_error[faces], axis=1)
    affected_vertex_error = vertex_error[changed_vertices]
    finite_affected_error = affected_vertex_error[np.isfinite(affected_vertex_error)]
    valid_vertex_error = vertex_error[np.isfinite(vertex_error)]
    correspondence_error_available = bool(np.all(mapping_valid))
    row = {
        "sample_id": f"{subject}_{state}", "subject": subject, "state": state, "split": split,
        "case_type": case_type, "error_family": family, "severity": level,
        "changed_vertices": int(np.sum(changed_vertices)), "affected_faces": int(np.sum(affected_faces)),
        "correspondence_error_applicable": correspondence_error_available,
        "correspondence_vertex_error_mean_native": float(np.mean(valid_vertex_error)) if correspondence_error_available else None,
        "correspondence_vertex_error_q95_native": float(np.quantile(valid_vertex_error, .95)) if correspondence_error_available else None,
        "correspondence_vertex_error_max_native": float(np.max(valid_vertex_error)) if correspondence_error_available else None,
        "affected_vertex_error_mean_native": float(finite_affected_error.mean()) if len(finite_affected_error) else None,
        "affected_vertex_error_q95_native": float(np.quantile(finite_affected_error, .95)) if len(finite_affected_error) else None,
        "affected_vertex_error_max_native": float(finite_affected_error.max()) if len(finite_affected_error) else None,
        "affected_face_fraction": float(np.mean(affected_faces)),
        "mapping_coverage_fraction": float(np.mean(mapping_valid)),
        "unique_target_coverage_fraction": float(len(np.unique(mapped_index[mapping_valid])) / len(mapped_index)),
        "missing_mapping_vertices": int(np.sum(~mapping_valid)),
        "invalid_jacobian_faces": int(np.sum(~metrics["valid"])),
        "normal_reversal_faces": int(np.sum(alerts["normal_reversal"])),
        "normal_angle_over_45_faces": int(np.sum(alerts["large_normal_change"])),
        "degenerate_faces": int(np.sum(alerts["degenerate"])),
        "strain_alert_faces": int(np.sum(alerts["strain"])),
        "sigma_min_area_weighted_mean": float(np.average(metrics["sigma_min"], weights=metrics["reference_area"])),
        "sigma_max_area_weighted_mean": float(np.average(metrics["sigma_max"], weights=metrics["reference_area"])),
        "area_ratio_area_weighted_mean": float(np.average(metrics["area_ratio"], weights=metrics["reference_area"])),
        "normal_angle_rad_area_weighted_mean": float(np.average(metrics["normal_angle_rad"], weights=metrics["reference_area"])),
    }
    for name, alert in incremental_alerts.items():
        row[f"new_{name}_alert_faces"] = (None if family == "coverage_missing" and name in GEOMETRIC_ALERTS
                                           else int(np.sum(alert)))
    if family == "coverage_missing":
        for key in ("invalid_jacobian_faces", "normal_reversal_faces", "normal_angle_over_45_faces",
                    "degenerate_faces", "strain_alert_faces", "sigma_min_area_weighted_mean",
                    "sigma_max_area_weighted_mean", "area_ratio_area_weighted_mean",
                    "normal_angle_rad_area_weighted_mean"):
            row[key] = None
    localization = []
    for name, alert in incremental_alerts.items():
        tp = int(np.sum(alert & affected_faces))
        fp = int(np.sum(alert & ~affected_faces))
        positives = int(np.sum(affected_faces))
        applicable = not (family == "coverage_missing" and name in GEOMETRIC_ALERTS)
        localization.append({
            "sample_id": row["sample_id"], "subject": subject, "state": state, "split": split,
            "case_type": case_type, "error_family": family, "severity": level, "probe": name,
            "probe_applicable": applicable,
            "affected_face_recall": (tp / positives if positives else None) if applicable else None,
            "outside_face_false_positive_rate": (fp / max(1, int(np.sum(~affected_faces)))) if applicable else None,
            "alert_precision_inside_affected_region": (tp / max(1, tp + fp)) if applicable else None,
            "affected_faces": positives, "true_positive_faces": tp, "false_positive_faces": fp,
        })
    return row, localization, metrics, alerts, face_error


def aggregate_curves(rows):
    curves = []
    for family in ERRORS:
        for level in LEVELS:
            selected = [r for r in rows if r["error_family"] == family and int(r["severity"]) == level]
            for split in ("development", "held_out"):
                group = [r for r in selected if r["split"] == split]
                if not group:
                    continue
                curves.append({
                    "error_family": family, "severity": level, "split": split, "sample_count": len(group),
                    "mean_affected_error_q95_native": _mean_non_null(r["affected_vertex_error_q95_native"] for r in group),
                    "mean_error_max_native": _mean_non_null(r["correspondence_vertex_error_max_native"] for r in group),
                    "mean_new_degenerate_alert_faces": _mean_non_null(r["new_degenerate_alert_faces"] for r in group),
                    "mean_new_normal_reversal_alert_faces": _mean_non_null(r["new_normal_reversal_alert_faces"] for r in group),
                    "mean_new_large_normal_change_alert_faces": _mean_non_null(r["new_large_normal_change_alert_faces"] for r in group),
                    "mean_new_strain_alert_faces": _mean_non_null(r["new_strain_alert_faces"] for r in group),
                    "mean_new_duplicate_mapping_alert_faces": _mean_non_null(r["new_duplicate_mapping_alert_faces"] for r in group),
                    "mean_new_coverage_missing_alert_faces": _mean_non_null(r["new_coverage_missing_alert_faces"] for r in group),
                    "mean_affected_face_fraction": float(np.mean([r["affected_face_fraction"] for r in group])),
                })
    return curves


def _mean_non_null(values):
    values = [float(value) for value in values if value not in (None, "")]
    return float(np.mean(values)) if values else None


def aggregate_alerts(localizations):
    groups = {}
    for row in localizations:
        key = (row["split"], row["case_type"], row["error_family"], row["severity"], row["probe"])
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        recalls = [float(row["affected_face_recall"]) for row in group if row["affected_face_recall"] not in (None, "")]
        applicable = [row for row in group if row.get("probe_applicable", True)]
        result.append({
            "split": key[0], "case_type": key[1], "error_family": key[2], "severity": key[3],
            "probe": key[4], "sample_count": len(group),
            "applicable_sample_count": len(applicable),
            "mean_affected_face_recall": _mean_non_null(row["affected_face_recall"] for row in applicable),
            "mean_outside_face_false_positive_rate": _mean_non_null(row["outside_face_false_positive_rate"] for row in applicable),
            "mean_alert_precision_inside_affected_region": _mean_non_null(row["alert_precision_inside_affected_region"] for row in applicable),
        })
    return result


def synthetic_self_intersection_control():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0],
                         [.2, .2, -1], [.2, .2, 1], [.8, .8, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    result = detect_self_intersections(vertices, faces)
    return {"case": "two_nonadjacent_triangles_with_known_crossing",
            "expected_crossing_pairs": 1, "detected_crossing_pairs": len(result["proper_crossing"]),
            "candidate_pairs": result["candidate_pairs"],
            "pass": len(result["proper_crossing"]) == 1}


def mesh_self_intersection_control(mesh, sample_id):
    vertices, faces = mesh.vertices.copy(), mesh.faces
    face_a = 1000
    shared = set(map(int, faces[face_a]))
    face_b = next(i for i in range(2000, len(faces)) if not shared.intersection(map(int, faces[i])))
    edge_scale = float(np.median(np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 0]], axis=1)) * 2)
    center = vertices[faces[face_a]].mean(axis=0)
    vertices[faces[face_a]] = center + edge_scale * np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    vertices[faces[face_b]] = center + edge_scale * np.array([[.2, .2, -1], [.2, .2, 1], [.8, .8, 0]])
    result = detect_self_intersections(vertices, faces)
    detected = {tuple(sorted(map(int, pair))) for pair in result["proper_crossing"]}
    expected = tuple(sorted((face_a, face_b)))
    return {"case": "known_crossing_in_full_AST_mesh", "sample_id": sample_id,
            "expected_face_pair": list(expected), "detected": expected in detected,
            "candidate_pairs": result["candidate_pairs"],
            "proper_crossing_pairs": len(result["proper_crossing"]),
            "pass": expected in detected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/ASTFace_Public")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "code/reports/quality_audit_s2")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--subjects", nargs="*", help="optional subject subset for a quick pilot")
    args = parser.parse_args()
    protocol = json.loads((PROJECT / "code/s0_protocol.json").read_text(encoding="utf-8"))
    subjects = args.subjects or protocol["subjects"]
    states = [state for state in protocol["pilot_states"] if state != protocol["neutral_state"]]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reader = ASTFaceReader(args.data_dir)
    inventory, _ = reader.inventory(subjects)
    rows, localizations = [], []
    for subject_index, subject in enumerate(subjects, 1):
        split = "development" if subject in protocol["development_subjects"] else "held_out"
        neutral = read_obj(inventory[(subject, protocol["neutral_state"])] ["mesh"])
        faces = neutral.faces
        neighbors = make_neighbors(len(neutral.vertices), faces)
        neutral_mask = np.zeros(len(neutral.vertices), dtype=bool)
        # Identity mapping and shared rigid transform are normal controls.
        clean_row, clean_loc, _, _, _ = evaluate_mapping(
            subject, protocol["neutral_state"], split, neutral.vertices, neutral.vertices, neutral.vertices,
            faces, neutral_mask, np.zeros(len(faces), dtype=bool), "clean_neutral")
        rows.append(clean_row)
        localizations.extend(clean_loc)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = np.array([2.3, -1.7, .8])
        rigid_row, rigid_loc, _, _, _ = evaluate_mapping(
            subject, protocol["neutral_state"], split, neutral.vertices @ rotation + translation,
            neutral.vertices @ rotation + translation, neutral.vertices @ rotation + translation,
            faces, neutral_mask, np.zeros(len(faces), dtype=bool), "rigid_invariance_control")
        rigid_row["rigid_feature_max_abs_delta"] = max(
            abs(rigid_row[key] - clean_row[key]) for key in (
                "sigma_min_area_weighted_mean", "sigma_max_area_weighted_mean",
                "area_ratio_area_weighted_mean", "normal_angle_rad_area_weighted_mean"))
        rows.append(rigid_row)
        localizations.extend(rigid_loc)
        for state in states:
            target = read_obj(inventory[(subject, state)]["mesh"]).vertices
            _, _, baseline_metrics, baseline_alerts, _ = evaluate_mapping(
                subject, state, split, neutral.vertices, target, target, faces,
                neutral_mask, np.zeros(len(faces), dtype=bool), "clean_expression_control")
            clean_row, clean_loc, _, _, _ = evaluate_mapping(
                subject, state, split, neutral.vertices, target, target, faces,
                neutral_mask, np.zeros(len(faces), dtype=bool), "clean_expression_control")
            rows.append(clean_row)
            localizations.extend(clean_loc)
            for family in ERRORS:
                for level in LEVELS:
                    mapped, changed, affected, metadata, mapped_index, mapping_valid = apply_corruption(
                        target, faces, neighbors, subject, family, level, args.seed)
                    row, localization, metrics, alerts, face_error = evaluate_mapping(
                        subject, state, split, neutral.vertices, target, mapped, faces,
                        changed, affected, "corruption", family, level, metadata, baseline_alerts,
                        mapped_index, mapping_valid)
                    row.update(metadata)
                    row["affected_correspondence_error_faces"] = (
                        None if family == "coverage_missing" else int(np.sum(face_error[affected] > 0)))
                    rows.append(row)
                    localizations.extend(localization)
        print(f"Processed {subject_index}/{len(subjects)} subjects", flush=True)
    curves = aggregate_curves(rows)
    alert_summary = aggregate_alerts(localizations)
    self_intersection = [synthetic_self_intersection_control()]
    for subject in ("ast005", "ast016"):
        if subject in subjects:
            neutral_mesh = read_obj(inventory[(subject, protocol["neutral_state"])] ["mesh"])
            self_intersection.append(mesh_self_intersection_control(neutral_mesh, f"{subject}_au0"))
    write_csv(output / "sample_results.csv", rows)
    write_csv(output / "localization_results.csv", localizations)
    write_csv(output / "severity_curves.csv", curves)
    write_csv(output / "alert_summary.csv", alert_summary)
    validation = {
        "schema_version": 1, "stage": "S2", "status": "RUN_COMPLETE" if all(x["pass"] for x in self_intersection) else "VALIDATION_FAILED",
        "seed": args.seed, "subjects": subjects, "expression_states": states,
        "split_policy": "subject split from S0; no thresholds tuned on held-out subjects",
        "mapping_assumption": "AST-Face shared vertex indexing is treated as synthetic correspondence oracle only",
        "normal_controls": {"clean_neutral": sum(r["case_type"] == "clean_neutral" for r in rows),
                            "clean_expression": sum(r["case_type"] == "clean_expression_control" for r in rows),
                            "rigid_invariance": sum(r["case_type"] == "rigid_invariance_control" for r in rows)},
        "corruption_counts": {family: sum(r["error_family"] == family for r in rows) for family in ERRORS},
        "sample_count": len(rows), "localization_rows": len(localizations),
        "self_intersection_controls": self_intersection,
        "thresholds": {"degenerate": "twice_area <= 1e-12 * max(1,bbox_diagonal)^2",
                       "normal_reversal": "normal dot < 0 versus source reference",
                       "large_normal_change": "angle > 45 degrees versus source reference",
                       "strain": "sigma_min<0.5 or sigma_max>2 or area_ratio outside [0.5,2]"},
        "limitations": ["Synthetic vertex-index correspondence is not independent semantic ground truth.",
                        "Geometric alert thresholds are fixed diagnostics, not optimized classifiers.",
                        "Smooth drift can preserve local intrinsic geometry and should be read as a known probe blind spot.",
                        "Anatomical mouth/eyelid shift was not isolated because landmark frame and scale are uncalibrated.",
                        "Self-intersection is checked on a known small crossing control, not as a per-corruption scan over all full meshes.",
                        "Held-out identities are evaluated descriptively and are not used to tune thresholds."],
    }
    write_json(output / "s2_validation.json", validation)
    sources = [Path(__file__), Path(__file__).with_name("s1_probes.py"),
               Path(__file__).with_name("self_intersection.py"), Path(__file__).with_name("audit_io.py"),
               Path(__file__).with_name("s0_protocol.json")]
    outputs = {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in sorted(output.iterdir())
               if p.is_file() and p.name != "run_manifest.json"}
    write_json(output / "run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "sources": {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in sources}, "outputs": outputs})
    print(json.dumps({"status": validation["status"], "rows": len(rows),
                      "corruptions": sum(validation["corruption_counts"].values()),
                      "self_intersection_controls": self_intersection, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
