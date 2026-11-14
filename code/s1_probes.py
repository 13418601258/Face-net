"""S1 deterministic local geometry probes for extracted AST-Face meshes."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from audit_io import ASTFaceReader, read_obj, sha256_file, topology_hash


PROJECT = Path(__file__).resolve().parents[1]


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        if not rows:
            return
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "min": float(np.min(values)),
            "q01": float(np.quantile(values, .01)), "q05": float(np.quantile(values, .05)),
            "median": float(np.median(values)), "q95": float(np.quantile(values, .95)),
            "q99": float(np.quantile(values, .99)), "max": float(np.max(values)),
            "mean": float(np.mean(values))}


def face_geometry(vertices, faces):
    p0, p1, p2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    cross = np.cross(e1, e2)
    twice_area = np.linalg.norm(cross, axis=1)
    area = twice_area / 2.0
    normals = np.zeros_like(cross)
    valid = twice_area > 0
    normals[valid] = cross[valid] / twice_area[valid, None]
    return p0, e1, e2, area, normals


def local_jacobian(reference, target, faces):
    """Return two singular values and absolute area ratio for each triangle.

    Both maps are surface-to-surface 3x2 maps. The target tangent basis is
    built with the target face normal, so determinant sign is not used as a
    fold oracle; normal changes are reported separately.
    """
    rp0, re1, re2, rarea, rn = face_geometry(reference, faces)
    tp0, te1, te2, tarea, tn = face_geometry(target, faces)
    n = len(faces)
    source = np.zeros((n, 2, 2), dtype=np.float64)
    target_coords = np.zeros((n, 2, 2), dtype=np.float64)
    source_len = np.linalg.norm(re1, axis=1)
    target_len = np.linalg.norm(te1, axis=1)
    source_u = np.divide(re1, source_len[:, None], out=np.zeros_like(re1), where=source_len[:, None] > 0)
    target_u = np.divide(te1, target_len[:, None], out=np.zeros_like(te1), where=target_len[:, None] > 0)
    source_v = np.cross(rn, source_u)
    target_v = np.cross(tn, target_u)
    source[:, 0, 0] = source_len
    source[:, 0, 1] = np.einsum("ij,ij->i", re2, source_u)
    source[:, 1, 1] = np.einsum("ij,ij->i", re2, source_v)
    target_coords[:, 0, 0] = target_len
    target_coords[:, 0, 1] = np.einsum("ij,ij->i", te2, target_u)
    target_coords[:, 1, 1] = np.einsum("ij,ij->i", te2, target_v)
    valid = (np.abs(source[:, 0, 0] * source[:, 1, 1]) > 1e-30) & (rarea > 0) & (tarea > 0)
    jac = np.zeros_like(source)
    jac[valid] = np.einsum("nij,njk->nik", target_coords[valid], np.linalg.inv(source[valid]))
    singular = np.linalg.svd(jac, compute_uv=False)
    sigma_min = singular[:, 1]
    sigma_max = singular[:, 0]
    area_ratio = np.divide(tarea, rarea, out=np.full(n, np.nan), where=rarea > 0)
    normal_dot = np.einsum("ij,ij->i", rn, tn)
    normal_dot = np.clip(normal_dot, -1.0, 1.0)
    normal_angle = np.arccos(normal_dot)
    normal_angle[normal_dot > 1.0 - 1e-12] = 0.0
    edge_ratios = np.column_stack((np.linalg.norm(te1, axis=1) / np.maximum(np.linalg.norm(re1, axis=1), 1e-30),
                                   np.linalg.norm(te2, axis=1) / np.maximum(np.linalg.norm(re2, axis=1), 1e-30),
                                   np.linalg.norm(te2 - te1, axis=1) /
                                   np.maximum(np.linalg.norm(re2 - re1, axis=1), 1e-30)))
    return {"sigma_min": sigma_min, "sigma_max": sigma_max, "area_ratio": area_ratio,
            "normal_dot": normal_dot, "normal_angle_rad": normal_angle,
            "edge_ratio_max": np.max(edge_ratios, axis=1), "edge_ratio_min": np.min(edge_ratios, axis=1),
            "reference_area": rarea, "target_area": tarea,
            "target_normals": tn, "reference_normals": rn, "valid": valid}


def aabb_candidate_screen(vertices, faces, max_pairs=250000):
    """Screen non-adjacent AABB overlaps with a deterministic spatial hash.

    It is deliberately a candidate screen, not an exact self-intersection
    predicate: touching and coplanar triangles are retained for later review.
    """
    tri = vertices[faces]
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    # A few local edge lengths per cell keeps candidate counts bounded while
    # remaining stable across expressions of one identity.
    edge_lengths = np.concatenate((np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                                   np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                                   np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)))
    cell = max(float(np.median(edge_lengths) * 4), diagonal / 256, 1e-12)
    origin = vertices.min(axis=0)
    buckets = {}
    for index in range(len(faces)):
        start = np.floor((lo[index] - origin) / cell).astype(np.int64)
        stop = np.floor((hi[index] - origin) / cell).astype(np.int64)
        for x in range(start[0], stop[0] + 1):
            for y in range(start[1], stop[1] + 1):
                for z in range(start[2], stop[2] + 1):
                    buckets.setdefault((int(x), int(y), int(z)), []).append(index)
    pairs = set()
    for indices in buckets.values():
        if len(indices) > 2000:
            return {"status": "candidate_overflow", "cell_size_native": cell,
                    "bucket_count": len(buckets), "max_bucket": len(indices),
                    "candidate_pairs": None, "nonadjacent_candidate_pairs": None,
                    "truncated": True}
        unique = sorted(set(indices))
        for left, i in enumerate(unique):
            for j in unique[left + 1:]:
                if np.intersect1d(faces[i], faces[j]).size:
                    continue
                if np.all(lo[i] <= hi[j]) and np.all(lo[j] <= hi[i]):
                    pairs.add((i, j))
                    if len(pairs) > max_pairs:
                        return {"status": "candidate_overflow", "cell_size_native": cell,
                                "bucket_count": len(buckets), "max_bucket": max(len(v) for v in buckets.values()),
                                "candidate_pairs": len(pairs), "nonadjacent_candidate_pairs": len(pairs),
                                "truncated": True}
    return {"status": "candidate_screen_complete", "cell_size_native": cell,
            "bucket_count": len(buckets), "max_bucket": max(len(v) for v in buckets.values()),
            "candidate_pairs": len(pairs), "nonadjacent_candidate_pairs": len(pairs), "truncated": False}


def probe(reference, target, faces, run_collision=False):
    metrics = local_jacobian(reference, target, faces)
    valid = metrics["valid"]
    summary = {
        "reference_area_native_squared": float(np.sum(metrics["reference_area"])),
        "target_area_native_squared": float(np.sum(metrics["target_area"])),
        "invalid_jacobian_faces": int(np.sum(~valid)),
        "degenerate_target_faces": int(np.sum(metrics["target_area"] <= 1e-12 * max(1.0, np.linalg.norm(np.ptp(target, axis=0))) ** 2)),
        "normal_reversal_faces": int(np.sum(metrics["normal_dot"] < 0)),
        "normal_angle_over_90_faces": int(np.sum(metrics["normal_dot"] < 0)),
        "normal_angle_over_45_faces": int(np.sum(metrics["normal_angle_rad"] > np.pi / 4)),
        "sigma_min_below_0_5_faces": int(np.sum(metrics["sigma_min"] < .5)),
        "sigma_max_over_2_faces": int(np.sum(metrics["sigma_max"] > 2)),
        "area_ratio_below_0_5_faces": int(np.sum(metrics["area_ratio"] < .5)),
        "area_ratio_over_2_faces": int(np.sum(metrics["area_ratio"] > 2)),
    }
    weights = metrics["reference_area"] / max(np.sum(metrics["reference_area"]), 1e-30)
    for name in ("sigma_min", "sigma_max", "area_ratio", "normal_dot", "normal_angle_rad", "edge_ratio_min", "edge_ratio_max"):
        finite = np.isfinite(metrics[name])
        summary[name] = quantiles(metrics[name][finite])
        summary[name + "_area_weighted_mean"] = float(np.sum(weights[finite] * metrics[name][finite]) / max(np.sum(weights[finite]), 1e-30))
    summary["collision"] = aabb_candidate_screen(target, faces) if run_collision else {
        "status": "not_run", "reason": "candidate screen is optional and not an exact self-intersection test"}
    return metrics, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/ASTFace_Public")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "code/reports/quality_audit_s1")
    parser.add_argument("--all-states", action="store_true", help="also include AU states; default is S0 pilot states")
    parser.add_argument("--collision", action="store_true", help="run the AABB candidate screen")
    args = parser.parse_args()
    protocol = json.loads((PROJECT / "code/s0_protocol.json").read_text(encoding="utf-8"))
    subjects = protocol["subjects"]
    states = protocol["expected_states"] if args.all_states else protocol["pilot_states"]
    reader = ASTFaceReader(args.data_dir)
    inventory, extras = reader.inventory(subjects)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    face_dir = output / "face_metrics"
    face_dir.mkdir(exist_ok=True)
    faces = read_obj(inventory[(protocol["template_subject"], protocol["neutral_state"])] ["mesh"]).faces
    rows, summaries, collision_rows = [], [], []
    for index, subject in enumerate(subjects, 1):
        neutral = read_obj(inventory[(subject, protocol["neutral_state"])] ["mesh"])
        if topology_hash(neutral) != topology_hash(read_obj(inventory[(protocol["template_subject"], protocol["neutral_state"])] ["mesh"])):
            raise ValueError(f"topology mismatch: {subject}")
        for state in states:
            key = (subject, state)
            if key not in inventory:
                continue
            target = read_obj(inventory[key]["mesh"])
            metrics, summary = probe(neutral.vertices, target.vertices, faces, args.collision)
            sample_id = f"{subject}_{state}"
            row = {"sample_id": sample_id, "subject": subject, "state": state,
                   "split": "development" if subject in protocol["development_subjects"] else "held_out",
                   "pilot": state in protocol["pilot_states"], "topology_hash": topology_hash(target),
                   "mapping_available": False, "coverage_status": "not_applicable_no_forward_backward_map",
                   "cycle_status": "not_applicable_no_forward_backward_map"}
            for name, value in summary.items():
                if name != "collision":
                    row[name] = value if not isinstance(value, dict) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            collision_rows.append({"sample_id": sample_id, **summary["collision"]})
            flat = {key: value for key, value in metrics.items() if key not in ("target_normals", "reference_normals")}
            np.savez_compressed(face_dir / f"{sample_id}.npz", **flat)
            row["face_metrics_path"] = (face_dir / f"{sample_id}.npz").relative_to(PROJECT).as_posix()
            row["face_metrics_sha256"] = sha256_file(face_dir / f"{sample_id}.npz")
            rows.append(row)
            summaries.append({"sample_id": sample_id, "subject": subject, "state": state,
                              "split": row["split"], **summary})
        print(f"Probed {index}/{len(subjects)} subjects", flush=True)
    summary_json = {
        "schema_version": 1, "stage": "S1", "status": "PARTIAL",
        "meaning": "deterministic geometric probe outputs; thresholds are descriptive, not quality labels",
        "states": states, "sample_count": len(rows), "collision_mode": "aabb_candidate_screen_only" if args.collision else "not_run",
        "completed_probes": ["surface_jacobian_singular_values", "area_ratio", "edge_length_ratio", "normal_change", "degeneracy"],
        "pending_probes": ["exact_triangle_self_intersection", "coverage", "cycle_consistency"],
        "mapping_dependent": {"coverage": "not_applicable", "cycle_consistency": "not_applicable"},
        "normal_interpretation": "normal reversal and angle are changes against each subject neutral, not proof of a fold; legal facial expression can be large",
        "reference": "each subject's neutral mesh; shared topology is checked, not semantic truth",
        "limitations": ["AABB overlaps are candidates and require exact triangle intersection confirmation.",
                        "No forward/backward predicted mapping is present, so coverage and cycle probes are not run.",
                        "No S1 threshold turns a legal expression into a failure label.",
                        "Landmarks are not used as independent ground truth."],
        "counts": {"development": sum(r["split"] == "development" for r in rows),
                   "held_out": sum(r["split"] == "held_out" for r in rows),
                   "neutral": sum(r["state"] == "au0" for r in rows),
                   "expressions": sum(r["state"] != "au0" for r in rows)},
        "topology_hashes": sorted({r["topology_hash"] for r in rows})
    }
    csv_write(output / "sample_probes.csv", rows)
    state_rows = []
    for state in states:
        selected = [row for row in rows if row["state"] == state]
        state_rows.append({
            "state": state,
            "sample_count": len(selected),
            "split_development": sum(row["split"] == "development" for row in selected),
            "split_held_out": sum(row["split"] == "held_out" for row in selected),
            "sigma_min_area_weighted_mean": float(np.mean([row["sigma_min_area_weighted_mean"] for row in selected])),
            "sigma_max_area_weighted_mean": float(np.mean([row["sigma_max_area_weighted_mean"] for row in selected])),
            "area_ratio_area_weighted_mean": float(np.mean([row["area_ratio_area_weighted_mean"] for row in selected])),
            "normal_angle_rad_area_weighted_mean": float(np.mean([row["normal_angle_rad_area_weighted_mean"] for row in selected])),
            "normal_reversal_faces_sum": sum(row["normal_reversal_faces"] for row in selected),
            "normal_angle_over_45_faces_sum": sum(row["normal_angle_over_45_faces"] for row in selected),
            "sigma_min_below_0_5_faces_sum": sum(row["sigma_min_below_0_5_faces"] for row in selected),
            "sigma_max_over_2_faces_sum": sum(row["sigma_max_over_2_faces"] for row in selected),
            "area_ratio_below_0_5_faces_sum": sum(row["area_ratio_below_0_5_faces"] for row in selected),
            "area_ratio_over_2_faces_sum": sum(row["area_ratio_over_2_faces"] for row in selected),
        })
    csv_write(output / "state_summary.csv", state_rows)
    csv_write(output / "collision_candidates.csv", collision_rows)
    json_write(output / "s1_summary.json", summary_json)
    json_write(output / "run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "sources": {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in
                    [Path(__file__), Path(__file__).with_name("audit_io.py"), Path(__file__).with_name("s0_protocol.json")]},
        "outputs": {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in sorted(output.glob("*"))
                    if p.is_file() and p.name != "run_manifest.json"}
    })
    print(json.dumps({"status": "PASS", "samples": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
