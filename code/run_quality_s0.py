"""Freeze the S0 input contract and audit extracted AST-Face asset integrity."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

from audit_io import ASTFaceReader, read_obj, rigid_transform, sha256_file, topology_hash


PROJECT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)


def component_count(vertex_count, edges, active=None):
    parent = np.arange(vertex_count)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    indices = range(vertex_count) if active is None else active
    return len({root(i) for i in indices})


def connectivity_stats(mesh):
    faces = mesh.faces
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    sorted_edges = np.sort(directed, axis=1)
    edges, inverse, counts = np.unique(sorted_edges, axis=0, return_inverse=True, return_counts=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    orientation_sum = np.bincount(inverse, weights=signs, minlength=len(edges))
    boundary = edges[counts == 1]
    used = np.unique(faces)
    boundary_vertices, degree = np.unique(boundary, return_counts=True)
    return {
        "vertex_count": len(mesh.vertices), "face_count": len(faces), "edge_count": len(edges),
        "boundary_edges": int(np.sum(counts == 1)),
        "boundary_components": component_count(len(mesh.vertices), boundary, boundary_vertices),
        "boundary_vertices_degree_not_two": int(np.sum(degree != 2)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "inconsistent_winding_edges": int(np.sum((counts == 2) & (orientation_sum != 0))),
        "duplicate_faces_ignoring_winding": len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0)),
        "repeated_index_faces": int(np.sum(np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0, axis=1))),
        "unused_vertices": len(mesh.vertices) - len(used),
        "connected_components_including_unused": component_count(len(mesh.vertices), edges),
        "euler_characteristic": len(mesh.vertices) - len(edges) + len(faces),
        "vertex_link_manifoldness": "not_checked; edge-manifoldness alone is insufficient",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("s0_protocol.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text(encoding="utf-8"))
    subjects = config["subjects"]
    development, held_out = set(config["development_subjects"]), set(config["held_out_subjects"])
    if development & held_out or development | held_out != set(subjects):
        raise ValueError("subject splits must be disjoint and cover the selected subjects")
    if config["template_subject"] not in development:
        raise ValueError("template must belong to development subjects")
    output = (PROJECT / (args.output_dir or config["output_dir"])).resolve()
    data_dir = (PROJECT / config["data_dir"]).resolve()
    if output == data_dir or data_dir in output.parents:
        raise ValueError("reports must not be written into the source dataset")
    reader = ASTFaceReader(data_dir)
    inventory, extras = reader.inventory(subjects)
    expected = {(s, state) for s in subjects for state in config["expected_states"]}
    neutral_state = config["neutral_state"]
    template_key = (config["template_subject"], neutral_state)
    template = read_obj(inventory[template_key]["mesh"])
    reference_hash = topology_hash(template)
    topology_cache, transforms, rows, assets_rows, errors = {}, {}, [], [], []
    if not np.isfinite(template.vertices).all() or template.faces.min() < 0 or template.faces.max() >= len(template.vertices):
        raise ValueError("invalid reference mesh")
    neutral_meshes = {}
    for subject in subjects:
        neutral_assets = inventory.get((subject, neutral_state), {})
        if "mesh" not in neutral_assets:
            errors.append({"sample": subject, "error": "missing neutral mesh"})
            continue
        try:
            neutral = read_obj(neutral_assets["mesh"])
            neutral_meshes[subject] = neutral
            if topology_hash(neutral) != reference_hash:
                raise ValueError("neutral topology differs from template; indexed rigid fit disabled")
            rotation, translation = rigid_transform(neutral.vertices, template.vertices)
            aligned = neutral.vertices @ rotation + translation
            inverse_translation = -translation @ rotation.T
            roundtrip = aligned @ rotation.T + inverse_translation
            transforms[subject] = {
                "reference": str(neutral_assets["mesh"].relative_to(PROJECT)).replace("\\", "/"),
                "rotation": rotation.tolist(), "translation": translation.tolist(),
                "inverse_rotation": rotation.T.tolist(), "inverse_translation": inverse_translation.tolist(),
                "det_rotation": float(np.linalg.det(rotation)), "scale": 1.0,
                "neutral_alignment_vertex_rmse_native": float(np.sqrt(np.mean(np.sum((aligned - template.vertices) ** 2, axis=1)))),
                "neutral_roundtrip_max_abs_native": float(np.max(np.abs(roundtrip - neutral.vertices))),
            }
        except (ValueError, OSError) as exc:
            errors.append({"sample": subject, "error": str(exc)})

    for number, (subject, state) in enumerate(sorted(expected | set(inventory)), 1):
        sample_id = subject + "_" + state
        paths = inventory.get((subject, state), {})
        split = "development" if subject in development else "held_out"
        row = {"sample_id": sample_id, "subject": subject, "state": state, "split": split,
               "pilot": state in config["pilot_states"], "source_kind": "dataset_registered_control"}
        issues = []
        for kind, path in sorted(paths.items()):
            assets_rows.append({"sample_id": sample_id, "subject": subject, "state": state, "split": split,
                                "kind": kind, "path": path.relative_to(PROJECT).as_posix(),
                                "bytes": path.stat().st_size, "sha256": sha256_file(path),
                                "source_state": path.stem[len(subject) + 1:]})
        if (subject, state) not in expected:
            issues.append("unexpected_state")
        try:
            sample = reader.read_sample(paths)
            mesh = sample["mesh"]
            if mesh is None:
                raise ValueError("missing mesh")
            v, f = mesh.vertices, mesh.faces
            row.update(vertex_count=len(v), face_count=len(f), finite_coordinates=bool(np.isfinite(v).all()),
                       indices_in_range=bool(f.min() >= 0 and f.max() < len(v)), topology_hash=topology_hash(mesh))
            row["topology_matches_template"] = row["topology_hash"] == reference_hash
            if not row["finite_coordinates"] or not row["indices_in_range"]:
                raise ValueError("nonfinite coordinates or out-of-range face indices")
            if not row["topology_matches_template"]:
                issues.append("topology_mismatch")
            digest = row["topology_hash"]
            if digest not in topology_cache:
                topology_cache[digest] = connectivity_stats(mesh)
            stats = topology_cache[digest]
            for name in ("nonmanifold_edges", "inconsistent_winding_edges", "duplicate_faces_ignoring_winding",
                         "repeated_index_faces", "unused_vertices", "boundary_vertices_degree_not_two"):
                row[name] = stats[name]
                if stats[name]:
                    issues.append(name)
            row["boundary_edges"] = stats["boundary_edges"]
            row["connected_components"] = stats["connected_components_including_unused"]
            if row["connected_components"] != 1:
                issues.append("disconnected_mesh")
            diagonal = float(np.linalg.norm(np.ptp(v, axis=0)))
            twice_area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
            threshold = config["degenerate_twice_area_relative_to_bbox_squared"] * diagonal ** 2
            row.update(bbox_diagonal_native=diagonal, min_twice_area_native_squared=float(twice_area.min()),
                       degeneracy_threshold_native_squared=threshold,
                       degenerate_faces=int(np.sum(twice_area <= threshold)),
                       surface_area_native_squared=float(twice_area.sum() / 2))
            if row["degenerate_faces"]:
                issues.append("degenerate_faces")
            neutral = neutral_meshes.get(subject)
            deformation = sample["deformation"]
            row["deformation_status"] = "not_applicable_neutral" if state == neutral_state else "missing"
            if state != neutral_state:
                if deformation is None:
                    issues.append("missing_deformation")
                elif neutral is None or deformation.shape != v.shape or neutral.vertices.shape != v.shape or not np.isfinite(deformation).all():
                    issues.append("invalid_deformation_or_neutral")
                    row["deformation_status"] = "invalid"
                else:
                    residual = deformation.astype(np.float64) - (v - neutral.vertices)
                    maximum = float(np.max(np.abs(residual)))
                    row["deformation_max_abs_error_native"] = maximum
                    row["deformation_status"] = "pass" if maximum <= config["deformation_absolute_tolerance_native"] else "mismatch"
                    if row["deformation_status"] != "pass":
                        issues.append("deformation_mismatch")
            elif deformation is not None:
                issues.append("unexpected_neutral_deformation")
            landmarks = sample["landmarks"]
            row["landmarks_status"] = "missing"
            if landmarks is None:
                issues.append("missing_landmarks")
            elif landmarks.shape[1] != 3 or not np.isfinite(landmarks).all():
                issues.append("invalid_landmarks")
                row["landmarks_status"] = "invalid"
            else:
                row["landmark_count"] = len(landmarks)
                row["landmark_bbox_to_mesh_bbox_ratio"] = float(np.linalg.norm(np.ptp(landmarks, axis=0)) / diagonal) if diagonal > 0 else None
                row["landmarks_status"] = "readable_frame_unverified_not_ground_truth"
            if subject in transforms:
                transform = transforms[subject]
                r, t = np.array(transform["rotation"]), np.array(transform["translation"])
                aligned = v @ r + t
                restored = aligned @ r.T + np.array(transform["inverse_translation"])
                error = float(np.max(np.abs(restored - v)))
                row["rigid_roundtrip_max_abs_native"] = error
                row["rigid_roundtrip_pass"] = error <= 1e-12 * max(1.0, diagonal)
                if not row["rigid_roundtrip_pass"]:
                    issues.append("rigid_roundtrip_failed")
            else:
                issues.append("alignment_unavailable")
        except (ValueError, OSError, IndexError) as exc:
            issues.append("read_or_validation_error")
            errors.append({"sample": sample_id, "error": str(exc)})
        row["status"] = "flagged" if issues else "pass"
        row["issues"] = "|".join(issues)
        rows.append(row)
        if number % 40 == 0:
            print(f"Checked {number}/{len(expected | set(inventory))} samples", flush=True)

    missing_samples = sorted(subject + "_" + state for subject, state in expected - set(inventory))
    counts = Counter(issue for row in rows for issue in row["issues"].split("|") if issue)
    landmark_ratios = [r["landmark_bbox_to_mesh_bbox_ratio"] for r in rows if r.get("landmark_bbox_to_mesh_bbox_ratio") is not None]
    max_deformation_error = max((r.get("deformation_max_abs_error_native", 0) for r in rows), default=0)
    passed = not errors and not counts and not missing_samples and not extras
    report = {
        "schema_version": 1, "stage": "S0", "status": "PASS" if passed else "REVIEW_REQUIRED",
        "meaning": "input-contract and integrity readiness only; not semantic quality or model performance",
        "summary": {"subjects": len(subjects), "samples": len(rows), "assets": len(assets_rows),
                    "asset_kinds": dict(Counter(a["kind"] for a in assets_rows)),
                    "sample_status": dict(Counter(r["status"] for r in rows)),
                    "split_samples": dict(Counter(r["split"] for r in rows)),
                    "pilot_samples": sum(r["pilot"] for r in rows), "unique_topologies": len(topology_cache),
                    "issue_counts": dict(counts), "missing_samples": missing_samples,
                    "deformation_max_abs_error_native": max_deformation_error,
                    "landmark_bbox_ratio_range": [min(landmark_ratios), max(landmark_ratios)] if landmark_ratios else None},
        "reference_topology_hash": reference_hash, "topologies": topology_cache,
        "errors": errors, "unclassified_assets": [p.relative_to(PROJECT).as_posix() for p in extras],
        "limitations": ["Dataset registration is a provisional reference, not independent correspondence ground truth.",
                        "Landmark physical units and coordinate frame are unverified; no automatic conversion or scoring.",
                        "No real model predictions or forward/backward maps have been supplied.",
                        "No self-intersection, fold or vertex-link manifoldness test in S0.",
                        "Held-out assets are checked for integrity only; no thresholds are fitted on them."]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "s0_protocol.json", config)
    write_json(output / "splits.json", {"development": sorted(development), "held_out": sorted(held_out),
                                      "unit": "subject", "inheritance": "all expressions, perturbations and seeds"})
    write_csv(output / "assets.csv", assets_rows)
    write_csv(output / "sample_checks.csv", rows)
    write_json(output / "transforms.json", transforms)
    write_json(output / "s0_validation.json", report)
    write_json(output / "run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "numpy": np.__version__, "command": sys.argv,
        "sources": {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in
                    [Path(__file__), Path(__file__).with_name("audit_io.py"), args.protocol.resolve()]},
        "outputs": {p.name: sha256_file(p) for p in sorted(output.iterdir())
                    if p.is_file() and p.name != "run_manifest.json"}})
    print(json.dumps({"status": report["status"], "summary": report["summary"], "output": str(output)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
