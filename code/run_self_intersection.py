"""Run BVH-based self-intersection checks on extracted AST-Face meshes."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from audit_io import ASTFaceReader, read_obj, sha256_file, topology_hash
from self_intersection import detect_self_intersections


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", help="optional subject IDs; default: all S0 subjects")
    parser.add_argument("--states", nargs="*", help="optional states; default: all S0 states")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "code/reports/quality_audit_s1/self_intersection")
    parser.add_argument("--leaf-size", type=int, default=8)
    parser.add_argument("--contact-tolerance-relative", type=float, default=1e-9)
    args = parser.parse_args()
    protocol = json.loads((PROJECT / "code/s0_protocol.json").read_text(encoding="utf-8"))
    subjects = args.subjects or protocol["subjects"]
    states = args.states or protocol["expected_states"]
    reader = ASTFaceReader(PROJECT / protocol["data_dir"])
    inventory, extras = reader.inventory(subjects)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, pair_rows = [], []
    for subject_index, subject in enumerate(subjects, 1):
        for state in states:
            assets = inventory.get((subject, state), {})
            if "mesh" not in assets:
                continue
            mesh = read_obj(assets["mesh"])
            diagonal = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
            tolerance = args.contact_tolerance_relative * diagonal
            result = detect_self_intersections(mesh.vertices, mesh.faces, tolerance, args.leaf_size)
            categories = {name: result[name] for name in ("proper_crossing", "coplanar_overlap", "touch_only")}
            affected = sorted({int(face) for values in categories.values() for pair in values for face in pair})
            row = {
                "sample_id": f"{subject}_{state}", "subject": subject, "state": state,
                "split": "development" if subject in protocol["development_subjects"] else "held_out",
                "mesh_path": assets["mesh"].relative_to(PROJECT).as_posix(),
                "topology_hash": topology_hash(mesh), "face_count": len(mesh.faces),
                "bvh_nodes": result["bvh_nodes"], "candidate_pairs": result["candidate_pairs"],
                "proper_crossing_pairs": len(categories["proper_crossing"]),
                "coplanar_overlap_pairs": len(categories["coplanar_overlap"]),
                "touch_only_pairs": len(categories["touch_only"]),
                "affected_faces": len(affected), "affected_face_fraction": len(affected) / len(mesh.faces),
                "bbox_diagonal_native": diagonal, "contact_tolerance_native": tolerance,
                "status": "candidate_screen_complete",
                "interpretation": "BVH + float64 triangle SAT; review topological/tolerance edge cases before treating as a geometric validity oracle",
            }
            rows.append(row)
            for category, pairs in categories.items():
                for face_a, face_b in pairs:
                    pair_rows.append({"sample_id": row["sample_id"], "category": category,
                                      "face_a": int(face_a), "face_b": int(face_b)})
        print(f"Checked {subject_index}/{len(subjects)} subjects", flush=True)
    with (output / "self_intersection.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    with (output / "intersection_pairs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "category", "face_a", "face_b"])
        writer.writeheader()
        writer.writerows(pair_rows)
    summary = {
        "schema_version": 1, "stage": "S1_self_intersection", "status": "CANDIDATE_SCREEN_COMPLETE",
        "algorithm": "median-split triangle BVH broad phase + float64 triangle SAT and plane-intersection narrow phase",
        "subjects": subjects, "states": states, "sample_count": len(rows),
        "parameters": {"leaf_size": args.leaf_size, "contact_tolerance_relative": args.contact_tolerance_relative},
        "totals": {key: sum(r[key] for r in rows) for key in
                   ("candidate_pairs", "proper_crossing_pairs", "coplanar_overlap_pairs", "touch_only_pairs", "affected_faces")},
        "notes": ["Shared-index triangle pairs are excluded as expected local adjacency.",
                  "AABB is only broad phase; triangles are classified by SAT and explicit plane intersection.",
                  "Floating-point predicates are not exact arithmetic; small-geometry tests are in test_self_intersection.py.",
                  "All detected pairs are written with face indices for visual inspection.",
                  "This detects mesh self-intersection, not correspondence correctness."],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sources = [Path(__file__), Path(__file__).with_name("self_intersection.py"),
               Path(__file__).with_name("audit_io.py"), Path(__file__).with_name("s0_protocol.json")]
    outputs = {p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(output.iterdir()) if p.is_file() and p.name != "run_manifest.json"}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
                "sources": {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in sources},
                "outputs": outputs}
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_count": len(rows), "totals": summary["totals"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
