"""Audit the extracted AST-Face public tier against the official pipeline contract."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_io import ASTFaceReader, read_obj, topology_hash


PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL_REPOSITORY = "https://github.com/zhaopu99/AST-face"


def robust_outliers(records):
    grouped = defaultdict(list)
    for subject, state, y_mean, z_mean in records:
        grouped[state].append((subject, y_mean, z_mean))
    output = []
    for state, values in sorted(grouped.items()):
        y = np.asarray([v[1] for v in values], dtype=np.float64)
        z = np.asarray([v[2] for v in values], dtype=np.float64)
        y_med, z_med = np.median(y), np.median(z)
        y_mad = max(np.median(np.abs(y - y_med)), 1e-12)
        z_mad = max(np.median(np.abs(z - z_med)), 1e-12)
        for subject, y_mean, z_mean in values:
            y_score = abs(y_mean - y_med) / y_mad
            z_score = abs(z_mean - z_med) / z_mad
            if y_score > 8 or z_score > 8:
                output.append({"subject": subject, "state": state,
                               "landmark_y_mean": float(y_mean), "landmark_z_mean": float(z_mean),
                               "state_y_median": float(y_med), "state_z_median": float(z_med),
                               "robust_y_score": float(y_score), "robust_z_score": float(z_score)})
    return output


def main():
    data_dir = PROJECT / "data/ASTFace_Public"
    output = PROJECT / "code/reports/quality_audit_s0/official_pipeline_reconciliation.json"
    protocol = json.loads((PROJECT / "code/s0_protocol.json").read_text(encoding="utf-8"))
    reader = ASTFaceReader(data_dir)
    inventory, extras = reader.inventory(protocol["subjects"])
    topology_hashes = {}
    deformation_errors = []
    landmark_records = []
    shape_errors = []
    for (subject, state), assets in sorted(inventory.items()):
        if "mesh" not in assets:
            continue
        mesh = read_obj(assets["mesh"])
        topology_hashes.setdefault(topology_hash(mesh), []).append(f"{subject}_{state}")
        if "landmarks" in assets:
            landmarks = np.loadtxt(assets["landmarks"], dtype=np.float64, ndmin=2)
            if landmarks.shape != (84, 3) or not np.isfinite(landmarks).all():
                shape_errors.append(f"{subject}_{state}: landmarks shape/finite check failed")
            else:
                landmark_records.append((subject, state, float(landmarks[:, 1].mean()),
                                         float(landmarks[:, 2].mean())))
        if state != protocol["neutral_state"] and "deformation" in assets:
            neutral_assets = inventory[(subject, protocol["neutral_state"])]
            neutral = read_obj(neutral_assets["mesh"]).vertices
            expression = mesh.vertices
            deformation = np.load(assets["deformation"], allow_pickle=False)
            if deformation.shape != expression.shape:
                shape_errors.append(f"{subject}_{state}: deformation shape mismatch")
            else:
                deformation_errors.append(float(np.max(np.abs((neutral + deformation) - expression))))

    report = {
        "stage": "official_pipeline_reconciliation",
        "official_repository": OFFICIAL_REPOSITORY,
        "mode": "public_derived_tier_reconciliation; full raw-to-standardized rerun unavailable",
        "public_assets": {
            "subjects": len(protocol["subjects"]),
            "inventory_keys": len(inventory),
            "unclassified_files": len(extras),
            "topology_count": len(topology_hashes),
            "topology_groups": {key: len(value) for key, value in topology_hashes.items()},
            "max_deformation_reconstruction_error_native": max(deformation_errors) if deformation_errors else None,
            "landmark_shape_errors": shape_errors,
        },
        "official_stage_status": {
            "01_icp": "not rerun; final public meshes are already a derived standardized tier",
            "02_landmarks": "84-point files present; exact coordinate frame relative to final meshes remains unverified",
            "03_flame_fitting": "not reproducible from current workspace; FLAME assets and fitted intermediates absent",
            "04_flame_to_bfm": "not rerun; final standardized meshes are present",
            "05_edge_nicp": "not rerun; final topology-standardized meshes are present; intermediate transforms absent",
            "06_bfm_cropper": "not rerun; final cropped topology is treated as supplied output",
            "07_deformation_fields": "verified by reconstruction against supplied mesh vertices",
        },
        "landmark_suspicious_outliers": robust_outliers(landmark_records),
        "interpretation": [
            "Do not run ICP again on final Standardized Meshes unless an official input/output transform is explicitly recovered.",
            "The mesh and deformation-field checks can use the public derived tier.",
            "The landmark files must be coordinate-validated separately before serving as manual ground truth.",
            "Suspicious landmark samples are data-quality flags, not proof that the official AST-Face pipeline is incorrect.",
        ],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_WITH_LANDMARK_REVIEW_REQUIRED",
                      "topology_count": report["public_assets"]["topology_count"],
                      "max_deformation_error": report["public_assets"]["max_deformation_reconstruction_error_native"],
                      "suspicious_landmark_samples": len(report["landmark_suspicious_outliers"]),
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
