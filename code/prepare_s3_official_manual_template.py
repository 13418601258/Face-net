"""Create a manual landmark template for final AST-Face standardized meshes."""

import csv
from pathlib import Path

import numpy as np

from audit_io import ASTFaceReader


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "code/reports/s3_manual_pilot/manual_sparse_pairs_official.csv"
PAIRS = (("ast001", "happy"), ("ast001", "surprise"),
         ("ast016", "happy"), ("ast016", "surprise"))
ANCHORS = (
    ("左眼内眼角", 1, "En (L)"),
    ("左眼外眼角", 5, "Ex (L)"),
    ("右眼内眼角", 9, "En (R)"),
    ("右眼外眼角", 13, "Ex (R)"),
    ("左嘴角", 49, "Ch (L)"),
    ("右嘴角", 55, "Ch (R)"),
    ("下巴尖", 76, "Po"),
    ("鼻尖", 84, "Nt"),
)


def main():
    rows = []
    reader = ASTFaceReader(PROJECT / "data/ASTFace_Public")
    inventory, _ = reader.inventory([f"ast{i:03d}" for i in range(1, 21)])
    for subject, expression in PAIRS:
        source_landmarks = np.loadtxt(inventory[(subject, "au0")]["landmarks"], dtype=np.float64)
        target_landmarks = np.loadtxt(inventory[(subject, expression)]["landmarks"], dtype=np.float64)
        for anchor_name, landmark_id, official_name in ANCHORS:
            rows.append({
                "pair_id": f"{subject}_au0_to_{expression}",
                "split": "development" if int(subject[-3:]) <= 15 else "held_out",
                "subject": subject,
                "source_mesh": f"data/ASTFace_Public/{subject}/Standardized Meshes/{subject}_au0.obj",
                "target_mesh": f"data/ASTFace_Public/{subject}/Standardized Meshes/{subject}_{expression}.obj",
                "semantic_anchor": anchor_name,
                "official_landmark_id_1based": landmark_id,
                "official_landmark_name": official_name,
                "source_xyz_official_raw": " ".join(f"{v:.8f}" for v in source_landmarks[landmark_id - 1]),
                "target_xyz_official_raw": " ".join(f"{v:.8f}" for v in target_landmarks[landmark_id - 1]),
                "source_xyz_manual_native": "",
                "target_xyz_manual_native": "",
                "pair_label": "",
                "confidence_1_to_5": "",
                "review_note": "",
            })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
