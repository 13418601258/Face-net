"""Create a manual landmark template for final AST-Face standardized meshes."""

import csv
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "code/reports/s3_manual_pilot/manual_sparse_pairs_official.csv"
PAIRS = (("ast001", "happy"), ("ast001", "surprise"),
         ("ast016", "happy"), ("ast016", "surprise"))
ANCHORS = ("左眼外眼角", "左眼内眼角", "右眼内眼角", "右眼外眼角",
           "鼻尖", "左嘴角", "右嘴角", "下巴尖")


def main():
    rows = []
    for subject, expression in PAIRS:
        for anchor in ANCHORS:
            rows.append({
                "pair_id": f"{subject}_au0_to_{expression}",
                "split": "development" if int(subject[-3:]) <= 15 else "held_out",
                "subject": subject,
                "source_mesh": f"data/ASTFace_Public/{subject}/Standardized Meshes/{subject}_au0.obj",
                "target_mesh": f"data/ASTFace_Public/{subject}/Standardized Meshes/{subject}_{expression}.obj",
                "semantic_anchor": anchor,
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
