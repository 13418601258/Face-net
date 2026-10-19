#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_bfm_cropper.py — Crop BFM mesh using a vertex-keep mapping file.

This utility reads:
- an input OBJ (BFM full head / face)
- a mapping text file that lists which vertex indices are kept, followed by faces (optional)

It writes a cropped OBJ containing:
- only kept vertices (reindexed)
- faces remapped to the new vertex indices (faces that reference removed vertices are skipped)

Typical usage:
  python 06_bfm_cropper.py --input_obj bfm_full.obj --mapping_txt data/cut_bfm.txt --output_obj bfm_cropped.obj

Mapping file format (expected):
- First line: something like "kept_indices: 1,2,3,..."  (1-based indices)
- Following lines: optional face lines in OBJ format, e.g. "f 1 2 3"
  (If faces are not provided, we will reuse faces from the input OBJ and filter them.)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def _require(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Not found: {Path(path).resolve()}")


def _read_obj_vertices_faces(obj_path: str) -> Tuple[List[str], List[List[int]]]:
    """
    Read vertex lines and face indices (v-only faces) from an OBJ.
    We preserve vertex lines as strings; faces are parsed as integer vertex indices (1-based).
    Faces with texture/normal indices are supported (we only keep the vertex index before '/').
    """
    verts: List[str] = []
    faces: List[List[int]] = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                verts.append(line.strip())
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idxs: List[int] = []
                for p in parts:
                    v_str = p.split("/")[0]
                    try:
                        idxs.append(int(v_str))
                    except ValueError:
                        pass
                if len(idxs) >= 3:
                    faces.append(idxs)
    if not verts:
        raise ValueError(f"No vertices found in OBJ: {obj_path}")
    return verts, faces


def _read_mapping(mapping_txt: str) -> Tuple[List[int], List[List[int]]]:
    """
    Read kept vertex indices and optional faces from mapping_txt.

    Supported first line examples:
      kept_indices: 1,2,3
      kept_indices: 1, 2, 3
    """
    kept: List[int] = []
    faces: List[List[int]] = []
    with open(mapping_txt, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        if ":" not in first:
            raise ValueError("Mapping first line must contain ':' (e.g., 'kept_indices: 1,2,3').")
        kept_part = first.split(":", 1)[1].strip()
        if kept_part:
            kept = [int(x.strip()) for x in kept_part.split(",") if x.strip()]

        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("f "):
                parts = line.split()[1:]
                idxs: List[int] = []
                for p in parts:
                    v_str = p.split("/")[0]
                    try:
                        idxs.append(int(v_str))
                    except ValueError:
                        pass
                if len(idxs) >= 3:
                    faces.append(idxs)

    if not kept:
        raise ValueError("No kept vertex indices found in mapping file.")
    return kept, faces


def crop_obj(
    input_obj: str,
    mapping_txt: str,
    output_obj: str,
    use_faces_from_mapping: bool = True
) -> None:
    """
    Crop an OBJ by keeping only a subset of vertices.

    Args:
        input_obj: source OBJ
        mapping_txt: mapping file containing kept vertex indices (and optional faces)
        output_obj: output OBJ path
        use_faces_from_mapping: if True and mapping provides faces, use them; otherwise use faces from input_obj
    """
    _require(input_obj)
    _require(mapping_txt)

    verts, faces_in = _read_obj_vertices_faces(input_obj)
    kept_indices, faces_map = _read_mapping(mapping_txt)

    # Create old->new index mapping (1-based)
    kept_set = set(i for i in kept_indices if i > 0)
    index_map: Dict[int, int] = {}
    new_idx = 1
    for old_idx in kept_indices:
        if old_idx in kept_set and old_idx not in index_map:
            index_map[old_idx] = new_idx
            new_idx += 1

    out_path = Path(output_obj)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Choose faces source
    faces_src = faces_map if (use_faces_from_mapping and faces_map) else faces_in

    # Remap faces; skip any face that references removed vertices
    faces_out: List[List[int]] = []
    for f in faces_src:
        if all(v in index_map for v in f):
            faces_out.append([index_map[v] for v in f])

    with open(out_path, "w", encoding="utf-8") as f:
        for old_idx in kept_indices:
            if old_idx in index_map:
                f.write(verts[old_idx - 1] + "\n")

        for tri in faces_out:
            # write as polygon (supports triangles/quads)
            f.write("f " + " ".join(str(v) for v in tri) + "\n")

    print(f"[BFM cropper] wrote: {out_path.resolve()}")
    print(f"[BFM cropper] kept vertices: {len(index_map)} | faces: {len(faces_out)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Crop BFM OBJ using a keep-index mapping file (AST-Face step 06).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--input_obj", type=str, required=True, help="Input OBJ (full BFM mesh).")
    ap.add_argument("--mapping_txt", type=str, required=True, help="Mapping txt (kept indices + optional faces).")
    ap.add_argument("--output_obj", type=str, required=True, help="Output OBJ (cropped).")
    ap.add_argument(
        "--use_faces_from_mapping",
        action="store_true",
        help="Use face list provided in mapping_txt if present; otherwise use faces from input_obj."
    )
    args = ap.parse_args()

    crop_obj(
        input_obj=args.input_obj,
        mapping_txt=args.mapping_txt,
        output_obj=args.output_obj,
        use_faces_from_mapping=args.use_faces_from_mapping
    )


if __name__ == "__main__":
    main()
