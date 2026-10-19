#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Edge-NICP registration for AST-Face topology unification.

Example:
  python run_edge_nicp.py \
    --template_mesh result/flame_to_bfm.obj \
    --target_mesh testdata/astface_show/show_au27.obj \
    --target_landmarks_txt testdata/astface_show/show_au27_landmarks.txt \
    --config config/fine_grain.json \
    --out_mesh result/out_mesh.obj \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch

import io3d
from bfm import MorphableModel
from utils import batch_vertex_sample
from edge_nicp import tranformAtoB, non_rigid_icp_edge


# -------------------------
# Default landmark mappings
# -------------------------
# NOTE: Keep these default lists consistent with your pipeline conventions.
LANDMARK_ID_84_DEFAULT: List[int] = [
    21, 25, 35, 31, 83, 40, 41, 43, 4, 3, 1, 0, 7, 5, 8, 9, 11, 12, 13, 15,
    60, 49, 50, 51, 52, 53, 64, 55, 57, 59, 61, 62, 63, 65, 66, 67
]
LANDMARK_ID_68_DEFAULT: List[int] = [
    17, 21, 22, 26, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 59, 61, 62, 63, 65, 66, 67
]


def _require_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")


def _load_json(path: str) -> dict:
    _require_file(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_int_list(s: str) -> List[int]:
    """
    Parse a comma-separated int list, e.g. "1,2,3".
    If s is empty, return [].
    """
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def _load_target_landmarks_txt(path: str, select_ids: List[int]) -> np.ndarray:
    """
    Load target landmarks from a text file and select indices.
    Expected: (N, 3) float.
    """
    _require_file(path)
    lm = np.loadtxt(path).astype(np.float32)
    if lm.ndim != 2 or lm.shape[1] != 3:
        raise ValueError(f"Expected landmarks shape (N,3), got {lm.shape} from {path}")
    if max(select_ids) >= lm.shape[0]:
        raise IndexError(
            f"landmark_id_84 has index {max(select_ids)} but file only has {lm.shape[0]} points"
        )
    return lm[select_ids]


def run(
    template_mesh_path: str,
    target_mesh_path: str,
    target_landmarks_txt: str,
    config_path: str,
    out_mesh_path: str,
    device_str: str = "cuda:0",
    landmark_id_84: List[int] | None = None,
    landmark_id_68: List[int] | None = None,
    verbose: bool = True,
) -> None:
    """
    Register template mesh (BFM topology) to target mesh using Edge-NICP.

    Args:
        template_mesh_path: Template mesh (BFM topology) .obj
        target_mesh_path: Target mesh .obj
        target_landmarks_txt: Target landmarks file (.txt), containing at least indices in landmark_id_84
        config_path: JSON config for Edge-NICP (fine_grain.json)
        out_mesh_path: Output registered mesh .obj
        device_str: torch device string, e.g. "cuda:0" or "cpu"
        landmark_id_84: indices to select from target landmark file
        landmark_id_68: indices to select keypoints from MorphableModel.kpt_inds
        verbose: print basic logs
    """
    landmark_id_84 = landmark_id_84 or LANDMARK_ID_84_DEFAULT
    landmark_id_68 = landmark_id_68 or LANDMARK_ID_68_DEFAULT

    # Validate files
    _require_file(template_mesh_path)
    _require_file(target_mesh_path)
    _require_file(target_landmarks_txt)
    _require_file(config_path)

    device = torch.device(device_str)
    if "cuda" in device_str and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA not available but device={device_str}")

    fine_config = _load_json(config_path)

    if verbose:
        print(f"[Edge-NICP] device          : {device}")
        print(f"[Edge-NICP] template_mesh   : {template_mesh_path}")
        print(f"[Edge-NICP] target_mesh     : {target_mesh_path}")
        print(f"[Edge-NICP] target_landmarks: {target_landmarks_txt}")
        print(f"[Edge-NICP] config          : {config_path}")
        print(f"[Edge-NICP] out_mesh        : {out_mesh_path}")

    # Load meshes
    template_mesh = io3d.load_obj_as_mesh(template_mesh_path, device=device)
    target_mesh = io3d.load_obj_as_mesh(target_mesh_path, device=device)

    # Load target landmarks (84 set) -> torch (1, K, 3)
    target_lm_np = _load_target_landmarks_txt(target_landmarks_txt, landmark_id_84)
    target_lm = torch.from_numpy(target_lm_np).unsqueeze(0).to(device)  # (1, K, 3)

    # Build template landmark indices (from BFM MorphableModel keypoints)
    model = MorphableModel(device=device)
    # model.kpt_inds: (num_kpt,) long indices
    bfm_lm_index = model.kpt_inds[landmark_id_68].unsqueeze(0).to(device)  # (1, K)

    # Sample template landmarks -> (1, K, 3)
    with torch.no_grad():
        bfm_lm = torch.tensor(
            batch_vertex_sample(bfm_lm_index, template_mesh.verts_padded()),
            dtype=torch.float32,
            device=device,
        )

        # Rigid align target to template landmark frame (or vice versa depending on your implementation)
        after_target, after_target_lm = tranformAtoB(
            template_mesh, target_mesh, bfm_lm, target_lm, device
        )

    # Non-rigid registration (Edge-NICP)
    registered_mesh = non_rigid_icp_edge(
        template_mesh,
        after_target,
        bfm_lm_index,
        after_target_lm,
        fine_config,
        device,
        with_edge=True,
    )

    # Save
    out_mesh_path = str(Path(out_mesh_path))
    Path(out_mesh_path).parent.mkdir(parents=True, exist_ok=True)
    io3d.save_meshes_as_objs([out_mesh_path], registered_mesh, save_textures=False)

    if verbose:
        print(f"[Edge-NICP] done. Saved to: {out_mesh_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Edge-NICP registration (AST-Face pipeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--template_mesh", type=str, default= 'testdata/astface_show/bfm_show_au27.obj', required=True,
                   help="Template mesh (BFM topology), e.g. result/flame_to_bfm.obj")
    p.add_argument("--target_mesh", type=str, required=True,
                   help="Target mesh to be registered, e.g. show_au27.obj")
    p.add_argument("--target_landmarks_txt", type=str, required=True,
                   help="Target landmarks txt file (N x 3).")
    p.add_argument("--config", type=str, required=True,
                   help="Edge-NICP config json, e.g. config/fine_grain.json")
    p.add_argument("--out_mesh", type=str, required=True,
                   help="Output registered mesh path (.obj)")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Torch device, e.g. cuda:0 or cpu")
    p.add_argument("--landmark_id_84", type=str, default="",
                   help="Override landmark_id_84 as comma-separated list. Empty = default.")
    p.add_argument("--landmark_id_68", type=str, default="",
                   help="Override landmark_id_68 as comma-separated list. Empty = default.")
    p.add_argument("--quiet", action="store_true", help="Disable verbose logs")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    landmark_id_84 = _parse_int_list(args.landmark_id_84) or None
    landmark_id_68 = _parse_int_list(args.landmark_id_68) or None

    run(
        template_mesh_path=args.template_mesh,
        target_mesh_path=args.target_mesh,
        target_landmarks_txt=args.target_landmarks_txt,
        config_path=args.config,
        out_mesh_path=args.out_mesh,
        device_str=args.device,
        landmark_id_84=landmark_id_84,
        landmark_id_68=landmark_id_68,
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()