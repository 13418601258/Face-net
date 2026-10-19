#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_deformation_fields.py — Compute per-vertex deformation fields relative to neutral.

Given:
  - neutral mesh (topology-unified) OBJ
  - expression mesh (same topology) OBJ

This script computes:
  D = V_expr - V_neutral   (N,3)

and saves it as a NumPy file (.npy).

Typical usage:
  python 07_deformation_fields.py --neutral neutral.obj --expression au27.obj --out au27_def.npy

Notes
- The two meshes must have the same vertex count and correspondence.
- Default backend uses trimesh (CPU). A PyTorch3D backend is also provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def _require(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Not found: {Path(path).resolve()}")


def load_vertices_trimesh(obj_path: str) -> np.ndarray:
    import trimesh
    m = trimesh.load(obj_path, process=False)
    v = np.asarray(m.vertices, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Invalid vertices from {obj_path}: {v.shape}")
    return v


def load_vertices_pytorch3d(obj_path: str, device: str = "cuda:0") -> np.ndarray:
    import torch
    from pytorch3d.io import load_objs_as_meshes
    dev = torch.device(device)
    meshes = load_objs_as_meshes([obj_path], device=dev)
    v = meshes.verts_packed().detach().cpu().numpy().astype(np.float32)
    return v


def compute_deformation(neutral_v: np.ndarray, expr_v: np.ndarray) -> np.ndarray:
    if neutral_v.shape != expr_v.shape:
        raise ValueError(f"Vertex shape mismatch: neutral {neutral_v.shape} vs expr {expr_v.shape}")
    return (expr_v - neutral_v).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute deformation fields (AST-Face step 07).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--neutral", type=str, required=True, help="Neutral OBJ path.")
    ap.add_argument("--expression", type=str, required=True, help="Expression OBJ path (same topology).")
    ap.add_argument("--out", type=str, required=True, help="Output .npy path.")
    ap.add_argument(
        "--backend",
        type=str,
        default="trimesh",
        choices=["trimesh", "pytorch3d"],
        help="Mesh loading backend."
    )
    ap.add_argument("--device", type=str, default="cuda:0", help="Device for pytorch3d backend.")
    args = ap.parse_args()

    _require(args.neutral)
    _require(args.expression)

    if args.backend == "trimesh":
        v0 = load_vertices_trimesh(args.neutral)
        v1 = load_vertices_trimesh(args.expression)
    else:
        v0 = load_vertices_pytorch3d(args.neutral, device=args.device)
        v1 = load_vertices_pytorch3d(args.expression, device=args.device)

    D = compute_deformation(v0, v1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), D)

    print(f"[Deformation] saved: {out_path.resolve()} | shape={D.shape} | dtype={D.dtype}")


if __name__ == "__main__":
    main()
