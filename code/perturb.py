"""Deterministic perturbations for the AST-Face correspondence pilot.

The functions operate on a mesh ``(vertices, faces)`` pair and return a
``PerturbedMesh``.  They are intentionally independent and can be composed
with :func:`apply_perturbations` using an explicit seed/configuration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PerturbedMesh:
    """A perturbed mesh and the mask needed by downstream metrics."""

    vertices: np.ndarray
    faces: np.ndarray
    valid_mask: np.ndarray
    metadata: Mapping[str, object]


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices 应为 (N, 3)，实际为 {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces 应为 (F, 3)，实际为 {faces.shape}")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces 存在越界索引")
    return vertices.copy(), faces.copy()


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _new_result(vertices, faces, valid_mask, kind: str, strength: float, seed: int | None, **extra):
    metadata = {"type": kind, "strength": float(strength), "seed": seed}
    metadata.update(extra)
    return PerturbedMesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int64),
        np.asarray(valid_mask, dtype=bool),
        metadata,
    )


def baseline(vertices: np.ndarray, faces: np.ndarray) -> PerturbedMesh:
    """Return an unchanged baseline with all vertices valid."""

    vertices, faces = _validate_mesh(vertices, faces)
    return _new_result(vertices, faces, np.ones(len(vertices), dtype=bool), "baseline", 0.0, None)


def shuffle_indices(
    vertices: np.ndarray,
    faces: np.ndarray,
    strength: float = 1.0,
    seed: int | None = 0,
) -> PerturbedMesh:
    """Shuffle vertex labels while preserving the mesh geometry.

    ``strength=0`` is the identity.  For positive strength, a fraction of
    vertex labels is permuted and faces are remapped with the inverse
    permutation.  The resulting surface is geometrically unchanged, while
    vertex index correspondence to another mesh is damaged.
    """

    vertices, faces = _validate_mesh(vertices, faces)
    if not 0 <= strength <= 1:
        raise ValueError("shuffle strength 必须在 [0, 1] 内")
    if strength == 0 or len(vertices) < 2:
        return _new_result(vertices, faces, np.ones(len(vertices), bool), "shuffle_indices", strength, seed)
    generator = _rng(seed)
    count = max(2, int(round(len(vertices) * strength)))
    count = min(count, len(vertices))
    chosen = generator.choice(len(vertices), size=count, replace=False)
    shuffled = chosen.copy()
    generator.shuffle(shuffled)
    permutation = np.arange(len(vertices), dtype=np.int64)
    permutation[chosen] = shuffled
    # output vertex i contains the old vertex permutation[i]; remap faces by
    # inverse permutation so each face still joins the same geometric points.
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(vertices), dtype=np.int64)
    output_vertices = vertices[permutation]
    output_faces = inverse[faces]
    return _new_result(
        output_vertices,
        output_faces,
        np.ones(len(vertices), bool),
        "shuffle_indices",
        strength,
        seed,
        shuffled_vertices=count,
    )


def coordinate_noise(
    vertices: np.ndarray,
    faces: np.ndarray,
    strength: float,
    seed: int | None = 0,
) -> PerturbedMesh:
    """Add isotropic Gaussian coordinate noise.

    ``strength`` is measured as a fraction of the mesh bounding-box diagonal.
    """

    vertices, faces = _validate_mesh(vertices, faces)
    if strength < 0:
        raise ValueError("noise strength 不能为负")
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    sigma = strength * diagonal
    noise = _rng(seed).normal(0.0, sigma, size=vertices.shape).astype(np.float32)
    return _new_result(vertices + noise, faces, np.ones(len(vertices), bool), "coordinate_noise", strength, seed, sigma=sigma)


def smooth_local_deformation(
    vertices: np.ndarray,
    faces: np.ndarray,
    strength: float,
    seed: int | None = 0,
    radius: float = 0.22,
) -> PerturbedMesh:
    """Apply a smooth Gaussian displacement to a local surface region.

    ``strength`` is a fraction of the bounding-box diagonal and ``radius`` is
    a fraction of that diagonal.  This is a controlled low-frequency
    deformation, not a self-intersection/folding perturbation.
    """

    vertices, faces = _validate_mesh(vertices, faces)
    if strength < 0 or radius <= 0:
        raise ValueError("strength 必须非负，radius 必须为正")
    generator = _rng(seed)
    bounds = vertices.max(axis=0) - vertices.min(axis=0)
    diagonal = float(np.linalg.norm(bounds))
    center = generator.uniform(vertices.min(axis=0), vertices.max(axis=0))
    direction = generator.normal(size=3)
    direction /= max(float(np.linalg.norm(direction)), np.finfo(np.float32).eps)
    distances = np.linalg.norm(vertices - center, axis=1)
    weights = np.exp(-0.5 * (distances / (radius * diagonal)) ** 2)
    displacement = (strength * diagonal * weights[:, None] * direction).astype(np.float32)
    return _new_result(
        vertices + displacement,
        faces,
        np.ones(len(vertices), bool),
        "smooth_local_deformation",
        strength,
        seed,
        radius=radius,
        center=center.tolist(),
        direction=direction.tolist(),
    )


def random_missing(
    vertices: np.ndarray,
    faces: np.ndarray,
    strength: float,
    seed: int | None = 0,
) -> PerturbedMesh:
    """Mark a random fraction of vertices as missing using NaN and a mask."""

    vertices, faces = _validate_mesh(vertices, faces)
    if not 0 <= strength < 1:
        raise ValueError("missing strength 必须在 [0, 1) 内")
    count = int(round(len(vertices) * strength))
    generator = _rng(seed)
    missing = generator.choice(len(vertices), size=count, replace=False) if count else np.array([], dtype=np.int64)
    valid = np.ones(len(vertices), dtype=bool)
    valid[missing] = False
    output = vertices.copy()
    output[missing] = np.nan
    return _new_result(output, faces, valid, "random_missing", strength, seed, missing_vertices=count)


def rigid_transform(
    vertices: np.ndarray,
    faces: np.ndarray,
    strength: float,
    seed: int | None = 0,
) -> PerturbedMesh:
    """Apply a global rotation and translation as a negative control.

    ``strength=1`` corresponds to a rotation of at most 15 degrees and a
    translation of 5% of the bounding-box diagonal.  The transform preserves
    all vertex indices and should not change correspondence quality.
    """

    vertices, faces = _validate_mesh(vertices, faces)
    if strength < 0:
        raise ValueError("rigid strength 不能为负")
    generator = _rng(seed)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    angles = generator.uniform(-np.pi / 12, np.pi / 12, size=3) * strength
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    rotation = rz @ ry @ rx
    translation = generator.uniform(-0.05, 0.05, size=3).astype(np.float32) * strength * diagonal
    center = vertices.mean(axis=0)
    output = (vertices - center) @ rotation.T + center + translation
    return _new_result(
        output,
        faces,
        np.ones(len(vertices), bool),
        "rigid_transform",
        strength,
        seed,
        angles=angles.tolist(),
        translation=translation.tolist(),
    )


def apply_perturbations(
    vertices: np.ndarray,
    faces: np.ndarray,
    operations: Sequence[Mapping[str, object]],
    seed: int | None = 0,
) -> PerturbedMesh:
    """Apply an explicit ordered list of perturbation configurations."""

    current = baseline(vertices, faces)
    valid = current.valid_mask.copy()
    metadata: list[dict] = []
    seed_sequence = np.random.SeedSequence(seed).spawn(len(operations))
    functions = {
        "baseline": baseline,
        "shuffle_indices": shuffle_indices,
        "coordinate_noise": coordinate_noise,
        "smooth_local_deformation": smooth_local_deformation,
        "random_missing": random_missing,
        "rigid_transform": rigid_transform,
    }
    for operation, child_seed in zip(operations, seed_sequence):
        kind = str(operation.get("type", ""))
        if kind not in functions:
            raise ValueError(f"未知扰动类型: {kind}")
        kwargs = dict(operation)
        kwargs.pop("type", None)
        strength = float(kwargs.get("strength", 0.0))
        kwargs["seed"] = int(child_seed.generate_state(1)[0])
        if kind == "baseline":
            current = baseline(current.vertices, current.faces)
        else:
            current = functions[kind](current.vertices, current.faces, strength, **{k: v for k, v in kwargs.items() if k != "strength"})
        valid &= current.valid_mask
        metadata.append(dict(current.metadata))
    return PerturbedMesh(current.vertices, current.faces, valid, {"type": "composed", "seed": seed, "operations": metadata})


def save_npz(result: PerturbedMesh, path: str | Path) -> None:
    """Save vertices, faces, valid mask and JSON metadata to a compact NPZ."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices=result.vertices,
        faces=result.faces,
        valid_mask=result.valid_mask,
        metadata=np.asarray(json.dumps(result.metadata, ensure_ascii=False)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成一个 AST-Face 网格的可复现扰动样本")
    parser.add_argument("--input", required=True, help="输入 OBJ 文件")
    parser.add_argument("--output", required=True, help="输出 NPZ 文件")
    parser.add_argument("--type", choices=("baseline", "shuffle_indices", "coordinate_noise", "smooth_local_deformation", "random_missing", "rigid_transform"), required=True)
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.22)
    return parser


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == "f":
            polygon = [int(x.split("/", 1)[0]) - 1 for x in fields[1:]]
            faces.extend((polygon[0], polygon[i], polygon[i + 1]) for i in range(1, len(polygon) - 1))
    return np.asarray(vertices, np.float32), np.asarray(faces, np.int64)


if __name__ == "__main__":
    args = _parser().parse_args()
    vertices, faces = _read_obj(Path(args.input))
    if args.type == "baseline":
        result = baseline(vertices, faces)
    elif args.type == "shuffle_indices":
        result = shuffle_indices(vertices, faces, args.strength, args.seed)
    elif args.type == "coordinate_noise":
        result = coordinate_noise(vertices, faces, args.strength, args.seed)
    elif args.type == "smooth_local_deformation":
        result = smooth_local_deformation(vertices, faces, args.strength, args.seed, args.radius)
    elif args.type == "random_missing":
        result = random_missing(vertices, faces, args.strength, args.seed)
    else:
        result = rigid_transform(vertices, faces, args.strength, args.seed)
    save_npz(result, args.output)
    print(json.dumps({"output": str(args.output), "vertices": list(result.vertices.shape), "faces": list(result.faces.shape), "valid_vertices": int(result.valid_mask.sum()), "metadata": result.metadata}, ensure_ascii=False, indent=2))
