"""Read extracted AST-Face assets without repairing or reordering meshes."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


ALIASES = {"disguest": "disgust", "superise": "surprise"}
KINDS = {"mesh": ".obj", "deformation": ".npy", "landmarks": ".txt"}


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_obj(path):
    vertices, faces = [], []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                if len(fields) != 4:
                    raise ValueError(f"line {number}: expected exactly three vertex coordinates")
                vertices.append([float(value) for value in fields[1:]])
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError(f"line {number}: non-triangular face; no auto-triangulation")
                face = []
                for field in fields[1:]:
                    index = int(field.split("/")[0])
                    if index == 0:
                        raise ValueError(f"line {number}: OBJ vertex index zero is invalid")
                    resolved = index - 1 if index > 0 else len(vertices) + index
                    if resolved < 0:
                        raise ValueError(f"line {number}: negative index precedes first vertex")
                    face.append(resolved)
                faces.append(face)
    if not vertices or not faces:
        raise ValueError("empty mesh")
    return Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))


def topology_hash(mesh):
    digest = hashlib.sha256()
    digest.update(np.asarray([len(mesh.vertices), len(mesh.faces)], dtype="<i8").tobytes())
    digest.update(np.asarray(mesh.faces, dtype="<i8").tobytes())
    return digest.hexdigest()


class ASTFaceReader:
    def __init__(self, data_dir):
        self.root = Path(data_dir).resolve()
        if not self.root.is_dir():
            raise ValueError("data_dir must be an extracted dataset directory, never an archive")

    def inventory(self, subjects):
        samples, extras = {}, []
        directories = {"mesh": "Standardized Meshes", "deformation": "Deformation Fields",
                       "landmarks": "landmarks"}
        for subject in subjects:
            folder = self.root / subject
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            for kind, directory in directories.items():
                for path in sorted(folder.glob(f"{directory.strip()}*/*")):
                    # Some released directory names contain trailing spaces.
                    if not path.is_file():
                        continue
                    if path.suffix.lower() != KINDS[kind] or not path.stem.startswith(subject + "_"):
                        extras.append(path)
                        continue
                    raw_state = path.stem[len(subject) + 1:]
                    state = ALIASES.get(raw_state, raw_state)
                    key = (subject, state)
                    assets = samples.setdefault(key, {})
                    if kind in assets:
                        raise ValueError(f"duplicate canonical asset: {key}, {kind}")
                    assets[kind] = path
        return samples, extras

    def read_sample(self, assets):
        return {
            "mesh": read_obj(assets["mesh"]) if "mesh" in assets else None,
            "deformation": np.load(assets["deformation"], allow_pickle=False)
            if "deformation" in assets else None,
            "landmarks": np.loadtxt(assets["landmarks"], dtype=np.float64, ndmin=2)
            if "landmarks" in assets else None,
        }


def rigid_transform(source, target):
    """Row-vector convention: aligned = source @ R + t; reflection forbidden."""
    if source.shape != target.shape or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("alignment requires equally shaped, finite point arrays")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    translation = target_mean - source_mean @ rotation
    return rotation, translation
