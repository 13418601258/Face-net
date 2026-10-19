"""Unified reader and validator for the public AST-Face dataset.

The reader works only with an extracted ``ASTFace_Public`` directory.  It
exposes the three public modalities using one sample interface:

* standardized mesh: vertices and zero-based triangular faces from OBJ;
* deformation field: an ``(N, 3)`` NumPy array;
* landmarks: an ``(L, 3)`` float array from TXT.

Examples
--------
    python code/astface_loader.py --data-dir data/ASTFace_Public --subjects 1-15 --validate
    python code/astface_loader.py --data-dir data/ASTFace_Public --subject ast001 --state angry

The default validation checks the archive index only.  Add ``--deep`` to
decode every selected file and verify shapes and shared mesh topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


MODALITIES = ("mesh", "deformation", "landmark")
MODALITY_DIRS = {
    "mesh": ("Standardized Meshes", ".obj"),
    "deformation": ("Deformation Fields", ".npy"),
    "landmark": ("landmarks", ".txt"),
}

# The public archive contains a few spelling variants.  They identify the
# same state, so the reader canonicalizes them but keeps the real source path.
STATE_ALIASES = {
    "disguest": "disgust",
    "superise": "surprise",
}
SUBJECT_RE = re.compile(r"(?:^|/)ast(?P<number>\d{3})(?:/|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Mesh:
    """A standardized mesh with zero-based triangular face indices."""

    vertices: np.ndarray
    faces: np.ndarray
    path: str


@dataclass(frozen=True)
class FaceSample:
    """All requested modalities for one subject/state pair.

    ``deformation`` and ``landmarks`` are optional because ``au0`` has no
    public deformation field in AST-Face.  Missing data is represented by
    ``None`` rather than silently substituting another state.
    """

    subject: str
    state: str
    mesh: Mesh | None
    deformation: np.ndarray | None
    landmarks: np.ndarray | None
    source_paths: Mapping[str, str]


@dataclass(frozen=True)
class ValidationReport:
    """Serializable summary returned by :meth:`ASTFaceReader.validate`."""

    subjects: tuple[str, ...]
    counts: Mapping[str, int]
    missing: Mapping[str, Mapping[str, tuple[str, ...]]]
    errors: tuple[str, ...]
    topology_hashes: Mapping[str, tuple[str, ...]]
    vertex_counts: Mapping[str, tuple[int, ...]]
    deformation_shapes: Mapping[str, tuple[tuple[int, ...], ...]]
    landmark_shapes: Mapping[str, tuple[tuple[int, ...], ...]]

    def to_dict(self) -> dict:
        return {
            "subjects": list(self.subjects),
            "counts": dict(self.counts),
            "missing": {
                modality: {subject: list(states) for subject, states in by_subject.items()}
                for modality, by_subject in self.missing.items()
            },
            "errors": list(self.errors),
            "topology_hashes": {s: list(v) for s, v in self.topology_hashes.items()},
            "vertex_counts": {s: list(v) for s, v in self.vertex_counts.items()},
            "deformation_shapes": {s: [list(x) for x in v] for s, v in self.deformation_shapes.items()},
            "landmark_shapes": {s: [list(x) for x in v] for s, v in self.landmark_shapes.items()},
        }


def canonical_state(raw_state: str) -> str:
    """Return the canonical lowercase state name used by the reader."""

    state = raw_state.strip().lower()
    return STATE_ALIASES.get(state, state)


def parse_subjects(value: str | Iterable[str | int]) -> tuple[str, ...]:
    """Parse ``1-15``, ``1,2,5`` or an iterable into sorted subject IDs."""

    if isinstance(value, str):
        tokens = [part.strip() for part in value.split(",") if part.strip()]
        numbers: list[int] = []
        for token in tokens:
            if "-" in token:
                start, end = token.split("-", 1)
                numbers.extend(range(int(start), int(end) + 1))
            else:
                numbers.append(int(token))
    else:
        numbers = []
        for item in value:
            text = str(item).strip().lower()
            match = re.fullmatch(r"ast(\d{3})", text)
            numbers.append(int(match.group(1)) if match else int(text))
    if not numbers:
        raise ValueError("至少指定一个身份，例如 1-15")
    if any(number < 1 or number > 999 for number in numbers):
        raise ValueError("身份编号必须在 1 到 999 之间")
    return tuple(f"ast{number:03d}" for number in sorted(set(numbers)))


class ASTFaceReader:
    """Read AST-Face files from an extracted directory."""

    def __init__(self, source: str | Path):
        self.source = Path(source).expanduser().resolve()
        if not self.source.exists():
            raise FileNotFoundError(f"找不到数据目录: {self.source}")
        if self.source.is_file():
            raise ValueError("只接受已解压的数据目录，不读取 ZIP 或其他文件")
        self._root = self.source / "ASTFace_Public" if (self.source / "ASTFace_Public").is_dir() else self.source
        if not self._root.is_dir():
            raise ValueError(f"目录中没有 ASTFace_Public: {self.source}")
        self._names = tuple(
            path.relative_to(self._root).as_posix()
            for path in self._root.rglob("*")
            if path.is_file()
        )
        self._index = self._build_index()

    def close(self) -> None:
        return None

    def __enter__(self) -> "ASTFaceReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def source_kind(self) -> str:
        return "directory"

    @property
    def file_count(self) -> int:
        return len(self._names)

    def _build_index(self) -> dict[str, dict[str, dict[str, str]]]:
        index: dict[str, dict[str, dict[str, str]]] = {
            modality: {} for modality in MODALITIES
        }
        for name in self._names:
            parts = name.replace("\\", "/").split("/")
            if len(parts) < 3:
                continue
            match = SUBJECT_RE.search("/".join(parts[:-2]) + "/")
            if match is None:
                continue
            subject = f"ast{int(match.group('number')):03d}"
            folder = parts[-2]
            filename = parts[-1]
            stem = Path(filename).stem
            prefix = subject + "_"
            if not stem.lower().startswith(prefix):
                continue
            raw_state = stem[len(prefix):]
            state = canonical_state(raw_state)
            modality = next(
                (key for key, (expected_folder, suffix) in MODALITY_DIRS.items()
                 if folder.lower() == expected_folder.lower() and filename.lower().endswith(suffix)),
                None,
            )
            if modality is None:
                continue
            subject_index = index[modality].setdefault(subject, {})
            if state in subject_index:
                raise ValueError(f"同一模态出现重复状态: {subject} / {modality} / {state}")
            subject_index[state] = name
        return index

    def subjects(self) -> tuple[str, ...]:
        values = set()
        for modality_index in self._index.values():
            values.update(modality_index)
        return tuple(sorted(values))

    def available_states(self, subject: str | int) -> dict[str, tuple[str, ...]]:
        subject_id = parse_subjects([subject])[0]
        return {
            modality: tuple(sorted(self._index[modality].get(subject_id, {})))
            for modality in MODALITIES
        }

    def path_for(self, subject: str | int, state: str, modality: str) -> str:
        subject_id = parse_subjects([subject])[0]
        if modality not in MODALITIES:
            raise ValueError(f"未知模态 {modality!r}，可选: {', '.join(MODALITIES)}")
        canonical = canonical_state(state)
        try:
            return self._index[modality][subject_id][canonical]
        except KeyError as exc:
            available = self.available_states(subject_id)[modality]
            raise FileNotFoundError(
                f"找不到 {subject_id} / {canonical} / {modality}; 可用状态: {', '.join(available)}"
            ) from exc

    @contextmanager
    def _open_binary(self, relative_name: str):
        with (self._root / Path(relative_name)).open("rb") as handle:
            yield handle

    def _read_bytes(self, relative_name: str) -> bytes:
        with self._open_binary(relative_name) as handle:
            return handle.read()

    def read_mesh(self, subject: str | int, state: str) -> Mesh:
        relative_name = self.path_for(subject, state, "mesh")
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        text = self._read_bytes(relative_name).decode("utf-8", errors="strict")
        for line_number, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"OBJ 顶点格式错误: {relative_name}:{line_number}")
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
            elif fields[0] == "f":
                if len(fields) < 4:
                    raise ValueError(f"OBJ 面片格式错误: {relative_name}:{line_number}")
                polygon = []
                for token in fields[1:]:
                    raw_index = token.split("/", 1)[0]
                    index = int(raw_index)
                    polygon.append(index - 1 if index > 0 else len(vertices) + index)
                for offset in range(1, len(polygon) - 1):
                    faces.append((polygon[0], polygon[offset], polygon[offset + 1]))
        vertex_array = np.asarray(vertices, dtype=np.float32)
        face_array = np.asarray(faces, dtype=np.int64)
        if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
            raise ValueError(f"OBJ 没有合法顶点: {relative_name}")
        if face_array.size and (face_array.min() < 0 or face_array.max() >= len(vertices)):
            raise ValueError(f"OBJ 面片索引越界: {relative_name}")
        return Mesh(vertex_array, face_array.reshape((-1, 3)), relative_name)

    def read_deformation(self, subject: str | int, state: str) -> np.ndarray:
        relative_name = self.path_for(subject, state, "deformation")
        with self._open_binary(relative_name) as handle:
            array = np.load(handle, allow_pickle=False)
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"形变场应为 (N, 3)，实际为 {array.shape}: {relative_name}")
        return array

    def read_landmarks(self, subject: str | int, state: str) -> np.ndarray:
        relative_name = self.path_for(subject, state, "landmark")
        text = self._read_bytes(relative_name).decode("utf-8-sig", errors="strict")
        rows: list[list[float]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            values = stripped.split()
            if len(values) != 3:
                raise ValueError(f"landmark 应有 3 列: {relative_name}:{line_number}")
            rows.append([float(value) for value in values])
        array = np.asarray(rows, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"landmark 应为 (L, 3)，实际为 {array.shape}: {relative_name}")
        return array

    def read_sample(
        self,
        subject: str | int,
        state: str,
        modalities: Sequence[str] = MODALITIES,
    ) -> FaceSample:
        subject_id = parse_subjects([subject])[0]
        canonical = canonical_state(state)
        requested = tuple(modalities)
        unknown = set(requested) - set(MODALITIES)
        if unknown:
            raise ValueError(f"未知模态: {', '.join(sorted(unknown))}")
        mesh = self.read_mesh(subject_id, canonical) if "mesh" in requested else None
        deformation = None
        if "deformation" in requested:
            try:
                deformation = self.read_deformation(subject_id, canonical)
            except FileNotFoundError:
                # au0 is intentionally absent from the public deformation-field set.
                deformation = None
        landmarks = None
        if "landmark" in requested:
            try:
                landmarks = self.read_landmarks(subject_id, canonical)
            except FileNotFoundError:
                landmarks = None
        paths = {}
        for modality in requested:
            try:
                paths[modality] = self.path_for(subject_id, canonical, modality)
            except FileNotFoundError:
                pass
        return FaceSample(subject_id, canonical, mesh, deformation, landmarks, paths)

    def iter_samples(
        self,
        subjects: Iterable[str | int],
        states: Iterable[str] | None = None,
        modalities: Sequence[str] = MODALITIES,
        skip_missing: bool = False,
    ) -> Iterator[FaceSample]:
        subject_ids = parse_subjects(subjects)
        requested_states = None if states is None else {canonical_state(item) for item in states}
        for subject in subject_ids:
            state_set = requested_states if requested_states is not None else set(self.available_states(subject)["mesh"])
            for state in sorted(state_set):
                try:
                    yield self.read_sample(subject, state, modalities)
                except FileNotFoundError:
                    if not skip_missing:
                        raise

    @staticmethod
    def topology_hash(mesh: Mesh) -> str:
        """Hash vertex count, face count and exact zero-based face order."""

        payload = b"|".join((str(len(mesh.vertices)).encode(), str(len(mesh.faces)).encode(), mesh.faces.tobytes()))
        return hashlib.sha256(payload).hexdigest()

    def validate(
        self,
        subjects: Iterable[str | int],
        deep: bool = False,
    ) -> ValidationReport:
        subject_ids = parse_subjects(subjects)
        missing: dict[str, dict[str, tuple[str, ...]]] = {modality: {} for modality in MODALITIES}
        errors: list[str] = []
        counts = {modality: 0 for modality in MODALITIES}
        topology_hashes: dict[str, tuple[str, ...]] = {}
        vertex_counts: dict[str, tuple[int, ...]] = {}
        deformation_shapes: dict[str, tuple[tuple[int, ...], ...]] = {}
        landmark_shapes: dict[str, tuple[tuple[int, ...], ...]] = {}
        for subject in subject_ids:
            states_by_modality = self.available_states(subject)
            expected = set(states_by_modality["mesh"])
            for modality in MODALITIES:
                counts[modality] += len(states_by_modality[modality])
                missing_states = tuple(sorted(expected - set(states_by_modality[modality])))
                if missing_states:
                    missing[modality][subject] = missing_states
            if not deep:
                continue
            subject_topology: list[str] = []
            subject_vertices: list[int] = []
            subject_deformation_shapes: list[tuple[int, ...]] = []
            subject_landmark_shapes: list[tuple[int, ...]] = []
            for state in states_by_modality["mesh"]:
                try:
                    mesh = self.read_mesh(subject, state)
                    subject_topology.append(self.topology_hash(mesh))
                    subject_vertices.append(len(mesh.vertices))
                    if state in states_by_modality["deformation"]:
                        deformation = self.read_deformation(subject, state)
                        subject_deformation_shapes.append(tuple(deformation.shape))
                        if len(deformation) != len(mesh.vertices):
                            errors.append(f"{subject}/{state}: deformation 行数 {len(deformation)} != mesh 顶点数 {len(mesh.vertices)}")
                    if state in states_by_modality["landmark"]:
                        landmarks = self.read_landmarks(subject, state)
                        subject_landmark_shapes.append(tuple(landmarks.shape))
                except (OSError, ValueError) as exc:
                    errors.append(f"{subject}/{state}: {exc}")
            topology_hashes[subject] = tuple(sorted(set(subject_topology)))
            vertex_counts[subject] = tuple(sorted(set(subject_vertices)))
            deformation_shapes[subject] = tuple(sorted(set(subject_deformation_shapes)))
            landmark_shapes[subject] = tuple(sorted(set(subject_landmark_shapes)))
            if len(topology_hashes[subject]) > 1:
                errors.append(f"{subject}: mesh 拓扑哈希不一致 ({len(topology_hashes[subject])} 种)")
            if len(vertex_counts[subject]) > 1:
                errors.append(f"{subject}: mesh 顶点数不一致 ({vertex_counts[subject]})")
        return ValidationReport(
            subjects=subject_ids,
            counts=counts,
            missing=missing,
            errors=tuple(errors),
            topology_hashes=topology_hashes,
            vertex_counts=vertex_counts,
            deformation_shapes=deformation_shapes,
            landmark_shapes=landmark_shapes,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取和校验 AST-Face 公共数据")
    parser.add_argument("--data-dir", required=True, help="已解压的 ASTFace_Public 目录，或包含该目录的 data 目录")
    parser.add_argument("--subjects", default="1-15", help="身份范围，例如 1-15 或 1,3,5")
    parser.add_argument("--validate", action="store_true", help="输出选定身份的模态清单")
    parser.add_argument("--deep", action="store_true", help="在校验时实际读取全部文件并检查形状/拓扑")
    parser.add_argument("--subject", help="读取单个样本时的身份，例如 ast001")
    parser.add_argument("--state", default="angry", help="读取单个样本时的状态，例如 angry 或 au0")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出验证结果或样本摘要")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    with ASTFaceReader(args.data_dir) as reader:
        if args.validate:
            report = reader.validate(parse_subjects(args.subjects), deep=args.deep)
            payload = report.to_dict()
            payload["source"] = str(reader.source)
            payload["source_kind"] = reader.source_kind
            payload["file_count"] = reader.file_count
            print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else _format_report(payload))
            return 1 if report.errors else 0
        if not args.subject:
            raise SystemExit("请使用 --validate，或同时提供 --subject 和 --state")
        sample = reader.read_sample(args.subject, args.state, modalities=MODALITIES)
        assert sample.mesh is not None
        summary = {
            "subject": sample.subject,
            "state": sample.state,
            "vertices": list(sample.mesh.vertices.shape),
            "faces": list(sample.mesh.faces.shape),
            "deformation": None if sample.deformation is None else list(sample.deformation.shape),
            "landmarks": None if sample.landmarks is None else list(sample.landmarks.shape),
            "paths": dict(sample.source_paths),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else _format_report(summary))
    return 0


def _format_report(payload: Mapping) -> str:
    lines = []
    if "source" in payload:
        lines.extend([
            f"数据源: {payload['source']}",
            f"读取方式: {payload['source_kind']}",
            f"索引文件数: {payload['file_count']}",
        ])
    if "counts" in payload:
        lines.append(f"身份: {', '.join(payload['subjects'])}")
        lines.append("模态文件数: " + ", ".join(f"{k}={v}" for k, v in payload["counts"].items()))
        for modality, by_subject in payload["missing"].items():
            for subject, states in by_subject.items():
                lines.append(f"缺失 {modality}: {subject}: {', '.join(states)}")
        if payload["topology_hashes"]:
            lines.append("拓扑哈希种类: " + ", ".join(f"{s}={len(v)}" for s, v in payload["topology_hashes"].items()))
            lines.append("顶点数: " + ", ".join(f"{s}={v}" for s, v in payload["vertex_counts"].items()))
        lines.append("状态: 通过" if not payload["errors"] else "状态: 有错误")
        lines.extend(f"错误: {error}" for error in payload["errors"])
    else:
        lines.append(f"样本: {payload['subject']} / {payload['state']}")
        lines.append(f"网格: vertices={payload['vertices']}, faces={payload['faces']}")
        lines.append(f"形变场: {payload['deformation']}")
        lines.append(f"landmark: {payload['landmarks']}")
        lines.extend(f"{key}: {value}" for key, value in payload["paths"].items())
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
