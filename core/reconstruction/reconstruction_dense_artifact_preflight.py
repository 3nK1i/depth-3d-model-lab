"""I6.7E1 read-only full dense PLY qualification, NOT final scene admission.

The full artifact is scanned in bounded chunks, not sampled. The returned digest,
point count and bounding box are evidence for a later, separate final-admission
step. This module does not modify providers, persistence or GPU/viewer state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import BinaryIO

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

I67E1_PHASE = "I6.7E1_FULL_DENSE_ARTIFACT_PREFLIGHT"
I67E1_CHUNK_POINTS = 65536
I67E1_ASCII_CHUNK_POINTS = 1024
I67E1_MAX_BINARY_CHUNK_BYTES = 8 * 1024 * 1024
I67E1_MAX_HEADER_BYTES = 65536

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


@dataclass(frozen=True, slots=True)
class DenseArtifactPreflight:
    path: str
    sha256: str
    file_bytes: int
    vertex_count: int
    has_rgb: bool
    ply_format: str
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    scanned_vertex_count: int
    is_final_admitted: bool = False
    is_persistent_scene: bool = False
    metric_scale_claimed: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1, "phase": I67E1_PHASE, "artifact_path": self.path,
            "artifact_sha256": self.sha256, "file_bytes": self.file_bytes,
            "vertex_count": self.vertex_count, "scanned_vertex_count": self.scanned_vertex_count,
            "has_rgb": self.has_rgb, "ply_format": self.ply_format,
            "bounds_min": list(self.bounds_min), "bounds_max": list(self.bounds_max),
            "final_dense_result_admitted": False, "persistent_dense_scene_created": False,
            "metric_scale_claimed": False,
        }


def _fail(message: str) -> None:
    raise ReconstructionValidationError("I6.7E1 full dense artifact preflight: " + message)


def _header(handle: BinaryIO, digest) -> tuple[str, int, np.dtype, bool]:
    total = 0
    lines: list[str] = []
    while True:
        raw = handle.readline(I67E1_MAX_HEADER_BYTES - total + 1)
        if not raw:
            _fail("PLY header truncated")
        total += len(raw)
        if total > I67E1_MAX_HEADER_BYTES:
            _fail("PLY header exceeds bounded parsing budget")
        digest.update(raw)
        try:
            text = raw.decode("ascii").strip()
        except UnicodeDecodeError:
            _fail("PLY header is not ASCII")
        lines.append(text)
        if text == "end_header":
            break
    if not lines or lines[0] != "ply":
        _fail("not a PLY file")
    fmt = ""
    count = None
    element = ""
    props: list[tuple[str, str]] = []
    vertex_seen = False
    for line in lines[1:-1]:
        parts = line.split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "format":
            if len(parts) != 3 or parts[2] != "1.0" or fmt:
                _fail("invalid or repeated PLY format")
            fmt = parts[1]
        elif parts[0] == "element":
            if len(parts) != 3:
                _fail("invalid PLY element header")
            element = parts[1]
            if element == "vertex":
                if vertex_seen:
                    _fail("repeated PLY vertex element")
                vertex_seen = True
                try:
                    count = int(parts[2])
                except ValueError:
                    _fail("invalid PLY vertex count")
        elif parts[0] == "property" and element == "vertex":
            if len(parts) != 3 or parts[1] not in _PLY_TYPES:
                _fail("unsupported or list-valued vertex property")
            props.append((parts[2], parts[1]))
    if fmt not in {"binary_little_endian", "binary_big_endian", "ascii"}:
        _fail("unsupported PLY format")
    if count is None or count <= 0:
        _fail("PLY must contain vertices")
    names = [name for name, _ in props]
    if len(set(names)) != len(names) or not {"x", "y", "z"}.issubset(names):
        _fail("PLY vertex properties are duplicated or lack XYZ")
    rgb = {"red", "green", "blue"}.issubset(names)
    if any(color in names for color in ("red", "green", "blue")) and not rgb:
        _fail("incomplete RGB triplet")
    if fmt == "ascii":
        # dtype remains a metadata type; ASCII payload is handled separately.
        dtype = np.dtype([(name, _PLY_TYPES[typ]) for name, typ in props])
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + _PLY_TYPES[typ]) for name, typ in props])
    return fmt, count, dtype, rgb


def _bounds_update(rows: np.ndarray, lower: np.ndarray, upper: np.ndarray, has_rgb: bool) -> None:
    xyz = np.column_stack((rows["x"], rows["y"], rows["z"]))
    if not np.isfinite(xyz).all():
        _fail("non-finite XYZ in full vertex population")
    np.minimum(lower, np.min(xyz, axis=0), out=lower)
    np.maximum(upper, np.max(xyz, axis=0), out=upper)
    if has_rgb:
        for name in ("red", "green", "blue"):
            values = rows[name]
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 255):
                _fail("RGB outside [0, 255] in full vertex population")


def _bounds_ascii(rows: list[tuple[float, ...]], names: tuple[str, ...],
                  lower: np.ndarray, upper: np.ndarray, has_rgb: bool) -> None:
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        _fail("non-finite value in ASCII vertex population")
    xyz = values[:, [names.index(name) for name in ("x", "y", "z")]]
    np.minimum(lower, np.min(xyz, axis=0), out=lower)
    np.maximum(upper, np.max(xyz, axis=0), out=upper)
    if has_rgb:
        rgb = values[:, [names.index(name) for name in ("red", "green", "blue")]]
        if np.any(rgb < 0) or np.any(rgb > 255):
            _fail("RGB outside [0, 255] in full ASCII vertex population")


def inspect_full_dense_ply(
    path: str | Path, *, expected_sha256: str, expected_root: str | Path,
    expected_vertex_count: int | None = None,
) -> DenseArtifactPreflight:
    """Scan all vertices with bounded allocations and verify exact artifact identity.

    This is deliberately NOT an admission or persistence API. `expected_root`
    must be the already-authorized dense provider output root, not an arbitrary
    user-selected parent. Caller must bind sparse/session/provider identities
    separately before admitting the completed result in a later I6.7E step.
    """
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        digit not in "0123456789abcdef" for digit in expected_sha256
    ):
        _fail("expected SHA256 must be 64 lowercase hexadecimal characters")
    root = Path(expected_root).expanduser().resolve(strict=True)
    file_path = Path(path).expanduser()
    if file_path.is_symlink():
        _fail("symlinked dense geometry is not allowed")
    source = file_path.resolve(strict=True)
    if not root.is_dir() or not source.is_file() or source.suffix.lower() != ".ply":
        _fail("dense geometry must be a regular PLY file under a valid output root")
    try:
        source.relative_to(root)
    except ValueError:
        _fail("dense artifact escapes admitted provider output root")
    if source == root:
        _fail("expected a dense geometry file, not the output directory")
    digest = hashlib.sha256()
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    with source.open("rb") as handle:
        before = source.stat()
        fmt, count, dtype, has_rgb = _header(handle, digest)
        if expected_vertex_count is not None and (
            isinstance(expected_vertex_count, bool) or count != expected_vertex_count
        ):
            _fail("full artifact vertex count disagrees with expected result")
        if fmt == "ascii":
            names = tuple(dtype.names or ())
            collected: list[tuple[float, ...]] = []
            for _ in range(count):
                line = handle.readline(4096)
                if not line or len(line) == 4096 and not line.endswith(b"\n"):
                    _fail("truncated or unbounded ASCII vertex record")
                digest.update(line)
                try:
                    tokens = line.decode("ascii").split()
                    if len(tokens) != len(names):
                        _fail("invalid ASCII vertex row width")
                    collected.append(tuple(float(v) for v in tokens))
                except (UnicodeDecodeError, ValueError, OverflowError):
                    _fail("invalid ASCII vertex record")
                if len(collected) == I67E1_ASCII_CHUNK_POINTS:
                    _bounds_ascii(collected, names, lower, upper, has_rgb)
                    collected.clear()
            if collected:
                _bounds_ascii(collected, names, lower, upper, has_rgb)
        else:
            rows_per_chunk = min(I67E1_CHUNK_POINTS, max(1, I67E1_MAX_BINARY_CHUNK_BYTES // dtype.itemsize))
            for start in range(0, count, rows_per_chunk):
                n = min(rows_per_chunk, count - start)
                data = handle.read(n * dtype.itemsize)
                if len(data) != n * dtype.itemsize:
                    _fail("truncated binary PLY vertex population")
                digest.update(data)
                _bounds_update(np.frombuffer(data, dtype=dtype, count=n), lower, upper, has_rgb)
        # Include all trailing file bytes in the digest; no silent truncation.
        while block := handle.read(1024 * 1024):
            digest.update(block)
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            _fail("dense artifact changed during qualification")
    actual = digest.hexdigest()
    if actual != expected_sha256:
        _fail("full dense artifact SHA256 mismatch")
    return DenseArtifactPreflight(
        path=str(source), sha256=actual, file_bytes=before.st_size,
        vertex_count=count, has_rgb=has_rgb, ply_format=fmt,
        bounds_min=tuple(float(value) for value in lower),
        bounds_max=tuple(float(value) for value in upper),
        scanned_vertex_count=count,
    )


__all__ = ["I67E1_PHASE", "I67E1_CHUNK_POINTS", "I67E1_ASCII_CHUNK_POINTS", "DenseArtifactPreflight", "inspect_full_dense_ply"]
