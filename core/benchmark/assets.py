"""Deterministic, dependency-light benchmark assets.

The benchmark inputs are generated from NumPy primitives rather than downloaded
photographs. This makes the performance workload reproducible on the RTX 2060
and RTX 5090 without network access or copyright-sensitive test assets.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ASSET_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BenchmarkAssetManifest:
    root: Path
    landscape: Path
    portrait: Path
    temporal: Path
    manifest: Path
    checksums: dict[str, str]

    def asset_path(self, key: str) -> Path:
        normalized = str(key).strip().lower()
        values = {
            "landscape": self.landscape,
            "portrait": self.portrait,
            "temporal": self.temporal,
        }
        try:
            return values[normalized]
        except KeyError:
            raise KeyError(f"Unknown benchmark asset key: {key!r}") from None


def ensure_benchmark_assets(
    root: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> BenchmarkAssetManifest:
    """Create or validate the deterministic Phase 6 benchmark dataset."""

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    landscape_path = root_path / "landscape_bgr.npy"
    portrait_path = root_path / "portrait_bgr.npy"
    temporal_path = root_path / "temporal_bgr.npy"
    manifest_path = root_path / "manifest.json"

    expected = (landscape_path, portrait_path, temporal_path, manifest_path)
    if overwrite or not all(path.is_file() for path in expected):
        landscape = _landscape_frame(width=640, height=384)
        portrait = np.ascontiguousarray(np.rot90(landscape, 1))
        temporal = _temporal_sequence(frame_count=12, width=640, height=384)
        _atomic_save_npy(landscape_path, landscape)
        _atomic_save_npy(portrait_path, portrait)
        _atomic_save_npy(temporal_path, temporal)
        checksums = {
            landscape_path.name: _sha256(landscape_path),
            portrait_path.name: _sha256(portrait_path),
            temporal_path.name: _sha256(temporal_path),
        }
        payload = {
            "schema_version": _ASSET_SCHEMA_VERSION,
            "color_order": "BGR",
            "dtype": "uint8",
            "assets": {
                "landscape": {
                    "file": landscape_path.name,
                    "shape": list(landscape.shape),
                    "sha256": checksums[landscape_path.name],
                },
                "portrait": {
                    "file": portrait_path.name,
                    "shape": list(portrait.shape),
                    "sha256": checksums[portrait_path.name],
                },
                "temporal": {
                    "file": temporal_path.name,
                    "shape": list(temporal.shape),
                    "sha256": checksums[temporal_path.name],
                },
            },
        }
        _atomic_write_json(manifest_path, payload)

    return validate_benchmark_assets(root_path)


def validate_benchmark_assets(root: str | os.PathLike[str]) -> BenchmarkAssetManifest:
    root_path = Path(root)
    manifest_path = root_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != _ASSET_SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark asset manifest schema.")
    if payload.get("color_order") != "BGR" or payload.get("dtype") != "uint8":
        raise ValueError("Benchmark asset format metadata is invalid.")

    paths: dict[str, Path] = {}
    checksums: dict[str, str] = {}
    for key in ("landscape", "portrait", "temporal"):
        metadata = payload.get("assets", {}).get(key)
        if not isinstance(metadata, dict):
            raise ValueError(f"Benchmark asset {key!r} is missing from the manifest.")
        path = root_path / str(metadata.get("file", ""))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        expected = str(metadata.get("sha256", ""))
        if actual != expected:
            raise ValueError(f"Checksum mismatch for benchmark asset {path.name}.")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(int(value) for value in metadata.get("shape", ()))
        if tuple(array.shape) != expected_shape or array.dtype != np.uint8:
            raise ValueError(f"Invalid benchmark array metadata for {path.name}.")
        paths[key] = path
        checksums[path.name] = actual

    return BenchmarkAssetManifest(
        root=root_path,
        landscape=paths["landscape"],
        portrait=paths["portrait"],
        temporal=paths["temporal"],
        manifest=manifest_path,
        checksums=checksums,
    )


def load_asset_frames(manifest: BenchmarkAssetManifest, key: str) -> tuple[np.ndarray, ...]:
    array = np.load(manifest.asset_path(key), allow_pickle=False)
    if array.dtype != np.uint8:
        raise ValueError("Benchmark inputs must be uint8 BGR arrays.")
    if array.ndim == 3:
        return (np.ascontiguousarray(array),)
    if array.ndim == 4:
        return tuple(np.ascontiguousarray(frame) for frame in array)
    raise ValueError(f"Unsupported benchmark array shape: {array.shape!r}")


def _landscape_frame(*, width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    x_norm = x.astype(np.float32) / max(1, width - 1)
    y_norm = y.astype(np.float32) / max(1, height - 1)

    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = np.clip(35 + 150 * x_norm, 0, 255).astype(np.uint8)
    frame[..., 1] = np.clip(25 + 145 * y_norm, 0, 255).astype(np.uint8)
    frame[..., 2] = np.clip(45 + 120 * (1.0 - x_norm * y_norm), 0, 255).astype(np.uint8)

    # Deterministic geometric objects create local edges and broad depth cues.
    _fill_rect(frame, 48, 70, 190, 315, (35, 180, 225))
    _fill_rect(frame, 235, 115, 410, 330, (210, 90, 45))
    _fill_rect(frame, 455, 55, 595, 260, (65, 210, 85))
    _fill_circle(frame, 335, 92, 52, (235, 220, 65))
    _fill_circle(frame, 545, 315, 38, (190, 70, 220))
    return np.ascontiguousarray(frame)


def _temporal_sequence(*, frame_count: int, width: int, height: int) -> np.ndarray:
    base = _landscape_frame(width=width, height=height)
    frames: list[np.ndarray] = []
    for index in range(frame_count):
        shift = index * 7
        frame = np.roll(base, shift=shift, axis=1).copy()
        left = 30 + index * 26
        top = 245 - index * 4
        _fill_rect(frame, left, top, left + 85, top + 62, (20, 35, 245))
        _fill_circle(frame, width - 70 - index * 12, 80 + index * 8, 24, (245, 190, 35))
        frames.append(np.ascontiguousarray(frame))
    return np.stack(frames, axis=0)


def _fill_rect(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    xa, xb = sorted((max(0, x0), min(width, x1)))
    ya, yb = sorted((max(0, y0), min(height, y1)))
    frame[ya:yb, xa:xb] = color


def _fill_circle(
    frame: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    y, x = np.ogrid[:height, :width]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    frame[mask] = color


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    handle = tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
