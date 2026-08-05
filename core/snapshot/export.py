from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.geometry.export import export_ply
from core.geometry.projection import depth_to_points


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def align_point_colors(
    *,
    rgb_bgr: np.ndarray,
    points: np.ndarray,
    colors: Any,
    point_stride: int,
    depth_shape: tuple[int, int] | None,
) -> np.ndarray:
    """Return flat uint8 RGB colors aligned one-to-one with a point grid.

    Adapters may return a spatially strided point grid to respect a point
    budget. When model-provided colors are absent, derive colors from the
    frozen BGR frame using the same stride. If an adapter uses a different
    native grid, resize the RGB image exactly to the point grid instead of
    repeating, truncating, or broadcasting pixels.
    """
    point_array = np.asarray(points)
    if point_array.ndim != 3 or point_array.shape[2] != 3:
        raise ValueError(f"Snapshot points must be an H×W×3 grid, got {point_array.shape}.")

    target_height, target_width = point_array.shape[:2]
    target_count = target_height * target_width

    if colors is not None:
        color_array = np.asarray(colors)
        if color_array.shape != point_array.shape:
            raise ValueError(
                "Snapshot point colors must match the point grid: "
                f"points={point_array.shape}, colors={color_array.shape}."
            )
        return np.ascontiguousarray(color_array, dtype=np.uint8).reshape(target_count, 3)

    rgb = np.ascontiguousarray(rgb_bgr[:, :, ::-1], dtype=np.uint8)
    stride = max(1, int(point_stride))

    if depth_shape is not None and rgb.shape[:2] != tuple(depth_shape):
        depth_height, depth_width = (int(depth_shape[0]), int(depth_shape[1]))
        rgb = np.asarray(
            Image.fromarray(rgb).resize(
                (depth_width, depth_height),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )

    sampled = rgb[::stride, ::stride]
    if sampled.shape[:2] != (target_height, target_width):
        sampled = np.asarray(
            Image.fromarray(rgb).resize(
                (target_width, target_height),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )

    return np.ascontiguousarray(sampled, dtype=np.uint8).reshape(target_count, 3)


def align_point_mask(
    *,
    valid_mask: Any,
    points: np.ndarray,
    point_stride: int,
) -> np.ndarray:
    """Return a flat boolean mask aligned one-to-one with a point grid."""
    point_array = np.asarray(points)
    if point_array.ndim != 3 or point_array.shape[2] != 3:
        raise ValueError(f"Snapshot points must be an H×W×3 grid, got {point_array.shape}.")

    target_height, target_width = point_array.shape[:2]
    target_count = target_height * target_width
    if valid_mask is None:
        return np.ones(target_count, dtype=bool)

    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Snapshot validity mask must be H×W, got {mask.shape}.")

    if mask.shape == (target_height, target_width):
        return np.ascontiguousarray(mask).reshape(-1)

    stride = max(1, int(point_stride))
    sampled = mask[::stride, ::stride]
    if sampled.shape != (target_height, target_width):
        sampled = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (target_width, target_height),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ) > 0

    return np.ascontiguousarray(sampled, dtype=bool).reshape(target_count)


def export_snapshot_package(destination: str | Path, result: Any) -> Path:
    """Atomically export all outputs supported by one normalized result."""
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        rgb = np.asarray(result.rgb, dtype=np.uint8)
        Image.fromarray(rgb[:, :, ::-1]).save(tmp / "rgb.png")

        depth = None if result.depth is None else np.asarray(result.depth, dtype=np.float32)
        if depth is not None:
            np.save(tmp / "depth.npy", depth)
            finite = np.isfinite(depth)
            if finite.any():
                lo, hi = np.percentile(depth[finite], [2, 98])
                if hi <= lo:
                    hi = lo + 1.0
                normalized = np.clip((depth - float(lo)) / float(hi - lo), 0.0, 1.0)
                preview = (normalized * 255.0).astype(np.uint8)
                if result.valid_mask is not None:
                    preview = np.where(np.asarray(result.valid_mask, dtype=bool), preview, 0).astype(np.uint8)
                Image.fromarray(preview).save(tmp / "depth_preview.png")

        optional_arrays = {
            "confidence.npy": result.confidence,
            "normals.npy": result.normals,
            "validity_mask.npy": result.valid_mask,
        }
        for name, value in optional_arrays.items():
            if value is not None:
                np.save(tmp / name, np.asarray(value))

        points = result.points
        colors = result.colors
        if points is None and depth is not None:
            points = depth_to_points(depth, result.intrinsics)
        if points is not None:
            point_array = np.asarray(points, dtype=np.float32)
            flat_points = point_array.reshape(-1, 3)
            flat_colors = align_point_colors(
                rgb_bgr=rgb,
                points=point_array,
                colors=colors,
                point_stride=int(result.point_stride),
                depth_shape=None if depth is None else depth.shape[:2],
            )
            valid = np.isfinite(flat_points).all(axis=1)
            valid &= align_point_mask(
                valid_mask=result.valid_mask,
                points=point_array,
                point_stride=int(result.point_stride),
            )
            export_ply(tmp / "point_cloud.ply", flat_points[valid], flat_colors[valid])

        metadata = {
            "model_id": result.model_id,
            "model_name": result.model_name,
            "precision": result.precision,
            "depth_scale": result.depth_scale,
            "is_metric": bool(result.is_metric),
            "intrinsics_source": result.intrinsics_source,
            "confidence_kind": result.confidence_kind,
            "latency_ms": float(result.latency_ms),
            "preprocess_ms": result.preprocess_ms,
            "postprocess_ms": result.postprocess_ms,
            "worker_total_ms": result.worker_total_ms,
            "point_stride": int(result.point_stride),
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape) if depth is not None else None,
            "intrinsics": _jsonable(result.intrinsics),
        }
        (tmp / "snapshot_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp, target)
        return target
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
