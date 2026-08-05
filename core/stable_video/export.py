"""Atomic Stable Video export.

The exporter owns no model logic. It receives already normalized RGB/depth
frames plus generic performance metadata and writes synchronized videos and a
machine-readable run manifest. A ``.partial`` directory is renamed only after
successful finalization so incomplete runs are never mistaken for completed
exports.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np



@dataclass(frozen=True, slots=True)
class StableVideoExportSummary:
    output_dir: Path
    frame_count: int
    duration_seconds: float
    completed: bool


class StableVideoExportSession:
    """Write one synchronized RGB/depth video run."""

    def __init__(
        self,
        output_dir: Path,
        *,
        fps: float,
        model_id: str,
        model_name: str,
        temporal_mode: str,
        smoothing_strength: float,
        export_numeric_depth: bool = False,
    ) -> None:
        if fps <= 0:
            raise ValueError("Export FPS must be positive.")
        self._final_dir = Path(output_dir)
        self._partial_dir = self._final_dir.with_name(self._final_dir.name + ".partial")
        if self._final_dir.exists() or self._partial_dir.exists():
            raise FileExistsError(f"Stable Video export already exists: {self._final_dir}")
        self._partial_dir.mkdir(parents=True, exist_ok=False)
        self._depth_dir = self._partial_dir / "depth_frames"
        self._export_numeric_depth = bool(export_numeric_depth)
        if self._export_numeric_depth:
            self._depth_dir.mkdir(parents=True, exist_ok=True)

        self._fps = float(fps)
        self._model_id = str(model_id)
        self._model_name = str(model_name)
        self._temporal_mode = str(temporal_mode)
        self._smoothing_strength = float(smoothing_strength)
        self._started_at = time.time()
        self._rgb_writer: cv2.VideoWriter | None = None
        self._depth_writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._frame_count = 0
        self._performance_rows: list[dict[str, Any]] = []
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def partial_dir(self) -> Path:
        return self._partial_dir

    def append(
        self,
        *,
        rgb_bgr: np.ndarray,
        depth: np.ndarray,
        performance: dict[str, Any],
        near: float | None = None,
        far: float | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Stable Video export session is already closed.")
        rgb = np.ascontiguousarray(rgb_bgr, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB frame must be HxWx3 uint8 BGR.")
        depth_array = np.asarray(depth, dtype=np.float32)
        if depth_array.ndim != 2:
            raise ValueError("Depth frame must be two-dimensional.")

        height, width = rgb.shape[:2]
        if self._size is None:
            self._open_writers(width, height)
        elif self._size != (width, height):
            target_width, target_height = self._size
            rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            height, width = target_height, target_width

        if depth_array.shape != (height, width):
            depth_array = cv2.resize(
                depth_array,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32, copy=False)
            if valid_mask is not None:
                valid_mask = cv2.resize(
                    np.asarray(valid_mask, dtype=np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

        finite = np.isfinite(depth_array) & (depth_array > 0.0)
        if valid_mask is not None:
            finite &= np.asarray(valid_mask, dtype=bool)
        if near is None or far is None or far <= near:
            values = depth_array[finite]
            if values.size:
                lo, hi = np.percentile(values, [2.0, 98.0])
                near = float(lo)
                far = float(hi if hi > lo else lo + 1.0)
            else:
                near, far = 0.0, 1.0
        span = max(float(far) - float(near), 1e-6)
        normalized = np.clip((depth_array - float(near)) / span, 0.0, 1.0)
        gray = np.asarray(normalized * 255.0, dtype=np.uint8)
        gray[~finite] = 0
        depth_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        depth_bgr[~finite] = 0
        depth_bgr = np.ascontiguousarray(depth_bgr, dtype=np.uint8)

        assert self._rgb_writer is not None and self._depth_writer is not None
        self._rgb_writer.write(rgb)
        self._depth_writer.write(depth_bgr)
        if self._export_numeric_depth:
            np.save(self._depth_dir / f"depth_{self._frame_count:06d}.npy", depth_array)

        row = dict(performance)
        row.setdefault("frame_index", self._frame_count)
        self._performance_rows.append(row)
        self._frame_count += 1

    def finalize(self, *, completed: bool = True, extra_manifest: dict[str, Any] | None = None) -> StableVideoExportSummary:
        if self._closed:
            raise RuntimeError("Stable Video export session is already finalized.")
        self._release_writers()
        ended_at = time.time()
        duration = max(0.0, ended_at - self._started_at)
        self._write_performance_csv()
        manifest = {
            "schema_version": 1,
            "status": "completed" if completed else "cancelled",
            "model_id": self._model_id,
            "model_name": self._model_name,
            "temporal_mode": self._temporal_mode,
            "smoothing_strength": self._smoothing_strength,
            "export_fps": self._fps,
            "frame_count": self._frame_count,
            "resolution": list(self._size) if self._size is not None else None,
            "started_at_unix": self._started_at,
            "ended_at_unix": ended_at,
            "duration_seconds": duration,
            "numeric_depth_exported": self._export_numeric_depth,
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        (self._partial_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._closed = True
        if completed:
            self._partial_dir.replace(self._final_dir)
            output = self._final_dir
        else:
            output = self._partial_dir
        return StableVideoExportSummary(output, self._frame_count, duration, completed)

    def abort(self) -> None:
        if self._closed:
            return
        self._release_writers()
        shutil.rmtree(self._partial_dir, ignore_errors=True)
        self._closed = True

    def _open_writers(self, width: int, height: int) -> None:
        self._size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._rgb_writer = cv2.VideoWriter(
            str(self._partial_dir / "rgb_video.mp4"), fourcc, self._fps, (width, height)
        )
        self._depth_writer = cv2.VideoWriter(
            str(self._partial_dir / "depth_video.mp4"), fourcc, self._fps, (width, height)
        )
        if not self._rgb_writer.isOpened() or not self._depth_writer.isOpened():
            self._release_writers()
            raise RuntimeError("OpenCV could not create Stable Video MP4 writers.")

    def _release_writers(self) -> None:
        for writer in (self._rgb_writer, self._depth_writer):
            if writer is not None:
                writer.release()
        self._rgb_writer = None
        self._depth_writer = None

    def _write_performance_csv(self) -> None:
        path = self._partial_dir / "performance.csv"
        if not self._performance_rows:
            path.write_text("frame_index\n", encoding="utf-8")
            return
        fieldnames: list[str] = []
        for row in self._performance_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._performance_rows)
