"""Release-time validation for Phase 7 operational-mode artifacts.

The validators are deliberately read-only. They verify the latest completed
Benchmark, Snapshot, and Stable Video outputs without importing model-specific
code or mutating the generated artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.runtime.release import ReleaseCheck


def latest_completed_export(path: str | Path, marker: str) -> Path | None:
    """Resolve *path* to a completed export directory containing *marker*.

    ``path`` may point directly at one export or at a parent containing many
    timestamped exports. Partial directories are ignored.
    """

    candidate = Path(path).expanduser()
    if candidate.is_dir() and (candidate / marker).is_file():
        return candidate.resolve()
    if not candidate.is_dir():
        return None

    matches = [
        child
        for child in candidate.iterdir()
        if child.is_dir()
        and not child.name.endswith(".partial")
        and (child / marker).is_file()
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: (item.stat().st_mtime_ns, item.name)).resolve()


def snapshot_export_check(
    path: str | Path,
) -> tuple[ReleaseCheck, dict[str, Any] | None]:
    """Validate the newest complete Snapshot export."""

    export_dir = latest_completed_export(path, "snapshot_manifest.json")
    if export_dir is None:
        return ReleaseCheck(
            "phase7_snapshot_export",
            False,
            f"no completed Snapshot export under {Path(path)}",
        ), None

    required = (
        "rgb.png",
        "depth.npy",
        "depth_preview.png",
        "point_cloud.ply",
        "result.npz",
        "snapshot_manifest.json",
    )
    missing = [name for name in required if not (export_dir / name).is_file()]
    if missing:
        return ReleaseCheck(
            "phase7_snapshot_export",
            False,
            f"{export_dir}: missing {', '.join(missing)}",
        ), None

    try:
        manifest = _read_json(export_dir / "snapshot_manifest.json")
        _require_text(manifest, "model_id")
        _require_text(manifest, "model_name")
        _require_text(manifest, "depth_scale")
        intrinsics_source = _require_text(manifest, "intrinsics_source")
        if intrinsics_source.strip().lower() == "calibrated":
            raise ValueError("Snapshot must not claim calibrated intrinsics in Phase 7.")
        if not isinstance(manifest.get("is_metric"), bool):
            raise ValueError("is_metric must be boolean")

        rgb = cv2.imread(str(export_dir / "rgb.png"), cv2.IMREAD_COLOR)
        preview = cv2.imread(str(export_dir / "depth_preview.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb.png is unreadable or not a three-channel image")
        if preview is None or preview.ndim != 2:
            raise ValueError("depth_preview.png is unreadable or not single-channel")

        depth = np.load(export_dir / "depth.npy", allow_pickle=False)
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError(f"depth.npy must be a non-empty 2D array, got {depth.shape}")
        if not np.isfinite(depth).any():
            raise ValueError("depth.npy contains no finite values")

        expected_depth_shape = manifest.get("depth_shape")
        if expected_depth_shape is not None and list(depth.shape) != list(expected_depth_shape):
            raise ValueError(
                f"depth shape mismatch: manifest={expected_depth_shape}, file={list(depth.shape)}"
            )
        expected_rgb_shape = manifest.get("rgb_shape")
        if expected_rgb_shape is not None and list(rgb.shape) != list(expected_rgb_shape):
            raise ValueError(
                f"RGB shape mismatch: manifest={expected_rgb_shape}, file={list(rgb.shape)}"
            )

        with np.load(export_dir / "result.npz", allow_pickle=False) as result:
            required_arrays = {"rgb", "depth", "points", "colors", "intrinsics", "valid_mask", "point_stride"}
            missing_arrays = sorted(required_arrays - set(result.files))
            if missing_arrays:
                raise ValueError("result.npz missing arrays: " + ", ".join(missing_arrays))

        ply = export_dir / "point_cloud.ply"
        if ply.stat().st_size <= 64:
            raise ValueError("point_cloud.ply is empty")
        header = ply.read_bytes()[:256]
        if not header.startswith(b"ply") or b"element vertex" not in header:
            raise ValueError("point_cloud.ply has an invalid PLY header")

        payload = {
            "path": str(export_dir),
            "model_id": manifest["model_id"],
            "depth_scale": manifest["depth_scale"],
            "is_metric": manifest["is_metric"],
            "intrinsics_source": intrinsics_source,
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
        }
        return ReleaseCheck(
            "phase7_snapshot_export",
            True,
            f"valid: {export_dir.name} · {manifest['model_id']} · {depth.shape[1]}×{depth.shape[0]}",
        ), payload
    except Exception as exc:  # noqa: BLE001
        return ReleaseCheck(
            "phase7_snapshot_export",
            False,
            f"invalid {export_dir}: {exc}",
        ), None


def stable_video_export_check(
    path: str | Path,
) -> tuple[ReleaseCheck, dict[str, Any] | None]:
    """Validate the newest completed Stable Video export and both MP4 streams."""

    export_dir = latest_completed_export(path, "run_manifest.json")
    if export_dir is None:
        return ReleaseCheck(
            "phase7_stable_video_export",
            False,
            f"no completed Stable Video export under {Path(path)}",
        ), None

    required = ("rgb_video.mp4", "depth_video.mp4", "performance.csv", "run_manifest.json")
    missing = [name for name in required if not (export_dir / name).is_file()]
    if missing:
        return ReleaseCheck(
            "phase7_stable_video_export",
            False,
            f"{export_dir}: missing {', '.join(missing)}",
        ), None

    try:
        manifest = _read_json(export_dir / "run_manifest.json")
        if manifest.get("status") != "completed":
            raise ValueError(f"status is {manifest.get('status')!r}, expected 'completed'")
        frame_count = int(manifest.get("frame_count", 0))
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        _require_text(manifest, "model_id")
        _require_text(manifest, "model_name")
        _require_text(manifest, "temporal_mode")
        resolution = manifest.get("resolution")
        if not (
            isinstance(resolution, list)
            and len(resolution) == 2
            and all(isinstance(value, int) and value > 0 for value in resolution)
        ):
            raise ValueError(f"invalid resolution: {resolution!r}")

        rows = _read_csv_rows(export_dir / "performance.csv")
        if len(rows) != frame_count:
            raise ValueError(
                f"performance.csv rows ({len(rows)}) do not match frame_count ({frame_count})"
            )

        rgb_info = _video_info(export_dir / "rgb_video.mp4")
        depth_info = _video_info(export_dir / "depth_video.mp4")
        width, height = int(resolution[0]), int(resolution[1])
        for label, info in (("RGB", rgb_info), ("depth", depth_info)):
            if info["width"] != width or info["height"] != height:
                raise ValueError(
                    f"{label} video resolution {info['width']}×{info['height']} "
                    f"does not match manifest {width}×{height}"
                )
            if info["frame_count"] <= 0:
                raise ValueError(f"{label} video contains no readable frames")
        if abs(rgb_info["frame_count"] - depth_info["frame_count"]) > 1:
            raise ValueError("RGB and depth videos have different frame counts")

        numeric_depth = bool(manifest.get("numeric_depth_exported"))
        if numeric_depth:
            depth_files = sorted((export_dir / "depth_frames").glob("depth_*.npy"))
            if len(depth_files) != frame_count:
                raise ValueError(
                    f"numeric depth count ({len(depth_files)}) does not match frame_count ({frame_count})"
                )

        payload = {
            "path": str(export_dir),
            "model_id": manifest["model_id"],
            "temporal_mode": manifest["temporal_mode"],
            "frame_count": frame_count,
            "resolution": resolution,
            "export_fps": manifest.get("export_fps"),
            "numeric_depth_exported": numeric_depth,
            "rgb_video": rgb_info,
            "depth_video": depth_info,
        }
        return ReleaseCheck(
            "phase7_stable_video_export",
            True,
            f"valid: {export_dir.name} · {frame_count} frames · {width}×{height}",
        ), payload
    except Exception as exc:  # noqa: BLE001
        return ReleaseCheck(
            "phase7_stable_video_export",
            False,
            f"invalid {export_dir}: {exc}",
        ), None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _require_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "frame_index" not in reader.fieldnames:
            raise ValueError("performance.csv must contain a frame_index column")
        return list(reader)


def _video_info(path: Path) -> dict[str, float | int]:
    if path.stat().st_size <= 64:
        raise ValueError(f"{path.name} is empty")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV cannot open {path.name}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        ok, _frame = capture.read()
        if not ok:
            raise ValueError(f"OpenCV cannot decode the first frame of {path.name}")
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": max(reported_frames, 1),
        }
    finally:
        capture.release()
