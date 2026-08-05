"""Release-gate validation for Snapshot and Stable Video artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.runtime.release import (
    latest_output_directory,
    snapshot_export_check,
    stable_video_export_check,
)


def _snapshot(path: Path) -> Path:
    path.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(path / "rgb.png")
    Image.fromarray(np.zeros((8, 10), dtype=np.uint8)).save(path / "depth_preview.png")
    np.save(path / "depth.npy", np.ones((8, 10), dtype=np.float32))
    (path / "point_cloud.ply").write_bytes(b"ply\n" + b"x" * 256)
    (path / "snapshot_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "development-mock",
                "model_name": "Development Mock",
                "depth_scale": "synthetic",
                "intrinsics_source": "none",
                "rgb_shape": [8, 10, 3],
            }
        ),
        encoding="utf-8",
    )
    return path


def _stable(path: Path, *, frames: int = 3) -> Path:
    path.mkdir(parents=True)
    (path / "rgb_video.mp4").write_bytes(b"rgb")
    (path / "depth_video.mp4").write_bytes(b"depth")
    with (path / "performance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "latency_ms"])
        writer.writeheader()
        for index in range(frames):
            writer.writerow({"frame_index": index, "latency_ms": 5.0})
    (path / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "model_id": "moge-2",
                "model_name": "MoGe-2",
                "frame_count": frames,
                "resolution": [32, 24],
                "numeric_depth_exported": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_snapshot_release_check_accepts_complete_package(tmp_path: Path) -> None:
    check, payload = snapshot_export_check(_snapshot(tmp_path / "snapshot_1"))
    assert check.passed, check.detail
    assert payload is not None
    assert "Development Mock" in check.detail


def test_snapshot_release_check_rejects_calibrated_claim(tmp_path: Path) -> None:
    path = _snapshot(tmp_path / "snapshot_1")
    manifest_path = path / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["intrinsics_source"] = "calibrated"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    check, _ = snapshot_export_check(path)
    assert not check.passed
    assert "calibrated" in check.detail


def test_stable_video_release_check_accepts_complete_package(tmp_path: Path) -> None:
    check, payload = stable_video_export_check(_stable(tmp_path / "stable_video_1"))
    assert check.passed, check.detail
    assert payload is not None
    assert "3 frames" in check.detail


def test_stable_video_release_check_rejects_row_count_drift(tmp_path: Path) -> None:
    path = _stable(tmp_path / "stable_video_1", frames=2)
    manifest_path = path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_count"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    check, _ = stable_video_export_check(path)
    assert not check.passed
    assert "performance rows" in check.detail


def test_latest_output_directory_ignores_partial_directories(tmp_path: Path) -> None:
    older = _snapshot(tmp_path / "snapshot_20260101_000000")
    newer = _snapshot(tmp_path / "snapshot_20260101_000001")
    partial = tmp_path / ".snapshot_20260101_000002.partial"
    partial.mkdir()
    older.touch()
    newer.touch()

    assert latest_output_directory(tmp_path, "snapshot_") == newer
