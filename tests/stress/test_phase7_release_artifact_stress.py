"""Repeated release-artifact validation stays deterministic and read-only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.runtime.release_artifacts import snapshot_export_check, stable_video_export_check
from core.snapshot.export import export_snapshot_package
from core.stable_video.export import StableVideoExportSession


def test_repeated_phase7_artifact_validation(tmp_path: Path) -> None:
    rgb = np.zeros((18, 24, 3), dtype=np.uint8)
    depth = np.ones((18, 24), dtype=np.float32)
    points = np.zeros((18, 24, 3), dtype=np.float32)
    points[..., 2] = 1.0
    result = SimpleNamespace(
        rgb=rgb,
        depth=depth,
        confidence=None,
        normals=None,
        valid_mask=np.ones((18, 24), dtype=bool),
        points=points,
        colors=rgb[:, :, ::-1],
        intrinsics=np.eye(3, dtype=np.float32),
        model_id="development-mock",
        model_name="Development Mock",
        precision="FP32",
        depth_scale="synthetic",
        is_metric=False,
        intrinsics_source="estimated",
        confidence_kind="none",
        latency_ms=1.0,
        preprocess_ms=0.1,
        postprocess_ms=0.1,
        worker_total_ms=1.2,
        point_stride=1,
    )
    snapshot = export_snapshot_package(tmp_path / "snapshots" / "snapshot", result)
    np.savez_compressed(
        snapshot / "result.npz",
        rgb=rgb,
        depth=depth,
        points=points,
        colors=rgb[:, :, ::-1],
        intrinsics=np.eye(3, dtype=np.float32),
        valid_mask=np.ones((18, 24), dtype=bool),
        point_stride=np.asarray([1], dtype=np.int32),
    )

    session = StableVideoExportSession(
        tmp_path / "stable" / "run",
        fps=15.0,
        model_id="development-mock",
        model_name="Development Mock",
        temporal_mode="generic_smoothing",
        smoothing_strength=0.5,
    )
    for index in range(30):
        session.append(
            rgb_bgr=np.full((18, 24, 3), index, dtype=np.uint8),
            depth=np.full((18, 24), 1.0 + index / 100.0, dtype=np.float32),
            performance={"frame_index": index, "latency_ms": 2.0},
        )
    session.finalize()

    snapshot_sizes = {path.name: path.stat().st_size for path in snapshot.iterdir() if path.is_file()}
    stable_dir = tmp_path / "stable" / "run"
    stable_sizes = {path.name: path.stat().st_size for path in stable_dir.iterdir() if path.is_file()}

    for _ in range(20):
        snapshot_check, _ = snapshot_export_check(tmp_path / "snapshots")
        stable_check, _ = stable_video_export_check(tmp_path / "stable")
        assert snapshot_check.passed, snapshot_check.detail
        assert stable_check.passed, stable_check.detail

    assert snapshot_sizes == {
        path.name: path.stat().st_size for path in snapshot.iterdir() if path.is_file()
    }
    assert stable_sizes == {
        path.name: path.stat().st_size for path in stable_dir.iterdir() if path.is_file()
    }
