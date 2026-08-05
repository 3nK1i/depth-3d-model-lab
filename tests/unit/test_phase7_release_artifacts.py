"""Snapshot and Stable Video release-artifact validation tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.runtime.release_artifacts import (
    latest_completed_export,
    snapshot_export_check,
    stable_video_export_check,
)
from core.snapshot.export import export_snapshot_package
from core.stable_video.export import StableVideoExportSession


def _snapshot_result() -> SimpleNamespace:
    height, width = 12, 16
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 1] = 130
    depth = np.linspace(0.5, 4.0, height * width, dtype=np.float32).reshape(height, width)
    yy, xx = np.mgrid[:height, :width]
    points = np.stack((xx * 0.01, yy * 0.01, depth), axis=-1).astype(np.float32)
    return SimpleNamespace(
        rgb=rgb,
        depth=depth,
        confidence=None,
        normals=None,
        valid_mask=np.ones((height, width), dtype=bool),
        points=points,
        colors=rgb[:, :, ::-1],
        intrinsics=np.asarray(
            [[100.0, 0.0, width / 2], [0.0, 100.0, height / 2], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        model_id="development-mock",
        model_name="Development Mock",
        precision="FP32",
        depth_scale="synthetic",
        is_metric=False,
        intrinsics_source="estimated",
        confidence_kind="none",
        latency_ms=1.0,
        preprocess_ms=0.1,
        postprocess_ms=0.2,
        worker_total_ms=1.3,
        point_stride=1,
    )


def _create_snapshot(path: Path) -> Path:
    result = _snapshot_result()
    output = export_snapshot_package(path, result)
    np.savez_compressed(
        output / "result.npz",
        rgb=result.rgb,
        depth=result.depth,
        points=result.points,
        colors=result.colors,
        intrinsics=result.intrinsics,
        valid_mask=result.valid_mask,
        point_stride=np.asarray([1], dtype=np.int32),
    )
    return output


def _create_stable_video(path: Path, *, frames: int = 4) -> Path:
    session = StableVideoExportSession(
        path,
        fps=12.0,
        model_id="development-mock",
        model_name="Development Mock",
        temporal_mode="generic_smoothing",
        smoothing_strength=0.5,
        export_numeric_depth=True,
    )
    for index in range(frames):
        rgb = np.full((24, 32, 3), index * 20, dtype=np.uint8)
        depth = np.full((24, 32), 1.0 + index * 0.1, dtype=np.float32)
        session.append(
            rgb_bgr=rgb,
            depth=depth,
            performance={"frame_index": index, "latency_ms": 2.0},
        )
    return session.finalize(completed=True).output_dir


def test_snapshot_export_validator_accepts_complete_package(tmp_path: Path) -> None:
    export = _create_snapshot(tmp_path / "snapshots" / "snapshot_1")

    check, payload = snapshot_export_check(tmp_path / "snapshots")

    assert check.passed, check.detail
    assert payload is not None
    assert payload["path"] == str(export.resolve())
    assert payload["intrinsics_source"] == "estimated"


def test_snapshot_export_validator_rejects_calibrated_claim(tmp_path: Path) -> None:
    export = _create_snapshot(tmp_path / "snapshot")
    manifest_path = export / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["intrinsics_source"] = "calibrated"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    check, _ = snapshot_export_check(export)

    assert not check.passed
    assert "calibrated" in check.detail


def test_stable_video_validator_accepts_completed_synchronized_export(tmp_path: Path) -> None:
    export = _create_stable_video(tmp_path / "stable" / "run_1", frames=5)

    check, payload = stable_video_export_check(tmp_path / "stable")

    assert check.passed, check.detail
    assert payload is not None
    assert payload["path"] == str(export.resolve())
    assert payload["frame_count"] == 5
    assert payload["numeric_depth_exported"] is True


def test_stable_video_validator_rejects_partial_or_row_mismatch(tmp_path: Path) -> None:
    export = _create_stable_video(tmp_path / "run", frames=3)
    (export / "performance.csv").write_text("frame_index\n0\n", encoding="utf-8")

    check, _ = stable_video_export_check(export)

    assert not check.passed
    assert "do not match" in check.detail


def test_latest_completed_export_ignores_partial_directories(tmp_path: Path) -> None:
    completed = tmp_path / "snapshot_1"
    completed.mkdir()
    (completed / "snapshot_manifest.json").write_text("{}", encoding="utf-8")
    partial = tmp_path / "snapshot_2.partial"
    partial.mkdir()
    (partial / "snapshot_manifest.json").write_text("{}", encoding="utf-8")

    assert latest_completed_export(tmp_path, "snapshot_manifest.json") == completed.resolve()
