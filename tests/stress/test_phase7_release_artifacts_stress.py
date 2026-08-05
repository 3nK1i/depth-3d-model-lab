"""Repeated dependency-light validation of Phase 7 finalized artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.runtime.release import snapshot_export_check, stable_video_export_check


def test_repeated_artifact_validation_is_stable(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot_1"
    snapshot.mkdir()
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(snapshot / "rgb.png")
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(snapshot / "depth_preview.png")
    np.save(snapshot / "depth.npy", np.ones((4, 5), dtype=np.float32))
    (snapshot / "point_cloud.ply").write_bytes(b"ply\n" + b"p" * 256)
    (snapshot / "snapshot_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "mock",
                "model_name": "Mock",
                "depth_scale": "relative",
                "intrinsics_source": "estimated",
                "rgb_shape": [4, 5, 3],
            }
        ),
        encoding="utf-8",
    )

    stable = tmp_path / "stable_video_1"
    stable.mkdir()
    (stable / "rgb_video.mp4").write_bytes(b"rgb")
    (stable / "depth_video.mp4").write_bytes(b"depth")
    with (stable / "performance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index"])
        writer.writeheader()
        for index in range(50):
            writer.writerow({"frame_index": index})
    (stable / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "model_id": "mock",
                "model_name": "Mock",
                "frame_count": 50,
                "resolution": [5, 4],
                "numeric_depth_exported": False,
            }
        ),
        encoding="utf-8",
    )

    for _ in range(100):
        assert snapshot_export_check(snapshot)[0].passed
        assert stable_video_export_check(stable)[0].passed
