from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NeuralSceneDataset,
    NeuralSceneFrame,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
)
from scripts.nerf_workers.common import export_nerfstudio_dataset


def test_e2_exported_dataset_round_trips_images_and_pose_convention(tmp_path: Path) -> None:
    frames = []
    for index in range(3):
        image = (tmp_path / f"image-{index}.ppm").resolve()
        image.write_bytes(b"P6\n8 6\n255\n" + bytes([50 + index, 90, 130]) * 48)
        pose = np.eye(4)
        pose[2, 3] = 1.0 + index * 0.1
        frames.append(
            NeuralSceneFrame(
                frame_id=f"f{index}", image_path=str(image), width=8, height=6,
                fx=7.0, fy=7.0, cx=4.0, cy=3.0,
                T_reconstruction_from_camera=pose, pose_layer="optimized_keyframe",
                camera_profile_id="camera",
            )
        )
    dataset = NeuralSceneDataset(
        session_id="integration-e2", scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="reconstruction", camera_profile_id="camera",
        scale_profile_id="arbitrary", metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY, world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None, relative_scale_uncertainty=None, frames=tuple(frames),
    )
    path = export_nerfstudio_dataset(dataset, tmp_path / "nerfstudio")
    payload = json.loads(path.read_text())
    assert len(payload["frames"]) == 3
    assert len(payload["train_filenames"]) == 2
    assert len(payload["test_filenames"]) == 1
    assert np.asarray(payload["frames"][0]["transform_matrix"])[2, 2] == -1.0
