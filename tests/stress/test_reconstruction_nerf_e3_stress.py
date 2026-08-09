from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NeuralSceneDataset,
    NeuralSceneFrame,
    NerfRealSceneValidationSpec,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    deterministic_nerf_validation_split,
)


def test_e3_deterministic_split_is_stable_across_repeated_scene_sizes() -> None:
    for count in range(12, 65):
        ids = tuple(f"f{index:03d}" for index in range(count))
        heldout_count = 3 if count < 20 else 4
        first = deterministic_nerf_validation_split(ids, validation_count=heldout_count)
        for _ in range(20):
            assert deterministic_nerf_validation_split(ids, validation_count=heldout_count) == first
        train, heldout = first
        assert len(set(train) | set(heldout)) == count
        assert not set(train) & set(heldout)


def test_e3_spec_fingerprint_is_stable_under_repeated_construction(tmp_path: Path) -> None:
    frames = []
    for index in range(12):
        image = (tmp_path / f"f{index}.png").resolve()
        Image.fromarray(np.full((64, 64, 3), index * 10, dtype=np.uint8), mode="RGB").save(image)
        pose = np.eye(4)
        pose[0, 3] = index * 0.02
        frames.append(
            NeuralSceneFrame(
                frame_id=f"f{index}", image_path=str(image), width=64, height=64,
                fx=55.0, fy=55.0, cx=32.0, cy=32.0,
                T_reconstruction_from_camera=pose, pose_layer="optimized_keyframe",
                camera_profile_id="camera",
            )
        )
    dataset = NeuralSceneDataset(
        session_id="stress-e3", scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="reconstruction", camera_profile_id="camera",
        scale_profile_id="arbitrary", metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY, world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None, relative_scale_uncertainty=None, frames=tuple(frames),
    )
    train, heldout = deterministic_nerf_validation_split(tuple(frame.frame_id for frame in frames), validation_count=3)
    values = set()
    for _ in range(100):
        spec = NerfRealSceneValidationSpec(
            validation_id="stress", source_label="real", dataset=dataset,
            training_frame_ids=train, validation_frame_ids=heldout, training_steps=128,
        )
        values.add(spec.validation_fingerprint_sha256)
    assert len(values) == 1
