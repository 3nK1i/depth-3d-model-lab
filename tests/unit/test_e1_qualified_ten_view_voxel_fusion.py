from __future__ import annotations

import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseSceneProfile,
    arbitrary_scale_reference,
    estimate_camera_geometry_from_horizontal_fov,
    fuse_dense_frames,
    reconstruction_unit_depth_alignment,
)
from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_e1_qualified_ten_view_voxel_fusion import (
    _derive_voxel_size,
    _support_histogram,
    _support_masks,
    _validate_d4_3_summary,
)


def _summary(*, pass_value: bool = True, qualified: bool = True, authorized: bool = True):
    candidate = {
        "pass": True,
        "workspace_orders": [5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
        "keyframe_ids": [f"kf-{i}" for i in range(10)],
        "scene_scale": {"robust_diagonal_reconstruction_units": 20.0},
    }
    return {
        "phase": "D4.3",
        "pass": pass_value,
        "status": "pass" if pass_value else "degraded",
        "selection": {
            "selected_candidate": "B",
            "ten_view_chain_qualified": qualified,
            "authorizes_next_fusion_milestone": authorized,
        },
        "candidates": {"B": candidate},
    }


def test_validate_d4_3_summary_accepts_only_authorized_qualified_chain():
    selected, candidate = _validate_d4_3_summary(_summary())
    assert selected == "B"
    assert candidate["workspace_orders"] == [5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_3_summary(_summary(qualified=False))
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_3_summary(_summary(authorized=False))


def test_derive_voxel_size_is_scene_relative_and_rejects_invalid_values():
    assert _derive_voxel_size(20.0, 0.004) == pytest.approx(0.08)
    with pytest.raises(ReconstructionValidationError):
        _derive_voxel_size(0.0, 0.004)
    with pytest.raises(ReconstructionValidationError):
        _derive_voxel_size(20.0, 0.0)


def test_support_masks_keep_multiview_and_high_support_voxels():
    counts = np.asarray([1, 2, 3, 4, 2, 1], dtype=np.uint32)
    supported, high = _support_masks(counts, minimum_supported_views=2, high_support_views=3)
    assert supported.tolist() == [False, True, True, True, True, False]
    assert high.tolist() == [False, False, True, True, False, False]


def test_support_histogram_is_stable_and_string_keyed():
    assert _support_histogram(np.asarray([3, 1, 2, 3, 2, 3], dtype=np.uint32)) == {
        "1": 1,
        "2": 2,
        "3": 3,
    }


def test_existing_dense_voxel_fusion_produces_real_multiview_support_for_e1_policy():
    width, height = 64, 48
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="e1-test-camera",
        camera_id="camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=70.0,
        focal_length_relative_uncertainty=0.2,
        camera_frame_id="camera-frame",
        reconstruction_frame_id="reconstruction",
    )
    frames = []
    alignment = reconstruction_unit_depth_alignment(provenance="already aligned synthetic E1 depth")
    for index, tx in enumerate((-0.05, 0.0, 0.05)):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = 50 + 20 * index
        rgb[..., 1] = 100
        rgb[..., 2] = 180
        depth = np.full((height, width), 2.0, dtype=np.float32)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = tx
        frames.append(
            DenseFusionFrame(
                frame_id=f"frame-{index}",
                timestamp_ns=1_000_000 + index * 100_000,
                rgb=rgb,
                depth=depth,
                T_reconstruction_from_camera=pose,
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                pose_layer="offline_sfm_optimized",
            )
        )
    result = fuse_dense_frames(
        frames,
        session_id="e1-synthetic",
        camera_profile=camera,
        scale_profile=arbitrary_scale_reference(profile_id="e1-arbitrary"),
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.08,
            pixel_stride=1,
            minimum_view_observations=1,
            max_voxels=100_000,
        ),
        reconstruction_frame_id="reconstruction",
    )
    supported, high = _support_masks(
        result.point_cloud.view_observation_counts,
        minimum_supported_views=2,
        high_support_views=3,
    )
    assert result.statistics.frames_integrated == 3
    assert result.statistics.dropped_new_voxels_capacity == 0
    assert np.count_nonzero(supported) > 100
    assert np.count_nonzero(high) > 50
    assert result.unit_to_metre is None
    assert result.certified_metrology is False


def test_validate_d4_3_summary_rejects_non_ten_view_candidate():
    payload = _summary()
    payload["candidates"]["B"]["workspace_orders"] = [1, 2, 3]
    with pytest.raises(ReconstructionValidationError):
        _validate_d4_3_summary(payload)
