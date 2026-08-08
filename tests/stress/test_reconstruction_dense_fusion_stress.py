from __future__ import annotations

import numpy as np

from core.contracts import ImageSize
from core.reconstruction import (
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseSceneProfile,
    DenseVoxelFusion,
    arbitrary_scale_reference,
    estimate_camera_geometry_from_horizontal_fov,
    reconstruction_unit_depth_alignment,
)


def test_dense_fusion_100_frame_sequence_remains_bounded() -> None:
    width, height = 96, 72
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="stress-camera",
        camera_id="stress-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.2,
        camera_frame_id="stress-camera-frame",
        reconstruction_frame_id="stress-reconstruction",
    )
    policy = DenseFusionPolicy(
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        voxel_size_reconstruction_units=0.025,
        pixel_stride=4,
        minimum_view_observations=1,
        max_voxels=12_000,
    )
    fusion = DenseVoxelFusion(
        session_id="stress-dense",
        camera_profile=camera,
        scale_profile=arbitrary_scale_reference(profile_id="raw"),
        policy=policy,
    )
    alignment = reconstruction_unit_depth_alignment(provenance="stress synthetic depth")
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 1] = 128
    for index in range(100):
        yy, xx = np.indices((height, width), dtype=np.float32)
        depth = 2.0 + 0.05 * np.sin(xx * 0.08 + index * 0.02) + 0.03 * np.cos(yy * 0.06)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.002
        fusion.integrate(
            DenseFusionFrame(
                frame_id=f"stress-{index:03d}",
                timestamp_ns=1_000_000_000 + index * 10_000_000,
                rgb=image,
                depth=depth,
                T_reconstruction_from_camera=pose,
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                pose_layer="optimized_propagated",
            )
        )
    result = fusion.finalize()
    assert result.statistics.frames_integrated == 100
    assert result.statistics.retained_voxels <= policy.max_voxels
    assert len(result.point_cloud.points_reconstruction) <= policy.max_voxels
    assert result.statistics.frame_voxel_observations > result.statistics.retained_voxels
    assert result.statistics.multi_view_voxels > 0
    assert result.unit_to_metre is None
