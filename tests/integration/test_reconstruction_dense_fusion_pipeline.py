from __future__ import annotations

import numpy as np

from core.contracts import ImageSize
from core.reconstruction import (
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseSceneProfile,
    MetricGeometryStatus,
    arbitrary_scale_reference,
    estimate_camera_geometry_from_horizontal_fov,
    fuse_dense_frames,
    reconstruction_unit_depth_alignment,
)


def test_estimated_camera_optimized_pose_multiview_dense_pipeline_remains_relative() -> None:
    width, height = 80, 60
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="integration-estimated-camera",
        camera_id="camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=75.0,
        focal_length_relative_uncertainty=0.20,
        camera_frame_id="camera-frame",
        reconstruction_frame_id="integration-reconstruction",
    )
    alignment = reconstruction_unit_depth_alignment(provenance="integration synthetic aligned depth")
    frames = []
    for index, tx in enumerate((-0.10, -0.05, 0.0, 0.05, 0.10)):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
        image[..., 1] = 90 + index
        image[..., 2] = 170
        depth = np.full((height, width), 2.5, dtype=np.float32)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = tx
        frames.append(
            DenseFusionFrame(
                frame_id=f"frame-{index}",
                timestamp_ns=1_000_000 + index * 50_000,
                rgb=image,
                depth=depth,
                T_reconstruction_from_camera=pose,
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                pose_layer="optimized_propagated" if index not in {0, 4} else "optimized_keyframe",
            )
        )
    result = fuse_dense_frames(
        frames,
        session_id="integration-dense",
        camera_profile=camera,
        scale_profile=arbitrary_scale_reference(profile_id="arbitrary-live"),
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.10,
            pixel_stride=2,
            minimum_view_observations=2,
            max_voxels=100_000,
        ),
        reconstruction_frame_id="integration-reconstruction",
    )
    assert result.metric_geometry_status is MetricGeometryStatus.ARBITRARY
    assert result.scale_status.value == "arbitrary"
    assert result.unit_to_metre is None
    assert result.certified_metrology is False
    assert result.statistics.frames_integrated == 5
    assert result.statistics.multi_view_voxels > 100
    assert result.statistics.retained_voxels > 100
    assert np.median(result.point_cloud.points_reconstruction[:, 2]) == 2.5
    assert set(result.source_pose_layers) == {"optimized_keyframe", "optimized_propagated"}
