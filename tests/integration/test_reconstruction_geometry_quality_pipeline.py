from __future__ import annotations

import numpy as np

from core.contracts import ImageSize
from core.reconstruction import (
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseGeometryEvidence,
    DenseSceneProfile,
    GeometryQualityLevel,
    MetricGeometryStatus,
    arbitrary_scale_reference,
    assess_dense_geometry_quality,
    estimate_camera_geometry_from_horizontal_fov,
    fuse_dense_frames,
    reconstruction_unit_depth_alignment,
)


def test_dense_fusion_to_quality_pipeline_preserves_relative_semantics() -> None:
    width, height = 80, 60
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="quality-integration-camera",
        camera_id="camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=74.0,
        focal_length_relative_uncertainty=0.15,
        camera_frame_id="camera-frame",
        reconstruction_frame_id="quality-integration-reconstruction",
    )
    scale = arbitrary_scale_reference(profile_id="quality-integration-scale")
    alignment = reconstruction_unit_depth_alignment(provenance="integration aligned depth")
    frames = []
    for index, tx in enumerate((-0.10, -0.05, 0.0, 0.05, 0.10)):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = tx
        frames.append(
            DenseFusionFrame(
                frame_id=f"frame-{index}",
                timestamp_ns=1_000_000 + index,
                rgb=np.zeros((height, width, 3), dtype=np.uint8),
                depth=np.full((height, width), 2.5, dtype=np.float32),
                T_reconstruction_from_camera=pose,
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                confidence=np.full((height, width), 0.95, dtype=np.float32),
                pose_layer="optimized_keyframe" if index in {0, 4} else "optimized_propagated",
            )
        )
    dense = fuse_dense_frames(
        frames,
        session_id="quality-integration",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.10,
            pixel_stride=2,
            minimum_view_observations=2,
        ),
        reconstruction_frame_id="quality-integration-reconstruction",
    )
    count = len(dense.point_cloud.points_reconstruction)
    report = assess_dense_geometry_quality(
        dense,
        camera_profile=camera,
        scale_profile=scale,
        evidence=DenseGeometryEvidence(
            point_count=count,
            depth_relative_residual=np.full(count, 0.01, np.float32),
            view_angle_span_degrees=np.full(count, 22.0, np.float32),
            visibility_ratio=np.full(count, 0.95, np.float32),
            provenance="integration deterministic evidence",
        ),
    )
    assert report.metric_geometry_status is MetricGeometryStatus.ARBITRARY
    assert report.summary.global_level is GeometryQualityLevel.HIGH
    assert report.summary.accepted_ratio > 0.9
    assert report.summary.multi_view_ratio > 0.9
    assert report.summary.median_local_uncertainty_m is None
    assert report.certified_metrology is False
    assert report.pose_convention == "T_reconstruction_from_camera"
