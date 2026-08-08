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
    arbitrary_scale_reference,
    assess_dense_geometry_quality,
    estimate_camera_geometry_from_horizontal_fov,
    fuse_dense_frames,
    generate_quality_aware_surface_mesh,
    reconstruction_unit_depth_alignment,
)


def _pose(tx: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = tx
    return matrix


def test_dense_quality_to_surface_mesh_pipeline_preserves_provenance_and_quality() -> None:
    width, height = 72, 54
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="surface-integration-camera",
        camera_id="surface-integration-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.12,
        camera_frame_id="surface-integration-camera-frame",
        reconstruction_frame_id="surface-integration-reconstruction",
    )
    scale = arbitrary_scale_reference(profile_id="surface-integration-scale")
    alignment = reconstruction_unit_depth_alignment(provenance="integration aligned depth")
    frames = []
    for index, tx in enumerate((-0.10, -0.05, 0.0, 0.05, 0.10)):
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[..., 0] = 30 + index * 20
        rgb[..., 1] = 110
        rgb[..., 2] = np.arange(width, dtype=np.uint8)[None, :]
        frames.append(
            DenseFusionFrame(
                frame_id=f"integration-{index}",
                timestamp_ns=1_000_000_000 + index * 50_000_000,
                rgb=rgb,
                depth=np.full((height, width), 2.2, np.float32),
                T_reconstruction_from_camera=_pose(tx),
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                confidence=np.full((height, width), 0.96, np.float32),
                pose_layer="optimized_keyframe" if index in {0, 4} else "optimized_propagated",
            )
        )
    dense = fuse_dense_frames(
        frames,
        session_id="surface-integration",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.08,
            pixel_stride=2,
            minimum_view_observations=2,
            max_voxels=50_000,
        ),
        reconstruction_frame_id="surface-integration-reconstruction",
    )
    point_count = len(dense.point_cloud.points_reconstruction)
    quality = assess_dense_geometry_quality(
        dense,
        camera_profile=camera,
        scale_profile=scale,
        evidence=DenseGeometryEvidence(
            point_count=point_count,
            depth_relative_residual=np.full(point_count, 0.01, np.float32),
            view_angle_span_degrees=np.full(point_count, 26.0, np.float32),
            visibility_ratio=np.full(point_count, 0.97, np.float32),
            provenance="integration quality evidence",
        ),
    )
    assert quality.summary.global_level is GeometryQualityLevel.HIGH
    surface = generate_quality_aware_surface_mesh(dense, quality)
    assert surface.session_id == dense.session_id == quality.session_id
    assert surface.camera_profile_id == camera.profile_id
    assert surface.scale_profile_id == scale.profile_id
    assert surface.pose_convention == "T_reconstruction_from_camera"
    assert surface.mesh.face_count > surface.mesh.vertex_count
    assert surface.statistics.quality_accepted_points == int(np.count_nonzero(quality.point_quality.accepted_mask))
    assert surface.certified_metrology is False
