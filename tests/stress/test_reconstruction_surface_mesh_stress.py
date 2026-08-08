from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.contracts import ImageSize
from core.reconstruction import (
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseGeometryEvidence,
    DenseSceneProfile,
    arbitrary_scale_reference,
    assess_dense_geometry_quality,
    default_surface_meshing_policy,
    estimate_camera_geometry_from_horizontal_fov,
    fuse_dense_frames,
    generate_quality_aware_surface_mesh,
    reconstruction_unit_depth_alignment,
)


def test_surface_meshing_is_bounded_on_larger_dense_cloud() -> None:
    width, height = 160, 120
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="surface-stress-camera",
        camera_id="surface-stress-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.15,
        camera_frame_id="surface-stress-camera-frame",
        reconstruction_frame_id="surface-stress-reconstruction",
    )
    scale = arbitrary_scale_reference(profile_id="surface-stress-scale")
    alignment = reconstruction_unit_depth_alignment(provenance="stress aligned depth")
    frames = []
    for index, tx in enumerate((-0.12, -0.06, 0.0, 0.06, 0.12)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = tx
        frames.append(
            DenseFusionFrame(
                frame_id=f"stress-{index}",
                timestamp_ns=1_000_000_000 + index * 50_000_000,
                rgb=np.full((height, width, 3), 120 + index, np.uint8),
                depth=np.full((height, width), 2.0, np.float32),
                T_reconstruction_from_camera=matrix,
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                confidence=np.full((height, width), 0.95, np.float32),
                pose_layer="optimized_propagated",
            )
        )
    dense = fuse_dense_frames(
        frames,
        session_id="surface-stress",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.035,
            pixel_stride=1,
            minimum_view_observations=1,
            max_voxels=80_000,
        ),
        reconstruction_frame_id="surface-stress-reconstruction",
    )
    count = len(dense.point_cloud.points_reconstruction)
    quality = assess_dense_geometry_quality(
        dense,
        camera_profile=camera,
        scale_profile=scale,
        evidence=DenseGeometryEvidence(
            point_count=count,
            depth_relative_residual=np.full(count, 0.015, np.float32),
            view_angle_span_degrees=np.full(count, 22.0, np.float32),
            visibility_ratio=np.full(count, 0.95, np.float32),
            provenance="stress quality evidence",
        ),
    )
    policy = replace(
        default_surface_meshing_policy(DenseSceneProfile.SPATIAL_DIGITAL_TWIN),
        max_vertices=2_000,
        max_faces=6_000,
        minimum_component_faces=1,
    )
    result = generate_quality_aware_surface_mesh(dense, quality, policy=policy)
    assert result.statistics.input_points >= result.statistics.selected_points_before_cleanup
    assert result.statistics.selected_points_before_cleanup <= 2_000
    assert result.mesh.vertex_count <= 2_000
    assert result.mesh.face_count <= 6_000
    assert result.mesh.face_count > 0
    assert result.certified_metrology is False
