from __future__ import annotations

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
    estimate_camera_geometry_from_horizontal_fov,
    fuse_dense_frames,
    reconstruction_unit_depth_alignment,
)


def test_geometry_quality_repeated_assessment_is_bounded_and_deterministic() -> None:
    width, height = 128, 96
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="quality-stress-camera",
        camera_id="stress-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.2,
        reconstruction_frame_id="quality-stress-reconstruction",
    )
    scale = arbitrary_scale_reference(profile_id="quality-stress-scale")
    alignment = reconstruction_unit_depth_alignment(provenance="stress aligned depth")
    frames = []
    yy, xx = np.indices((height, width), dtype=np.float32)
    for index in range(40):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = (index - 20) * 0.006
        depth = 2.0 + 0.03 * np.sin(xx * 0.06 + index * 0.01) + 0.02 * np.cos(yy * 0.05)
        frames.append(
            DenseFusionFrame(
                frame_id=f"stress-{index:03d}",
                timestamp_ns=1_000_000_000 + index * 1_000_000,
                rgb=np.zeros((height, width, 3), dtype=np.uint8),
                depth=depth,
                T_reconstruction_from_camera=pose,
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                confidence=np.full((height, width), 0.9, dtype=np.float32),
                pose_layer="optimized_propagated",
            )
        )
    dense = fuse_dense_frames(
        frames,
        session_id="quality-stress",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.03,
            pixel_stride=3,
            minimum_view_observations=2,
            max_voxels=15_000,
        ),
        reconstruction_frame_id="quality-stress-reconstruction",
    )
    count = len(dense.point_cloud.points_reconstruction)
    evidence = DenseGeometryEvidence(
        point_count=count,
        depth_relative_residual=np.full(count, 0.02, np.float32),
        view_angle_span_degrees=np.full(count, 16.0, np.float32),
        visibility_ratio=np.full(count, 0.9, np.float32),
        provenance="stress deterministic evidence",
    )
    first = assess_dense_geometry_quality(
        dense, camera_profile=camera, scale_profile=scale, evidence=evidence
    )
    first_scores = first.point_quality.quality_scores.copy()
    first_uncertainty = first.point_quality.local_uncertainty_reconstruction_units.copy()
    for _ in range(25):
        report = assess_dense_geometry_quality(
            dense, camera_profile=camera, scale_profile=scale, evidence=evidence
        )
        assert np.array_equal(report.point_quality.quality_scores, first_scores)
        assert np.array_equal(
            report.point_quality.local_uncertainty_reconstruction_units, first_uncertainty
        )
        assert report.point_quality.point_count == count
    assert count <= 15_000
    assert dense.unit_to_metre is None
