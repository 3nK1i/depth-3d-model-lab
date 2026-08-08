from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseGeometryEvidence,
    DenseGeometryQualityPolicy,
    DenseSceneProfile,
    MetricGeometryStatus,
    ReconstructionValidationError,
    SurfaceMesh,
    SurfaceMeshingMethod,
    SurfaceMeshingPolicy,
    arbitrary_scale_reference,
    assess_dense_geometry_quality,
    default_surface_meshing_policy,
    estimate_camera_geometry_from_horizontal_fov,
    fuse_dense_frames,
    generate_quality_aware_surface_mesh,
    known_dimension_scale_reference,
    metric_depth_alignment,
    reconstruction_unit_depth_alignment,
    write_surface_mesh_obj,
)


def _pose(tx: float = 0.0) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = tx
    return matrix


def _camera(width: int = 64, height: int = 48):
    return estimate_camera_geometry_from_horizontal_fov(
        profile_id="surface-estimated",
        camera_id="surface-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=70.0,
        focal_length_relative_uncertainty=0.12,
        camera_frame_id="surface-camera-frame",
        reconstruction_frame_id="surface-reconstruction",
    )


def _good_evidence(count: int) -> DenseGeometryEvidence:
    return DenseGeometryEvidence(
        point_count=count,
        depth_relative_residual=np.full(count, 0.01, np.float32),
        view_angle_span_degrees=np.full(count, 30.0, np.float32),
        visibility_ratio=np.full(count, 0.98, np.float32),
        provenance="surface unit evidence",
    )


def _arbitrary_fixture(*, width: int = 64, height: int = 48):
    camera = _camera(width, height)
    scale = arbitrary_scale_reference(profile_id="surface-arbitrary")
    alignment = reconstruction_unit_depth_alignment(provenance="surface-plane")
    frames = []
    for index, tx in enumerate((-0.08, 0.0, 0.08)):
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[..., 0] = 40 + index * 30
        rgb[..., 1] = np.arange(width, dtype=np.uint8)[None, :]
        rgb[..., 2] = 180
        frames.append(
            DenseFusionFrame(
                frame_id=f"surface-{index}",
                timestamp_ns=1_000_000_000 + index * 50_000_000,
                rgb=rgb,
                depth=np.full((height, width), 2.0, np.float32),
                T_reconstruction_from_camera=_pose(tx),
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                confidence=np.full((height, width), 0.96, np.float32),
                pose_layer="optimized_keyframe" if index == 0 else "optimized_propagated",
            )
        )
    dense = fuse_dense_frames(
        frames,
        session_id="surface-arbitrary-session",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.09,
            pixel_stride=2,
            minimum_view_observations=1,
            max_voxels=30_000,
        ),
        reconstruction_frame_id="surface-reconstruction",
    )
    quality = assess_dense_geometry_quality(
        dense,
        camera_profile=camera,
        scale_profile=scale,
        evidence=_good_evidence(len(dense.point_cloud.points_reconstruction)),
    )
    return dense, quality, camera, scale


def _metric_fixture(*, width: int = 48, height: int = 36):
    camera = _camera(width, height)
    scale = known_dimension_scale_reference(
        profile_id="surface-metric",
        reference_id="known-20mm",
        known_length_m=0.02,
        measured_length_reconstruction_units=2.0,
        known_length_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.02,
        provenance="surface metric fixture",
    )
    frames = []
    for index, tx in enumerate((-2.5, 0.0, 2.5)):
        frames.append(
            DenseFusionFrame(
                frame_id=f"metric-{index}",
                timestamp_ns=2_000_000_000 + index * 50_000_000,
                rgb=np.full((height, width, 3), (70, 140, 210), np.uint8),
                depth=np.full((height, width), 1.25, np.float32),
                T_reconstruction_from_camera=_pose(tx),
                depth_unit=DenseDepthUnit.METRES,
                depth_alignment=metric_depth_alignment(scale, provenance="surface metric depth"),
                confidence=np.full((height, width), 0.96, np.float32),
                pose_layer="optimized_keyframe",
            )
        )
    dense = fuse_dense_frames(
        frames,
        session_id="surface-metric-session",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            voxel_size_reconstruction_units=7.0,
            pixel_stride=2,
            minimum_view_observations=1,
            max_voxels=30_000,
        ),
        reconstruction_frame_id="surface-reconstruction",
    )
    quality = assess_dense_geometry_quality(
        dense,
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseGeometryQualityPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            target_view_observations=2,
            minimum_view_observations=1,
            minimum_confidence=0.60,
            desired_view_angle_span_degrees=20.0,
            maximum_depth_relative_residual=0.05,
            high_score_threshold=0.78,
            medium_score_threshold=0.58,
            minimum_input_depth_coverage=0.60,
        ),
        evidence=_good_evidence(len(dense.point_cloud.points_reconstruction)),
    )
    return dense, quality, camera, scale


def test_default_surface_policies_are_product_aware() -> None:
    object_policy = default_surface_meshing_policy(DenseSceneProfile.OBJECT_INSPECTION)
    twin_policy = default_surface_meshing_policy(DenseSceneProfile.SPATIAL_DIGITAL_TWIN)
    assert object_policy.minimum_vertex_quality_score > twin_policy.minimum_vertex_quality_score
    assert object_policy.method is SurfaceMeshingMethod.LOCAL_TANGENT_FAN
    assert twin_policy.max_vertices >= object_policy.max_vertices


def test_surface_policy_rejects_invalid_bounds() -> None:
    base = default_surface_meshing_policy(DenseSceneProfile.OBJECT_INSPECTION)
    with pytest.raises(ReconstructionValidationError):
        replace(base, max_faces=0)
    with pytest.raises(ReconstructionValidationError):
        replace(base, minimum_vertex_quality_score=1.1)
    with pytest.raises(ReconstructionValidationError):
        replace(base, maximum_fan_angle_degrees=200.0)


def test_quality_aware_surface_mesh_generates_triangles() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    result = generate_quality_aware_surface_mesh(dense, quality)
    assert result.mesh.vertex_count > 100
    assert result.mesh.face_count > 100
    assert result.mesh.component_count >= 1
    assert result.metric_geometry_status is MetricGeometryStatus.ARBITRARY
    assert result.certified_metrology is False


def test_surface_mesh_preserves_original_dense_vertex_positions() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    result = generate_quality_aware_surface_mesh(dense, quality)
    expected = dense.point_cloud.points_reconstruction[result.mesh.source_point_indices.astype(np.int64)]
    assert np.array_equal(result.mesh.vertices_reconstruction, expected)


def test_surface_mesh_carries_quality_and_uncertainty_per_vertex() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    result = generate_quality_aware_surface_mesh(dense, quality)
    indices = result.mesh.source_point_indices.astype(np.int64)
    assert np.array_equal(result.mesh.vertex_quality_scores, quality.point_quality.quality_scores[indices])
    assert np.array_equal(
        result.mesh.vertex_uncertainty_reconstruction_units,
        quality.point_quality.local_uncertainty_reconstruction_units[indices],
    )
    assert float(np.min(result.mesh.vertex_quality_scores)) >= result.policy.minimum_vertex_quality_score


def test_surface_mesh_normals_are_unit_length() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    result = generate_quality_aware_surface_mesh(dense, quality)
    lengths = np.linalg.norm(result.mesh.normals_reconstruction, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)


def test_arbitrary_surface_refuses_metric_conversion() -> None:
    dense, quality, _, scale = _arbitrary_fixture()
    result = generate_quality_aware_surface_mesh(dense, quality)
    with pytest.raises(ReconstructionValidationError):
        result.vertices_metres(scale)
    with pytest.raises(ReconstructionValidationError):
        result.uncertainty_metres(scale)


def test_metric_surface_preserves_provisional_metric_semantics() -> None:
    dense, quality, _, scale = _metric_fixture()
    result = generate_quality_aware_surface_mesh(dense, quality)
    assert result.metric_geometry_status is MetricGeometryStatus.PROVISIONAL_METRIC
    assert result.unit_to_metre == pytest.approx(0.01)
    vertices_m = result.vertices_metres(scale)
    uncertainty_m = result.uncertainty_metres(scale)
    assert np.isfinite(vertices_m).all()
    assert np.all(uncertainty_m > 0.0)
    assert np.median(vertices_m[:, 2]) == pytest.approx(1.25, abs=0.1)
    assert result.certified_metrology is False


def test_metric_surface_rejects_wrong_scale_profile() -> None:
    dense, quality, _, scale = _metric_fixture()
    result = generate_quality_aware_surface_mesh(dense, quality)
    wrong = known_dimension_scale_reference(
        profile_id="wrong",
        reference_id="other",
        known_length_m=0.02,
        measured_length_reconstruction_units=2.0,
        provenance="wrong profile",
    )
    with pytest.raises(ReconstructionValidationError):
        result.vertices_metres(wrong)
    assert scale.profile_id == result.scale_profile_id


def test_meshing_is_deterministic() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    first = generate_quality_aware_surface_mesh(dense, quality)
    second = generate_quality_aware_surface_mesh(dense, quality)
    assert np.array_equal(first.mesh.vertices_reconstruction, second.mesh.vertices_reconstruction)
    assert np.array_equal(first.mesh.faces, second.mesh.faces)
    assert np.array_equal(first.mesh.normals_reconstruction, second.mesh.normals_reconstruction)


def test_meshing_respects_vertex_capacity() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    policy = replace(
        default_surface_meshing_policy(dense.scene_profile),
        max_vertices=160,
        minimum_component_faces=1,
    )
    result = generate_quality_aware_surface_mesh(dense, quality, policy=policy)
    assert result.mesh.vertex_count <= 160
    assert result.statistics.selected_points_before_cleanup == 160


def test_meshing_respects_face_capacity() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    policy = replace(
        default_surface_meshing_policy(dense.scene_profile),
        max_faces=120,
        minimum_component_faces=1,
    )
    result = generate_quality_aware_surface_mesh(dense, quality, policy=policy)
    assert result.mesh.face_count <= 120
    assert result.statistics.dropped_faces_capacity > 0


def test_low_quality_geometry_is_not_silently_meshed() -> None:
    dense, _, camera, scale = _arbitrary_fixture()
    count = len(dense.point_cloud.points_reconstruction)
    degraded = assess_dense_geometry_quality(
        dense,
        camera_profile=camera,
        scale_profile=scale,
        evidence=DenseGeometryEvidence(
            point_count=count,
            depth_relative_residual=np.full(count, 0.4, np.float32),
            view_angle_span_degrees=np.full(count, 0.1, np.float32),
            visibility_ratio=np.zeros(count, np.float32),
            provenance="deliberately unusable",
        ),
    )
    assert degraded.summary.accepted_ratio == 0.0
    with pytest.raises(ReconstructionValidationError):
        generate_quality_aware_surface_mesh(dense, degraded)


def test_meshing_rejects_session_mismatch() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    mismatched = replace(quality, session_id="other-session")
    with pytest.raises(ReconstructionValidationError):
        generate_quality_aware_surface_mesh(dense, mismatched)


def test_meshing_rejects_metric_semantics_mismatch() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    mismatched = replace(quality, metric_geometry_status=MetricGeometryStatus.PROVISIONAL_METRIC)
    with pytest.raises(ReconstructionValidationError):
        generate_quality_aware_surface_mesh(dense, mismatched)


def test_meshing_rejects_policy_scene_mismatch() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    wrong_policy = default_surface_meshing_policy(DenseSceneProfile.OBJECT_INSPECTION)
    with pytest.raises(ReconstructionValidationError):
        generate_quality_aware_surface_mesh(dense, quality, policy=wrong_policy)


def test_surface_mesh_arrays_are_read_only() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    result = generate_quality_aware_surface_mesh(dense, quality)
    with pytest.raises(ValueError):
        result.mesh.vertices_reconstruction[0, 0] += 1.0
    with pytest.raises(ValueError):
        result.mesh.faces[0, 0] = 0


def test_surface_mesh_contract_rejects_invalid_face_indices() -> None:
    with pytest.raises(ReconstructionValidationError):
        SurfaceMesh(
            vertices_reconstruction=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64),
            faces=np.array([[0, 1, 3]], np.uint32),
            colors_rgb=np.zeros((3, 3), np.uint8),
            normals_reconstruction=np.tile(np.array([[0, 0, 1]], np.float32), (3, 1)),
            vertex_quality_scores=np.ones(3, np.float32),
            vertex_quality_codes=np.full(3, 2, np.uint8),
            vertex_uncertainty_reconstruction_units=np.ones(3, np.float32),
            source_point_indices=np.arange(3, dtype=np.uint32),
            face_component_ids=np.zeros(1, np.uint32),
        )


def test_surface_result_preserves_pose_and_source_provenance() -> None:
    dense, quality, _, _ = _arbitrary_fixture()
    result = generate_quality_aware_surface_mesh(dense, quality)
    assert result.pose_convention == "T_reconstruction_from_camera"
    assert result.source_frame_ids == dense.source_frame_ids
    assert result.source_pose_layers == dense.source_pose_layers
    assert result.source_quality_schema_version == quality.schema_version
    assert result.quality_limiters == quality.quality_limiters


def test_obj_validation_export_writes_faces_and_no_metrology_claim(tmp_path: Path) -> None:
    dense, quality, _, _ = _arbitrary_fixture(width=40, height=30)
    result = generate_quality_aware_surface_mesh(dense, quality)
    path = write_surface_mesh_obj(result, tmp_path / "surface.obj")
    text = path.read_text(encoding="utf-8")
    assert "certified_metrology=false" in text
    assert "metric_geometry_status=arbitrary" in text
    assert "\nv " in text
    assert "\nf " in text
