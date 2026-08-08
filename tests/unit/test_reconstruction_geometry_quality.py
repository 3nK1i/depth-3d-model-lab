from __future__ import annotations

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
    GeometryQualityCode,
    GeometryQualityLevel,
    MetricGeometryStatus,
    ReconstructionValidationError,
    arbitrary_scale_reference,
    assess_dense_geometry_quality,
    default_dense_geometry_quality_policy,
    estimate_camera_geometry_from_horizontal_fov,
    fuse_dense_frames,
    known_dimension_scale_reference,
    metric_depth_alignment,
    reconstruction_unit_depth_alignment,
)


def _camera(width: int = 64, height: int = 48):
    return estimate_camera_geometry_from_horizontal_fov(
        profile_id="quality-camera",
        camera_id="quality-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.15,
        camera_frame_id="quality-camera-frame",
        reconstruction_frame_id="quality-reconstruction",
    )


def _pose(tx: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = tx
    return matrix


def _arbitrary_result(*, confidence: float = 0.95, minimum_views: int = 2):
    width, height = 64, 48
    camera = _camera(width, height)
    scale = arbitrary_scale_reference(profile_id="quality-arbitrary")
    alignment = reconstruction_unit_depth_alignment(provenance="unit test aligned depth")
    frames = []
    for index, tx in enumerate((-0.12, -0.06, 0.0, 0.06, 0.12)):
        frames.append(
            DenseFusionFrame(
                frame_id=f"quality-{index}",
                timestamp_ns=1_000_000 + index,
                rgb=np.zeros((height, width, 3), dtype=np.uint8),
                depth=np.full((height, width), 2.0, dtype=np.float32),
                T_reconstruction_from_camera=_pose(tx),
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                confidence=np.full((height, width), confidence, dtype=np.float32),
                pose_layer="optimized_keyframe" if index in {0, 4} else "optimized_propagated",
            )
        )
    result = fuse_dense_frames(
        frames,
        session_id="quality-session",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.08,
            pixel_stride=2,
            minimum_view_observations=minimum_views,
            max_voxels=50_000,
        ),
        reconstruction_frame_id="quality-reconstruction",
    )
    return result, camera, scale


def _good_evidence(point_count: int) -> DenseGeometryEvidence:
    return DenseGeometryEvidence(
        point_count=point_count,
        depth_relative_residual=np.full(point_count, 0.01, dtype=np.float32),
        view_angle_span_degrees=np.full(point_count, 24.0, dtype=np.float32),
        visibility_ratio=np.full(point_count, 0.95, dtype=np.float32),
        provenance="synthetic multiview evidence",
    )


def test_default_profiles_are_stricter_for_object_inspection() -> None:
    obj = default_dense_geometry_quality_policy(DenseSceneProfile.OBJECT_INSPECTION)
    spatial = default_dense_geometry_quality_policy(DenseSceneProfile.SPATIAL_DIGITAL_TWIN)
    assert obj.target_view_observations > spatial.target_view_observations
    assert obj.minimum_confidence > spatial.minimum_confidence
    assert obj.maximum_depth_relative_residual < spatial.maximum_depth_relative_residual
    assert obj.high_score_threshold > spatial.high_score_threshold


def test_policy_rejects_invalid_threshold_order() -> None:
    with pytest.raises(ReconstructionValidationError):
        DenseGeometryQualityPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            target_view_observations=3,
            minimum_view_observations=2,
            minimum_confidence=0.5,
            desired_view_angle_span_degrees=20.0,
            maximum_depth_relative_residual=0.05,
            high_score_threshold=0.6,
            medium_score_threshold=0.7,
            minimum_input_depth_coverage=0.5,
        )


def test_evidence_requires_one_value_per_point() -> None:
    with pytest.raises(ReconstructionValidationError):
        DenseGeometryEvidence(point_count=5, visibility_ratio=np.ones(4, np.float32))


def test_evidence_rejects_invalid_visibility() -> None:
    with pytest.raises(ReconstructionValidationError):
        DenseGeometryEvidence(point_count=2, visibility_ratio=np.array([0.5, 1.2]))


def test_good_multiview_evidence_produces_high_quality_relative_geometry() -> None:
    result, camera, scale = _arbitrary_result()
    report = assess_dense_geometry_quality(
        result,
        camera_profile=camera,
        scale_profile=scale,
        evidence=_good_evidence(len(result.point_cloud.points_reconstruction)),
    )
    assert report.summary.global_level is GeometryQualityLevel.HIGH
    assert report.summary.accepted_ratio > 0.9
    assert report.summary.high_quality_ratio > 0.5
    assert report.metric_geometry_status is MetricGeometryStatus.ARBITRARY
    assert report.summary.median_local_uncertainty_m is None
    assert "arbitrary_scale_no_metric_claim" in report.quality_limiters
    assert report.certified_metrology is False


def test_missing_consistency_evidence_is_conservative() -> None:
    result, camera, scale = _arbitrary_result()
    report = assess_dense_geometry_quality(
        result,
        camera_profile=camera,
        scale_profile=scale,
    )
    assert report.summary.evidence_completeness_ratio == 0.0
    assert report.summary.global_level in {GeometryQualityLevel.MEDIUM, GeometryQualityLevel.LOW}
    assert "depth_consistency_evidence_unavailable" in report.quality_limiters
    assert "view_diversity_evidence_unavailable" in report.quality_limiters
    assert "visibility_evidence_unavailable" in report.quality_limiters


def test_bad_consistency_and_visibility_degrade_quality() -> None:
    result, camera, scale = _arbitrary_result()
    count = len(result.point_cloud.points_reconstruction)
    evidence = DenseGeometryEvidence(
        point_count=count,
        depth_relative_residual=np.full(count, 0.25, dtype=np.float32),
        view_angle_span_degrees=np.full(count, 1.0, dtype=np.float32),
        visibility_ratio=np.full(count, 0.15, dtype=np.float32),
        provenance="intentionally degraded evidence",
    )
    report = assess_dense_geometry_quality(
        result,
        camera_profile=camera,
        scale_profile=scale,
        evidence=evidence,
    )
    assert report.summary.global_level is GeometryQualityLevel.LOW
    assert report.summary.low_quality_ratio > 0.5


def test_low_confidence_points_are_not_accepted() -> None:
    result, camera, scale = _arbitrary_result(confidence=0.2)
    report = assess_dense_geometry_quality(
        result,
        camera_profile=camera,
        scale_profile=scale,
        evidence=_good_evidence(len(result.point_cloud.points_reconstruction)),
    )
    assert report.summary.accepted_ratio == 0.0
    assert report.summary.global_level is GeometryQualityLevel.LOW


def test_point_quality_arrays_are_readonly() -> None:
    result, camera, scale = _arbitrary_result()
    report = assess_dense_geometry_quality(
        result,
        camera_profile=camera,
        scale_profile=scale,
        evidence=_good_evidence(len(result.point_cloud.points_reconstruction)),
    )
    with pytest.raises(ValueError):
        report.point_quality.quality_scores[0] = 0.0
    with pytest.raises(ValueError):
        report.point_quality.accepted_mask[0] = False


def test_arbitrary_quality_refuses_metric_uncertainty_conversion() -> None:
    result, camera, scale = _arbitrary_result()
    report = assess_dense_geometry_quality(
        result,
        camera_profile=camera,
        scale_profile=scale,
        evidence=_good_evidence(len(result.point_cloud.points_reconstruction)),
    )
    with pytest.raises(ReconstructionValidationError):
        report.point_quality.local_uncertainty_metres(scale)


def test_metric_quality_keeps_local_and_scale_uncertainty_separate() -> None:
    width, height = 64, 48
    camera = _camera(width, height)
    scale = known_dimension_scale_reference(
        profile_id="quality-known-dimension",
        reference_id="known-20mm",
        known_length_m=0.02,
        measured_length_reconstruction_units=2.0,
        known_length_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.02,
        provenance="unit test known dimension",
    )
    frames = []
    for index, tx in enumerate((-3.0, 0.0, 3.0)):
        frames.append(
            DenseFusionFrame(
                frame_id=f"metric-{index}",
                timestamp_ns=2_000_000 + index,
                rgb=np.zeros((height, width, 3), dtype=np.uint8),
                depth=np.full((height, width), 1.5, dtype=np.float32),
                T_reconstruction_from_camera=_pose(tx),
                depth_unit=DenseDepthUnit.METRES,
                depth_alignment=metric_depth_alignment(scale, provenance="metric depth"),
                confidence=np.full((height, width), 0.95, dtype=np.float32),
                pose_layer="optimized_keyframe",
            )
        )
    result = fuse_dense_frames(
        frames,
        session_id="metric-quality",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            voxel_size_reconstruction_units=8.0,
            pixel_stride=2,
            minimum_view_observations=1,
        ),
        reconstruction_frame_id="quality-reconstruction",
    )
    report = assess_dense_geometry_quality(
        result,
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseGeometryQualityPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            target_view_observations=2,
            minimum_view_observations=1,
            minimum_confidence=0.65,
            desired_view_angle_span_degrees=20.0,
            maximum_depth_relative_residual=0.04,
            high_score_threshold=0.80,
            medium_score_threshold=0.60,
            minimum_input_depth_coverage=0.65,
        ),
        evidence=_good_evidence(len(result.point_cloud.points_reconstruction)),
    )
    local_m = report.point_quality.local_uncertainty_metres(scale)
    assert report.metric_geometry_status is MetricGeometryStatus.PROVISIONAL_METRIC
    assert report.summary.median_local_uncertainty_m is not None
    assert np.median(local_m) == pytest.approx(report.summary.median_local_uncertainty_m)
    assert report.summary.relative_scale_uncertainty == pytest.approx(
        scale.gauge.relative_scale_uncertainty
    )
    assert "metric_scale_uncertainty_present" in report.quality_limiters
    assert report.certified_metrology is False


def test_mismatched_camera_profile_is_rejected() -> None:
    result, _, scale = _arbitrary_result()
    with pytest.raises(ReconstructionValidationError):
        assess_dense_geometry_quality(
            result,
            camera_profile=estimate_camera_geometry_from_horizontal_fov(
                profile_id="other-camera",
                camera_id="other",
                image_size=ImageSize(64, 48),
                horizontal_fov_degrees=70.0,
                focal_length_relative_uncertainty=0.2,
            ),
            scale_profile=scale,
        )


def test_mismatched_scale_profile_is_rejected() -> None:
    result, camera, _ = _arbitrary_result()
    with pytest.raises(ReconstructionValidationError):
        assess_dense_geometry_quality(
            result,
            camera_profile=camera,
            scale_profile=arbitrary_scale_reference(profile_id="other-scale"),
        )


def test_policy_scene_profile_must_match_result() -> None:
    result, camera, scale = _arbitrary_result()
    with pytest.raises(ReconstructionValidationError):
        assess_dense_geometry_quality(
            result,
            camera_profile=camera,
            scale_profile=scale,
            policy=default_dense_geometry_quality_policy(DenseSceneProfile.OBJECT_INSPECTION),
        )


def test_quality_codes_only_use_observed_bands() -> None:
    result, camera, scale = _arbitrary_result()
    report = assess_dense_geometry_quality(
        result,
        camera_profile=camera,
        scale_profile=scale,
        evidence=_good_evidence(len(result.point_cloud.points_reconstruction)),
    )
    assert set(np.unique(report.point_quality.quality_codes)).issubset(
        {int(GeometryQualityCode.LOW), int(GeometryQualityCode.MEDIUM), int(GeometryQualityCode.HIGH)}
    )
    assert report.unobserved_semantics == "sampled_input_depth_only"


def test_report_summary_ratios_partition_observed_points() -> None:
    result, camera, scale = _arbitrary_result()
    report = assess_dense_geometry_quality(
        result,
        camera_profile=camera,
        scale_profile=scale,
        evidence=_good_evidence(len(result.point_cloud.points_reconstruction)),
    )
    total = (
        report.summary.high_quality_ratio
        + report.summary.medium_quality_ratio
        + report.summary.low_quality_ratio
    )
    assert total == pytest.approx(1.0)


def test_estimated_camera_is_explicit_quality_limiter() -> None:
    result, camera, scale = _arbitrary_result()
    report = assess_dense_geometry_quality(
        result,
        camera_profile=camera,
        scale_profile=scale,
        evidence=_good_evidence(len(result.point_cloud.points_reconstruction)),
    )
    assert "estimated_camera_intrinsics" in report.quality_limiters
    assert report.summary.camera_reliability_score < 1.0


def test_local_uncertainty_is_positive_and_voxel_aware() -> None:
    result, camera, scale = _arbitrary_result()
    report = assess_dense_geometry_quality(
        result,
        camera_profile=camera,
        scale_profile=scale,
        evidence=_good_evidence(len(result.point_cloud.points_reconstruction)),
    )
    uncertainty = report.point_quality.local_uncertainty_reconstruction_units
    assert np.all(uncertainty > 0.0)
    assert report.summary.p95_local_uncertainty_reconstruction_units >= report.summary.median_local_uncertainty_reconstruction_units
