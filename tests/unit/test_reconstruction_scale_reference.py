from __future__ import annotations

import json

import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    MetricGeometryStatus,
    ScaleReferencePurpose,
    SfmScaleStatus,
    arbitrary_scale_reference,
    assess_reconstruction_geometry,
    camera_geometry_from_calibrated_camera,
    estimate_camera_geometry_from_horizontal_fov,
    external_metric_prior_scale_reference,
    known_dimension_scale_reference,
    reconstruction_distance_to_metres,
    reconstruction_points_to_metres,
    reference_target_scale_reference,
    scale_reference_profile_from_json,
    scale_reference_profile_to_json,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.reconstruction_live_helpers import make_calibrated_live_camera


def _estimated_camera():
    return estimate_camera_geometry_from_horizontal_fov(
        profile_id="estimated-camera",
        camera_id="camera-0",
        image_size=ImageSize(1280, 720),
        horizontal_fov_degrees=79.0,
        focal_length_relative_uncertainty=0.20,
    )


def _known_dimension():
    return known_dimension_scale_reference(
        profile_id="known-20mm",
        reference_id="connector-width",
        known_length_m=0.020,
        measured_length_reconstruction_units=2.5,
        known_length_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.02,
        provenance="operator-entered connector drawing dimension",
    )


def _validated_target():
    return reference_target_scale_reference(
        profile_id="target-40mm",
        reference_id="precision-target-40mm-v1",
        target_length_m=0.040,
        measured_length_reconstruction_units=5.0,
        target_relative_uncertainty=0.001,
        measurement_relative_uncertainty=0.01,
        validated_target=True,
        provenance="calibration-target-record:precision-target-40mm-v1",
    )


def test_arbitrary_scale_is_default_and_cannot_convert_to_metres() -> None:
    profile = arbitrary_scale_reference(profile_id="raw")
    assert profile.gauge.scale_status is SfmScaleStatus.ARBITRARY
    assert profile.gauge.unit_to_metre is None
    assert profile.assess(ScaleReferencePurpose.RELATIVE_GEOMETRY).allowed is True
    assert profile.assess(ScaleReferencePurpose.METRIC_COORDINATES).allowed is False
    with pytest.raises(ReconstructionValidationError, match="without a metric scale reference"):
        reconstruction_distance_to_metres(1.0, profile)


def test_known_dimension_establishes_provisional_metric_scale() -> None:
    profile = _known_dimension()
    assert profile.gauge.scale_status is SfmScaleStatus.ESTIMATED_METRIC
    assert profile.gauge.unit_to_metre == pytest.approx(0.008)
    assert profile.is_calibrated_metric is False
    assert profile.assess(ScaleReferencePurpose.METRIC_COORDINATES).allowed is True
    assert profile.assess(ScaleReferencePurpose.METRIC_COORDINATES).provisional is True
    assert profile.assess(ScaleReferencePurpose.CALIBRATED_METRIC_GEOMETRY).allowed is False


def test_known_dimension_combines_reference_and_measurement_uncertainty() -> None:
    profile = _known_dimension()
    assert profile.gauge.relative_scale_uncertainty == pytest.approx(
        np.hypot(0.005, 0.02)
    )


def test_known_dimension_rejects_invalid_lengths() -> None:
    with pytest.raises(ReconstructionValidationError, match="known_length_m"):
        known_dimension_scale_reference(
            profile_id="bad",
            reference_id="bad",
            known_length_m=0.0,
            measured_length_reconstruction_units=1.0,
            provenance="test",
        )
    with pytest.raises(ReconstructionValidationError, match="measured_length_reconstruction_units"):
        known_dimension_scale_reference(
            profile_id="bad",
            reference_id="bad",
            known_length_m=0.1,
            measured_length_reconstruction_units=-1.0,
            provenance="test",
        )


def test_validated_reference_target_establishes_calibrated_metric_scale_only() -> None:
    profile = _validated_target()
    assert profile.gauge.scale_status is SfmScaleStatus.CALIBRATED_METRIC
    assert profile.gauge.unit_to_metre == pytest.approx(0.008)
    assert profile.is_calibrated_metric is True
    assessment = profile.assess(ScaleReferencePurpose.CALIBRATED_METRIC_GEOMETRY)
    assert assessment.allowed is True
    assert assessment.provisional is False


def test_unvalidated_reference_target_remains_estimated_metric() -> None:
    profile = reference_target_scale_reference(
        profile_id="target-unvalidated",
        reference_id="printed-target",
        target_length_m=0.1,
        measured_length_reconstruction_units=10.0,
        target_relative_uncertainty=0.01,
        measurement_relative_uncertainty=0.02,
        validated_target=False,
        provenance="printed shop-floor target",
    )
    assert profile.gauge.scale_status is SfmScaleStatus.ESTIMATED_METRIC
    assert profile.is_calibrated_metric is False


def test_external_metric_prior_supports_future_pose_sources_without_robot_dependency() -> None:
    profile = external_metric_prior_scale_reference(
        profile_id="external-prior",
        reference_id="future-pose-source",
        unit_to_metre=0.001,
        relative_scale_uncertainty=0.03,
        validated_prior=False,
        provenance="external metric pose prior",
    )
    assert profile.gauge.scale_status is SfmScaleStatus.ESTIMATED_METRIC
    assert profile.gauge.unit_to_metre == pytest.approx(0.001)
    assert profile.reference_length_m is None
    assert profile.measured_length_reconstruction_units is None


def test_distance_and_point_conversion_use_only_explicit_scale_factor() -> None:
    profile = _known_dimension()
    assert reconstruction_distance_to_metres(2.5, profile) == pytest.approx(0.020)
    points = reconstruction_points_to_metres(
        np.array([[0.0, 0.0, 0.0], [2.5, -1.0, 3.0]], dtype=np.float64),
        profile,
    )
    np.testing.assert_allclose(points[1], [0.020, -0.008, 0.024])
    assert points.flags.writeable is False


def test_point_conversion_rejects_nonfinite_or_wrong_shape() -> None:
    profile = _known_dimension()
    with pytest.raises(ReconstructionValidationError, match="final dimension 3"):
        reconstruction_points_to_metres([[1.0, 2.0]], profile)
    with pytest.raises(ReconstructionValidationError, match="final dimension 3"):
        reconstruction_points_to_metres([[1.0, np.nan, 3.0]], profile)


def test_estimated_camera_plus_known_dimension_is_provisional_metric_not_calibrated() -> None:
    assessment = assess_reconstruction_geometry(_estimated_camera(), _known_dimension())
    assert assessment.metric_status is MetricGeometryStatus.PROVISIONAL_METRIC
    assert assessment.relative_reconstruction_allowed is True
    assert assessment.metric_coordinates_allowed is True
    assert assessment.calibrated_metric_geometry_allowed is False
    assert assessment.provisional is True
    assert assessment.certified_metrology is False


def test_estimated_camera_plus_calibrated_scale_is_still_provisional_metric() -> None:
    assessment = assess_reconstruction_geometry(_estimated_camera(), _validated_target())
    assert assessment.metric_status is MetricGeometryStatus.PROVISIONAL_METRIC
    assert assessment.metric_coordinates_allowed is True
    assert assessment.calibrated_metric_geometry_allowed is False
    assert assessment.certified_metrology is False


def test_calibrated_camera_plus_validated_target_enables_calibrated_metric_geometry_not_certified_metrology() -> None:
    camera = camera_geometry_from_calibrated_camera(
        profile_id="calibrated-camera",
        camera=make_calibrated_live_camera(),
    )
    assessment = assess_reconstruction_geometry(camera, _validated_target())
    assert assessment.metric_status is MetricGeometryStatus.CALIBRATED_METRIC
    assert assessment.metric_coordinates_allowed is True
    assert assessment.calibrated_metric_geometry_allowed is True
    assert assessment.provisional is False
    assert assessment.certified_metrology is False


def test_profile_json_round_trip_is_lossless() -> None:
    profile = _known_dimension()
    payload = scale_reference_profile_to_json(profile)
    restored = scale_reference_profile_from_json(payload)
    assert restored.as_dict() == profile.as_dict()
    assert json.loads(payload)["gauge"]["scale_status"] == "estimated_metric"


def test_tampered_serialized_unit_conversion_is_rejected() -> None:
    payload = json.loads(scale_reference_profile_to_json(_known_dimension()))
    payload["gauge"]["unit_to_metre"] = 999.0
    with pytest.raises(ReconstructionValidationError, match="must equal"):
        scale_reference_profile_from_json(json.dumps(payload))
