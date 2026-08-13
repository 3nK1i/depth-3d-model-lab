from __future__ import annotations

import json

import numpy as np
import pytest

from core.reconstruction import (
    MetricEvidenceKind,
    MetricEvidenceValidationState,
    MetricGeometryStatus,
    MetricQualificationEvidence,
    MetricQualificationState,
    assess_metric_qualification,
    assess_reconstruction_geometry,
    camera_geometry_from_calibrated_camera,
    external_metric_prior_evidence,
    future_metric_source_evidence,
    known_distance_reference_evidence,
    known_size_benchmark_supporting_evidence,
    known_size_reference_evidence,
    metric_depth_scale_evidence,
    metric_qualification_evidence_from_json,
    metric_qualification_evidence_to_json,
    reference_target_metric_evidence,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.reconstruction_live_helpers import make_calibrated_live_camera


FIXED_TIME = "2026-08-13T17:00:00+00:00"


def _known_size(**overrides):
    values = dict(
        evidence_id="known-size-1",
        reconstruction_frame_id="reconstruction",
        reference_id="box-width",
        known_length_m=0.4,
        measured_length_reconstruction_units=5.0,
        known_length_relative_uncertainty=0.002,
        measurement_relative_uncertainty=0.015,
        provenance="operator-measured box width with caliper",
        created_at_utc=FIXED_TIME,
    )
    values.update(overrides)
    return known_size_reference_evidence(**values)


def _target(state=MetricEvidenceValidationState.VALIDATED):
    return reference_target_metric_evidence(
        evidence_id="target-1",
        reconstruction_frame_id="reconstruction",
        reference_id="precision-target-100mm",
        target_length_m=0.1,
        measured_length_reconstruction_units=12.5,
        target_relative_uncertainty=0.001,
        measurement_relative_uncertainty=0.008,
        provenance="target-certificate:precision-target-100mm",
        validation_state=state,
        validation_record_id=("target-validation-2026-08-13" if state is MetricEvidenceValidationState.VALIDATED else ""),
        created_at_utc=FIXED_TIME,
    )


def test_known_size_requires_explicit_uncertainty_and_stays_provisional() -> None:
    evidence = _known_size()
    assessment = assess_metric_qualification(evidence, scale_profile_id="scale-known-size")

    assert assessment.state is MetricQualificationState.PROVISIONAL
    assert assessment.authorizes_metric_coordinates is True
    assert assessment.authorizes_calibrated_metric_scale is False
    assert assessment.certified_metrology is False
    assert assessment.scale_profile is not None
    assert assessment.scale_profile.gauge.unit_to_metre == pytest.approx(0.08)
    assert assessment.relative_scale_uncertainty == pytest.approx(np.hypot(0.002, 0.015))
    assert assessment.scale_profile.validated_reference is False


def test_known_size_cannot_omit_reference_uncertainty() -> None:
    with pytest.raises(ReconstructionValidationError, match="explicit physical length"):
        MetricQualificationEvidence(
            evidence_id="missing-uncertainty",
            kind=MetricEvidenceKind.KNOWN_SIZE_REFERENCE,
            validation_state=MetricEvidenceValidationState.UNVALIDATED,
            provenance="test",
            reconstruction_frame_id="reconstruction",
            reference_id="box",
            reference_length_m=0.4,
            measured_length_reconstruction_units=5.0,
            measurement_relative_uncertainty=0.02,
            created_at_utc=FIXED_TIME,
        )


def test_known_distance_uses_existing_known_dimension_scale_authority() -> None:
    evidence = known_distance_reference_evidence(
        evidence_id="distance-1",
        reconstruction_frame_id="reconstruction",
        reference_id="camera-to-target-baseline",
        known_distance_m=1.5,
        measured_distance_reconstruction_units=30.0,
        reference_relative_uncertainty=0.003,
        measurement_relative_uncertainty=0.01,
        provenance="surveyed baseline",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="baseline-check-1",
        created_at_utc=FIXED_TIME,
    )
    assessment = assess_metric_qualification(evidence, scale_profile_id="scale-distance")
    assert assessment.state is MetricQualificationState.PROVISIONAL
    assert assessment.scale_profile is not None
    assert assessment.scale_profile.gauge.unit_to_metre == pytest.approx(0.05)
    assert assessment.scale_profile.validated_reference is False


def test_validated_reference_target_qualifies_scale() -> None:
    assessment = assess_metric_qualification(_target(), scale_profile_id="scale-target")
    assert assessment.state is MetricQualificationState.QUALIFIED
    assert assessment.authorizes_metric_coordinates is True
    assert assessment.authorizes_calibrated_metric_scale is True
    assert assessment.scale_profile is not None
    assert assessment.scale_profile.is_calibrated_metric is True
    assert assessment.certified_metrology is False


def test_unvalidated_reference_target_is_provisional() -> None:
    assessment = assess_metric_qualification(
        _target(MetricEvidenceValidationState.UNVALIDATED),
        scale_profile_id="scale-target-unvalidated",
    )
    assert assessment.state is MetricQualificationState.PROVISIONAL
    assert assessment.authorizes_calibrated_metric_scale is False
    assert assessment.scale_profile is not None
    assert assessment.scale_profile.is_calibrated_metric is False


def test_validated_reference_target_requires_validation_record_id() -> None:
    with pytest.raises(ReconstructionValidationError, match="validation_record_id"):
        reference_target_metric_evidence(
            evidence_id="target-missing-record",
            reconstruction_frame_id="reconstruction",
            reference_id="target",
            target_length_m=0.1,
            measured_length_reconstruction_units=10.0,
            target_relative_uncertainty=0.001,
            measurement_relative_uncertainty=0.01,
            provenance="target provenance",
            validation_state=MetricEvidenceValidationState.VALIDATED,
            created_at_utc=FIXED_TIME,
        )


def test_validated_metric_depth_can_qualify_scale_but_requires_explicit_mapping() -> None:
    evidence = metric_depth_scale_evidence(
        evidence_id="metric-depth-1",
        reconstruction_frame_id="reconstruction",
        reference_id="metric3d-qualified-run",
        unit_to_metre=1.0,
        relative_scale_uncertainty=0.035,
        provenance="qualified metric-depth validation record",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="metric-depth-validation-1",
        created_at_utc=FIXED_TIME,
    )
    assessment = assess_metric_qualification(evidence, scale_profile_id="scale-metric-depth")
    assert assessment.state is MetricQualificationState.QUALIFIED
    assert assessment.scale_profile is not None
    assert assessment.scale_profile.gauge.unit_to_metre == pytest.approx(1.0)
    assert assessment.relative_scale_uncertainty == pytest.approx(0.035)
    assert assessment.authorizes_calibrated_metric_scale is True


def test_unvalidated_metric_depth_is_only_provisional() -> None:
    evidence = metric_depth_scale_evidence(
        evidence_id="metric-depth-unvalidated",
        reconstruction_frame_id="reconstruction",
        reference_id="metric-depth-run",
        unit_to_metre=1.0,
        relative_scale_uncertainty=0.08,
        provenance="unqualified metric-depth output",
        validation_state=MetricEvidenceValidationState.UNVALIDATED,
        created_at_utc=FIXED_TIME,
    )
    assessment = assess_metric_qualification(evidence, scale_profile_id="scale-md-unvalidated")
    assert assessment.state is MetricQualificationState.PROVISIONAL
    assert assessment.authorizes_metric_coordinates is True
    assert assessment.authorizes_calibrated_metric_scale is False


def test_direct_metric_prior_requires_explicit_uncertainty() -> None:
    with pytest.raises(ReconstructionValidationError, match="unit_to_metre and relative_scale_uncertainty"):
        MetricQualificationEvidence(
            evidence_id="bad-direct",
            kind=MetricEvidenceKind.EXTERNAL_METRIC_PRIOR,
            validation_state=MetricEvidenceValidationState.UNVALIDATED,
            provenance="external prior",
            reconstruction_frame_id="reconstruction",
            reference_id="prior",
            unit_to_metre=0.001,
            created_at_utc=FIXED_TIME,
        )


def test_external_metric_prior_preserves_provisional_or_qualified_state() -> None:
    provisional = external_metric_prior_evidence(
        evidence_id="prior-p",
        reconstruction_frame_id="reconstruction",
        reference_id="pose-source",
        unit_to_metre=0.001,
        relative_scale_uncertainty=0.04,
        provenance="external pose source",
        validation_state=MetricEvidenceValidationState.UNVALIDATED,
        created_at_utc=FIXED_TIME,
    )
    qualified = external_metric_prior_evidence(
        evidence_id="prior-q",
        reconstruction_frame_id="reconstruction",
        reference_id="pose-source-qualified",
        unit_to_metre=0.001,
        relative_scale_uncertainty=0.015,
        provenance="qualified external pose source",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="pose-source-validation",
        created_at_utc=FIXED_TIME,
    )
    assert assess_metric_qualification(provisional, scale_profile_id="p").state is MetricQualificationState.PROVISIONAL
    assert assess_metric_qualification(qualified, scale_profile_id="q").state is MetricQualificationState.QUALIFIED


def test_rejected_evidence_can_record_failure_before_scale_measurement_exists() -> None:
    evidence = MetricQualificationEvidence(
        evidence_id="target-detection-rejected",
        kind=MetricEvidenceKind.REFERENCE_TARGET,
        validation_state=MetricEvidenceValidationState.REJECTED,
        provenance="target detector",
        reconstruction_frame_id="reconstruction",
        reference_id="target-100mm",
        rejection_reasons=("target_not_detected",),
        created_at_utc=FIXED_TIME,
    )
    assessment = assess_metric_qualification(evidence, scale_profile_id="never-emitted")
    assert assessment.state is MetricQualificationState.REJECTED
    assert assessment.scale_profile is None
    assert assessment.authorizes_metric_coordinates is False


def test_rejected_evidence_never_emits_scale_profile() -> None:
    evidence = MetricQualificationEvidence(
        evidence_id="rejected-target",
        kind=MetricEvidenceKind.REFERENCE_TARGET,
        validation_state=MetricEvidenceValidationState.REJECTED,
        provenance="target inspection",
        reconstruction_frame_id="reconstruction",
        reference_id="damaged-target",
        reference_length_m=0.1,
        measured_length_reconstruction_units=10.0,
        reference_relative_uncertainty=0.002,
        measurement_relative_uncertainty=0.02,
        rejection_reasons=("target_damage_detected",),
        created_at_utc=FIXED_TIME,
    )
    assessment = assess_metric_qualification(evidence, scale_profile_id="must-not-exist")
    assert assessment.state is MetricQualificationState.REJECTED
    assert assessment.scale_profile is None
    assert assessment.authorizes_metric_coordinates is False
    assert assessment.rejection_reasons == ("target_damage_detected",)


def test_future_stereo_and_rgbd_are_reserved_but_fail_closed() -> None:
    for kind in (MetricEvidenceKind.FUTURE_STEREO, MetricEvidenceKind.FUTURE_RGBD):
        evidence = future_metric_source_evidence(
            evidence_id=f"future-{kind.value}",
            kind=kind,
            provenance="reserved future provider",
            created_at_utc=FIXED_TIME,
        )
        assessment = assess_metric_qualification(evidence, scale_profile_id="future-scale")
        assert assessment.state is MetricQualificationState.UNSUPPORTED
        assert assessment.scale_profile is None
        assert assessment.authorizes_metric_coordinates is False


def test_future_sources_cannot_sneak_in_scale_values() -> None:
    with pytest.raises(ReconstructionValidationError, match="cannot declare scale"):
        MetricQualificationEvidence(
            evidence_id="future-bad",
            kind=MetricEvidenceKind.FUTURE_RGBD,
            validation_state=MetricEvidenceValidationState.UNVALIDATED,
            provenance="future",
            unit_to_metre=1.0,
            relative_scale_uncertainty=0.01,
            created_at_utc=FIXED_TIME,
        )


def test_passing_known_size_benchmark_is_supporting_only() -> None:
    evidence = known_size_benchmark_supporting_evidence(
        {
            "schema_version": 1,
            "passed": True,
            "summary": {"case_count": 3, "available_dimension_count": 9},
            "certified_metrology": False,
        },
        evidence_id="benchmark-1",
        provenance="phase7.5 known-size report",
        validation_record_id="benchmark-report-1",
        created_at_utc=FIXED_TIME,
    )
    assessment = assess_metric_qualification(evidence, scale_profile_id="benchmark-must-not-scale")
    assert evidence.validation_state is MetricEvidenceValidationState.VALIDATED
    assert assessment.state is MetricQualificationState.SUPPORTING_ONLY
    assert assessment.scale_profile is None
    assert assessment.authorizes_metric_coordinates is False
    assert evidence.source_report_sha256


def test_failed_known_size_benchmark_is_rejected_support() -> None:
    evidence = known_size_benchmark_supporting_evidence(
        {
            "schema_version": 1,
            "passed": False,
            "summary": {"case_count": 3, "available_dimension_count": 7},
            "certified_metrology": False,
        },
        evidence_id="benchmark-failed",
        provenance="failed benchmark",
        created_at_utc=FIXED_TIME,
    )
    assessment = assess_metric_qualification(evidence, scale_profile_id="benchmark-no-scale")
    assert evidence.validation_state is MetricEvidenceValidationState.REJECTED
    assert assessment.state is MetricQualificationState.REJECTED
    assert assessment.authorizes_metric_coordinates is False


def test_known_size_benchmark_cannot_import_certified_metrology_claim() -> None:
    with pytest.raises(ReconstructionValidationError, match="must not claim certified metrology"):
        known_size_benchmark_supporting_evidence(
            {"passed": True, "summary": {}, "certified_metrology": True},
            evidence_id="bad-report",
            provenance="tampered report",
            created_at_utc=FIXED_TIME,
        )


def test_evidence_json_round_trip_is_lossless() -> None:
    evidence = _known_size(validation_record_id="known-size-review-1")
    payload = metric_qualification_evidence_to_json(evidence)
    restored = metric_qualification_evidence_from_json(payload)
    assert restored.as_dict() == evidence.as_dict()
    assert json.loads(payload)["kind"] == "known_size_reference"


def test_calibrated_camera_plus_qualified_target_enables_calibrated_metric_not_metrology() -> None:
    camera = camera_geometry_from_calibrated_camera(
        profile_id="camera-calibrated",
        camera=make_calibrated_live_camera(),
    )
    qualification = assess_metric_qualification(_target(), scale_profile_id="scale-target")
    assert qualification.scale_profile is not None
    geometry = assess_reconstruction_geometry(camera, qualification.scale_profile)
    assert geometry.metric_status is MetricGeometryStatus.CALIBRATED_METRIC
    assert geometry.metric_coordinates_allowed is True
    assert geometry.calibrated_metric_geometry_allowed is True
    assert geometry.certified_metrology is False
