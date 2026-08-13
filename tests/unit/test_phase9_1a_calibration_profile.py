from __future__ import annotations

from dataclasses import replace
import json

import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION,
    CalibrationQualityMetrics,
    CameraCalibrationMethod,
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
    MetricGeometryStatus,
    ReconstructionValidationError,
    arbitrary_scale_reference,
    assess_reconstruction_geometry,
    calibration_source_binding_from_metadata,
    camera_calibration_profile_from_json,
    load_camera_calibration_profile,
    camera_calibration_profile_to_json,
    camera_geometry_from_calibrated_camera,
    make_usb_source_metadata,
    save_camera_calibration_profile,
)
from tests.reconstruction_live_helpers import make_calibrated_live_camera


CREATED_AT = "2026-08-13T16:00:00+00:00"


def _manual_profile() -> CameraCalibrationProfile:
    source = make_usb_source_metadata(0, width=320, height=240, frame_rate_hz=30.0)
    return CameraCalibrationProfile(
        profile_id="usb0-manual-320x240",
        camera=make_calibrated_live_camera(),
        method=CameraCalibrationMethod.MANUAL,
        validation_state=CameraCalibrationValidationState.VALIDATED,
        created_at_utc=CREATED_AT,
        provenance="pytest-manual-entry",
        source_binding=calibration_source_binding_from_metadata(source),
        notes="known manufacturer intrinsics",
    )


def test_profile_round_trip_is_deterministic_and_reuses_phase8_camera_schema() -> None:
    profile = _manual_profile()
    payload = camera_calibration_profile_to_json(profile)
    restored = camera_calibration_profile_from_json(payload)

    assert restored.as_dict() == profile.as_dict()
    assert camera_calibration_profile_to_json(restored) == payload
    raw = json.loads(payload)
    assert raw["schema_version"] == CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION
    assert raw["camera_calibration"]["schema_version"] == 1
    assert raw["camera_calibration"]["calibration_record_id"] == "calibration-live-test-1"



def test_profile_save_load_round_trip(tmp_path) -> None:
    profile = _manual_profile()
    path = save_camera_calibration_profile(profile, tmp_path / "camera-profile.json")
    restored = load_camera_calibration_profile(path)
    assert restored.as_dict() == profile.as_dict()


def test_serialized_fractional_view_count_is_rejected() -> None:
    raw = _manual_profile().as_dict()
    raw["quality"] = {
        "rms_reprojection_error_px": 0.2,
        "mean_reprojection_error_px": 0.1,
        "max_reprojection_error_px": 0.3,
        "accepted_view_count": 3.5,
        "rejected_view_count": 0,
    }
    raw["method"] = CameraCalibrationMethod.CHECKERBOARD.value
    raw["validation_state"] = CameraCalibrationValidationState.VALIDATED.value
    with pytest.raises(ReconstructionValidationError, match="view counts"):
        camera_calibration_profile_from_json(json.dumps(raw))

def test_profile_keeps_camera_calibration_separate_from_metric_scale() -> None:
    profile = _manual_profile()
    raw = profile.as_dict()

    assert "scale_reference" not in raw
    assert "unit_to_metre" not in json.dumps(raw)

    camera_geometry = camera_geometry_from_calibrated_camera(
        profile_id="camera-geometry",
        camera=profile.camera,
    )
    assessment = assess_reconstruction_geometry(
        camera_geometry,
        arbitrary_scale_reference(profile_id="arbitrary"),
    )
    assert assessment.metric_status is MetricGeometryStatus.ARBITRARY
    assert assessment.metric_coordinates_allowed is False
    assert assessment.certified_metrology is False


def test_source_binding_preserves_identity_without_source_uri() -> None:
    source = make_usb_source_metadata(2, width=320, height=240, frame_rate_hz=25.0)
    binding = calibration_source_binding_from_metadata(source)

    assert binding.source_id == source.source_id
    assert binding.kind == source.kind
    assert binding.expected_image_size == ImageSize(320, 240)
    assert binding.fingerprint == source.fingerprint
    assert "source_uri" not in binding.as_dict()


def test_profile_rejects_source_resolution_mismatch() -> None:
    source = make_usb_source_metadata(0, width=640, height=480, frame_rate_hz=30.0)
    with pytest.raises(ReconstructionValidationError, match="resolution"):
        CameraCalibrationProfile(
            profile_id="mismatch",
            camera=make_calibrated_live_camera(),
            method=CameraCalibrationMethod.MANUAL,
            validation_state=CameraCalibrationValidationState.UNVALIDATED,
            created_at_utc=CREATED_AT,
            provenance="pytest",
            source_binding=calibration_source_binding_from_metadata(source),
        )


def test_validated_automatic_profile_requires_reprojection_evidence() -> None:
    with pytest.raises(ReconstructionValidationError, match="reprojection"):
        CameraCalibrationProfile(
            profile_id="checkerboard-no-quality",
            camera=make_calibrated_live_camera(),
            method=CameraCalibrationMethod.CHECKERBOARD,
            validation_state=CameraCalibrationValidationState.VALIDATED,
            created_at_utc=CREATED_AT,
            provenance="pytest",
        )

    profile = CameraCalibrationProfile(
        profile_id="checkerboard-with-quality",
        camera=make_calibrated_live_camera(),
        method=CameraCalibrationMethod.CHECKERBOARD,
        validation_state=CameraCalibrationValidationState.VALIDATED,
        created_at_utc=CREATED_AT,
        provenance="pytest",
        quality=CalibrationQualityMetrics(
            rms_reprojection_error_px=0.31,
            mean_reprojection_error_px=0.24,
            max_reprojection_error_px=0.65,
            accepted_view_count=18,
            rejected_view_count=2,
        ),
    )
    assert profile.has_phase9_validation_evidence is True
    assert profile.is_accepted is True


def test_quality_metrics_reject_invalid_values() -> None:
    with pytest.raises(ReconstructionValidationError, match="non-negative"):
        CalibrationQualityMetrics(rms_reprojection_error_px=-0.1)
    with pytest.raises(ReconstructionValidationError, match="cannot exceed"):
        CalibrationQualityMetrics(
            mean_reprojection_error_px=0.5,
            max_reprojection_error_px=0.2,
        )


def test_legacy_state_is_reserved_for_legacy_import_method() -> None:
    profile = _manual_profile()
    with pytest.raises(ReconstructionValidationError, match="reserved"):
        replace(
            profile,
            validation_state=CameraCalibrationValidationState.LEGACY_COMPATIBLE,
        )
