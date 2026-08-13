from __future__ import annotations

from core.reconstruction import (
    CameraCalibrationMethod,
    CameraCalibrationValidationState,
    camera_calibration_profile_from_live_calibration_dict,
    camera_calibration_profile_from_json,
    camera_calibration_profile_to_json,
    camera_calibration_profile_to_live_calibration_dict,
    live_camera_calibration_dict,
    make_usb_source_metadata,
)
from tests.reconstruction_live_helpers import make_calibrated_live_camera


CREATED_AT = "2026-08-13T16:00:00+00:00"


def test_phase8_live_calibration_import_export_is_lossless() -> None:
    legacy = live_camera_calibration_dict(make_calibrated_live_camera())
    profile = camera_calibration_profile_from_live_calibration_dict(
        legacy,
        source=make_usb_source_metadata(0, width=320, height=240, frame_rate_hz=30.0),
        created_at_utc=CREATED_AT,
    )

    assert profile.method is CameraCalibrationMethod.LEGACY_LIVE_JSON
    assert profile.validation_state is CameraCalibrationValidationState.LEGACY_COMPATIBLE
    assert profile.is_accepted is True
    assert profile.has_phase9_validation_evidence is False
    assert camera_calibration_profile_to_live_calibration_dict(profile) == legacy


def test_legacy_compatibility_survives_new_profile_json_round_trip() -> None:
    legacy = live_camera_calibration_dict(make_calibrated_live_camera())
    profile = camera_calibration_profile_from_live_calibration_dict(
        legacy,
        created_at_utc=CREATED_AT,
    )
    restored = camera_calibration_profile_from_json(
        camera_calibration_profile_to_json(profile)
    )

    assert restored.as_dict() == profile.as_dict()
    assert camera_calibration_profile_to_live_calibration_dict(restored) == legacy
