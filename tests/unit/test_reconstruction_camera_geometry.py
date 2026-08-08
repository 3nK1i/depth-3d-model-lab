from __future__ import annotations

from dataclasses import replace
from math import isclose, pi, tan

import numpy as np
import pytest

from core.contracts import CalibrationStatus, CameraIntrinsics, ImageSize
from core.models.metadata import IntrinsicsSource
from core.reconstruction import (
    CameraGeometryMethod,
    CameraGeometryMode,
    CameraGeometryPurpose,
    CameraIntrinsicsPolicy,
    LiveSlamRequest,
    ReconstructionValidationError,
    camera_geometry_from_calibrated_camera,
    camera_geometry_profile_from_json,
    camera_geometry_profile_to_json,
    estimate_camera_geometry_from_horizontal_fov,
    stella_monocular_config_text,
)
from tests.reconstruction_helpers import SESSION_ID
from tests.reconstruction_live_helpers import make_calibrated_live_camera
from core.reconstruction import ReconstructionQualityProfile, make_usb_source_metadata


def _estimated_profile():
    return estimate_camera_geometry_from_horizontal_fov(
        profile_id="estimated-usb-720p",
        camera_id="usb-camera-0",
        image_size=ImageSize(1280, 720),
        horizontal_fov_degrees=70.0,
        focal_length_relative_uncertainty=0.15,
    )


def test_estimated_fov_profile_is_explicitly_provisional() -> None:
    profile = _estimated_profile()
    camera = profile.camera
    expected_f = 1280.0 / (2.0 * tan(70.0 * pi / 360.0))
    assert profile.mode is CameraGeometryMode.ESTIMATED
    assert profile.method is CameraGeometryMethod.ASSUMED_HORIZONTAL_FOV
    assert profile.is_provisional is True
    assert camera.intrinsics.source is IntrinsicsSource.ESTIMATED
    assert camera.intrinsics.calibration_status is CalibrationStatus.NOT_CALIBRATED
    assert camera.intrinsics.calibration_record_id == ""
    assert isclose(float(camera.intrinsics.matrix[0, 0]), expected_f, rel_tol=1e-12)
    assert isclose(float(camera.intrinsics.matrix[1, 1]), expected_f, rel_tol=1e-12)
    assert camera.distortion_coefficients == ()
    assert profile.distortion_assumed_zero is True


def test_estimated_profile_allows_tracking_and_relative_geometry_only() -> None:
    profile = _estimated_profile()
    tracking = profile.assess(CameraGeometryPurpose.VISUAL_TRACKING)
    relative = profile.assess(CameraGeometryPurpose.RELATIVE_RECONSTRUCTION)
    rays = profile.assess(CameraGeometryPurpose.CALIBRATED_RAY_GEOMETRY)
    assert tracking.allowed and tracking.provisional
    assert relative.allowed and relative.provisional
    assert rays.allowed is False and rays.provisional is True


def test_calibrated_profile_preserves_calibration_record() -> None:
    camera = make_calibrated_live_camera()
    profile = camera_geometry_from_calibrated_camera(
        profile_id="calibrated-live",
        camera=camera,
    )
    assert profile.mode is CameraGeometryMode.CALIBRATED
    assert profile.is_calibrated is True
    assert profile.camera.intrinsics.calibration_record_id == "calibration-live-test-1"
    assessment = profile.assess(CameraGeometryPurpose.CALIBRATED_RAY_GEOMETRY)
    assert assessment.allowed is True
    assert assessment.provisional is False
    assert "scale" in assessment.message.lower()


def test_camera_geometry_json_round_trip_is_deterministic() -> None:
    profile = _estimated_profile()
    payload = camera_geometry_profile_to_json(profile)
    restored = camera_geometry_profile_from_json(payload)
    assert restored.as_dict() == profile.as_dict()
    assert camera_geometry_profile_to_json(restored) == payload


def test_estimated_profile_rejects_missing_uncertainty() -> None:
    profile = _estimated_profile()
    with pytest.raises(ReconstructionValidationError, match="uncertainty"):
        replace(profile, focal_length_relative_uncertainty=None)


def test_default_live_request_still_rejects_estimated_camera() -> None:
    profile = _estimated_profile()
    with pytest.raises(ReconstructionValidationError, match="calibrated"):
        LiveSlamRequest(
            session_id=SESSION_ID,
            source=make_usb_source_metadata(0, width=1280, height=720, frame_rate_hz=30.0),
            camera=profile.camera,
            profile=ReconstructionQualityProfile.EFFICIENT,
            backend_id="mock",
        )


def test_explicit_provisional_policy_allows_estimated_camera_for_live_tracking() -> None:
    profile = _estimated_profile()
    request = LiveSlamRequest(
        session_id=SESSION_ID,
        source=make_usb_source_metadata(0, width=1280, height=720, frame_rate_hz=30.0),
        camera=profile.camera,
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id="mock",
        camera_intrinsics_policy=CameraIntrinsicsPolicy.ALLOW_ESTIMATED,
    )
    assert request.camera_intrinsics_policy is CameraIntrinsicsPolicy.ALLOW_ESTIMATED
    assert request.as_dict()["camera_intrinsics_policy"] == "allow_estimated"
    assert request.camera.intrinsics.calibration_status is CalibrationStatus.NOT_CALIBRATED


def test_provisional_policy_does_not_accept_model_predicted_intrinsics() -> None:
    profile = _estimated_profile()
    predicted = CameraIntrinsics(
        profile.camera.intrinsics.matrix,
        profile.camera.intrinsics.pixel_space_id,
        IntrinsicsSource.MODEL_PREDICTED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    camera = replace(profile.camera, intrinsics=predicted)
    with pytest.raises(ReconstructionValidationError, match="ESTIMATED"):
        LiveSlamRequest(
            session_id=SESSION_ID,
            source=make_usb_source_metadata(0, width=1280, height=720, frame_rate_hz=30.0),
            camera=camera,
            profile=ReconstructionQualityProfile.EFFICIENT,
            backend_id="mock",
            camera_intrinsics_policy=CameraIntrinsicsPolicy.ALLOW_ESTIMATED,
        )


def test_stella_config_keeps_provisional_provenance_visible() -> None:
    profile = estimate_camera_geometry_from_horizontal_fov(
        profile_id="estimated-live",
        camera_id="camera",
        image_size=ImageSize(320, 240),
        horizontal_fov_degrees=70.0,
        focal_length_relative_uncertainty=0.15,
    )
    request = LiveSlamRequest(
        session_id=SESSION_ID,
        source=make_usb_source_metadata(0, width=320, height=240, frame_rate_hz=30.0),
        camera=profile.camera,
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id="mock",
        camera_intrinsics_policy=CameraIntrinsicsPolicy.ALLOW_ESTIMATED,
    )
    text = stella_monocular_config_text(request)
    assert "intrinsics_source=estimated" in text
    assert "calibration_status=not_calibrated" in text
    assert "camera_intrinsics_policy=allow_estimated" in text
    assert "unit_to_metre" not in text
