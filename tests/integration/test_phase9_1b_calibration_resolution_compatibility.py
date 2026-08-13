from __future__ import annotations

from core.contracts import ImageSize
from core.reconstruction import (
    CameraCalibrationMethod,
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
    LiveSlamRequest,
    ReconstructionQualityProfile,
    apply_camera_calibration_profile,
    make_usb_source_metadata,
)
from tests.reconstruction_helpers import SESSION_ID
from tests.reconstruction_live_helpers import make_calibrated_live_camera


def test_rescaled_calibration_satisfies_existing_live_resolution_guard_without_weakening_it() -> None:
    profile = CameraCalibrationProfile(
        profile_id="phase9-1b-live-profile",
        camera=make_calibrated_live_camera(),
        method=CameraCalibrationMethod.MANUAL,
        validation_state=CameraCalibrationValidationState.VALIDATED,
        provenance="pytest-phase9-1b-live",
        created_at_utc="2026-08-13T16:00:00+00:00",
    )
    target_size = ImageSize(640, 480)
    applied = apply_camera_calibration_profile(
        profile,
        target_size,
        allow_same_aspect_full_frame_resize=True,
        target_pixel_space_id="live-rgb-640x480",
    )
    source = make_usb_source_metadata(
        0,
        width=target_size.width,
        height=target_size.height,
        frame_rate_hz=30.0,
    )

    request = LiveSlamRequest(
        session_id=SESSION_ID,
        source=source,
        camera=applied.camera,
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id="development-live-slam-mock",
    )

    assert request.camera.image_size == target_size
    assert request.source.expected_image_size == target_size
    assert request.camera.intrinsics.calibration_record_id == profile.calibration_record_id
