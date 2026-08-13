from __future__ import annotations

import numpy as np
import pytest

from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageSize,
)
from core.models.metadata import IntrinsicsSource
from core.reconstruction import (
    CalibrationApplicationMode,
    CameraCalibrationMethod,
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
    CameraDistortionModel,
    ReconstructionCamera,
    ReconstructionValidationError,
    apply_camera_calibration_profile,
    arbitrary_scale_reference,
    assess_reconstruction_geometry,
    camera_geometry_from_calibrated_camera,
)


def _profile(size: ImageSize = ImageSize(640, 480)) -> CameraCalibrationProfile:
    camera = ReconstructionCamera(
        camera_id="phase9-1b-camera",
        image_size=size,
        intrinsics=CameraIntrinsics(
            np.array(
                [[520.0, 0.0, 320.0], [0.0, 522.0, 240.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "phase9-1b-rgb",
            IntrinsicsSource.CALIBRATED,
            CalibrationStatus.CALIBRATED,
            calibration_record_id="phase9-1b-record",
        ),
        camera_frame=CoordinateFrame(
            "phase9-1b-camera-frame",
            CoordinateFrameKind.CAMERA_GENERIC,
            AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
            parent_frame_id="phase9-1b-reconstruction",
        ),
        distortion_model=CameraDistortionModel.BROWN_CONRADY,
        distortion_coefficients=(-0.01, 0.002, 0.0, 0.0, 0.0),
    )
    return CameraCalibrationProfile(
        profile_id="phase9-1b-profile",
        camera=camera,
        method=CameraCalibrationMethod.MANUAL,
        validation_state=CameraCalibrationValidationState.VALIDATED,
        provenance="pytest-phase9-1b",
        created_at_utc="2026-08-13T16:00:00+00:00",
    )


def test_exact_resolution_preserves_camera_authority_and_record_id() -> None:
    profile = _profile()
    applied = apply_camera_calibration_profile(profile, ImageSize(640, 480))

    assert applied.mode is CalibrationApplicationMode.EXACT_RESOLUTION
    assert applied.was_rescaled is False
    assert applied.camera.image_size == profile.camera.image_size
    assert applied.camera.intrinsics.matrix == pytest.approx(profile.camera.intrinsics.matrix)
    assert applied.camera.intrinsics.calibration_record_id == profile.calibration_record_id
    assert applied.camera.distortion_coefficients == profile.camera.distortion_coefficients
    assert applied.warnings == ()


def test_same_aspect_full_frame_resize_scales_intrinsics_and_preserves_distortion() -> None:
    profile = _profile()
    applied = apply_camera_calibration_profile(
        profile,
        ImageSize(320, 240),
        allow_same_aspect_full_frame_resize=True,
    )

    assert applied.mode is CalibrationApplicationMode.SAME_ASPECT_FULL_FRAME_RESIZE
    assert applied.was_rescaled is True
    assert applied.camera.intrinsics.matrix == pytest.approx(
        np.array([[260.0, 0.0, 160.0], [0.0, 261.0, 120.0], [0.0, 0.0, 1.0]])
    )
    assert applied.camera.intrinsics.calibration_record_id == profile.calibration_record_id
    assert applied.camera.distortion_model is CameraDistortionModel.BROWN_CONRADY
    assert applied.camera.distortion_coefficients == profile.camera.distortion_coefficients
    assert applied.warnings == ("calibration_intrinsics_rescaled_from_full_frame_profile",)


def test_resolution_change_is_rejected_without_explicit_policy() -> None:
    with pytest.raises(ReconstructionValidationError, match="explicit qualified resize policy"):
        apply_camera_calibration_profile(_profile(), ImageSize(320, 240))


def test_aspect_ratio_change_is_rejected_even_when_resize_policy_is_enabled() -> None:
    with pytest.raises(ReconstructionValidationError, match="same aspect ratio"):
        apply_camera_calibration_profile(
            _profile(),
            ImageSize(640, 360),
            allow_same_aspect_full_frame_resize=True,
        )


def test_custom_target_pixel_space_is_explicit_and_profile_remains_immutable() -> None:
    profile = _profile()
    original = profile.as_dict()
    applied = apply_camera_calibration_profile(
        profile,
        ImageSize(1280, 960),
        allow_same_aspect_full_frame_resize=True,
        target_pixel_space_id="usb0-runtime-rgb",
    )

    assert applied.camera.intrinsics.pixel_space_id == "usb0-runtime-rgb"
    assert profile.as_dict() == original
    assert applied.as_dict()["pixel_transform"]["destination_space"]["size"] == {
        "width": 1280,
        "height": 960,
    }


def test_resolution_application_does_not_promote_metric_authority() -> None:
    profile = _profile()
    applied = apply_camera_calibration_profile(
        profile,
        ImageSize(320, 240),
        allow_same_aspect_full_frame_resize=True,
    )
    geometry = camera_geometry_from_calibrated_camera(
        profile_id="phase9-1b-camera-geometry",
        camera=applied.camera,
    )
    assessment = assess_reconstruction_geometry(
        geometry,
        arbitrary_scale_reference(profile_id="phase9-1b-arbitrary-scale"),
    )
    assert assessment.metric_coordinates_allowed is False
    assert assessment.certified_metrology is False
