from __future__ import annotations

import numpy as np
import pytest

from core.alignment import (
    declare_full_frame_resize,
    depth_input_from_inference_result,
    make_depth_space,
    transform_intrinsics,
)
from core.alignment.calibration import rectify_bgr_with_camera_calibration
from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    FrameReference,
    ImageSize,
    PixelCoordinateSpace,
    PixelSpaceRole,
)
from core.models.metadata import IntrinsicsSource
from core.reconstruction import (
    CameraCalibrationMethod,
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
    CameraDistortionModel,
    ReconstructionCamera,
)
from core.runtime.messages import InferenceResult


def _profile(*, distortion_model=CameraDistortionModel.BROWN_CONRADY, coefficients=(-0.08, 0.01, 0.0, 0.0, 0.0)):
    size = ImageSize(640, 480)
    camera = ReconstructionCamera(
        camera_id="phase9-1g-camera",
        image_size=size,
        intrinsics=CameraIntrinsics(
            np.array([[520.0, 0.0, 320.0], [0.0, 522.0, 240.0], [0.0, 0.0, 1.0]]),
            "phase9-1g-calibration-rgb",
            IntrinsicsSource.CALIBRATED,
            CalibrationStatus.CALIBRATED,
            calibration_record_id="phase9-1g-record",
        ),
        camera_frame=CoordinateFrame(
            "phase9-1g-camera-frame",
            CoordinateFrameKind.CAMERA_GENERIC,
            AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
            parent_frame_id="phase9-1g-reconstruction",
        ),
        distortion_model=distortion_model,
        distortion_coefficients=coefficients,
    )
    return CameraCalibrationProfile(
        profile_id="phase9-1g-profile",
        camera=camera,
        method=CameraCalibrationMethod.MANUAL,
        validation_state=CameraCalibrationValidationState.UNVALIDATED,
        provenance="pytest-phase9-1g",
        created_at_utc="2026-08-13T18:00:00+00:00",
        quality=None,
    )


def test_rectification_keeps_same_grid_and_calibrated_authority() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[100:380, 120:520] = 255
    prepared = rectify_bgr_with_camera_calibration(image, _profile())

    assert prepared.image_bgr.shape == image.shape
    assert prepared.support_mask.shape == image.shape[:2]
    assert prepared.support_mask.dtype == np.bool_
    assert prepared.rectified is True
    assert prepared.application.camera.intrinsics.pixel_space_id == "rgb-rectified"
    assert prepared.application.camera.intrinsics.calibration_status is CalibrationStatus.CALIBRATED
    assert prepared.application.camera.intrinsics.calibration_record_id == "phase9-1g-record"
    assert "camera_distortion_rectified_for_measurement" in prepared.warnings


def test_no_distortion_does_not_modify_pixels_or_claim_rectification() -> None:
    image = np.arange(32 * 24 * 3, dtype=np.uint8).reshape(24, 32, 3)
    # Build a size-matched no-distortion profile through a resized version of the helper.
    base = _profile(distortion_model=CameraDistortionModel.NONE, coefficients=())
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    prepared = rectify_bgr_with_camera_calibration(image, base)

    assert prepared.rectified is False
    assert np.array_equal(prepared.image_bgr, image)
    assert np.all(prepared.support_mask)
    assert prepared.application.target_image_size == ImageSize(320, 240)


def test_calibrated_intrinsics_transform_to_native_depth_grid_and_override_model_intrinsics() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    prepared = rectify_bgr_with_camera_calibration(image, _profile())
    frame = FrameReference(7, 9, 0.0, ImageSize(640, 480))
    rgb_space = PixelCoordinateSpace(
        prepared.application.camera.intrinsics.pixel_space_id,
        frame,
        ImageSize(640, 480),
        PixelSpaceRole.RGB,
    )
    depth = np.ones((240, 320), dtype=np.float32)
    depth_space = make_depth_space(frame, depth)
    transform = declare_full_frame_resize(rgb_space, depth_space)
    calibrated_depth_intrinsics = transform_intrinsics(
        prepared.application.camera.intrinsics,
        transform,
    )
    inference = InferenceResult(
        session_id=7,
        frame_id=9,
        model_id="depth-pro",
        depth=depth,
        latency_ms=1.0,
        intrinsics=None,
        depth_scale="metric",
        intrinsics_source="none",
    )

    depth_input = depth_input_from_inference_result(
        inference,
        frame,
        intrinsics_override=calibrated_depth_intrinsics,
    )

    assert depth_input.intrinsics.pixel_space_id == "depth"
    assert depth_input.intrinsics.source is IntrinsicsSource.CALIBRATED
    assert depth_input.intrinsics.calibration_status is CalibrationStatus.CALIBRATED
    assert depth_input.intrinsics.calibration_record_id == "phase9-1g-record"
    assert depth_input.intrinsics.matrix == pytest.approx(
        np.array([[260.0, 0.0, 160.0], [0.0, 261.0, 120.0], [0.0, 0.0, 1.0]])
    )
    assert depth_input.depth_scale.value == "metric"


def test_fisheye_rectification_uses_same_calibrated_pinhole_output_grid() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    profile = _profile(
        distortion_model=CameraDistortionModel.OPENCV_FISHEYE,
        coefficients=(-0.02, 0.003, 0.0, 0.0),
    )
    prepared = rectify_bgr_with_camera_calibration(image, profile)
    assert prepared.rectified is True
    assert prepared.image_bgr.shape == image.shape
    assert prepared.application.camera.intrinsics.matrix == pytest.approx(profile.camera.intrinsics.matrix)
    assert np.count_nonzero(prepared.support_mask) > 0


def test_default_depth_factory_still_rejects_untrusted_calibrated_model_claim() -> None:
    frame = FrameReference(1, 1, 0.0, ImageSize(8, 6))
    result = InferenceResult(
        session_id=1,
        frame_id=1,
        model_id="model",
        depth=np.ones((6, 8), dtype=np.float32),
        latency_ms=1.0,
        intrinsics=np.eye(3),
        intrinsics_source="calibrated",
    )
    from core.contracts.validation import ContractValidationError

    with pytest.raises(ContractValidationError, match="without a calibration record"):
        depth_input_from_inference_result(result, frame)


def test_override_requires_native_depth_pixel_space() -> None:
    frame = FrameReference(1, 1, 0.0, ImageSize(8, 6))
    result = InferenceResult(
        session_id=1,
        frame_id=1,
        model_id="model",
        depth=np.ones((6, 8), dtype=np.float32),
        latency_ms=1.0,
    )
    wrong = CameraIntrinsics(
        np.eye(3),
        "rgb",
        IntrinsicsSource.CALIBRATED,
        CalibrationStatus.CALIBRATED,
        calibration_record_id="trusted-record",
    )
    from core.contracts.validation import ContractValidationError

    with pytest.raises(ContractValidationError, match="native depth pixel space"):
        depth_input_from_inference_result(result, frame, intrinsics_override=wrong)
