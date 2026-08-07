from __future__ import annotations

import numpy as np

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
    CameraDistortionModel,
    LiveSlamRequest,
    ReconstructionCamera,
    ReconstructionQualityProfile,
    make_usb_source_metadata,
)
from tests.reconstruction_helpers import SESSION_ID

LIVE_IMAGE_SIZE = ImageSize(320, 240)


def make_calibrated_live_camera() -> ReconstructionCamera:
    intrinsics = CameraIntrinsics(
        np.array(
            [[250.0, 0.0, 160.0], [0.0, 251.0, 120.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "live-rgb",
        IntrinsicsSource.CALIBRATED,
        CalibrationStatus.CALIBRATED,
        calibration_record_id="calibration-live-test-1",
    )
    frame = CoordinateFrame(
        "live-camera-frame",
        CoordinateFrameKind.CAMERA_GENERIC,
        AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        parent_frame_id="live-reconstruction",
    )
    return ReconstructionCamera(
        camera_id="live-camera-1",
        image_size=LIVE_IMAGE_SIZE,
        intrinsics=intrinsics,
        camera_frame=frame,
        distortion_model=CameraDistortionModel.BROWN_CONRADY,
        distortion_coefficients=(-0.01, 0.002, 0.0, 0.0, 0.0),
    )


def make_live_request(*, backend_id: str = "development-live-slam-mock") -> LiveSlamRequest:
    return LiveSlamRequest(
        session_id=SESSION_ID,
        source=make_usb_source_metadata(
            0,
            width=LIVE_IMAGE_SIZE.width,
            height=LIVE_IMAGE_SIZE.height,
            frame_rate_hz=30.0,
        ),
        camera=make_calibrated_live_camera(),
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id=backend_id,
    )
