#!/usr/bin/env python3
"""Dependency-light Phase 9.1B calibration resolution-application smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    CameraCalibrationMethod,
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
    CameraDistortionModel,
    ReconstructionCamera,
    ReconstructionValidationError,
    apply_camera_calibration_profile,
)


def main() -> int:
    source_size = ImageSize(1920, 1080)
    target_size = ImageSize(1280, 720)
    camera = ReconstructionCamera(
        camera_id="phase9-1b-smoke-camera",
        image_size=source_size,
        intrinsics=CameraIntrinsics(
            np.array(
                [[1400.0, 0.0, 960.0], [0.0, 1404.0, 540.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "phase9-1b-smoke-rgb",
            IntrinsicsSource.CALIBRATED,
            CalibrationStatus.CALIBRATED,
            calibration_record_id="phase9-1b-smoke-record",
        ),
        camera_frame=CoordinateFrame(
            "phase9-1b-smoke-frame",
            CoordinateFrameKind.CAMERA_GENERIC,
            AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
            parent_frame_id="phase9-1b-smoke-reconstruction",
        ),
        distortion_model=CameraDistortionModel.BROWN_CONRADY,
        distortion_coefficients=(-0.02, 0.003, 0.0, 0.0, 0.0),
    )
    profile = CameraCalibrationProfile(
        profile_id="phase9-1b-smoke-profile",
        camera=camera,
        method=CameraCalibrationMethod.MANUAL,
        validation_state=CameraCalibrationValidationState.VALIDATED,
        provenance="phase9_1b_dependency_light_smoke",
        created_at_utc="2026-08-13T16:00:00+00:00",
    )
    applied = apply_camera_calibration_profile(
        profile,
        target_size,
        allow_same_aspect_full_frame_resize=True,
    )
    expected_fx = 1400.0 * (1280.0 / 1920.0)
    expected_fy = 1404.0 * (720.0 / 1080.0)
    if not np.isclose(applied.camera.intrinsics.matrix[0, 0], expected_fx):
        raise RuntimeError("fx was not rescaled correctly.")
    if not np.isclose(applied.camera.intrinsics.matrix[1, 1], expected_fy):
        raise RuntimeError("fy was not rescaled correctly.")
    if applied.camera.intrinsics.calibration_record_id != profile.calibration_record_id:
        raise RuntimeError("Calibration record provenance was not preserved.")

    rejected_aspect_change = False
    try:
        apply_camera_calibration_profile(
            profile,
            ImageSize(1280, 800),
            allow_same_aspect_full_frame_resize=True,
        )
    except ReconstructionValidationError:
        rejected_aspect_change = True
    if not rejected_aspect_change:
        raise RuntimeError("Unsafe aspect-ratio change was not rejected.")

    print(
        json.dumps(
            {
                "status": "PASS",
                "profile_id": profile.profile_id,
                "calibration_record_id": profile.calibration_record_id,
                "calibration_resolution": source_size.as_dict(),
                "target_resolution": target_size.as_dict(),
                "application_mode": applied.mode.value,
                "fx": float(applied.camera.intrinsics.matrix[0, 0]),
                "fy": float(applied.camera.intrinsics.matrix[1, 1]),
                "unsafe_aspect_ratio_change_rejected": rejected_aspect_change,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
