#!/usr/bin/env python3
"""Dependency-light Phase 9.1A calibration-profile compatibility smoke."""

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
    arbitrary_scale_reference,
    assess_reconstruction_geometry,
    camera_calibration_profile_from_json,
    camera_calibration_profile_to_json,
    camera_geometry_from_calibrated_camera,
)


def main() -> int:
    size = ImageSize(640, 480)
    intrinsics = CameraIntrinsics(
        np.array(
            [[520.0, 0.0, 320.0], [0.0, 521.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "phase9-1a-smoke-rgb",
        IntrinsicsSource.CALIBRATED,
        CalibrationStatus.CALIBRATED,
        calibration_record_id="phase9-1a-smoke-calibration",
    )
    camera = ReconstructionCamera(
        camera_id="phase9-1a-smoke-camera",
        image_size=size,
        intrinsics=intrinsics,
        camera_frame=CoordinateFrame(
            "phase9-1a-smoke-camera-frame",
            CoordinateFrameKind.CAMERA_GENERIC,
            AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
            parent_frame_id="phase9-1a-smoke-reconstruction",
        ),
        distortion_model=CameraDistortionModel.BROWN_CONRADY,
        distortion_coefficients=(-0.01, 0.001, 0.0, 0.0, 0.0),
    )
    profile = CameraCalibrationProfile(
        profile_id="phase9-1a-smoke-profile",
        camera=camera,
        method=CameraCalibrationMethod.MANUAL,
        validation_state=CameraCalibrationValidationState.UNVALIDATED,
        provenance="phase9_1a_dependency_light_smoke",
        created_at_utc="2026-08-13T16:00:00+00:00",
    )
    restored = camera_calibration_profile_from_json(
        camera_calibration_profile_to_json(profile)
    )
    if restored.as_dict() != profile.as_dict():
        raise RuntimeError("Calibration-profile JSON round trip changed the profile.")

    geometry = camera_geometry_from_calibrated_camera(
        profile_id="phase9-1a-smoke-camera-geometry",
        camera=restored.camera,
    )
    assessment = assess_reconstruction_geometry(
        geometry,
        arbitrary_scale_reference(profile_id="phase9-1a-smoke-arbitrary-scale"),
    )
    if assessment.metric_coordinates_allowed:
        raise RuntimeError("Camera calibration incorrectly promoted arbitrary scale to metric.")
    if assessment.certified_metrology:
        raise RuntimeError("Camera calibration incorrectly claimed certified metrology.")

    print(
        json.dumps(
            {
                "status": "PASS",
                "profile_id": restored.profile_id,
                "calibration_record_id": restored.calibration_record_id,
                "validation_state": restored.validation_state.value,
                "metric_status": assessment.metric_status.value,
                "metric_coordinates_allowed": assessment.metric_coordinates_allowed,
                "certified_metrology": assessment.certified_metrology,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
