"""Explicit calibrated-camera JSON loader for live SLAM.

Phase 8.4C deliberately does not infer camera intrinsics. A live session must
reference an explicit calibration record that matches the capture resolution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

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
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera
from core.reconstruction.errors import ReconstructionValidationError

LIVE_CALIBRATION_SCHEMA_VERSION = 1


def load_live_camera_calibration(path: str | Path) -> ReconstructionCamera:
    calibration_path = Path(path).expanduser().resolve()
    if not calibration_path.is_file():
        raise ReconstructionValidationError(
            f"Live camera calibration file does not exist: {calibration_path}"
        )
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError("Could not parse live camera calibration JSON.") from exc
    if not isinstance(payload, Mapping):
        raise ReconstructionValidationError("Live camera calibration must be a JSON object.")
    return live_camera_from_calibration_dict(payload)


def live_camera_from_calibration_dict(payload: Mapping[str, Any]) -> ReconstructionCamera:
    if int(payload.get("schema_version", 0)) != LIVE_CALIBRATION_SCHEMA_VERSION:
        raise ReconstructionValidationError("Unsupported live camera calibration schema version.")
    try:
        camera_id = str(payload["camera_id"])
        record_id = str(payload["calibration_record_id"])
        width = int(payload["width"])
        height = int(payload["height"])
        fx = float(payload["fx"])
        fy = float(payload["fy"])
        cx = float(payload["cx"])
        cy = float(payload["cy"])
        distortion_model = CameraDistortionModel(str(payload.get("distortion_model", "none")))
        coefficients = tuple(float(value) for value in payload.get("distortion_coefficients", ()))
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionValidationError("Invalid live camera calibration fields.") from exc
    if not record_id.strip():
        raise ReconstructionValidationError("Live camera calibration requires calibration_record_id.")
    image_size = ImageSize(width, height)
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    intrinsics = CameraIntrinsics(
        matrix,
        f"live-calibration-{width}x{height}",
        IntrinsicsSource.CALIBRATED,
        CalibrationStatus.CALIBRATED,
        calibration_record_id=record_id,
    )
    camera_frame = CoordinateFrame(
        str(payload.get("camera_frame_id", f"{camera_id}-frame")),
        CoordinateFrameKind.CAMERA_GENERIC,
        AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        parent_frame_id=str(payload.get("reconstruction_frame_id", "live-reconstruction")),
    )
    return ReconstructionCamera(
        camera_id=camera_id,
        image_size=image_size,
        intrinsics=intrinsics,
        camera_frame=camera_frame,
        distortion_model=distortion_model,
        distortion_coefficients=coefficients,
    )


def live_camera_calibration_dict(camera: ReconstructionCamera) -> dict[str, Any]:
    if not isinstance(camera, ReconstructionCamera):
        raise TypeError("camera must be ReconstructionCamera.")
    if camera.intrinsics.calibration_status is not CalibrationStatus.CALIBRATED:
        raise ReconstructionValidationError("Only explicitly calibrated cameras can be exported.")
    matrix = camera.intrinsics.matrix
    return {
        "schema_version": LIVE_CALIBRATION_SCHEMA_VERSION,
        "camera_id": camera.camera_id,
        "calibration_record_id": camera.intrinsics.calibration_record_id,
        "width": camera.image_size.width,
        "height": camera.image_size.height,
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
        "distortion_model": camera.distortion_model.value,
        "distortion_coefficients": list(camera.distortion_coefficients),
        "camera_frame_id": camera.camera_frame.frame_id,
        "reconstruction_frame_id": camera.camera_frame.parent_frame_id,
    }
