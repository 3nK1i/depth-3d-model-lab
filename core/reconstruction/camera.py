"""Camera and distortion contracts for reconstruction backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from core.contracts import (
    AxisConvention,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageSize,
)
from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError


class CameraDistortionModel(str, Enum):
    NONE = "none"
    BROWN_CONRADY = "brown_conrady"
    OPENCV_FISHEYE = "opencv_fisheye"
    BACKEND_SPECIFIC = "backend_specific"


@dataclass(frozen=True, slots=True)
class ReconstructionCamera:
    """One camera model using the canonical optical-axis convention.

    Reconstruction camera coordinates are right-handed with X right, Y down,
    and Z forward. Intrinsics are expressed in the declared image pixel space.
    """

    camera_id: str
    image_size: ImageSize
    intrinsics: CameraIntrinsics
    camera_frame: CoordinateFrame
    distortion_model: CameraDistortionModel = CameraDistortionModel.NONE
    distortion_coefficients: tuple[float, ...] = ()
    backend_distortion_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "camera_id", require_nonempty(self.camera_id, field_name="camera_id")
        )
        if not isinstance(self.image_size, ImageSize):
            raise TypeError("image_size must be an ImageSize.")
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise TypeError("intrinsics must be CameraIntrinsics.")
        if not isinstance(self.camera_frame, CoordinateFrame):
            raise TypeError("camera_frame must be a CoordinateFrame.")
        if self.camera_frame.kind not in {
            CoordinateFrameKind.CAMERA_GENERIC,
            CoordinateFrameKind.CAMERA_RGB,
            CoordinateFrameKind.CAMERA_DEPTH,
        }:
            raise ReconstructionValidationError(
                "camera_frame must use a camera coordinate-frame kind."
            )
        if self.camera_frame.axis_convention is not AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD:
            raise ReconstructionValidationError(
                "Reconstruction cameras must use X-right, Y-down, Z-forward axes."
            )
        if not isinstance(self.distortion_model, CameraDistortionModel):
            raise TypeError("distortion_model must be a CameraDistortionModel.")

        coefficients = tuple(float(value) for value in self.distortion_coefficients)
        if not np.isfinite(np.asarray(coefficients, dtype=np.float64)).all():
            raise ReconstructionValidationError(
                "distortion_coefficients must contain only finite values."
            )
        allowed_counts = {
            CameraDistortionModel.NONE: {0},
            CameraDistortionModel.BROWN_CONRADY: {4, 5, 8, 12, 14},
            CameraDistortionModel.OPENCV_FISHEYE: {4},
        }
        if self.distortion_model in allowed_counts:
            expected = allowed_counts[self.distortion_model]
            if len(coefficients) not in expected:
                raise ReconstructionValidationError(
                    f"{self.distortion_model.value} distortion expects coefficient counts "
                    f"{sorted(expected)}; received {len(coefficients)}."
                )
        backend_name = str(self.backend_distortion_name).strip()
        if self.distortion_model is CameraDistortionModel.BACKEND_SPECIFIC:
            if not coefficients:
                raise ReconstructionValidationError(
                    "Backend-specific distortion requires coefficients."
                )
            backend_name = require_nonempty(
                backend_name, field_name="backend_distortion_name"
            )
        elif backend_name:
            raise ReconstructionValidationError(
                "backend_distortion_name is valid only for backend-specific distortion."
            )

        matrix = self.intrinsics.matrix
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        if not (0.0 <= cx < self.image_size.width and 0.0 <= cy < self.image_size.height):
            raise ReconstructionValidationError(
                "Camera principal point must lie inside the declared image size."
            )
        object.__setattr__(self, "distortion_coefficients", coefficients)
        object.__setattr__(self, "backend_distortion_name", backend_name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "image_size": self.image_size.as_dict(),
            "intrinsics": {
                "matrix": self.intrinsics.matrix.tolist(),
                "pixel_space_id": self.intrinsics.pixel_space_id,
                "source": self.intrinsics.source.value,
                "calibration_status": self.intrinsics.calibration_status.value,
                "calibration_record_id": self.intrinsics.calibration_record_id,
            },
            "camera_frame": {
                "frame_id": self.camera_frame.frame_id,
                "kind": self.camera_frame.kind.value,
                "axis_convention": self.camera_frame.axis_convention.value,
                "parent_frame_id": self.camera_frame.parent_frame_id,
                "length_unit": self.camera_frame.length_unit.value,
            },
            "distortion_model": self.distortion_model.value,
            "distortion_coefficients": list(self.distortion_coefficients),
            "backend_distortion_name": self.backend_distortion_name,
        }
