"""Authoritative pixel-space, camera and 3D coordinate-frame contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose
from typing import Any

import numpy as np

from core.contracts.common import FrameReference, ImageSize
from core.contracts.validation import (
    ContractValidationError,
    readonly_array,
    require_nonempty,
)
from core.models.metadata import IntrinsicsSource


class PixelSpaceRole(str, Enum):
    RGB = "rgb"
    MODEL_INPUT = "model_input"
    MASK = "mask"
    DEPTH = "depth"
    CONFIDENCE = "confidence"
    ANNOTATION = "annotation"
    OTHER = "other"


class PixelOrigin(str, Enum):
    TOP_LEFT = "top_left"


class PixelCoordinateConvention(str, Enum):
    PIXEL_CENTER = "pixel_center"
    PIXEL_EDGE = "pixel_edge"


class CoordinateFrameKind(str, Enum):
    CAMERA_RGB = "camera_rgb"
    CAMERA_DEPTH = "camera_depth"
    OBJECT_LOCAL = "object_local"
    WORLD = "world"
    RECONSTRUCTION_MAP = "reconstruction_map"
    CAMERA_GENERIC = "camera_generic"


class AxisConvention(str, Enum):
    X_RIGHT_Y_DOWN_Z_FORWARD = "x_right_y_down_z_forward"
    X_RIGHT_Y_UP_Z_BACK = "x_right_y_up_z_back"
    CUSTOM = "custom"


class LengthUnit(str, Enum):
    METRE = "m"


class CalibrationStatus(str, Enum):
    CALIBRATED = "calibrated"
    NOT_CALIBRATED = "not_calibrated"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PixelCoordinateSpace:
    space_id: str
    frame_reference: FrameReference
    size: ImageSize
    role: PixelSpaceRole
    origin: PixelOrigin = PixelOrigin.TOP_LEFT
    coordinate_convention: PixelCoordinateConvention = PixelCoordinateConvention.PIXEL_EDGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "space_id", require_nonempty(self.space_id, field_name="space_id"))
        if not isinstance(self.frame_reference, FrameReference):
            raise TypeError("frame_reference must be a FrameReference.")
        if not isinstance(self.size, ImageSize):
            raise TypeError("size must be an ImageSize.")
        if not isinstance(self.role, PixelSpaceRole):
            raise TypeError("role must be a PixelSpaceRole.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "frame_reference": self.frame_reference.as_dict(),
            "size": self.size.as_dict(),
            "role": self.role.value,
            "origin": self.origin.value,
            "coordinate_convention": self.coordinate_convention.value,
        }


@dataclass(frozen=True, slots=True)
class ImageTransform2D:
    """One authoritative homogeneous pixel transform.

    Descriptive operations use the fixed order: crop source, resize cropped
    content, then add destination padding. Lens distortion is deliberately not
    represented here and must remain a separate future operation.
    """

    source_space: PixelCoordinateSpace
    destination_space: PixelCoordinateSpace
    matrix: np.ndarray
    crop_xyxy: tuple[float, float, float, float] | None = None
    resized_size: ImageSize | None = None
    padding_ltrb: tuple[int, int, int, int] = (0, 0, 0, 0)
    interpolation: str = "unspecified"

    def __post_init__(self) -> None:
        if not isinstance(self.source_space, PixelCoordinateSpace):
            raise TypeError("source_space must be a PixelCoordinateSpace.")
        if not isinstance(self.destination_space, PixelCoordinateSpace):
            raise TypeError("destination_space must be a PixelCoordinateSpace.")
        if self.source_space.frame_reference != self.destination_space.frame_reference:
            raise ContractValidationError(
                "source_space and destination_space must refer to the same captured frame."
            )
        matrix = readonly_array(
            self.matrix,
            field_name="matrix",
            dtype=np.float64,
            shape=(3, 3),
            finite=True,
        )
        if not np.allclose(matrix[2], np.array([0.0, 0.0, 1.0]), atol=1e-9):
            raise ContractValidationError(
                "ImageTransform2D matrix must have homogeneous final row [0, 0, 1]."
            )
        if abs(float(np.linalg.det(matrix))) <= 1e-12:
            raise ContractValidationError("ImageTransform2D matrix must be invertible.")

        source_w = float(self.source_space.size.width)
        source_h = float(self.source_space.size.height)
        crop = self.crop_xyxy or (0.0, 0.0, source_w, source_h)
        if len(crop) != 4 or not np.isfinite(np.asarray(crop, dtype=float)).all():
            raise ContractValidationError("crop_xyxy must contain four finite coordinates.")
        x0, y0, x1, y1 = (float(value) for value in crop)
        if x0 < 0.0 or y0 < 0.0 or x1 > source_w or y1 > source_h:
            raise ContractValidationError("crop_xyxy must be inside the source pixel space.")
        if x1 <= x0 or y1 <= y0:
            raise ContractValidationError("crop_xyxy must have positive width and height.")

        resized = self.resized_size or ImageSize(
            width=int(round(x1 - x0)),
            height=int(round(y1 - y0)),
        )
        if not isinstance(resized, ImageSize):
            raise TypeError("resized_size must be an ImageSize or None.")
        if len(self.padding_ltrb) != 4:
            raise ContractValidationError("padding_ltrb must contain four integers.")
        padding = tuple(int(value) for value in self.padding_ltrb)
        if any(value < 0 for value in padding):
            raise ContractValidationError("padding_ltrb values must not be negative.")
        left, top, right, bottom = padding
        expected_width = resized.width + left + right
        expected_height = resized.height + top + bottom
        if (
            expected_width != self.destination_space.size.width
            or expected_height != self.destination_space.size.height
        ):
            raise ContractValidationError(
                "Descriptive crop/resize/padding dimensions disagree with the destination space."
            )

        scale_x = resized.width / (x1 - x0)
        scale_y = resized.height / (y1 - y0)
        expected_matrix = np.array(
            [
                [scale_x, 0.0, left - scale_x * x0],
                [0.0, scale_y, top - scale_y * y0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        if not np.allclose(matrix, expected_matrix, rtol=1e-9, atol=1e-9):
            raise ContractValidationError(
                "Authoritative matrix disagrees with descriptive crop/resize/padding operations."
            )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "crop_xyxy", (x0, y0, x1, y1))
        object.__setattr__(self, "resized_size", resized)
        object.__setattr__(self, "padding_ltrb", padding)
        object.__setattr__(self, "interpolation", str(self.interpolation).strip())

    @property
    def inverse_matrix(self) -> np.ndarray:
        inverse = np.linalg.inv(self.matrix)
        inverse.setflags(write=False)
        return inverse


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    frame_id: str
    kind: CoordinateFrameKind
    axis_convention: AxisConvention
    parent_frame_id: str = ""
    length_unit: LengthUnit = LengthUnit.METRE
    source_frame: FrameReference | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        object.__setattr__(self, "parent_frame_id", str(self.parent_frame_id).strip())
        if self.parent_frame_id == self.frame_id:
            raise ContractValidationError("A coordinate frame cannot be its own parent.")
        if not isinstance(self.kind, CoordinateFrameKind):
            raise TypeError("kind must be a CoordinateFrameKind.")
        if not isinstance(self.axis_convention, AxisConvention):
            raise TypeError("axis_convention must be an AxisConvention.")


@dataclass(frozen=True, slots=True)
class RigidTransform3D:
    source_frame_id: str
    destination_frame_id: str
    matrix: np.ndarray
    method: str = "declared"
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        source = require_nonempty(self.source_frame_id, field_name="source_frame_id")
        destination = require_nonempty(
            self.destination_frame_id, field_name="destination_frame_id"
        )
        matrix = readonly_array(
            self.matrix,
            field_name="matrix",
            dtype=np.float64,
            shape=(4, 4),
            finite=True,
        )
        tolerance = float(self.tolerance)
        if tolerance <= 0.0 or not np.isfinite(tolerance):
            raise ContractValidationError("tolerance must be positive and finite.")
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=tolerance):
            raise ContractValidationError(
                "RigidTransform3D matrix must have homogeneous final row [0, 0, 0, 1]."
            )
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance):
            raise ContractValidationError("RigidTransform3D rotation must be orthonormal.")
        determinant = float(np.linalg.det(rotation))
        if determinant <= 0.0 or not isclose(determinant, 1.0, abs_tol=tolerance):
            raise ContractValidationError(
                "RigidTransform3D rotation must have a positive unit determinant."
            )
        identity = np.allclose(matrix, np.eye(4), atol=tolerance)
        if source == destination and not identity:
            raise ContractValidationError(
                "Equal source and destination frame IDs are allowed only for identity transforms."
            )
        if not identity and source == destination:
            raise ContractValidationError(
                "Non-identity transforms require different frame IDs."
            )
        object.__setattr__(self, "source_frame_id", source)
        object.__setattr__(self, "destination_frame_id", destination)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "method", require_nonempty(self.method, field_name="method"))
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    matrix: np.ndarray
    pixel_space_id: str
    source: IntrinsicsSource
    calibration_status: CalibrationStatus
    calibration_record_id: str = ""

    def __post_init__(self) -> None:
        matrix = readonly_array(
            self.matrix,
            field_name="matrix",
            dtype=np.float64,
            shape=(3, 3),
            finite=True,
        )
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ContractValidationError("Camera focal lengths must be positive.")
        if not np.allclose(matrix[2], np.array([0.0, 0.0, 1.0]), atol=1e-9):
            raise ContractValidationError(
                "CameraIntrinsics matrix must have homogeneous final row [0, 0, 1]."
            )
        record_id = str(self.calibration_record_id).strip()
        if self.source in {IntrinsicsSource.MODEL_PREDICTED, IntrinsicsSource.ESTIMATED}:
            if self.calibration_status is not CalibrationStatus.NOT_CALIBRATED:
                raise ContractValidationError(
                    "Model-predicted and estimated intrinsics must remain Not Calibrated."
                )
        if self.calibration_status is CalibrationStatus.CALIBRATED:
            if self.source is not IntrinsicsSource.CALIBRATED or not record_id:
                raise ContractValidationError(
                    "Calibrated intrinsics require IntrinsicsSource.CALIBRATED and a record ID."
                )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(
            self, "pixel_space_id", require_nonempty(self.pixel_space_id, field_name="pixel_space_id")
        )
        object.__setattr__(self, "calibration_record_id", record_id)
