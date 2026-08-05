"""Public dependency-light shared contracts for Phase 7.5."""

from core.contracts.common import FrameReference, ImageSize, ModelProvenance, ProcessingStep
from core.contracts.spatial import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageTransform2D,
    LengthUnit,
    PixelCoordinateConvention,
    PixelCoordinateSpace,
    PixelOrigin,
    PixelSpaceRole,
    RigidTransform3D,
)
from core.contracts.validation import ContractValidationError

__all__ = [
    "AxisConvention",
    "CalibrationStatus",
    "CameraIntrinsics",
    "ContractValidationError",
    "CoordinateFrame",
    "CoordinateFrameKind",
    "FrameReference",
    "ImageSize",
    "ImageTransform2D",
    "LengthUnit",
    "ModelProvenance",
    "PixelCoordinateConvention",
    "PixelCoordinateSpace",
    "PixelOrigin",
    "PixelSpaceRole",
    "ProcessingStep",
    "RigidTransform3D",
]
