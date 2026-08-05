"""Factories that deliberately declare pixel transforms and depth inputs."""

from __future__ import annotations

import numpy as np

from core.alignment.contracts import DepthAlignmentInput
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import (
    CalibrationStatus,
    CameraIntrinsics,
    ImageTransform2D,
    PixelCoordinateSpace,
    PixelSpaceRole,
)
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind, DepthScale, IntrinsicsSource
from core.runtime.messages import InferenceResult


def declare_identity_transform(
    source_space: PixelCoordinateSpace,
    destination_space: PixelCoordinateSpace,
    *,
    interpolation: str = "identity",
) -> ImageTransform2D:
    """Declare an explicit identity correspondence between equal-sized spaces."""

    if source_space.size != destination_space.size:
        raise ContractValidationError("Identity transforms require equal source/destination sizes.")
    return ImageTransform2D(
        source_space=source_space,
        destination_space=destination_space,
        matrix=np.eye(3, dtype=np.float64),
        crop_xyxy=(0.0, 0.0, float(source_space.size.width), float(source_space.size.height)),
        resized_size=source_space.size,
        padding_ltrb=(0, 0, 0, 0),
        interpolation=interpolation,
    )


def declare_full_frame_resize(
    source_space: PixelCoordinateSpace,
    destination_space: PixelCoordinateSpace,
    *,
    interpolation: str = "auto",
) -> ImageTransform2D:
    """Explicitly declare full-frame resize correspondence without crop or padding."""

    sx = destination_space.size.width / source_space.size.width
    sy = destination_space.size.height / source_space.size.height
    return ImageTransform2D(
        source_space=source_space,
        destination_space=destination_space,
        matrix=np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]),
        crop_xyxy=(0.0, 0.0, float(source_space.size.width), float(source_space.size.height)),
        resized_size=destination_space.size,
        padding_ltrb=(0, 0, 0, 0),
        interpolation=interpolation,
    )


def declare_crop_resize_padding(
    source_space: PixelCoordinateSpace,
    destination_space: PixelCoordinateSpace,
    *,
    crop_xyxy: tuple[float, float, float, float],
    resized_size: ImageSize,
    padding_ltrb: tuple[int, int, int, int],
    interpolation: str = "auto",
) -> ImageTransform2D:
    """Declare crop -> resize -> padding and derive its authoritative matrix."""

    x0, y0, x1, y1 = (float(value) for value in crop_xyxy)
    left, top, _, _ = (int(value) for value in padding_ltrb)
    sx = resized_size.width / (x1 - x0)
    sy = resized_size.height / (y1 - y0)
    matrix = np.array(
        [
            [sx, 0.0, left - sx * x0],
            [0.0, sy, top - sy * y0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ImageTransform2D(
        source_space=source_space,
        destination_space=destination_space,
        matrix=matrix,
        crop_xyxy=crop_xyxy,
        resized_size=resized_size,
        padding_ltrb=padding_ltrb,
        interpolation=interpolation,
    )


def make_depth_space(
    frame_reference: FrameReference,
    depth: np.ndarray,
    *,
    space_id: str = "depth",
) -> PixelCoordinateSpace:
    array = np.asarray(depth)
    if array.ndim != 2:
        raise ContractValidationError("depth must be HxW to construct a depth pixel space.")
    return PixelCoordinateSpace(
        space_id=space_id,
        frame_reference=frame_reference,
        size=ImageSize(width=int(array.shape[1]), height=int(array.shape[0])),
        role=PixelSpaceRole.DEPTH,
    )


def depth_input_from_inference_result(
    result: InferenceResult,
    frame_reference: FrameReference,
    *,
    depth_space_id: str = "depth",
    model_provenance: ModelProvenance | None = None,
) -> DepthAlignmentInput:
    """Convert the append-only Phase 6/7 IPC result without changing it."""

    if result.session_id != frame_reference.session_id or result.frame_id != frame_reference.frame_id:
        raise ContractValidationError("InferenceResult does not match frame_reference.")
    if result.depth is None:
        raise ContractValidationError("InferenceResult has no depth map.")
    if result.intrinsics is None:
        raise ContractValidationError("InferenceResult has no camera intrinsics.")

    try:
        depth_scale = DepthScale(result.depth_scale or ("metric" if result.is_metric else "unknown"))
    except ValueError as exc:
        raise ContractValidationError(f"Unknown depth scale: {result.depth_scale!r}.") from exc
    try:
        intrinsics_source = IntrinsicsSource(result.intrinsics_source or "none")
    except ValueError as exc:
        raise ContractValidationError(
            f"Unknown intrinsics source: {result.intrinsics_source!r}."
        ) from exc
    try:
        confidence_kind = ConfidenceKind(result.confidence_kind or "none")
    except ValueError as exc:
        raise ContractValidationError(
            f"Unknown confidence kind: {result.confidence_kind!r}."
        ) from exc

    if intrinsics_source is IntrinsicsSource.CALIBRATED:
        raise ContractValidationError(
            "Step 3 cannot construct calibrated intrinsics without a calibration record."
        )
    calibration_status = (
        CalibrationStatus.NOT_CALIBRATED
        if intrinsics_source in {IntrinsicsSource.MODEL_PREDICTED, IntrinsicsSource.ESTIMATED}
        else CalibrationStatus.UNKNOWN
    )
    depth_space = make_depth_space(frame_reference, result.depth, space_id=depth_space_id)
    intrinsics = CameraIntrinsics(
        matrix=result.intrinsics,
        pixel_space_id=depth_space.space_id,
        source=intrinsics_source,
        calibration_status=calibration_status,
    )
    provenance = model_provenance or ModelProvenance(
        model_id=result.model_id or "unknown-depth-model",
        display_name=result.model_name or result.model_id or "Unknown depth model",
        task="depth",
        device="cuda" if result.vram_total_gib is not None else "",
        precision=result.precision,
    )
    return DepthAlignmentInput(
        frame_reference=frame_reference,
        depth_space=depth_space,
        depth=result.depth,
        valid_mask=result.valid_mask,
        confidence=result.confidence,
        confidence_kind=confidence_kind,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        depth_model_provenance=provenance,
        point_stride=result.point_stride,
    )
