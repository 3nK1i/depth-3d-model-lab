"""Authoritative semantic validation helpers for Step 3 alignment."""

from __future__ import annotations

import numpy as np

from core.alignment.contracts import AlignmentRequest, AlignedFrameBundle
from core.contracts.validation import ContractValidationError, readonly_array
from core.models.metadata import ConfidenceKind


def compose_depth_valid_mask(
    depth: np.ndarray,
    model_valid_mask: np.ndarray | None,
) -> np.ndarray:
    values = np.asarray(depth)
    valid = np.isfinite(values) & (values > 0.0)
    if model_valid_mask is not None:
        mask = np.asarray(model_valid_mask, dtype=np.bool_)
        if mask.shape != values.shape:
            raise ContractValidationError("model_valid_mask shape does not match depth.")
        valid &= mask
    return readonly_array(valid, field_name="depth_valid_mask", dtype=np.bool_, ndim=2)


def compose_confidence_valid_mask(
    confidence: np.ndarray | None,
    confidence_kind: ConfidenceKind,
    shape: tuple[int, int],
) -> np.ndarray:
    if confidence is None:
        if confidence_kind is not ConfidenceKind.NONE:
            raise ContractValidationError(
                "confidence is required when confidence_kind is not NONE."
            )
        return readonly_array(
            np.zeros(shape, dtype=np.bool_),
            field_name="confidence_valid_mask",
            dtype=np.bool_,
            shape=shape,
        )
    values = np.asarray(confidence)
    if values.shape != shape:
        raise ContractValidationError("confidence shape does not match depth.")
    valid = np.isfinite(values)
    if confidence_kind is ConfidenceKind.CONFIDENCE:
        valid &= (values >= 0.0) & (values <= 1.0)
    elif confidence_kind is ConfidenceKind.UNCERTAINTY:
        valid &= values >= 0.0
    else:
        raise ContractValidationError(
            "confidence array is incompatible with ConfidenceKind.NONE."
        )
    return readonly_array(valid, field_name="confidence_valid_mask", dtype=np.bool_, ndim=2)


def validate_request_identity(request: AlignmentRequest) -> None:
    reference = request.frame_reference
    if request.rgb_space.frame_reference != reference:
        raise ContractValidationError("RGB frame identity mismatch.")
    if request.depth_input.frame_reference != reference:
        raise ContractValidationError("Depth frame identity mismatch.")
    if request.perception_result.frame_reference != reference:
        raise ContractValidationError("Perception frame identity mismatch.")


def validate_aligned_bundle(bundle: AlignedFrameBundle) -> None:
    shape = (
        bundle.target_depth_space.size.height,
        bundle.target_depth_space.size.width,
    )
    if bundle.depth.shape != shape:
        raise ContractValidationError("Aligned bundle depth shape mismatch.")
    if bundle.aligned_rgb.shape[:2] != shape:
        raise ContractValidationError("Aligned bundle RGB shape mismatch.")
    if bundle.depth_valid_mask.shape != shape:
        raise ContractValidationError("Aligned bundle validity shape mismatch.")
