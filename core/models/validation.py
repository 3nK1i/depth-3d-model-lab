"""Validation and normalization for adapter results and metadata."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from core.geometry.projection import intrinsics_from_virtual_fov
from core.models.metadata import IntrinsicsSource, ModelDescriptor
from core.models.result import GeometryResult
from core.runtime.messages import FrameRequest


class ModelOutputContractError(RuntimeError):
    """Raised when an adapter violates the normalized geometry contract."""


def normalize_geometry_result(
    result: GeometryResult,
    descriptor: ModelDescriptor,
    frame: FrameRequest,
) -> GeometryResult:
    """Validate adapter output and apply registry-authoritative semantics."""

    if not isinstance(result, GeometryResult):
        raise ModelOutputContractError(
            f"Adapter {descriptor.model_id!r} returned {type(result).__name__}, "
            "expected GeometryResult."
        )
    if result.session_id != frame.session_id or result.frame_id != frame.frame_id:
        raise ModelOutputContractError(
            f"Adapter {descriptor.model_id!r} returned stale identifiers "
            f"({result.session_id}, {result.frame_id}) for request "
            f"({frame.session_id}, {frame.frame_id})."
        )
    if result.model_id and result.model_id != descriptor.model_id:
        raise ModelOutputContractError(
            f"Adapter registered as {descriptor.model_id!r} returned model id "
            f"{result.model_id!r}."
        )

    _validate_rgb(result.rgb)
    _validate_depth(result.depth)
    _validate_optional_outputs(result, descriptor)

    capabilities = descriptor.capabilities
    intrinsics, intrinsics_source = _normalized_intrinsics(result, descriptor)
    return replace(
        result,
        model_id=descriptor.model_id,
        model_name=result.model_name or descriptor.display_name,
        is_metric=capabilities.is_metric,
        depth_scale=capabilities.depth_scale.value,
        intrinsics=intrinsics,
        intrinsics_source=intrinsics_source.value,
        confidence_kind=(
            capabilities.confidence_kind.value
            if result.confidence is not None
            else "none"
        ),
    )


def _validate_rgb(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ModelOutputContractError("GeometryResult.rgb must be an H×W×3 NumPy array.")
    if rgb.dtype != np.uint8:
        raise ModelOutputContractError("GeometryResult.rgb must use uint8 BGR pixels.")


def _validate_depth(depth: np.ndarray | None) -> None:
    if depth is None:
        raise ModelOutputContractError("Every registered model must return a depth map.")
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        raise ModelOutputContractError("GeometryResult.depth must be a 2D NumPy array.")
    if not np.issubdtype(depth.dtype, np.number):
        raise ModelOutputContractError("GeometryResult.depth must use a numeric dtype.")


def _validate_optional_outputs(result: GeometryResult, descriptor: ModelDescriptor) -> None:
    depth = result.depth
    assert depth is not None
    outputs = descriptor.capabilities.outputs

    optional: tuple[tuple[str, Any, bool], ...] = (
        ("points", result.points, outputs.points),
        ("colors", result.colors, outputs.colors),
        ("normals", result.normals, outputs.normals),
        ("intrinsics", result.intrinsics, outputs.intrinsics),
        ("confidence", result.confidence, outputs.confidence),
        ("valid_mask", result.valid_mask, outputs.valid_mask),
    )
    for name, value, declared in optional:
        if value is not None and not declared:
            raise ModelOutputContractError(
                f"Adapter {descriptor.model_id!r} returned undeclared output {name!r}."
            )

    if result.valid_mask is not None:
        if result.valid_mask.shape != depth.shape:
            raise ModelOutputContractError("valid_mask must have the same H×W shape as depth.")
        if result.valid_mask.dtype != np.bool_:
            raise ModelOutputContractError("valid_mask must use a boolean dtype.")
    if result.confidence is not None:
        if result.confidence.shape != depth.shape:
            raise ModelOutputContractError("confidence must have the same H×W shape as depth.")
        if not np.issubdtype(result.confidence.dtype, np.number):
            raise ModelOutputContractError("confidence must use a numeric dtype.")
    if result.intrinsics is not None:
        _validate_intrinsics(result.intrinsics)
    if int(result.point_stride) < 1:
        raise ModelOutputContractError("point_stride must be at least 1.")
    if result.points is not None:
        _validate_vector_map("points", result.points)
        expected_shape = depth[:: int(result.point_stride), :: int(result.point_stride)].shape
        if result.points.shape[:2] != expected_shape:
            raise ModelOutputContractError(
                "points must match depth sampled by point_stride: "
                f"expected {expected_shape}, got {result.points.shape[:2]}."
            )
    if result.normals is not None:
        _validate_vector_map("normals", result.normals)
        if result.points is not None and result.normals.shape != result.points.shape:
            raise ModelOutputContractError("normals and points must use the same sampled shape.")
    if result.colors is not None:
        _validate_vector_map("colors", result.colors)
        if result.colors.dtype != np.uint8:
            raise ModelOutputContractError("colors must use uint8 RGB values.")
        if result.points is not None and result.colors.shape != result.points.shape:
            raise ModelOutputContractError("colors and points must use the same sampled shape.")


def _validate_vector_map(name: str, value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[2] != 3:
        raise ModelOutputContractError(f"{name} must be an H×W×3 NumPy array.")


def _validate_intrinsics(value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray) or value.shape != (3, 3):
        raise ModelOutputContractError("intrinsics must be a 3×3 matrix.")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ModelOutputContractError("intrinsics must contain finite numeric values.")
    if float(value[0, 0]) <= 0.0 or float(value[1, 1]) <= 0.0:
        raise ModelOutputContractError("intrinsics focal lengths fx and fy must be positive.")


def _normalized_intrinsics(
    result: GeometryResult,
    descriptor: ModelDescriptor,
) -> tuple[np.ndarray | None, IntrinsicsSource]:
    if result.intrinsics is not None:
        return result.intrinsics, descriptor.capabilities.intrinsics_source

    capabilities = descriptor.capabilities
    source = capabilities.intrinsics_source
    fallback = capabilities.intrinsics_fallback
    should_estimate = (
        source is IntrinsicsSource.ESTIMATED
        or fallback is IntrinsicsSource.ESTIMATED
    )
    if capabilities.outputs.intrinsics and should_estimate:
        assert result.depth is not None
        height, width = result.depth.shape
        return intrinsics_from_virtual_fov(width, height), IntrinsicsSource.ESTIMATED
    return None, IntrinsicsSource.NONE

