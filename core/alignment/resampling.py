"""Semantic-specific NumPy/OpenCV resampling on declared pixel spaces."""

from __future__ import annotations

import cv2
import numpy as np

from core.alignment.contracts import ResamplingMethod
from core.alignment.transforms import is_full_frame_resize
from core.contracts.spatial import ImageTransform2D
from core.contracts.validation import ContractValidationError, readonly_array


def _validate_source_shape(array: np.ndarray, transform: ImageTransform2D) -> None:
    expected = (
        transform.source_space.size.height,
        transform.source_space.size.width,
    )
    if array.shape[:2] != expected:
        raise ContractValidationError(
            f"Source array shape {array.shape[:2]} does not match pixel space {expected}."
        )


def _cv_interpolation(method: ResamplingMethod, *, reducing: bool = False) -> int:
    if method is ResamplingMethod.NEAREST:
        return cv2.INTER_NEAREST
    if method is ResamplingMethod.AREA:
        return cv2.INTER_AREA
    if method is ResamplingMethod.LINEAR:
        return cv2.INTER_LINEAR
    return cv2.INTER_AREA if reducing else cv2.INTER_LINEAR


def resample_rgb(
    image: np.ndarray,
    transform: ImageTransform2D,
    *,
    method: ResamplingMethod = ResamplingMethod.AUTO,
) -> np.ndarray:
    source = np.asarray(image)
    _validate_source_shape(source, transform)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ContractValidationError("RGB source must have shape HxWx3.")
    destination_size = (
        transform.destination_space.size.width,
        transform.destination_space.size.height,
    )
    reducing = (
        transform.destination_space.size.width * transform.destination_space.size.height
        < transform.source_space.size.width * transform.source_space.size.height
    )
    interpolation = _cv_interpolation(method, reducing=reducing)
    if is_full_frame_resize(transform):
        output = cv2.resize(source, destination_size, interpolation=interpolation)
    else:
        # INTER_AREA is not reliable for arbitrary perspective/crop warps.
        if interpolation == cv2.INTER_AREA:
            interpolation = cv2.INTER_LINEAR
        output = cv2.warpPerspective(
            source,
            transform.matrix,
            destination_size,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    return readonly_array(output, field_name="aligned_rgb", dtype=np.uint8, ndim=3)


def resample_boolean_mask(mask: np.ndarray, transform: ImageTransform2D) -> np.ndarray:
    source = np.asarray(mask, dtype=np.bool_)
    _validate_source_shape(source, transform)
    if source.ndim != 2:
        raise ContractValidationError("Boolean mask must have shape HxW.")
    destination_size = (
        transform.destination_space.size.width,
        transform.destination_space.size.height,
    )
    output = cv2.warpPerspective(
        source.astype(np.uint8),
        transform.matrix,
        destination_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.bool_)
    return readonly_array(output, field_name="aligned_mask", dtype=np.bool_, ndim=2)


def resample_continuous_map(
    values: np.ndarray,
    transform: ImageTransform2D,
    *,
    method: ResamplingMethod = ResamplingMethod.LINEAR,
    border_value: float = float("nan"),
) -> np.ndarray:
    source = np.asarray(values)
    _validate_source_shape(source, transform)
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.number):
        raise ContractValidationError("Continuous map must be a numeric HxW array.")
    destination_size = (
        transform.destination_space.size.width,
        transform.destination_space.size.height,
    )
    output = cv2.warpPerspective(
        source.astype(np.float32),
        transform.matrix,
        destination_size,
        flags=_cv_interpolation(method),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )
    return readonly_array(output, field_name="aligned_continuous_map", ndim=2)


def resample_support_mask(transform: ImageTransform2D) -> np.ndarray:
    """Return the destination region supported by declared crop/resize/padding.

    This uses the descriptive operations already validated against the
    authoritative matrix, avoiding pixel-centre edge ambiguity at padding
    boundaries.
    """

    height = transform.destination_space.size.height
    width = transform.destination_space.size.width
    support = np.zeros((height, width), dtype=np.bool_)
    left, top, _, _ = transform.padding_ltrb
    resized = transform.resized_size
    assert resized is not None
    support[top : top + resized.height, left : left + resized.width] = True
    return readonly_array(
        support, field_name="rgb_support_mask", dtype=np.bool_, ndim=2
    )
