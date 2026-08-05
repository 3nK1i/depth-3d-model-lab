"""Boolean-mask preparation for object point extraction."""

from __future__ import annotations

import cv2
import numpy as np

from core.contracts.validation import ContractValidationError


def largest_connected_component(mask: np.ndarray, *, connectivity: int = 8) -> np.ndarray:
    array = np.asarray(mask, dtype=np.bool_)
    if array.ndim != 2:
        raise ContractValidationError("mask must be HxW.")
    if connectivity not in (4, 8):
        raise ContractValidationError("connectivity must be 4 or 8.")
    if not np.any(array):
        return np.zeros_like(array, dtype=np.bool_)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        array.astype(np.uint8), connectivity=connectivity
    )
    if count <= 1:
        return np.zeros_like(array, dtype=np.bool_)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    return labels == largest_label


def erode_mask_conservatively(
    mask: np.ndarray,
    *,
    pixels: int,
    minimum_pixels: int,
    connectivity: int = 8,
) -> tuple[np.ndarray, bool]:
    """Erode a mask, but preserve the input if erosion destroys useful support."""

    array = np.asarray(mask, dtype=np.bool_)
    if pixels <= 0 or not np.any(array):
        return np.array(array, copy=True), False
    kernel_size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(array.astype(np.uint8), kernel, iterations=1).astype(bool)
    if np.count_nonzero(eroded) < int(minimum_pixels):
        return np.array(array, copy=True), True
    eroded = largest_connected_component(eroded, connectivity=connectivity)
    if np.count_nonzero(eroded) < int(minimum_pixels):
        return np.array(array, copy=True), True
    return eroded, False


def exterior_ring(mask: np.ndarray, *, pixels: int) -> np.ndarray:
    array = np.asarray(mask, dtype=np.bool_)
    if array.ndim != 2:
        raise ContractValidationError("mask must be HxW.")
    if pixels <= 0:
        raise ContractValidationError("pixels must be positive.")
    kernel_size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(array.astype(np.uint8), kernel, iterations=1).astype(bool)
    return dilated & ~array


def mask_centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    y, x = np.nonzero(np.asarray(mask, dtype=np.bool_))
    if x.size == 0:
        return None
    return float(np.mean(x)), float(np.mean(y))
