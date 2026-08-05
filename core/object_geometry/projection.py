"""Direct selected-pixel back-projection without dense viewer geometry."""

from __future__ import annotations

import numpy as np

from core.contracts.spatial import CameraIntrinsics
from core.contracts.validation import ContractValidationError


def project_selected_pixels(
    depth: np.ndarray,
    pixel_indices_xy: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    depth_array = np.asarray(depth)
    indices = np.asarray(pixel_indices_xy, dtype=np.int64)
    if depth_array.ndim != 2:
        raise ContractValidationError("depth must be HxW.")
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ContractValidationError("pixel_indices_xy must have shape Nx2.")
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    x = indices[:, 0]
    y = indices[:, 1]
    if np.any(x < 0) or np.any(x >= depth_array.shape[1]) or np.any(y < 0) or np.any(y >= depth_array.shape[0]):
        raise ContractValidationError("pixel_indices_xy contain out-of-bounds coordinates.")
    z = depth_array[y, x].astype(np.float64, copy=False)
    if not np.isfinite(z).all() or np.any(z <= 0.0):
        raise ContractValidationError("Selected depth values must be finite and positive.")
    matrix = intrinsics.matrix
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    points_x = (x.astype(np.float64) - cx) * z / fx
    points_y = (y.astype(np.float64) - cy) * z / fy
    return np.column_stack((points_x, points_y, z))


def pixel_indices_from_mask(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(np.asarray(mask, dtype=np.bool_))
    return np.column_stack((x, y)).astype(np.int64, copy=False)
