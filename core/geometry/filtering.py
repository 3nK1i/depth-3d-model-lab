"""Point-cloud validity and depth-edge filtering."""

from __future__ import annotations

import numpy as np


def valid_mask(points: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """True where point coordinates and positive depth are finite."""
    return (
        np.isfinite(points).all(axis=-1)
        & np.isfinite(depth)
        & (depth > 0)
    )


def remove_invalid(
    points: np.ndarray,
    colors: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = valid_mask(points, depth)
    return points[mask], colors[mask]


def edge_filter_mask(depth: np.ndarray, strength: float) -> np.ndarray:
    """Suppress sharp depth discontinuities without propagating NaN/Inf."""
    depth_array = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth_array) & (depth_array > 0)
    if not finite.any():
        return np.zeros(depth_array.shape, dtype=bool)
    if strength <= 0:
        return finite

    fill_value = float(np.median(depth_array[finite]))
    safe_depth = np.where(finite, depth_array, fill_value)
    grad_y, grad_x = np.gradient(safe_depth)
    gradient_magnitude = np.hypot(grad_x, grad_y)

    finite_values = depth_array[finite]
    depth_scale = max(float(finite_values.max() - finite_values.min()), 1e-6)
    normalized_gradient = gradient_magnitude / depth_scale

    strength = min(max(float(strength), 0.0), 1.0)
    threshold = 0.5 * (1.0 - strength) + 0.05 * strength
    return finite & np.isfinite(normalized_gradient) & (normalized_gradient <= threshold)
