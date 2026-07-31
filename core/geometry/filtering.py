"""Point-cloud filtering: invalid geometry removal and edge filtering.

Applied after projection, before sampling — invalid points (NaN/Inf,
non-positive depth) must never reach the renderer or an export file.
"""

from __future__ import annotations

import numpy as np


def valid_mask(points: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Boolean (H, W) mask: True where the point is finite and in front
    of the camera (depth > 0)."""
    finite = np.isfinite(points).all(axis=-1)
    positive_depth = depth > 0
    return finite & positive_depth


def remove_invalid(
    points: np.ndarray, colors: np.ndarray, depth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to (N, 3)/(N, 3) and drop every invalid point.

    Takes the dense (H, W, 3) grids from projection/colorization and
    returns flat arrays ready for :mod:`core.geometry.sampling` — this
    is the one place the grid becomes a flat point list.
    """
    mask = valid_mask(points, depth)
    flat_points = points[mask]
    flat_colors = colors[mask]
    return flat_points, flat_colors


def edge_filter_mask(depth: np.ndarray, strength: float) -> np.ndarray:
    """Boolean (H, W) mask suppressing pixels at sharp depth
    discontinuities ("flying pixels" at object edges).

    ``strength`` in [0, 1]: 0 disables filtering entirely (mask is all
    True); higher values reject a wider gradient-magnitude band. Uses a
    simple finite-difference gradient rather than a full edge detector —
    adequate for suppressing the worst flying-pixel artifacts without
    extra dependencies.
    """
    if strength <= 0:
        return np.ones(depth.shape, dtype=bool)

    grad_y, grad_x = np.gradient(depth.astype(np.float32))
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    depth_scale = max(float(np.nanmax(depth)) - float(np.nanmin(depth)), 1e-6)
    normalized_gradient = gradient_magnitude / depth_scale

    # strength=1.0 -> reject anything above ~5% of the depth range's
    # gradient per pixel; strength=0.1 -> a much more permissive 50%.
    threshold = 0.5 * (1.0 - strength) + 0.05 * strength
    return normalized_gradient <= threshold