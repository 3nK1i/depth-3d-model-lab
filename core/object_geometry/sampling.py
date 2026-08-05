"""Deterministic object-point budget enforcement."""

from __future__ import annotations

import numpy as np


def deterministic_spatial_sample_indices(points_xyz: np.ndarray, maximum_points: int) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    count = len(points)
    if count <= int(maximum_points):
        return np.arange(count, dtype=np.int64)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    positions = np.linspace(0, count - 1, num=int(maximum_points), dtype=np.int64)
    return order[positions]
