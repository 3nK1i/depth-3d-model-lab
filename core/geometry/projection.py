"""Depth-to-point projection (spec §9, §16 "Virtual FOV").

For models that return points directly, :func:`passthrough_points` is
used as-is. For depth-only models (every adapter so far, including the
Phase 3 mock), :func:`depth_to_points` back-projects each pixel through
a virtual pinhole camera derived from a configurable horizontal FOV —
there is no calibration step in this version (spec: "do not require
camera calibration for the first version").

Returns a dense (H, W, 3) grid, not yet flattened or filtered — that is
:mod:`core.geometry.filtering` and :mod:`core.geometry.sampling`'s job,
kept separate so each stage stays independently testable.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_VIRTUAL_FOV_DEGREES = 60.0


def intrinsics_from_virtual_fov(
    width: int, height: int, fov_degrees: float = DEFAULT_VIRTUAL_FOV_DEGREES
) -> np.ndarray:
    """Build a simple pinhole intrinsics matrix from a horizontal FOV.

    Square pixels are assumed (fx == fy) — reasonable for a virtual
    camera with no real calibration data.
    """
    fov_rad = math.radians(fov_degrees)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def depth_to_points(
    depth: np.ndarray,
    intrinsics: np.ndarray | None = None,
    virtual_fov_degrees: float = DEFAULT_VIRTUAL_FOV_DEGREES,
) -> np.ndarray:
    """Back-project a depth map to a dense (H, W, 3) point grid.

    Uses ``intrinsics`` if a model predicted one; otherwise derives a
    virtual pinhole camera from ``virtual_fov_degrees``. ``depth`` is
    read-only here — the returned grid is a new array.
    """
    height, width = depth.shape[:2]
    if intrinsics is None:
        intrinsics = intrinsics_from_virtual_fov(width, height, virtual_fov_degrees)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(u, v)  # both (H, W)

    z = depth.astype(np.float32)
    x = (grid_u - cx) * z / fx
    y = (grid_v - cy) * z / fy

    return np.stack([x, y, z], axis=-1)  # (H, W, 3)

def passthrough_points(points: np.ndarray) -> np.ndarray:
    """For models that return points directly — kept as a named function
    (rather than callers using the array as-is) so the projection
    module remains the single place both code paths are documented."""
    return points