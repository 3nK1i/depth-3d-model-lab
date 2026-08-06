"""Screen-space projection helpers for compact 3D orientation overlays.

The inspection viewport uses these functions to draw an XYZ triad in a fixed
corner while still reflecting the active turntable-camera orientation.  This
module is deliberately independent from Qt and VisPy so the projection math can
be validated in ordinary unit tests.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

_AXIS_VECTORS: Mapping[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}


def project_turntable_axes(
    azimuth_deg: float,
    elevation_deg: float,
    roll_deg: float = 0.0,
) -> dict[str, tuple[float, float, float]]:
    """Project world XYZ basis vectors into normalized screen coordinates.

    Returns ``axis -> (screen_x, screen_y, camera_depth)``.  ``screen_y`` uses
    Qt's convention where positive values point downward.  The depth component
    is useful for deterministic back-to-front drawing of the three arrows.

    The calculation mirrors a conventional orbit/turntable camera and is kept
    bounded and finite even near the poles.
    """

    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    roll = math.radians(float(roll_deg))

    # Camera position relative to the look-at centre.  The exact absolute
    # distance is irrelevant for an orientation-only overlay.
    eye = np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )
    forward = _normalise(-eye, fallback=np.array([0.0, -1.0, 0.0]))

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = _normalise(right, fallback=np.array([1.0, 0.0, 0.0]))
    up = _normalise(np.cross(right, forward), fallback=world_up)

    # Apply camera roll around the forward axis to the screen basis.
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    rolled_right = (right * cos_roll) + (up * sin_roll)
    rolled_up = (-right * sin_roll) + (up * cos_roll)

    projected: dict[str, tuple[float, float, float]] = {}
    for name, axis in _AXIS_VECTORS.items():
        screen_x = float(np.dot(axis, rolled_right))
        screen_y = float(-np.dot(axis, rolled_up))
        depth = float(np.dot(axis, forward))
        projected[name] = (
            _finite_clamped(screen_x),
            _finite_clamped(screen_y),
            _finite_clamped(depth),
        )
    return projected


def _normalise(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(vector / norm, dtype=np.float64)


def _finite_clamped(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, float(value)))
