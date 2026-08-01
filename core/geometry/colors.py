"""Depth-map colorization helpers.

Invalid/occluded pixels are rendered black and never participate in integer LUT
index conversion. The numerical depth array is never modified.
"""

from __future__ import annotations

import numpy as np
from vispy.color import get_colormap

COLORMAP_NAMES: dict[str, str] = {
    "Turbo": "turbo",
    "Inferno": "inferno",
    "Magma": "magma",
    "Viridis": "viridis",
    "Greyscale": "grays",
}

_LUT_SIZE = 256
_lut_cache: dict[str, np.ndarray] = {}


def _lookup_table(vispy_colormap_name: str) -> np.ndarray:
    if vispy_colormap_name not in _lut_cache:
        colormap = get_colormap(vispy_colormap_name)
        positions = np.linspace(0.0, 1.0, _LUT_SIZE)
        colors = colormap.map(positions)
        _lut_cache[vispy_colormap_name] = (colors[:, :3] * 255).astype(np.uint8)
    return _lut_cache[vispy_colormap_name]


def apply_colormap(
    depth: np.ndarray,
    colormap_name: str = "Turbo",
    near: float = 0.0,
    far: float = 1.0,
    invert: bool = False,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map a depth array to a fresh RGB uint8 image.

    ``valid_mask`` is optional; non-finite depth is invalid regardless of it.
    Invalid pixels are black, which is safer for real model masks than allowing
    NaN/Inf values to become arbitrary LUT indices.
    """
    depth_array = np.asarray(depth)
    finite = np.isfinite(depth_array)
    valid = finite.copy()
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != depth_array.shape:
            raise ValueError("valid_mask must have the same shape as depth")
        valid &= supplied

    safe_depth = np.where(finite, depth_array, near).astype(np.float32, copy=False)
    span = far - near
    if span <= 1e-9:
        normalized = np.zeros(depth_array.shape, dtype=np.float32)
    else:
        normalized = (safe_depth - near) / span
    normalized = np.clip(normalized, 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized

    lut = _lookup_table(COLORMAP_NAMES.get(colormap_name, "turbo"))
    indices = (normalized * (_LUT_SIZE - 1)).astype(np.int32)
    rgb = lut[indices].copy()
    rgb[~valid] = 0
    return rgb


def colormap_choices() -> list[str]:
    return list(COLORMAP_NAMES.keys())
