"""Depth colorization (spec §11).

Colormapping must never mutate the original depth array — every
function here reads ``depth`` and returns a brand-new RGB array,
leaving the caller's depth data exactly as it was for any downstream
numerical use (projection, export, etc.).

Built on VisPy's own colormap LUTs (turbo/inferno/magma/viridis/grays
are all built in) rather than adding matplotlib as a dependency just
for this.
"""

from __future__ import annotations

import numpy as np
from vispy.color import get_colormap

# UI display name -> VisPy colormap name.
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
    """A cached 256x3 uint8 lookup table for one colormap."""
    if vispy_colormap_name not in _lut_cache:
        colormap = get_colormap(vispy_colormap_name)
        positions = np.linspace(0.0, 1.0, _LUT_SIZE)
        colors = colormap.map(positions)  # Nx4 float RGBA in [0, 1]
        _lut_cache[vispy_colormap_name] = (colors[:, :3] * 255).astype(np.uint8)
    return _lut_cache[vispy_colormap_name]


def apply_colormap(
    depth: np.ndarray,
    colormap_name: str = "Turbo",
    near: float = 0.0,
    far: float = 1.0,
    invert: bool = False,
) -> np.ndarray:
    """Map a depth array to an RGB uint8 image via ``colormap_name``.

    ``depth`` is never modified — a new array is always returned.
    Values are normalized against [near, far] (clipped to that range)
    before being mapped, so the near/far sliders control contrast
    without needing to touch the underlying depth data.
    """
    vispy_name = COLORMAP_NAMES.get(colormap_name, "turbo")
    lut = _lookup_table(vispy_name)

    span = far - near
    if span <= 1e-9:
        normalized = np.zeros_like(depth, dtype=np.float32)
    else:
        normalized = (depth.astype(np.float32) - near) / span
    normalized = np.clip(normalized, 0.0, 1.0)

    if invert:
        normalized = 1.0 - normalized

    indices = (normalized * (_LUT_SIZE - 1)).astype(np.int32)
    return lut[indices]  # fancy-indexing returns a fresh array


def colormap_choices() -> list[str]:
    """UI-facing colormap names, in display order."""
    return list(COLORMAP_NAMES.keys())