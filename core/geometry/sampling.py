"""Point-cloud sampling: stride decimation and point-budget enforcement.

Two independent controls, matching the sidebar's Point Size/Density and
the GPU-profile point budgets from the original spec:

- ``decimate_grid``: regular stride sampling on the still-dense (H, W)
  grid, applied BEFORE flattening — cheaper than random sampling after
  the fact, and gives an even spatial distribution rather than a random
  hole pattern.
- ``enforce_point_budget``: a hard cap on the already-flat point count,
  for when even a decimated grid still exceeds what the GPU profile
  allows.
"""

from __future__ import annotations

import numpy as np


def decimate_grid(points: np.ndarray, colors: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Take every ``stride``-th row/column of a dense (H, W, 3) grid.

    ``stride=1`` returns the grid unchanged (still dense — caller
    flattens separately); ``stride=2`` keeps a quarter of the points,
    etc. Never mutates the inputs.
    """
    stride = max(1, int(stride))
    return points[::stride, ::stride], colors[::stride, ::stride]


def enforce_point_budget(
    points: np.ndarray, colors: np.ndarray, budget: int, *, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subsample flat (N, 3) arrays down to at most ``budget``
    points. A fixed ``seed`` makes the subsample reproducible (useful
    for tests); the default (None) varies frame to frame, which is
    preferable in the live viewer so a static hole pattern doesn't
    appear to "stick" to the geometry.
    """
    count = points.shape[0]
    if count <= budget:
        return points, colors

    rng = np.random.default_rng(seed)
    indices = rng.choice(count, size=budget, replace=False)
    return points[indices], colors[indices]


def adaptive_stride(base_stride: int, is_interacting: bool, interaction_factor: int = 3) -> int:
    """Coarsen the stride while the user is actively orbiting/panning,
    restoring the base stride once interaction stops (spec §10:
    "reduce point count while the user is actively rotating")."""
    if is_interacting:
        return max(1, base_stride * interaction_factor)
    return max(1, base_stride)