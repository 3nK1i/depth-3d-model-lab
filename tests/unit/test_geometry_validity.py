"""Finite-value safeguards for real model depth and point outputs."""

from __future__ import annotations

import numpy as np

from core.geometry.colors import apply_colormap
from core.geometry.filtering import edge_filter_mask, valid_mask


def test_colormap_blacks_out_masked_and_nonfinite_pixels() -> None:
    depth = np.array([[1.0, np.inf], [np.nan, 2.0]], dtype=np.float32)
    model_mask = np.array([[True, True], [True, False]])

    rgb = apply_colormap(depth, near=0.1, far=8.0, valid_mask=model_mask)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert np.any(rgb[0, 0] != 0)
    assert np.all(rgb[0, 1] == 0)
    assert np.all(rgb[1, 0] == 0)
    assert np.all(rgb[1, 1] == 0)


def test_edge_filter_handles_all_invalid_depth() -> None:
    depth = np.full((4, 5), np.inf, dtype=np.float32)
    mask = edge_filter_mask(depth, 0.6)
    assert mask.shape == depth.shape
    assert not mask.any()


def test_valid_mask_requires_finite_positive_depth() -> None:
    points = np.zeros((2, 2, 3), dtype=np.float32)
    points[0, 1, 0] = np.nan
    depth = np.array([[1.0, 1.0], [np.inf, -1.0]], dtype=np.float32)
    assert valid_mask(points, depth).tolist() == [[True, False], [False, False]]
