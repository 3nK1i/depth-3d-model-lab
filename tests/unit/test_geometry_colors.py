"""Depth colormap correctness and invalid-pixel safety tests."""

from __future__ import annotations

import numpy as np
import pytest

from core.geometry.colors import apply_colormap, colormap_choices


def test_supported_colormap_names_are_stable() -> None:
    assert colormap_choices() == ["Turbo", "Inferno", "Magma", "Viridis", "Greyscale"]


def test_invert_reverses_depth_order() -> None:
    depth = np.array([[0.0, 1.0]], dtype=np.float32)
    normal = apply_colormap(depth, near=0.0, far=1.0, invert=False)
    inverted = apply_colormap(depth, near=0.0, far=1.0, invert=True)

    np.testing.assert_array_equal(normal[0, 0], inverted[0, 1])
    np.testing.assert_array_equal(normal[0, 1], inverted[0, 0])


def test_degenerate_range_remains_finite_uint8() -> None:
    rgb = apply_colormap(np.ones((2, 2), dtype=np.float32), near=1.0, far=1.0)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (2, 2, 3)


def test_mask_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        apply_colormap(
            np.ones((2, 2), dtype=np.float32),
            valid_mask=np.ones((3, 3), dtype=bool),
        )


def test_unknown_colormap_falls_back_without_mutating_depth() -> None:
    depth = np.array([[0.25, 0.75]], dtype=np.float32)
    original = depth.copy()
    rgb = apply_colormap(depth, colormap_name="Unknown", near=0.0, far=1.0)
    assert rgb.shape == (1, 2, 3)
    np.testing.assert_array_equal(depth, original)
