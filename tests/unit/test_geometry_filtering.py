"""Point/depth validity, edge masking, and aligned removal tests."""

from __future__ import annotations

import numpy as np

from core.geometry.filtering import edge_filter_mask, remove_invalid, valid_mask


def test_remove_invalid_preserves_point_color_alignment() -> None:
    points = np.array(
        [[[0, 0, 1], [1, 0, 2]], [[0, 1, np.nan], [1, 1, 4]]],
        dtype=np.float32,
    )
    colors = np.array(
        [[[10, 11, 12], [20, 21, 22]], [[30, 31, 32], [40, 41, 42]]],
        dtype=np.uint8,
    )
    depth = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)

    clean_points, clean_colors = remove_invalid(points, colors, depth)

    assert clean_points.shape == (2, 3)
    np.testing.assert_array_equal(clean_colors, [[10, 11, 12], [40, 41, 42]])


def test_zero_strength_keeps_all_positive_finite_depth() -> None:
    depth = np.array([[1.0, 2.0], [0.0, np.nan]], dtype=np.float32)
    assert edge_filter_mask(depth, 0.0).tolist() == [[True, True], [False, False]]


def test_stronger_edge_filter_rejects_at_least_as_many_discontinuities() -> None:
    depth = np.ones((8, 8), dtype=np.float32)
    depth[:, 4:] = 10.0

    weak = edge_filter_mask(depth, 0.1)
    strong = edge_filter_mask(depth, 1.0)

    assert strong.sum() <= weak.sum()
    assert not strong[:, 3:5].all()


def test_valid_mask_rejects_nonfinite_coordinates_even_with_valid_depth() -> None:
    points = np.zeros((1, 2, 3), dtype=np.float32)
    points[0, 1, 1] = np.inf
    depth = np.ones((1, 2), dtype=np.float32)
    assert valid_mask(points, depth).tolist() == [[True, False]]
