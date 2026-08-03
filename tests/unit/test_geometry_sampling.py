"""Regular-grid and hard point-budget sampling tests."""

from __future__ import annotations

import numpy as np

from core.geometry.sampling import adaptive_stride, decimate_grid, enforce_point_budget


def _grid(height: int = 4, width: int = 6):
    points = np.arange(height * width * 3, dtype=np.float32).reshape(height, width, 3)
    colors = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    return points, colors


def test_decimate_grid_keeps_aligned_regular_samples() -> None:
    points, colors = _grid()
    sampled_points, sampled_colors = decimate_grid(points, colors, stride=2)

    np.testing.assert_array_equal(sampled_points, points[::2, ::2])
    np.testing.assert_array_equal(sampled_colors, colors[::2, ::2])
    assert sampled_points.shape == (2, 3, 3)


def test_decimate_grid_clamps_stride_to_one() -> None:
    points, colors = _grid(2, 2)
    sampled_points, sampled_colors = decimate_grid(points, colors, stride=0)
    np.testing.assert_array_equal(sampled_points, points)
    np.testing.assert_array_equal(sampled_colors, colors)


def test_enforce_budget_is_reproducible_and_keeps_alignment() -> None:
    points = np.arange(300, dtype=np.float32).reshape(100, 3)
    colors = np.arange(300, dtype=np.uint8).reshape(100, 3)

    a_points, a_colors = enforce_point_budget(points, colors, 20, seed=7)
    b_points, b_colors = enforce_point_budget(points, colors, 20, seed=7)

    np.testing.assert_array_equal(a_points, b_points)
    np.testing.assert_array_equal(a_colors, b_colors)
    for point, color in zip(a_points, a_colors, strict=True):
        index = np.flatnonzero(np.all(points == point, axis=1))[0]
        np.testing.assert_array_equal(color, colors[index])


def test_budget_larger_than_cloud_returns_original_arrays() -> None:
    points = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)
    out_points, out_colors = enforce_point_budget(points, colors, 10)
    assert out_points is points
    assert out_colors is colors


def test_adaptive_stride_only_coarsens_during_interaction() -> None:
    assert adaptive_stride(2, False) == 2
    assert adaptive_stride(2, True, interaction_factor=3) == 6
    assert adaptive_stride(0, False) == 1
