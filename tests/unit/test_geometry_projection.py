"""Virtual-camera projection tests used by depth-only adapters and the mock."""

from __future__ import annotations

import numpy as np
import pytest

from core.geometry.projection import (
    depth_to_points,
    intrinsics_from_virtual_fov,
    passthrough_points,
)


def test_intrinsics_from_90_degree_fov_are_predictable() -> None:
    intrinsics = intrinsics_from_virtual_fov(width=4, height=2, fov_degrees=90.0)

    np.testing.assert_allclose(
        intrinsics,
        np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        atol=1e-6,
    )


def test_depth_to_points_uses_supplied_intrinsics() -> None:
    depth = np.full((2, 3), 2.0, dtype=np.float32)
    intrinsics = np.array(
        [[2.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    points = depth_to_points(depth, intrinsics)

    assert points.shape == (2, 3, 3)
    np.testing.assert_allclose(points[0, 1], [0.0, -0.5, 2.0])
    np.testing.assert_allclose(points[1, 2], [1.0, 0.5, 2.0])
    np.testing.assert_allclose(points[..., 2], depth)


def test_projection_does_not_mutate_depth() -> None:
    depth = np.array([[1.0, 2.0]], dtype=np.float32)
    original = depth.copy()
    _ = depth_to_points(depth, virtual_fov_degrees=60.0)
    np.testing.assert_array_equal(depth, original)


def test_passthrough_keeps_direct_model_points() -> None:
    points = np.zeros((2, 2, 3), dtype=np.float32)
    assert passthrough_points(points) is points


def test_zero_focal_length_produces_nonfinite_coordinates_not_silent_shape_change() -> None:
    depth = np.ones((1, 1), dtype=np.float32)
    intrinsics = np.zeros((3, 3), dtype=np.float32)
    intrinsics[2, 2] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        points = depth_to_points(depth, intrinsics)
    assert points.shape == (1, 1, 3)
    assert not np.isfinite(points[..., :2]).all()
