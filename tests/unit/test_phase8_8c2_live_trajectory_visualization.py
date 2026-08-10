from __future__ import annotations

import numpy as np
import pytest

from app.reconstruction_visualization import (
    adaptive_world_axis_length,
    orientation_gizmo_projection,
    trajectory_display_indices,
)


def test_c2_orientation_gizmo_projection_is_finite_and_foreshortened() -> None:
    projection = orientation_gizmo_projection(0.0, 0.0)
    assert projection.shape == (3, 2)
    assert projection.dtype == np.float32
    assert np.isfinite(projection).all()
    # At zero azimuth/elevation, viewer X is screen-right and viewer Z is screen-up.
    assert projection[0, 0] > 0.99
    assert abs(float(projection[0, 1])) < 1e-6
    assert projection[2, 1] < -0.99
    # Viewer Y points approximately into the screen and is therefore foreshortened.
    assert float(np.linalg.norm(projection[1])) < 1e-5


def test_c2_orientation_gizmo_projection_changes_with_turntable_view() -> None:
    first = orientation_gizmo_projection(32.0, 24.0)
    second = orientation_gizmo_projection(95.0, 15.0)
    assert not np.allclose(first, second)
    assert np.all(np.linalg.norm(first, axis=1) <= 1.00001)
    assert np.all(np.linalg.norm(second, axis=1) <= 1.00001)


def test_c2_adaptive_world_axes_are_restrained_and_path_scaled() -> None:
    assert adaptive_world_axis_length(None) == pytest.approx(0.28)
    one = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    assert adaptive_world_axis_length(one) == pytest.approx(0.28)

    short = np.asarray([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]], dtype=np.float64)
    assert adaptive_world_axis_length(short) == pytest.approx(0.08)

    medium = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.4, 0.3]], dtype=np.float64)
    length = adaptive_world_axis_length(medium)
    assert 0.08 < length < 0.34

    large = np.asarray([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float64)
    assert adaptive_world_axis_length(large) == pytest.approx(0.34)


def test_c2_trajectory_visual_decimation_preserves_order_and_endpoints() -> None:
    small = trajectory_display_indices(5, max_points=16)
    assert np.array_equal(small, np.arange(5))

    indices = trajectory_display_indices(20_000, max_points=4096)
    assert indices.dtype == np.int64
    assert 2 <= indices.size <= 4096
    assert int(indices[0]) == 0
    assert int(indices[-1]) == 19_999
    assert np.all(np.diff(indices) > 0)


def test_c2_visualization_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        orientation_gizmo_projection(float("nan"), 0.0)
    with pytest.raises(ValueError, match=r"finite \(N, 3\)"):
        adaptive_world_axis_length(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="max_points"):
        trajectory_display_indices(10, max_points=1)
    with pytest.raises(ValueError, match="non-negative"):
        trajectory_display_indices(-1)
