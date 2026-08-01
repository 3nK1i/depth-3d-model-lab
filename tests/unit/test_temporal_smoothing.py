"""Unit tests for low-cost aligned geometry temporal smoothing."""

from __future__ import annotations

import numpy as np

from core.geometry.temporal import TemporalGeometrySmoother


def _depth(value: float) -> np.ndarray:
    return np.full((2, 3), value, dtype=np.float32)


def test_first_frame_is_returned_unchanged() -> None:
    smoother = TemporalGeometrySmoother()
    depth = _depth(2.0)
    points = np.dstack((depth, depth, depth)).astype(np.float32)

    result = smoother.apply(
        depth=depth,
        points=points,
        normals=None,
        strength=0.7,
        signature=("model", depth.shape, points.shape),
    )

    np.testing.assert_allclose(result.depth, depth)
    np.testing.assert_allclose(result.points, points)
    assert result.normals is None
    assert result.depth is not depth


def test_second_frame_uses_exponential_blend() -> None:
    smoother = TemporalGeometrySmoother()
    signature = ("model", (2, 3))
    smoother.apply(
        depth=_depth(2.0),
        points=None,
        normals=None,
        strength=0.5,
        signature=signature,
    )

    result = smoother.apply(
        depth=_depth(4.0),
        points=None,
        normals=None,
        strength=0.5,
        signature=signature,
    )

    np.testing.assert_allclose(result.depth, 3.0)


def test_invalid_current_values_are_not_filled_from_history() -> None:
    smoother = TemporalGeometrySmoother()
    signature = ("model", (2, 3))
    smoother.apply(
        depth=_depth(2.0),
        points=None,
        normals=None,
        strength=0.8,
        signature=signature,
    )
    current = _depth(4.0)
    current[0, 1] = np.nan
    current[1, 2] = 0.0

    result = smoother.apply(
        depth=current,
        points=None,
        normals=None,
        strength=0.8,
        signature=signature,
    )

    assert np.isnan(result.depth[0, 1])
    assert result.depth[1, 2] == 0.0
    assert np.isclose(result.depth[0, 0], 2.4)


def test_signature_change_resets_history() -> None:
    smoother = TemporalGeometrySmoother()
    smoother.apply(
        depth=_depth(2.0),
        points=None,
        normals=None,
        strength=0.9,
        signature=("source-a",),
    )

    result = smoother.apply(
        depth=_depth(8.0),
        points=None,
        normals=None,
        strength=0.9,
        signature=("source-b",),
    )

    np.testing.assert_allclose(result.depth, 8.0)


def test_smoothed_normals_are_renormalized() -> None:
    smoother = TemporalGeometrySmoother()
    signature = ("normal", (1, 1, 3))
    first = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    second = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)

    smoother.apply(
        depth=np.ones((1, 1), dtype=np.float32),
        points=None,
        normals=first,
        strength=0.5,
        signature=signature,
    )
    result = smoother.apply(
        depth=np.ones((1, 1), dtype=np.float32),
        points=None,
        normals=second,
        strength=0.5,
        signature=signature,
    )

    assert result.normals is not None
    np.testing.assert_allclose(
        np.linalg.norm(result.normals, axis=-1),
        1.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.normals[0, 0],
        np.array([2**-0.5, 2**-0.5, 0.0], dtype=np.float32),
        atol=1e-6,
    )
