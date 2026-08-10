from __future__ import annotations

import numpy as np
import pytest

from app.reconstruction_visualization import (
    sparse_landmark_display_indices,
    sparse_landmarks_to_viewer,
    world_axis_arrowhead_segments,
)
from core.reconstruction.live_contracts import LiveMapGeometrySnapshot, LiveMapSnapshot


def test_c3_sparse_map_snapshot_extends_counts_without_rewriting_base_contract() -> None:
    base = LiveMapSnapshot(3, 4, 20)
    assert base.as_dict() == {"revision": 3, "keyframe_count": 4, "landmark_count": 20}

    snapshot = LiveMapGeometrySnapshot(
        revision=3,
        keyframe_count=4,
        landmark_count=20,
        landmarks_reconstruction=((1.0, 2.0, 3.0), (-1.0, 0.5, 4.0)),
    )
    assert isinstance(snapshot, LiveMapSnapshot)
    assert len(snapshot.landmarks_reconstruction) == 2
    assert snapshot.as_dict()["landmarks_reconstruction"][0] == [1.0, 2.0, 3.0]


def test_c3_sparse_map_snapshot_rejects_invalid_or_overclaimed_samples() -> None:
    with pytest.raises(ValueError, match="finite 3-vector"):
        LiveMapGeometrySnapshot(1, 1, 2, ((1.0, 2.0, float("nan")),))
    with pytest.raises(ValueError, match="cannot exceed"):
        LiveMapGeometrySnapshot(1, 1, 1, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))


def test_c3_sparse_landmarks_use_same_z_up_presentation_transform() -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
    viewer = sparse_landmarks_to_viewer(points)
    assert viewer.dtype == np.float32
    assert np.allclose(viewer[0], [1.0, 3.0, -2.0])
    assert np.allclose(viewer[1], [-1.0, -3.0, 2.0])


def test_c3_sparse_landmark_display_decimation_is_bounded_and_deterministic() -> None:
    small = sparse_landmark_display_indices(5, max_points=8)
    assert np.array_equal(small, np.arange(5))
    large = sparse_landmark_display_indices(50_000, max_points=8192)
    assert large.size == 8192
    assert int(large[0]) == 0
    assert int(large[-1]) == 49_999
    assert np.all(np.diff(large) > 0)


def test_c3_world_axes_have_true_scene_space_arrowheads() -> None:
    for axis in range(3):
        segments = world_axis_arrowhead_segments(axis, 0.2)
        assert segments.shape == (8, 3)
        assert segments.dtype == np.float32
        tips = segments[0::2]
        expected = np.zeros(3, dtype=np.float32)
        expected[axis] = 0.2
        assert np.allclose(tips, expected)
        assert np.isfinite(segments).all()


def test_c3_visual_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="max_points"):
        sparse_landmark_display_indices(5, max_points=0)
    with pytest.raises(ValueError, match="finite"):
        sparse_landmarks_to_viewer(np.asarray([[0.0, np.nan, 1.0]]))
    with pytest.raises(ValueError, match="axis_index"):
        world_axis_arrowhead_segments(3, 0.2)
