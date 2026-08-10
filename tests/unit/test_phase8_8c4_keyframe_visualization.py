from __future__ import annotations

import numpy as np

from app.reconstruction_visualization import (
    keyframe_centers_in_viewer,
    keyframe_display_indices,
    keyframe_frustum_segments_in_viewer,
)


def _poses(count: int) -> np.ndarray:
    values = []
    for index in range(count):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [index * 0.1, index * 0.02, 0.4]
        values.append(pose)
    return np.asarray(values, dtype=np.float64)


def test_c4_keyframe_display_decimation_is_bounded_and_preserves_endpoints() -> None:
    indices = keyframe_display_indices(300, max_keyframes=96)
    assert indices.shape == (96,)
    assert indices[0] == 0
    assert indices[-1] == 299
    assert np.all(np.diff(indices) > 0)


def test_c4_keyframe_centres_follow_reconstruction_to_viewer_convention() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    centers = keyframe_centers_in_viewer(pose[None, ...])
    assert centers.shape == (1, 3)
    assert np.allclose(centers[0], [1.0, 3.0, -2.0])


def test_c4_keyframe_frustums_are_scene_space_segments() -> None:
    segments = keyframe_frustum_segments_in_viewer(_poses(4), depth=0.08)
    assert segments.shape == (4 * 16, 3)
    assert np.isfinite(segments).all()
