from __future__ import annotations

import numpy as np
import pytest

from app.reconstruction_visualization import (
    camera_frustum_segments_in_viewer,
    camera_position_in_viewer,
    reconstruction_point_to_viewer,
    reconstruction_points_to_viewer,
)


def _pose(*, translation=(0.0, 0.0, 0.0), yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    value[:3, 3] = np.asarray(translation, dtype=np.float64)
    return value


def test_c1_reconstruction_to_viewer_transform_is_z_up_and_deterministic() -> None:
    point = reconstruction_point_to_viewer(np.asarray([1.0, 2.0, 3.0]))
    assert np.allclose(point, [1.0, 3.0, -2.0])

    cloud = reconstruction_points_to_viewer(
        np.asarray([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float64)
    )
    assert np.allclose(cloud, [[1.0, 3.0, -2.0], [-4.0, -6.0, -5.0]])


def test_c1_camera_position_preserves_authoritative_pose_translation() -> None:
    pose = _pose(translation=(2.5, -1.0, 7.0), yaw_deg=22.0)
    position = camera_position_in_viewer(pose)
    assert np.allclose(position, [2.5, 7.0, 1.0])
    # Presentation conversion must never mutate the authoritative matrix.
    assert np.allclose(pose[:3, 3], [2.5, -1.0, 7.0])


def test_c1_symbolic_frustum_is_finite_and_anchored_at_camera_position() -> None:
    pose = _pose(translation=(1.0, 2.0, 3.0), yaw_deg=15.0)
    segments = camera_frustum_segments_in_viewer(pose)
    assert segments.shape == (16, 3)
    assert segments.dtype == np.float32
    assert np.isfinite(segments).all()
    camera = camera_position_in_viewer(pose)
    # The first four independent segments all start at the camera centre.
    assert np.allclose(segments[[0, 2, 4, 6]], camera)


def test_c1_calibrated_frustum_accepts_pinhole_intrinsics_without_metric_claim() -> None:
    K = np.asarray(
        [
            [600.0, 0.0, 319.5],
            [0.0, 600.0, 239.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    segments = camera_frustum_segments_in_viewer(
        _pose(), intrinsics_matrix=K, image_size=(640, 480), depth=0.4
    )
    assert segments.shape == (16, 3)
    assert np.isfinite(segments).all()
    # Frustum depth is only a display size; it does not alter any scale authority.
    assert float(np.max(np.linalg.norm(segments, axis=1))) < 1.0


def test_c1_visualization_helpers_reject_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="finite 3-vector"):
        reconstruction_point_to_viewer(np.asarray([1.0, np.nan, 3.0]))
    with pytest.raises(ValueError, match="finite \\(N, 3\\)"):
        reconstruction_points_to_viewer(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="finite 4x4"):
        camera_position_in_viewer(np.eye(3))

    invalid_rotation = np.eye(4)
    invalid_rotation[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        camera_frustum_segments_in_viewer(invalid_rotation)

    with pytest.raises(ValueError, match="supplied together"):
        camera_frustum_segments_in_viewer(_pose(), image_size=(640, 480))
