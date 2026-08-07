from __future__ import annotations

import numpy as np

from core.reconstruction import (
    colmap_camera_center_reconstruction,
    colmap_camera_from_reconstruction_to_internal_pose,
    colmap_qvec_tvec_to_camera_from_reconstruction,
    internal_pose_to_colmap_camera_from_reconstruction,
)


def test_colmap_pose_inversion_round_trip() -> None:
    angle = np.deg2rad(30.0)
    c, s = np.cos(angle), np.sin(angle)
    camera_from_map = np.array(
        [
            [c, 0.0, s, 1.5],
            [0.0, 1.0, 0.0, -0.25],
            [-s, 0.0, c, 0.75],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    internal = colmap_camera_from_reconstruction_to_internal_pose(camera_from_map)
    restored = internal_pose_to_colmap_camera_from_reconstruction(internal)
    assert np.allclose(restored, camera_from_map)


def test_colmap_camera_center_matches_inverse_translation() -> None:
    camera_from_map = np.eye(4, dtype=np.float64)
    camera_from_map[:3, 3] = [2.0, -3.0, 4.0]
    center = colmap_camera_center_reconstruction(camera_from_map)
    assert np.allclose(center, [-2.0, 3.0, -4.0])


def test_colmap_text_qvec_wxyz_translation_contract() -> None:
    camera_from_map = colmap_qvec_tvec_to_camera_from_reconstruction(
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
    )
    assert np.allclose(camera_from_map[:3, :3], np.eye(3))
    assert np.allclose(camera_from_map[:3, 3], [1.0, 2.0, 3.0])
