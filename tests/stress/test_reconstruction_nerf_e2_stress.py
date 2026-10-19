from __future__ import annotations

import numpy as np

from core.reconstruction import nerfstudio_camera_to_world_from_depth3d_pose


def test_pose_conversion_is_deterministic_under_repetition() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [0.3, -0.2, 1.7]
    expected = nerfstudio_camera_to_world_from_depth3d_pose(pose)
    for _ in range(1000):
        assert nerfstudio_camera_to_world_from_depth3d_pose(pose) == expected
