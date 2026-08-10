from __future__ import annotations

import numpy as np
import pytest

from app.offline_reconstruction_inspection import (
    build_offline_camera_inspection,
    format_offline_structure_summary,
)
from app.offline_reconstruction_visualization import (
    OFFLINE_VISUALIZATION_SCHEMA,
    parse_offline_sparse_visualization,
)


def _pose(x: float, y: float = 0.0, z: float = 0.0) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    return matrix.tolist()


def _visual():
    return parse_offline_sparse_visualization(
        {
            "schema": OFFLINE_VISUALIZATION_SCHEMA,
            "session_id": "d2-session",
            "backend_id": "pycolmap-offline-sfm",
            "backend_version": "3.13",
            "pose_convention": "T_reconstruction_from_camera",
            "scale_status": "arbitrary",
            "alignment_status": "arbitrary",
            "registered_camera_total": 5,
            "registered_cameras": [
                {"keyframe_id": "kf-0", "T_reconstruction_from_camera": _pose(0.0)},
                {"keyframe_id": "kf-1", "T_reconstruction_from_camera": _pose(0.3, 0.4)},
                {"keyframe_id": "kf-2", "T_reconstruction_from_camera": _pose(1.0, 0.4)},
            ],
            "landmark_total": 1000,
            "landmarks_reconstruction": [[0, 0, 1], [1, 0, 1]],
            "landmark_colors_rgb": [[200, 200, 200], [220, 220, 220]],
            "mean_reprojection_error_px": 0.51,
        }
    )


def test_d2_camera_inspection_is_explicitly_arbitrary_gauge() -> None:
    visual = _visual()
    item = build_offline_camera_inspection(visual, 1)
    assert item.camera_id == "kf-1"
    assert item.ordinal_text == "2/3"
    assert item.position_reconstruction == pytest.approx((0.3, 0.4, 0.0))
    assert item.distance_from_previous_gauge == pytest.approx(0.5)
    assert "gauge" in item.position_text
    assert "gauge from previous" in item.delta_previous_text


def test_d2_first_camera_has_no_fake_previous_baseline() -> None:
    item = build_offline_camera_inspection(_visual(), 0)
    assert item.distance_from_previous_gauge is None
    assert item.delta_previous_text == "First registered camera"


def test_d2_structure_summary_separates_sampled_and_authoritative_counts() -> None:
    text = format_offline_structure_summary(_visual())
    assert text == "3/5 cameras • 2/1,000 points shown"


def test_d2_camera_inspection_rejects_out_of_range_selection() -> None:
    visual = _visual()
    with pytest.raises(IndexError, match="out of range"):
        build_offline_camera_inspection(visual, -1)
    with pytest.raises(IndexError, match="out of range"):
        build_offline_camera_inspection(visual, 3)
