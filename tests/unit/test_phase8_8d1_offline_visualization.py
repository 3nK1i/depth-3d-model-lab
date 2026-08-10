from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.offline_reconstruction_visualization import (
    OFFLINE_VISUALIZATION_SCHEMA,
    load_offline_sparse_visualization,
    parse_offline_sparse_visualization,
)


def _pose(x: float) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    return matrix.tolist()


def _payload() -> dict[str, object]:
    return {
        "schema": OFFLINE_VISUALIZATION_SCHEMA,
        "session_id": "11111111-1111-4111-8111-111111111111",
        "backend_id": "pycolmap-offline-sfm",
        "backend_version": "3.13",
        "pose_convention": "T_reconstruction_from_camera",
        "scale_status": "arbitrary",
        "alignment_status": "arbitrary",
        "registered_camera_total": 2,
        "registered_cameras": [
            {"estimate_id": "cam-0", "keyframe_id": "kf-0", "T_reconstruction_from_camera": _pose(0.0)},
            {"estimate_id": "cam-1", "keyframe_id": "kf-1", "T_reconstruction_from_camera": _pose(1.0)},
        ],
        "landmark_total": 3,
        "landmarks_reconstruction": [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
        "landmark_colors_rgb": [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        "mean_reprojection_error_px": 0.42,
    }


def test_d1_parser_preserves_sparse_geometry_camera_structure_and_arbitrary_gauge() -> None:
    visual = parse_offline_sparse_visualization(_payload())

    assert visual.scale_status == "arbitrary"
    assert visual.pose_convention == "T_reconstruction_from_camera"
    assert visual.registered_camera_total == 2
    assert visual.landmark_total == 3
    assert visual.camera_matrices_reconstruction.shape == (2, 4, 4)
    assert visual.landmark_points_reconstruction.shape == (3, 3)
    assert visual.landmark_colors_rgb is not None
    assert visual.landmark_colors_rgb.dtype == np.uint8
    assert visual.has_geometry is True


def test_d1_loader_requires_absolute_bounded_artifact(tmp_path: Path) -> None:
    path = tmp_path / "offline_visualization.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    visual = load_offline_sparse_visualization(path.resolve())
    assert visual.camera_ids == ("kf-0", "kf-1")

    with pytest.raises(ValueError, match="absolute"):
        load_offline_sparse_visualization(Path("relative.json"))


def test_d1_parser_rejects_invalid_pose_or_mismatched_colors() -> None:
    payload = _payload()
    payload["registered_cameras"][0]["T_reconstruction_from_camera"] = [[1, 0], [0, 1]]  # type: ignore[index]
    with pytest.raises(ValueError, match="4x4"):
        parse_offline_sparse_visualization(payload)

    payload = _payload()
    payload["landmark_colors_rgb"] = [[255, 0, 0]]
    with pytest.raises(ValueError, match="match"):
        parse_offline_sparse_visualization(payload)


def test_d1_parser_does_not_accept_metric_claims_outside_known_scale_states() -> None:
    payload = _payload()
    payload["scale_status"] = "calibrated_metric"
    with pytest.raises(ValueError, match="arbitrary scale"):
        parse_offline_sparse_visualization(payload)
