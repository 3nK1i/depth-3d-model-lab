from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.main_window import MainWindow
from app.offline_reconstruction_visualization import OFFLINE_VISUALIZATION_SCHEMA


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _pose(x: float) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    return matrix.tolist()


def _result(root: Path, sid: str) -> dict[str, object]:
    visual = root / "offline_visualization.json"
    visual.write_text(
        json.dumps(
            {
                "schema": OFFLINE_VISUALIZATION_SCHEMA,
                "session_id": sid,
                "backend_id": "pycolmap-offline-sfm",
                "backend_version": "i6.2c-test",
                "pose_convention": "T_reconstruction_from_camera",
                "scale_status": "arbitrary",
                "alignment_status": "arbitrary",
                "registered_camera_total": 2,
                "registered_cameras": [
                    {"keyframe_id": "kf-0", "estimate_id": "c0", "T_reconstruction_from_camera": _pose(0.0)},
                    {"keyframe_id": "kf-1", "estimate_id": "c1", "T_reconstruction_from_camera": _pose(1.0)},
                ],
                "landmark_total": 3,
                "landmarks_reconstruction": [[0, 0, 1], [1, 0, 1], [0.5, 1, 1]],
                "landmark_colors_rgb": [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                "mean_reprojection_error_px": 0.4,
            }
        ),
        encoding="utf-8",
    )
    sparse = root / "sparse_reconstruction.ply"
    sparse.write_text("ply\n", encoding="utf-8")
    payload = {
        "status": "completed",
        "pass": True,
        "session_id": sid,
        "camera_pose_available": True,
        "canonical_geometry_authoritative": False,
        "persistent_scene_ready": False,
        "certified_metrology": False,
        "sfm_summary": {
            "registered_keyframes": 2,
            "input_keyframes": 2,
            "landmarks": 3,
            "mean_reprojection_error_px": 0.4,
            "gauge": {"scale_status": "arbitrary"},
        },
        "artifacts": {
            "offline_visualization_json": str(visual.resolve()),
            "sparse_reconstruction_ply": str(sparse.resolve()),
        },
    }
    (root / "offline_result.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_admitted_terminal_result_reaches_workspace_and_viewer(window, tmp_path: Path) -> None:
    root = (tmp_path / "run").resolve()
    root.mkdir()
    sid = "c2c2c2c2-2222-4222-8222-c2c2c2c2c2c2"
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    controller.offline_controller._session_id = sid
    controller.offline_controller._output_dir = root
    handoffs = []
    controller.offline_result_handoff_ready.connect(handoffs.append)

    controller._on_offline_result_ready(_result(root, sid), str(root))

    assert len(handoffs) == 1
    assert handoffs[0].partial_result_available is True
    assert handoffs[0].viewer_ready is True
    assert controller.persistence_controller.scene_availability.value == "partial"
    assert page.offline_workspace_snapshot is not None
    assert page.offline_workspace_snapshot.partial_result_available is True
    assert page._workspace_value.text() == "Partial result"
    assert page._reconstruction_canvas.offline_visualization_loaded is True
    assert page._reconstruction_canvas.sparse_landmark_count == 3
    assert page._viewport_state.text() == "Offline sparse reconstruction"


def test_rejected_terminal_result_never_reaches_viewer(window, tmp_path: Path) -> None:
    root = (tmp_path / "rejected").resolve()
    root.mkdir()
    admitted_sid = "c2c2c2c2-2222-4222-8222-c2c2c2c2c2c2"
    payload_sid = "d3d3d3d3-3333-4333-8333-d3d3d3d3d3d3"
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    controller.offline_controller._session_id = admitted_sid
    controller.offline_controller._output_dir = root

    controller._on_offline_result_ready(_result(root, payload_sid), str(root))

    assert controller.last_offline_result_handoff is None
    assert controller.persistence_controller.offline_result is None
    assert page._reconstruction_canvas.offline_visualization_loaded is False
