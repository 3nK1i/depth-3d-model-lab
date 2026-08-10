from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.offline_reconstruction_visualization import OFFLINE_VISUALIZATION_SCHEMA
from app.reconstruction_controller import ReconstructionWorkflow


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _pose(x: float, y: float = 0.0) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    matrix[1, 3] = y
    return matrix.tolist()


def _visualization(path: Path, session_id: str) -> Path:
    payload = {
        "schema": OFFLINE_VISUALIZATION_SCHEMA,
        "session_id": session_id,
        "backend_id": "pycolmap-offline-sfm",
        "backend_version": "3.13",
        "pose_convention": "T_reconstruction_from_camera",
        "scale_status": "arbitrary",
        "alignment_status": "arbitrary",
        "registered_camera_total": 3,
        "registered_cameras": [
            {"keyframe_id": "kf-0", "estimate_id": "c0", "T_reconstruction_from_camera": _pose(0.0)},
            {"keyframe_id": "kf-1", "estimate_id": "c1", "T_reconstruction_from_camera": _pose(0.5)},
            {"keyframe_id": "kf-2", "estimate_id": "c2", "T_reconstruction_from_camera": _pose(1.0)},
        ],
        "landmark_total": 4,
        "landmarks_reconstruction": [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        "landmark_colors_rgb": [[255, 0, 0], [0, 255, 0], [0, 0, 255], [220, 220, 220]],
        "mean_reprojection_error_px": 0.48,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.resolve()


def _result(session_id: str, artifact: Path) -> dict[str, object]:
    return {
        "status": "completed",
        "session_id": session_id,
        "certified_metrology": False,
        "canonical_geometry_authoritative": False,
        "sfm_summary": {
            "registered_keyframes": 3,
            "input_keyframes": 3,
            "landmarks": 4,
            "mean_reprojection_error_px": 0.48,
            "gauge": {"scale_status": "arbitrary"},
        },
        "artifacts": {"offline_visualization_json": str(artifact)},
    }


def test_d1_completed_offline_result_populates_real_viewer_primitives(window, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    session_id = "33333333-3333-4333-8333-333333333333"
    artifact = _visualization(tmp_path / "offline_visualization.json", session_id)

    controller.offline_result_ready.emit(_result(session_id, artifact), str(tmp_path))

    canvas = page._reconstruction_canvas
    assert canvas.offline_visualization_loaded is True
    assert canvas.offline_session_id == session_id
    assert canvas.offline_scale_status == "arbitrary"
    assert canvas.sparse_landmark_count == 4
    assert canvas.sparse_landmark_display_count == 4
    assert canvas.keyframe_count == 3
    assert canvas.keyframe_display_count == 3
    assert canvas.trajectory_count == 0
    assert canvas.has_pose is False
    assert page._viewport_placeholder.isHidden() is True
    assert page._viewport_state.text() == "Offline sparse reconstruction"
    assert page._telemetry_value.text() == "3/3 keyframes • 4 landmarks • 0.48 px"
    assert canvas.fit_calls == 1


def test_d1_starting_new_offline_run_clears_previous_visual_result(window, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    session_id = "44444444-4444-4444-8444-444444444444"
    artifact = _visualization(tmp_path / "offline_visualization.json", session_id)
    controller.offline_result_ready.emit(_result(session_id, artifact), str(tmp_path))
    assert page._reconstruction_canvas.offline_visualization_loaded is True

    page._prepare_runtime_presentation_for_start(ReconstructionWorkflow.IMAGE_SEQUENCE)

    assert page._reconstruction_canvas.offline_visualization_loaded is False
    assert page._reconstruction_canvas.sparse_landmark_count == 0
    assert page._reconstruction_canvas.keyframe_count == 0
    assert page._viewport_placeholder.isHidden() is False


def test_d1_bad_presentation_artifact_does_not_change_completed_offline_authority(window, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller._offline_controller._state = OfflineReconstructionRunState.COMPLETED
    controller._offline_controller._run_finalized = True
    controller.offline_state_changed.emit(OfflineReconstructionRunState.COMPLETED)
    controller.offline_result_ready.emit(
        {
            "status": "completed",
            "session_id": "55555555-5555-4555-8555-555555555555",
            "sfm_summary": {"registered_keyframes": 3, "input_keyframes": 3, "landmarks": 50},
            "artifacts": {"offline_visualization_json": str((tmp_path / "missing.json").resolve())},
        },
        str(tmp_path),
    )

    assert controller.offline_state is OfflineReconstructionRunState.COMPLETED
    assert page._offline_value.text() == "Completed"
    assert page._offline_visualization_error
    assert page._viewport_state.text() == "Offline visualization unavailable"
    assert page._runtime_message.text() == "Offline reconstruction completed."


def test_d1_offline_colored_landmarks_are_presentation_only(window, tmp_path: Path) -> None:
    page = window._reconstruction_page
    session_id = "66666666-6666-4666-8666-666666666666"
    artifact = _visualization(tmp_path / "offline_visualization.json", session_id)
    window._reconstruction_controller.offline_result_ready.emit(_result(session_id, artifact), str(tmp_path))

    colors = page._reconstruction_canvas._sparse_landmark_colors
    assert colors is not None
    assert colors.shape == (4, 3)
    assert page._reconstruction_canvas.offline_scale_status == "arbitrary"
