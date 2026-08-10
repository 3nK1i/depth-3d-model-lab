from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.main_window import MainWindow
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
            {"keyframe_id": "kf-0", "T_reconstruction_from_camera": _pose(0.0)},
            {"keyframe_id": "kf-1", "T_reconstruction_from_camera": _pose(0.5)},
            {"keyframe_id": "kf-2", "T_reconstruction_from_camera": _pose(1.0)},
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


def _load_result(window, tmp_path: Path):
    page = window._reconstruction_page
    session_id = "77777777-7777-4777-8777-777777777777"
    artifact = _visualization(tmp_path / "offline_visualization.json", session_id)
    window._reconstruction_controller.offline_result_ready.emit(_result(session_id, artifact), str(tmp_path))
    return page


def test_d2_offline_result_populates_inspection_selector_and_structure(window, tmp_path: Path) -> None:
    page = _load_result(window, tmp_path)
    assert page._offline_inspection_group.isHidden() is False
    assert page._offline_camera_combo.isEnabled() is True
    assert page._offline_camera_combo.count() == 4
    assert page._offline_structure_value.text() == "3/3 cameras • 4/4 points shown"
    assert page._reconstruction_canvas.offline_camera_path_visible is True
    assert page._reconstruction_canvas.offline_camera_path_display_count == 3


def test_d2_camera_selection_highlights_readout_and_focuses_without_changing_authority(window, tmp_path: Path) -> None:
    page = _load_result(window, tmp_path)
    page._offline_camera_combo.setCurrentIndex(2)

    canvas = page._reconstruction_canvas
    assert canvas.selected_offline_camera_index == 1
    assert canvas.selected_offline_camera_id == "kf-1"
    assert canvas.selected_offline_camera_visible is True
    assert "gauge" in page._offline_camera_position_value.text()
    assert "gauge from previous" in page._offline_camera_delta_value.text()
    assert canvas.has_pose is False
    assert canvas.offline_scale_status == "arbitrary"

    before = canvas.focus_calls
    page._offline_focus_camera_button.click()
    assert canvas.focus_calls == before + 1


def test_d2_prev_next_and_clear_are_presentation_only(window, tmp_path: Path) -> None:
    page = _load_result(window, tmp_path)
    page._offline_camera_combo.setCurrentIndex(2)
    page._offline_next_camera_button.click()
    assert page._reconstruction_canvas.selected_offline_camera_id == "kf-2"
    page._offline_prev_camera_button.click()
    assert page._reconstruction_canvas.selected_offline_camera_id == "kf-1"
    page._offline_clear_camera_button.click()
    assert page._reconstruction_canvas.selected_offline_camera_id is None
    assert page._offline_camera_position_value.text() == "—"


def test_d2_new_offline_run_clears_inspection_and_selected_camera(window, tmp_path: Path) -> None:
    page = _load_result(window, tmp_path)
    page._offline_camera_combo.setCurrentIndex(2)
    assert page._reconstruction_canvas.selected_offline_camera_id == "kf-1"

    page._prepare_runtime_presentation_for_start(ReconstructionWorkflow.IMAGE_SEQUENCE)

    assert page._offline_visualization is None
    assert page._offline_inspection_group.isHidden() is True
    assert page._reconstruction_canvas.selected_offline_camera_id is None
