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


def _pose(x: float) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    return matrix.tolist()


def _artifact(path: Path, session_id: str) -> Path:
    payload = {
        "schema": OFFLINE_VISUALIZATION_SCHEMA,
        "session_id": session_id,
        "backend_id": "pycolmap-offline-sfm",
        "backend_version": "test",
        "pose_convention": "T_reconstruction_from_camera",
        "scale_status": "arbitrary",
        "alignment_status": "arbitrary",
        "registered_camera_total": 4,
        "registered_cameras": [
            {"keyframe_id": f"kf-{index}", "T_reconstruction_from_camera": _pose(index * 0.25)}
            for index in range(4)
        ],
        "landmark_total": 100,
        "landmarks_reconstruction": [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
        "landmark_colors_rgb": [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        "mean_reprojection_error_px": 0.5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.resolve()


def _result(session_id: str, artifact: Path, *, status: str = "accepted", reasons=()) -> dict[str, object]:
    registered = 4
    total = 5
    landmarks = 100
    observations = 400
    return {
        "status": "completed",
        "session_id": session_id,
        "certified_metrology": False,
        "canonical_geometry_authoritative": False,
        "sfm_summary": {
            "registered_keyframes": registered,
            "input_keyframes": total,
            "landmarks": landmarks,
            "observations": observations,
            "mean_reprojection_error_px": 0.5,
            "quality_assessment": {
                "status": status,
                "usable": status != "rejected",
                "registration_ratio": registered / total,
                "landmarks_per_registered_keyframe": landmarks / registered,
                "mean_track_length": observations / landmarks,
                "mean_reprojection_error_px": 0.5,
                "reasons": list(reasons),
            },
            "gauge": {"scale_status": "arbitrary"},
        },
        "artifacts": {"offline_visualization_json": str(artifact)},
    }


def _load(window, tmp_path: Path, *, status: str = "accepted", reasons=()):
    session_id = "99999999-9999-4999-8999-999999999999"
    artifact = _artifact(tmp_path / "offline_visualization.json", session_id)
    window._reconstruction_controller.offline_result_ready.emit(
        _result(session_id, artifact, status=status, reasons=reasons),
        str(tmp_path),
    )
    return window._reconstruction_page


def test_d3_completed_offline_result_surfaces_backend_health(window, tmp_path: Path) -> None:
    page = _load(window, tmp_path)
    assert page._offline_inspection_group.isHidden() is False
    assert page._offline_quality_status_value.text() == "ACCEPTED"
    assert page._offline_quality_registration_value.text() == "4/5 • 80.0%"
    assert page._offline_quality_reprojection_value.text() == "0.500 px mean"
    assert "4.00 views / point" in page._offline_quality_support_value.text()
    assert "25.0 points / camera" in page._offline_quality_support_value.text()
    assert page._offline_quality_gate_value.text() == "No sparse-SfM gate warnings"


def test_d3_degraded_result_uses_warning_and_backend_reason(window, tmp_path: Path) -> None:
    page = _load(
        window,
        tmp_path,
        status="degraded",
        reasons=("Registration ratio 80.0% is below the preferred threshold.",),
    )
    assert page._offline_quality_status_value.text() == "DEGRADED"
    assert page._offline_quality_status_value.property("statusKind") == "warning"
    assert "preferred threshold" in page._offline_quality_gate_value.text()
    assert "preferred threshold" in page._offline_quality_gate_value.toolTip()


def test_d3_health_survives_optional_visualization_failure(window, tmp_path: Path) -> None:
    page = window._reconstruction_page
    session_id = "99999999-9999-4999-8999-999999999999"
    missing = tmp_path / "missing.json"
    window._reconstruction_controller.offline_result_ready.emit(
        _result(session_id, missing),
        str(tmp_path),
    )
    assert page._offline_quality_status_value.text() == "ACCEPTED"
    assert page._offline_inspection_group.isHidden() is False
    assert page._offline_visualization is None
    assert page._viewport_state.text() == "Offline visualization unavailable"


def test_d3_new_offline_run_clears_previous_health(window, tmp_path: Path) -> None:
    page = _load(window, tmp_path)
    assert page._offline_sfm_health is not None
    page._prepare_runtime_presentation_for_start(ReconstructionWorkflow.IMAGE_SEQUENCE)
    assert page._offline_sfm_health is None
    assert page._offline_quality_status_value.text() == "—"
    assert page._offline_inspection_group.isHidden() is True
