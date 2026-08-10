from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy

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


def _pose(angle: float, radius: float = 2.0) -> list[list[float]]:
    theta = np.deg2rad(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [radius * np.cos(theta), 0.1 * np.sin(2.0 * theta), radius * np.sin(theta)]
    return matrix.tolist()


def _artifact(path: Path, session_id: str) -> Path:
    rng = np.random.default_rng(884)
    points = rng.normal(size=(300, 3)).astype(np.float64)
    points *= np.asarray([1.5, 0.6, 1.0])
    cameras = [
        {"keyframe_id": f"kf-{index:02d}", "T_reconstruction_from_camera": _pose(index * 45.0)}
        for index in range(8)
    ]
    path.write_text(
        json.dumps(
            {
                "schema": OFFLINE_VISUALIZATION_SCHEMA,
                "session_id": session_id,
                "backend_id": "pycolmap-offline-sfm",
                "backend_version": "test",
                "pose_convention": "T_reconstruction_from_camera",
                "scale_status": "arbitrary",
                "alignment_status": "arbitrary",
                "registered_camera_total": len(cameras),
                "registered_cameras": cameras,
                "landmark_total": 5000,
                "landmarks_reconstruction": points.tolist(),
                "mean_reprojection_error_px": 0.5,
            }
        ),
        encoding="utf-8",
    )
    return path.resolve()


def _result(session_id: str, artifact: Path) -> dict[str, object]:
    registered = 8
    total = 8
    landmarks = 5000
    observations = 25_000
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
                "status": "accepted",
                "usable": True,
                "registration_ratio": 1.0,
                "landmarks_per_registered_keyframe": landmarks / registered,
                "mean_track_length": observations / landmarks,
                "mean_reprojection_error_px": 0.5,
                "reasons": [],
            },
            "gauge": {"scale_status": "arbitrary"},
        },
        "artifacts": {"offline_visualization_json": str(artifact)},
    }


def _load(window, tmp_path: Path):
    session_id = "44444444-4444-4444-8444-444444444444"
    artifact = _artifact(tmp_path / "offline_visualization.json", session_id)
    window._reconstruction_controller.offline_result_ready.emit(
        _result(session_id, artifact),
        str(tmp_path),
    )
    return window._reconstruction_page


def test_d4_completed_offline_result_surfaces_unit_honest_spatial_coverage(window, tmp_path: Path) -> None:
    page = _load(window, tmp_path)
    canvas = page._reconstruction_canvas

    assert page._offline_spatial_coverage is not None
    assert page._offline_spatial_coverage.has_envelope is True
    assert "X " in page._offline_spatial_envelope_value.text()
    assert "Y " in page._offline_spatial_envelope_value.text()
    assert "Z " in page._offline_spatial_envelope_value.text()
    assert "gauge" in page._offline_spatial_envelope_value.text()
    assert "centers" in page._offline_spatial_camera_span_value.text()
    assert "gauge span" in page._offline_spatial_camera_span_value.text()
    assert "gauge min / med / max" in page._offline_spatial_standoff_value.text()
    assert "presentation envelope only" in page._offline_spatial_basis_value.text()
    assert "not object dimensions" in page._offline_spatial_envelope_value.toolTip()
    assert canvas.offline_sparse_envelope_available is True
    assert canvas.offline_sparse_envelope_visible is True
    assert canvas.offline_sparse_envelope_bounds is not None


def test_d4_envelope_toolbar_toggle_is_presentation_only(window, qtbot, tmp_path: Path) -> None:
    page = _load(window, tmp_path)
    canvas = page._reconstruction_canvas
    button = page._viewport_tool_buttons["offline_envelope"]

    assert button.isChecked() is True
    assert canvas.offline_sparse_envelope_visible is True
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert canvas.offline_sparse_envelope_visible is False
    assert page._offline_spatial_coverage is not None
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert canvas.offline_sparse_envelope_visible is True


def test_d4_new_offline_run_clears_previous_spatial_coverage(window, tmp_path: Path) -> None:
    page = _load(window, tmp_path)
    assert page._offline_spatial_coverage is not None
    assert page._reconstruction_canvas.offline_sparse_envelope_available is True

    page._prepare_runtime_presentation_for_start(ReconstructionWorkflow.IMAGE_SEQUENCE)

    assert page._offline_spatial_coverage is None
    assert page._offline_spatial_envelope_value.text() == "—"
    assert "not object dimensions" in page._offline_spatial_basis_value.text()
    assert page._reconstruction_canvas.offline_sparse_envelope_available is False
    assert page._offline_inspection_group.isHidden() is True


def test_d4_live_start_cannot_retain_stale_offline_envelope(window, tmp_path: Path) -> None:
    page = _load(window, tmp_path)
    assert page._reconstruction_canvas.offline_sparse_envelope_available is True

    # Entering a new live session clears the canvas via the existing live STARTING lifecycle.
    from core.reconstruction import LiveCaptureRuntimeState

    page.set_live_state(LiveCaptureRuntimeState.STARTING)
    assert page._reconstruction_canvas.offline_sparse_envelope_available is False


def test_d4_settings_sidebar_scrolls_instead_of_crushing_offline_inspection(
    window, qtbot, tmp_path: Path
) -> None:
    page = _load(window, tmp_path)
    window.resize(1366, 768)
    window.show()
    qtbot.wait(50)

    scroll = page._settings_scroll
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert scroll.widgetResizable() is True
    assert scroll.verticalScrollBar().maximum() > 0
    # Reserve a visual gutter for Linux/Qt styles whose scrollbar overlays
    # the viewport; settings values must wrap before that scrollbar.
    assert scroll.widget().layout().contentsMargins().right() >= 12
    # Do not compare the live body/viewport geometry here: offscreen Qt can
    # transiently squeeze a QSplitter child below the designed sidebar width.
    # Instead verify the actual contracts that prevent horizontal clipping:
    # content controls and value labels must not contribute an unshrinkable
    # minimum width while horizontal scrolling remains disabled.
    assert page._offline_camera_combo.minimumWidth() == 0
    assert page._offline_camera_combo.minimumContentsLength() <= 4
    assert (
        page._offline_camera_combo.sizePolicy().horizontalPolicy()
        == QSizePolicy.Policy.Ignored
    )
    representative_values = (
        page._workflow_input_value,
        page._session_value,
        page._offline_camera_position_value,
        page._offline_structure_value,
        page._offline_quality_support_value,
        page._offline_spatial_basis_value,
    )
    for value in representative_values:
        assert value.minimumWidth() == 0
        assert value.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored

    # Do not assert live QLabel heights here. Under the offscreen Qt platform
    # the QSplitter can transiently squeeze this sidebar to a viewport only a
    # few dozen pixels wide; word-wrapped labels then legitimately report very
    # large heights that never occur at the application's designed >=290 px
    # settings-panel width. Verify the actual D4 row-allocation contract
    # instead: every representative value must be the stretch-bearing second
    # widget in a two-widget row, with no expanding spacer ahead of it.
    def _find_row(root_layout, target):
        for i in range(root_layout.count()):
            item = root_layout.itemAt(i)
            widget = item.widget()
            if widget is target:
                return root_layout

            # Rows live inside QGroupBox widgets.  QLayoutItem.layout() is
            # None for a widget item, so recurse through the widget's own
            # layout as well as directly nested layout items.
            child_layout = item.layout()
            if child_layout is not None:
                found = _find_row(child_layout, target)
                if found is not None:
                    return found

            if widget is not None and widget.layout() is not None:
                found = _find_row(widget.layout(), target)
                if found is not None:
                    return found
        return None

    settings_layout = scroll.widget().layout()
    for value in representative_values:
        row = _find_row(settings_layout, value)
        assert row is not None
        assert row.count() == 2
        assert row.itemAt(0).widget() is not None
        assert row.itemAt(1).widget() is value
        assert row.stretch(0) == 0
        assert row.stretch(1) == 1

    for button in (
        page._offline_prev_camera_button,
        page._offline_focus_camera_button,
        page._offline_next_camera_button,
        page._offline_clear_camera_button,
    ):
        assert button.minimumWidth() == 0
        assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored

    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    qtbot.wait(20)
    assert page._offline_inspection_group.height() >= page._offline_inspection_group.minimumSizeHint().height()
