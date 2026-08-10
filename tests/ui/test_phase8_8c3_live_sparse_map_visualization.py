from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import LiveCaptureRuntimeState


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    win._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    yield win
    win.close()


def _observation(*, x: float, landmarks: tuple[tuple[float, float, float], ...], revision: int = 1):
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return SimpleNamespace(
        pose=SimpleNamespace(T_reconstruction_from_camera=pose),
        is_keyframe=True,
        map_snapshot=SimpleNamespace(
            revision=revision,
            keyframe_count=3,
            landmark_count=max(10, len(landmarks)),
            landmarks_reconstruction=landmarks,
        ),
    )


def test_c3_live_observation_coalesces_sparse_map_into_canvas(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    points = tuple((index * 0.01, (index % 3) * 0.02, 0.2 + index * 0.005) for index in range(20))
    page.set_live_observation(_observation(x=0.1, landmarks=points, revision=4))
    qtbot.wait(55)

    canvas = page._reconstruction_canvas
    assert canvas.has_sparse_map is True
    assert canvas.sparse_landmark_display_count == 20
    assert canvas.sparse_landmark_count == 20
    assert canvas.sparse_map_revision == 4
    assert canvas.sparse_landmarks_visible is True
    assert page._viewport_placeholder.isHidden() is True


def test_c3_sparse_map_toggle_is_presentation_only(window, qtbot) -> None:
    page = window._reconstruction_page
    button = page._viewport_tool_buttons["landmarks"]
    assert button.isChecked() is True
    assert page._reconstruction_canvas.sparse_landmarks_visible is True
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert page._reconstruction_canvas.sparse_landmarks_visible is False
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert page._reconstruction_canvas.sparse_landmarks_visible is True


def test_c3_new_session_clears_sparse_map_but_keeps_user_visibility_preference(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    page.set_live_observation(_observation(x=0.1, landmarks=((0.0, 0.0, 1.0),)))
    qtbot.wait(55)
    assert page._reconstruction_canvas.has_sparse_map is True

    page._viewport_tool_buttons["landmarks"].click()
    assert page._reconstruction_canvas.sparse_landmarks_visible is False
    page.set_live_state(LiveCaptureRuntimeState.STOPPED)
    page.set_live_state(LiveCaptureRuntimeState.STARTING)
    assert page._reconstruction_canvas.has_sparse_map is False
    assert page._reconstruction_canvas.sparse_landmarks_visible is False


def test_c3_world_axis_arrowhead_visuals_are_bound_to_axes_visibility(window) -> None:
    canvas = window._reconstruction_page._reconstruction_canvas
    for name in ("_axis_x_head", "_axis_y_head", "_axis_z_head"):
        assert hasattr(canvas, name)
    canvas.set_axes_visible(False)
    assert canvas.axes_visible is False
    canvas.set_axes_visible(True)
    assert canvas.axes_visible is True
