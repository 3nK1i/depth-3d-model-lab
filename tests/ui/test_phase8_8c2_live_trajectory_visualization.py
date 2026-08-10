from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from app.widgets.reconstruction_orientation_gizmo import ReconstructionOrientationGizmo
from core.reconstruction import LiveCaptureRuntimeState


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    win._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    yield win
    win.close()


def _observation(x: float, y: float, z: float):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [x, y, z]
    return SimpleNamespace(
        pose=SimpleNamespace(T_reconstruction_from_camera=pose),
        is_keyframe=False,
        map_snapshot=None,
    )


def test_c2_trajectory_is_visible_by_default_and_has_explicit_toggle(window) -> None:
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    button = page._viewport_tool_buttons["trajectory"]

    assert button.isChecked() is True
    assert canvas.trajectory_visible is True

    button.click()
    assert button.isChecked() is False
    assert canvas.trajectory_visible is False
    button.click()
    assert canvas.trajectory_visible is True


def test_c2_live_pose_coalescer_builds_visible_camera_path(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)

    page.set_live_observation(_observation(0.0, 0.0, 0.0))
    qtbot.wait(50)
    page.set_live_observation(_observation(0.10, 0.0, 0.02))
    qtbot.wait(50)
    page.set_live_observation(_observation(0.20, 0.01, 0.05))
    qtbot.wait(50)

    canvas = page._reconstruction_canvas
    assert canvas.trajectory_count == 3
    assert canvas.trajectory_display_count == 3
    assert canvas.trajectory_visible is True
    assert canvas.has_pose is True
    assert canvas.world_axis_length < 0.28
    assert page._viewport_placeholder.isHidden() is True


def test_c2_new_live_session_clears_old_path_but_preserves_visibility_preference(window, qtbot) -> None:
    page = window._reconstruction_page
    button = page._viewport_tool_buttons["trajectory"]
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert page._reconstruction_canvas.trajectory_visible is False

    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    page.set_live_observation(_observation(0.1, 0.0, 0.0))
    qtbot.wait(50)
    assert page._reconstruction_canvas.trajectory_count == 1

    page.set_live_state(LiveCaptureRuntimeState.STOPPED)
    page.set_live_state(LiveCaptureRuntimeState.STARTING)
    assert page._reconstruction_canvas.trajectory_count == 0
    assert page._reconstruction_canvas.trajectory_visible is False
    assert button.isChecked() is False


def test_c2_orientation_gizmo_is_fixed_bottom_right_presentation_widget(window) -> None:
    page = window._reconstruction_page
    gizmo = page._orientation_gizmo
    assert isinstance(gizmo, ReconstructionOrientationGizmo)
    assert gizmo.width() == 84
    assert gizmo.height() == 84
    assert gizmo.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    projection = page._reconstruction_canvas.orientation_gizmo_axes()
    assert projection.shape == (3, 2)
    assert np.isfinite(projection).all()
