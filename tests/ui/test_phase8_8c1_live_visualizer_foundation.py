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


def _observation(x: float, y: float, z: float):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [x, y, z]
    return SimpleNamespace(
        pose=SimpleNamespace(T_reconstruction_from_camera=pose),
        is_keyframe=False,
        map_snapshot=None,
    )


def test_c1_reconstruction_page_owns_real_viewer_foundation(window) -> None:
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    assert canvas.grid_visible is True
    assert canvas.axes_visible is True
    assert canvas.trajectory_visible is False
    assert canvas.has_pose is False
    assert page._viewport_placeholder.isHidden() is False
    assert page._viewport_state.text() == "Awaiting scene"
    assert "ARBITRARY GAUGE" in page.findChild(type(page._viewport_state), "reconstructionViewportSafety").text()


def test_c1_live_observation_updates_current_camera_via_coalesced_ui_timer(window, qtbot) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)

    controller.live_observation_ready.emit(_observation(1.0, 2.0, 3.0))
    controller.live_observation_ready.emit(_observation(4.0, 5.0, 6.0))
    # The 33 ms coalescer should render only the newest pending pose.
    qtbot.wait(55)

    canvas = page._reconstruction_canvas
    assert canvas.has_pose is True
    assert canvas.trajectory_count == 1
    assert np.allclose(canvas.current_camera_position, [4.0, 6.0, -5.0])
    assert canvas.trajectory_visible is False
    assert page._viewport_placeholder.isHidden() is True
    assert page._viewport_state.text() == "Live camera pose"
    assert canvas.fit_calls == 1


def test_c1_new_live_session_clears_previous_visual_pose(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    page.set_live_observation(_observation(1.0, 0.0, 0.0))
    qtbot.wait(55)
    assert page._reconstruction_canvas.has_pose is True

    page.set_live_state(LiveCaptureRuntimeState.STOPPED)
    assert page._viewport_state.text() == "Live camera pose retained"
    page.set_live_state(LiveCaptureRuntimeState.STARTING)
    assert page._reconstruction_canvas.has_pose is False
    assert page._reconstruction_canvas.trajectory_count == 0
    assert page._viewport_placeholder.isHidden() is False
    assert page._viewport_state.text() == "Waiting for live pose"


def test_c1_viewport_navigation_controls_drive_dedicated_canvas(window, qtbot) -> None:
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas

    qtbot.mouseClick(page._viewport_tool_buttons["pan"], Qt.MouseButton.LeftButton)
    assert canvas.interaction_mode == "pan"
    qtbot.mouseClick(page._viewport_tool_buttons["zoom"], Qt.MouseButton.LeftButton)
    assert canvas.interaction_mode == "zoom"
    qtbot.mouseClick(page._viewport_tool_buttons["orbit"], Qt.MouseButton.LeftButton)
    assert canvas.interaction_mode == "orbit"

    grid_button = page._viewport_tool_buttons["grid"]
    qtbot.mouseClick(grid_button, Qt.MouseButton.LeftButton)
    assert canvas.grid_visible is False
    qtbot.mouseClick(grid_button, Qt.MouseButton.LeftButton)
    assert canvas.grid_visible is True

    buttons = page.findChildren(type(grid_button))
    by_tip = {button.toolTip(): button for button in buttons}
    qtbot.mouseClick(by_tip["Fit known camera path to view"], Qt.MouseButton.LeftButton)
    qtbot.mouseClick(by_tip["Focus current camera"], Qt.MouseButton.LeftButton)
    qtbot.mouseClick(by_tip["Reset reconstruction view"], Qt.MouseButton.LeftButton)
    assert canvas.fit_calls >= 1
    assert canvas.focus_calls == 1
    assert canvas.reset_calls == 1


def test_c1_viewport_shutdown_is_idempotent_and_ignores_late_observation(window, qtbot) -> None:
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    page.shutdown_viewport()
    page.shutdown_viewport()
    assert canvas.closed is True

    before = canvas.trajectory_count
    page.set_live_observation(_observation(3.0, 2.0, 1.0))
    qtbot.wait(55)
    assert canvas.trajectory_count == before
