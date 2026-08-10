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


def _keyframe(index: int, x: float):
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return SimpleNamespace(
        keyframe_id=f"stella-keyframe-{index}",
        T_reconstruction_from_camera=pose,
    )


def _observation(count: int = 6):
    pose = np.eye(4, dtype=np.float64)
    pose[2, 3] = 0.3
    keyframes = tuple(_keyframe(index, index * 0.08) for index in range(count))
    return SimpleNamespace(
        pose=SimpleNamespace(T_reconstruction_from_camera=pose),
        is_keyframe=True,
        map_snapshot=SimpleNamespace(
            revision=9,
            keyframe_count=count,
            landmark_count=20,
            landmarks_reconstruction=tuple((i * 0.01, 0.0, 0.4) for i in range(20)),
            keyframes_reconstruction=keyframes,
        ),
    )


def test_c4_live_observation_coalesces_historical_keyframes_into_canvas(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    page.set_live_observation(_observation(7))
    qtbot.wait(55)

    canvas = page._reconstruction_canvas
    assert canvas.has_keyframes is True
    assert canvas.keyframe_count == 7
    assert canvas.keyframe_display_count == 7
    assert canvas.keyframe_map_revision == 9
    assert canvas.keyframe_ids[0] == "stella-keyframe-0"
    assert canvas.keyframes_visible is True
    assert page._viewport_placeholder.isHidden() is True


def test_c4_keyframe_toggle_is_presentation_only(window, qtbot) -> None:
    page = window._reconstruction_page
    button = page._viewport_tool_buttons["keyframes"]
    assert button.isChecked() is True
    assert page._reconstruction_canvas.keyframes_visible is True
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert page._reconstruction_canvas.keyframes_visible is False
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert page._reconstruction_canvas.keyframes_visible is True


def test_c4_new_session_clears_keyframes_but_preserves_visibility_preference(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    page.set_live_observation(_observation(5))
    qtbot.wait(55)
    assert page._reconstruction_canvas.has_keyframes is True

    page._viewport_tool_buttons["keyframes"].click()
    assert page._reconstruction_canvas.keyframes_visible is False
    page.set_live_state(LiveCaptureRuntimeState.STOPPED)
    page.set_live_state(LiveCaptureRuntimeState.STARTING)
    assert page._reconstruction_canvas.has_keyframes is False
    assert page._reconstruction_canvas.keyframes_visible is False


def test_c4_ui_keyframe_display_is_bounded(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    page.set_live_observation(_observation(140))
    qtbot.wait(55)
    canvas = page._reconstruction_canvas
    assert canvas.keyframe_count == 140
    assert canvas.keyframe_display_count == canvas.MAX_KEYFRAME_DISPLAY_FRUSTUMS == 96
    assert canvas.keyframe_ids[0] == "stella-keyframe-0"
    assert canvas.keyframe_ids[-1] == "stella-keyframe-139"
