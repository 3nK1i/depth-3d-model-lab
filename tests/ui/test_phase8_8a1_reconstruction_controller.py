from __future__ import annotations

import pytest

from app.main_window import MainWindow
from app.reconstruction_controller import ReconstructionController
from app.widgets.mode_tabs import AppMode
from core.reconstruction import LiveCaptureRuntimeState


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_main_window_owns_dormant_reconstruction_controller(window) -> None:
    assert isinstance(window._reconstruction_controller, ReconstructionController)
    assert (
        window._live_slam_controller
        is window._reconstruction_controller.live_slam_controller
    )
    assert window._reconstruction_controller.live_state is LiveCaptureRuntimeState.IDLE
    assert window._reconstruction_controller.is_active is False


def test_a1_controller_boundary_remains_idle_when_b5_navigation_is_enabled(window) -> None:
    reconstruction_button = window._mode_tabs._buttons[AppMode.RECONSTRUCTION]

    assert reconstruction_button.isEnabled() is True
    assert window._mode_stack.count() == 6
    assert window._mode_stack.indexOf(window._reconstruction_page) >= 0
    assert window._mode_stack.currentWidget() is window._page
