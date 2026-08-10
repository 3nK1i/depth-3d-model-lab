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


def test_a1_does_not_enable_or_add_reconstruction_ui(window) -> None:
    reconstruction_button = window._mode_tabs._buttons[AppMode.RECONSTRUCTION]

    assert reconstruction_button.isEnabled() is False
    assert window._mode_stack.count() == 5
    assert window._mode_stack.currentWidget() is window._page
