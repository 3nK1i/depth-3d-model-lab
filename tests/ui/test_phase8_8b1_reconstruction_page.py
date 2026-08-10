from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QPushButton

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability
from app.widgets.mode_tabs import AppMode
from app.widgets.reconstruction_page import ReconstructionPage
from core.reconstruction import LiveCaptureRuntimeState


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_b1_adds_reconstruction_page_but_keeps_navigation_disabled(window) -> None:
    assert isinstance(window._reconstruction_page, ReconstructionPage)
    assert window._mode_stack.count() == 6
    assert window._mode_stack.indexOf(window._reconstruction_page) >= 0
    assert window._mode_stack.currentWidget() is window._page

    button = window._mode_tabs._buttons[AppMode.RECONSTRUCTION]
    assert button.isEnabled() is False
    with pytest.raises(ValueError, match="Mode is not enabled"):
        window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)


def test_b1_page_binding_is_passive_and_starts_no_backend(window) -> None:
    controller = window._reconstruction_controller
    page = window._reconstruction_page

    assert controller.live_state is LiveCaptureRuntimeState.IDLE
    assert controller.offline_state is OfflineReconstructionRunState.IDLE
    assert controller.scene_availability is ReconstructionSceneAvailability.NONE
    assert controller.is_active is False
    assert page._live_value.text() == "Idle"
    assert page._offline_value.text() == "Idle"
    assert page._scene_value.text() == "None"

    actions = page.findChildren(QPushButton)
    assert actions
    runtime_actions = [
        button for button in actions if button.objectName() == "reconstructionDisabledAction"
    ]
    assert len(runtime_actions) == 3
    assert all(not button.isEnabled() for button in runtime_actions)
    assert controller.is_active is False


def test_b1_page_passively_tracks_qualified_controller_signals(window, qtbot) -> None:
    controller = window._reconstruction_controller
    page = window._reconstruction_page

    controller.live_state_changed.emit(LiveCaptureRuntimeState.RUNNING)
    controller.offline_state_changed.emit(OfflineReconstructionRunState.STARTING)
    controller.scene_availability_changed.emit(ReconstructionSceneAvailability.PARTIAL)
    qtbot.wait(1)

    assert page._live_value.text() == "Running"
    assert page._offline_value.text() == "Starting"
    assert page._scene_value.text() == "Partial"
    assert page._viewport_state.text() == "Awaiting scene"
    assert controller.is_active is False


def test_b1_page_can_be_inspected_directly_without_starting_reconstruction(window) -> None:
    window._mode_stack.setCurrentWidget(window._reconstruction_page)
    assert window._mode_stack.currentWidget() is window._reconstruction_page
    assert window._mode_tabs.active_mode is AppMode.LIVE_3D
    assert window._reconstruction_controller.is_active is False
