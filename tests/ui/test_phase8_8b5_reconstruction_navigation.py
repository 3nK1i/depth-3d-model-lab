from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_controller import ReconstructionWorkflow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import LiveCaptureRuntimeState


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _page_for_mode(window: MainWindow, mode: AppMode):
    return {
        AppMode.LIVE_3D: window._page,
        AppMode.INSPECTION: window._inspection_page,
        AppMode.STABLE_VIDEO: window._stable_video_page,
        AppMode.SNAPSHOT: window._snapshot_page,
        AppMode.RECONSTRUCTION: window._reconstruction_page,
        AppMode.BENCHMARK: window._benchmark_page,
    }[mode]


def _images(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    root.mkdir()
    for index in range(4):
        (root / f"frame-{index:02d}.jpg").write_bytes(b"image")
    return root


def test_b5_reconstruction_tab_routes_normally_without_backend_autostart(window, qtbot, monkeypatch) -> None:
    controller = window._reconstruction_controller
    calls: list[str] = []
    monkeypatch.setattr(controller, "start_selected_live", lambda **_kwargs: calls.append("live"))
    monkeypatch.setattr(controller, "start_selected_offline", lambda **_kwargs: calls.append("offline"))

    button = window._mode_tabs._buttons[AppMode.RECONSTRUCTION]
    assert button.isEnabled() is True
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert window._mode_tabs.active_mode is AppMode.RECONSTRUCTION
    assert window._mode_stack.currentWidget() is window._reconstruction_page
    assert controller.live_state is LiveCaptureRuntimeState.IDLE
    assert controller.offline_state is OfflineReconstructionRunState.IDLE
    assert controller.is_active is False
    assert calls == []


def test_b5_repeated_navigation_maps_all_six_modes_without_starting_reconstruction(window, qtbot) -> None:
    sequence = (
        AppMode.RECONSTRUCTION,
        AppMode.LIVE_3D,
        AppMode.INSPECTION,
        AppMode.SNAPSHOT,
        AppMode.BENCHMARK,
        AppMode.STABLE_VIDEO,
        AppMode.RECONSTRUCTION,
    )
    for _ in range(12):
        for mode in sequence:
            qtbot.mouseClick(window._mode_tabs._buttons[mode], Qt.MouseButton.LeftButton)
            assert window._mode_tabs.active_mode is mode
            assert window._mode_stack.currentWidget() is _page_for_mode(window, mode)
            assert window._reconstruction_controller.is_active is False


def test_b5_reconstruction_source_selection_survives_safe_mode_switching(window, qtbot, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path))
    selection = controller.source_selection
    assert selection.workflow is ReconstructionWorkflow.IMAGE_SEQUENCE
    assert selection.ready is True

    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    assert window._mode_stack.currentWidget() is window._reconstruction_page
    window._mode_tabs.set_active_mode(AppMode.LIVE_3D)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)

    assert controller.source_selection == selection
    assert window._reconstruction_page._source_readiness_value.text() == "Ready"
    assert controller.is_active is False


def test_b5_active_reconstruction_temporarily_locks_other_primary_modes(window, qtbot) -> None:
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    controller = window._reconstruction_controller
    offline = controller.offline_controller

    offline._run_finalized = False
    offline._state = OfflineReconstructionRunState.RUNNING
    controller.offline_state_changed.emit(OfflineReconstructionRunState.RUNNING)
    qtbot.wait(1)

    for mode, button in window._mode_tabs._buttons.items():
        assert button.isEnabled() is (mode is AppMode.RECONSTRUCTION)

    offline._run_finalized = True
    offline._state = OfflineReconstructionRunState.COMPLETED
    controller.offline_state_changed.emit(OfflineReconstructionRunState.COMPLETED)
    controller.offline_run_finished.emit({"status": "completed"})
    qtbot.wait(1)

    assert all(button.isEnabled() for button in window._mode_tabs._buttons.values())


def test_b5_window_can_close_cleanly_from_reconstruction_mode(window, qtbot, monkeypatch) -> None:
    shutdown_calls: list[bool] = []
    monkeypatch.setattr(
        window._reconstruction_controller,
        "shutdown",
        lambda: shutdown_calls.append(True),
    )
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    assert window._mode_stack.currentWidget() is window._reconstruction_page
    window.close()
    qtbot.wait(1)
    assert shutdown_calls
