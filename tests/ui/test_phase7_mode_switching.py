"""Repeated cross-mode navigation validation for the Phase 7 release."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    monkeypatch.setattr(win._inference_controller._process_manager, "shutdown", lambda *_a, **_k: None)
    return win


def test_phase7_modes_remain_enabled_and_reconstruction_disabled(window) -> None:
    enabled = {AppMode.LIVE_3D, AppMode.STABLE_VIDEO, AppMode.SNAPSHOT, AppMode.BENCHMARK}
    for mode, button in window._mode_tabs._buttons.items():
        assert button.isEnabled() is (mode in enabled)


def test_repeated_operational_mode_switching_keeps_correct_pages(window, qtbot) -> None:
    expected = {
        AppMode.LIVE_3D: window._page,
        AppMode.STABLE_VIDEO: window._stable_video_page,
        AppMode.SNAPSHOT: window._snapshot_page,
        AppMode.BENCHMARK: window._benchmark_page,
    }
    order = tuple(expected)
    for _ in range(20):
        for mode in order:
            qtbot.mouseClick(window._mode_tabs._buttons[mode], Qt.MouseButton.LeftButton)
            assert window._mode_tabs.active_mode is mode
            assert window._mode_stack.currentWidget() is expected[mode]
