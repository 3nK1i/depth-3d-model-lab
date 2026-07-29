"""Phase 1 UI shell tests.

These cover only what Phase 1 claims: the window builds, the layout
survives the minimum supported resolution, unfinished nav modes stay
disabled, and the demo connect/start/stop lifecycle drives button
states correctly. Real state-machine/session-id tests arrive in Phase 3
once there is a real runtime to test against.
"""

from __future__ import annotations

import pytest

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode


@pytest.fixture
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_window_launches(window):
    assert window.windowTitle() == "EIA TECH — Depth & 3D Model Lab"


def test_minimum_resolution_supported(window):
    window.resize(1366, 768)
    assert window.size().width() >= 1366
    assert window.size().height() >= 768


def test_only_live_3d_mode_enabled(window):
    tabs = window._mode_tabs
    for mode, btn in tabs._buttons.items():
        if mode is AppMode.LIVE_3D:
            assert btn.isEnabled()
        else:
            assert not btn.isEnabled()


def test_initial_button_states_are_disconnected(window, qtbot):
    toolbar = window._runtime_toolbar
    assert toolbar._connect_btn.isEnabled()
    assert not toolbar._disconnect_btn.isEnabled()
    assert not toolbar._start_btn.isEnabled()
    assert not toolbar._pause_resume_btn.isEnabled()
    assert not toolbar._stop_btn.isEnabled()


def test_connect_start_stop_demo_lifecycle(window, qtbot):
    toolbar = window._runtime_toolbar

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    assert not toolbar._connect_btn.isEnabled()
    assert toolbar._disconnect_btn.isEnabled()
    assert toolbar._start_btn.isEnabled()

    qtbot.mouseClick(toolbar._start_btn, _left_button())
    assert not toolbar._start_btn.isEnabled()
    assert toolbar._pause_resume_btn.isEnabled()
    assert toolbar._stop_btn.isEnabled()

    qtbot.mouseClick(toolbar._stop_btn, _left_button())
    assert toolbar._start_btn.isEnabled()
    assert not toolbar._pause_resume_btn.isEnabled()
    assert not toolbar._stop_btn.isEnabled()
    # Source stays connected/live after Stop Inference (spec §5, §13).
    assert toolbar._disconnect_btn.isEnabled()


def test_stop_never_disconnects_source(window, qtbot):
    toolbar = window._runtime_toolbar
    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    qtbot.mouseClick(toolbar._stop_btn, _left_button())
    assert window._is_connected is True
    assert window._is_running is False


def _left_button():
    from PyQt6.QtCore import Qt

    return Qt.MouseButton.LeftButton
