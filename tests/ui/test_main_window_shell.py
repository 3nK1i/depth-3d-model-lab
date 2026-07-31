"""Main-window shell and runtime lifecycle regression tests."""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.models.adapters.mock_adapter import MOCK_MODEL_ID
from core.runtime.messages import Ack, CommandType, InferenceResult
from core.runtime.state_machine import ApplicationState


@pytest.fixture
def window(qtbot, monkeypatch):
    # Avoid real camera probing and source/GPU process work in UI tests.
    monkeypatch.setattr(
        "app.application_controller.CameraEnumerator.start",
        lambda self: None,
    )

    win = MainWindow()
    qtbot.addWidget(win)

    controller = win._inference_controller
    monkeypatch.setattr(win._controller, "connect_active_source", lambda: None)
    monkeypatch.setattr(win._controller, "disconnect_active_source", lambda: None)
    monkeypatch.setattr(controller._process_manager, "start_worker", lambda: None)
    monkeypatch.setattr(controller._process_manager, "load_model", lambda *_a, **_k: None)
    monkeypatch.setattr(controller._process_manager, "resume", lambda: None)
    monkeypatch.setattr(controller._process_manager, "stop", lambda *_a, **_k: None)
    monkeypatch.setattr(controller._process_manager, "shutdown", lambda *_a, **_k: None)
    monkeypatch.setattr(
        type(controller._process_manager),
        "is_running",
        property(lambda self: True),
    )
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
        assert btn.isEnabled() is (mode is AppMode.LIVE_3D)


def test_initial_button_states_are_disconnected(window):
    toolbar = window._runtime_toolbar
    assert toolbar._connect_btn.isEnabled()
    assert not toolbar._disconnect_btn.isEnabled()
    assert not toolbar._start_btn.isEnabled()
    assert not toolbar._pause_resume_btn.isEnabled()
    assert not toolbar._stop_btn.isEnabled()


def test_connect_start_stop_lifecycle(window, qtbot):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    assert controller.state_machine.state is ApplicationState.SOURCE_READY

    qtbot.mouseClick(toolbar._start_btn, _left_button())
    assert controller.state_machine.state is ApplicationState.LOADING_MODEL

    controller._on_ack_received(Ack(CommandType.LOAD_MODEL, True))
    assert controller.state_machine.state is ApplicationState.RUNNING
    assert toolbar._pause_resume_btn.isEnabled()
    assert toolbar._stop_btn.isEnabled()

    qtbot.mouseClick(toolbar._stop_btn, _left_button())
    assert controller.state_machine.state is ApplicationState.MODEL_READY
    assert toolbar._disconnect_btn.isEnabled()
    assert toolbar._start_btn.isEnabled()


def test_restart_from_model_ready_reuses_loaded_model(window, qtbot, monkeypatch):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    load_calls: list[str] = []
    monkeypatch.setattr(
        controller._process_manager,
        "load_model",
        lambda model_id, _config=None: load_calls.append(model_id),
    )

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(Ack(CommandType.LOAD_MODEL, True))
    assert len(load_calls) == 1

    qtbot.mouseClick(toolbar._stop_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())

    assert controller.state_machine.state is ApplicationState.RUNNING
    assert len(load_calls) == 1


def test_stop_never_disconnects_source(window, qtbot):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(Ack(CommandType.LOAD_MODEL, True))
    qtbot.mouseClick(toolbar._stop_btn, _left_button())

    assert controller.state_machine.state is ApplicationState.MODEL_READY
    assert toolbar._disconnect_btn.isEnabled()


def test_mock_performance_snapshot_reports_real_point_count(window):
    controller = window._inference_controller
    controller._run_started_at = time.monotonic()
    result = InferenceResult(
        session_id=1,
        frame_id=0,
        model_id=MOCK_MODEL_ID,
        depth=np.ones((24, 32), dtype=np.float32),
        latency_ms=12.5,
    )

    controller._update_performance_snapshot(result, point_count=12_345)

    assert window._performance_bar._points._value.text() == "12,345"
    assert window._performance_bar._precision._value.text() == "FP32"


def _left_button():
    from PyQt6.QtCore import Qt

    return Qt.MouseButton.LeftButton
