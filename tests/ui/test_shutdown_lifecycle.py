"""Application shutdown is safe from every asynchronous runtime state."""

from __future__ import annotations

import numpy as np
import pytest

from app.main_window import MainWindow
from core.models.catalog import MOGE2_MODEL_ID
from core.runtime.messages import Ack, CommandType, InferenceResult, ModelStage, ModelStageEvent
from core.runtime.state_machine import ApplicationState


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    controller = win._inference_controller
    monkeypatch.setattr(win._controller, "disconnect_active_source", lambda: None)
    return win


@pytest.mark.parametrize(
    "state",
    [state for state in ApplicationState if state is not ApplicationState.SHUTTING_DOWN],
)
def test_shutdown_is_legal_and_idempotent_from_every_state(
    window, monkeypatch, state
) -> None:
    controller = window._inference_controller
    controller._state_machine._state = state
    shutdown_calls: list[bool] = []
    stop_calls: list[bool] = []
    monkeypatch.setattr(
        controller._process_manager,
        "shutdown",
        lambda *_a, **_k: shutdown_calls.append(True),
    )
    monkeypatch.setattr(
        controller._process_manager,
        "stop",
        lambda *_a, **_k: stop_calls.append(True),
    )

    controller.shutdown()
    controller.shutdown()

    assert controller.state_machine.state is ApplicationState.SHUTTING_DOWN
    assert shutdown_calls == [True]
    assert stop_calls == []
    assert controller._shutdown_started
    assert not controller._pipeline_timer.isActive()
    assert not controller._inference_watchdog_timer.isActive()
    assert not controller._settings_refresh_timer.isActive()
    assert not controller._uptime_timer.isActive()


def test_late_worker_events_are_ignored_after_shutdown(window, monkeypatch) -> None:
    controller = window._inference_controller
    restart_calls: list[bool] = []
    monkeypatch.setattr(controller._process_manager, "shutdown", lambda *_a, **_k: None)
    monkeypatch.setattr(
        controller._process_manager,
        "restart_worker",
        lambda: restart_calls.append(True),
    )

    controller.shutdown()
    controller._on_model_stage_changed(
        ModelStageEvent(ModelStage.WARMING_UP, MOGE2_MODEL_ID, "late warm-up")
    )
    controller._on_ack_received(
        Ack(CommandType.LOAD_MODEL, True, model_id=MOGE2_MODEL_ID, metadata={"id": MOGE2_MODEL_ID})
    )
    controller._on_result_ready(
        InferenceResult(
            session_id=1,
            frame_id=1,
            model_id=MOGE2_MODEL_ID,
            depth=np.ones((2, 2), dtype=np.float32),
            latency_ms=1.0,
        )
    )
    controller._on_worker_crashed()
    controller._on_worker_unresponsive(CommandType.LOAD_MODEL)
    controller._on_inference_result_timeout()

    assert controller.state_machine.state is ApplicationState.SHUTTING_DOWN
    assert not controller._model_loaded
    assert restart_calls == []


def test_main_window_close_stops_source_controller_once(window, monkeypatch) -> None:
    source_controller = window._controller
    inference = window._inference_controller
    source_calls: list[bool] = []
    inference_calls: list[bool] = []
    monkeypatch.setattr(
        source_controller,
        "shutdown",
        lambda: source_calls.append(True),
    )
    monkeypatch.setattr(
        inference,
        "shutdown",
        lambda: inference_calls.append(True),
    )

    window.close()

    assert inference_calls == [True]
    assert source_calls == [True]


def test_source_controller_shutdown_is_idempotent(window, monkeypatch) -> None:
    source_controller = window._controller
    disconnect_calls: list[bool] = []
    enumerator_calls: list[bool] = []
    monkeypatch.setattr(
        source_controller,
        "disconnect_active_source",
        lambda: disconnect_calls.append(True),
    )
    monkeypatch.setattr(
        source_controller._enumerator,
        "stop",
        lambda *_a, **_k: enumerator_calls.append(True) or True,
    )

    source_controller.shutdown()
    source_controller.shutdown()

    assert source_controller._shutdown_started
    assert not source_controller._preview_timer.isActive()
    assert disconnect_calls == [True]
    assert enumerator_calls == [True]
