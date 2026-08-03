"""Real spawned-worker integration tests using only the development mock.

These tests do not download MoGe-2 or require a GPU. They validate the actual
spawn process, pipe, queues, worker protocol, latest-frame result, Stop, and
shutdown path that every real model depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.inference.process_manager import InferenceProcessManager
from core.models.catalog import MOCK_MODEL_ID
from core.runtime.failures import FailureKind
from core.runtime.messages import CommandType

pytestmark = pytest.mark.integration


def _wait_for_ack(qtbot, manager, action, expected: CommandType, timeout=10_000):
    with qtbot.waitSignal(manager.ack_received, timeout=timeout) as signal:
        action()
    ack = signal.args[0]
    assert ack.command_type is expected
    assert ack.success, ack.message
    return ack


def test_spawned_mock_worker_loads_runs_stops_and_shuts_down(qtbot) -> None:
    manager = InferenceProcessManager(stop_timeout_ms=3_000)
    try:
        manager.start_worker()
        qtbot.waitUntil(lambda: manager.is_running, timeout=5_000)

        load_ack = _wait_for_ack(
            qtbot,
            manager,
            lambda: manager.load_model(
                MOCK_MODEL_ID,
                {"device": "cpu", "precision": "fp32", "latency_ms": 0.0},
            ),
            CommandType.LOAD_MODEL,
        )
        assert load_ack.model_id == MOCK_MODEL_ID
        assert load_ack.metadata and load_ack.metadata["id"] == MOCK_MODEL_ID

        _wait_for_ack(qtbot, manager, manager.resume, CommandType.RESUME)

        image = np.zeros((24, 32, 3), dtype=np.uint8)
        with qtbot.waitSignal(manager.result_ready, timeout=10_000) as result_signal:
            assert manager.submit_frame(1, 0, image, point_budget=500)
        result = result_signal.args[0]
        assert result.session_id == 1
        assert result.frame_id == 0
        assert result.model_id == MOCK_MODEL_ID
        assert result.depth is not None and result.depth.shape == (24, 32)
        assert result.error is None

        _wait_for_ack(qtbot, manager, lambda: manager.stop(1), CommandType.STOP)
        assert not manager._watchdog.is_watching
    finally:
        manager.shutdown(timeout_s=3.0)

    assert not manager.is_running
    assert manager._parent_conn is None
    assert manager._frame_queue is None
    assert manager._result_queue is None


def test_worker_can_restart_with_fresh_ipc(qtbot) -> None:
    manager = InferenceProcessManager()
    try:
        manager.start_worker()
        qtbot.waitUntil(lambda: manager.is_running, timeout=5_000)
        first_process = manager._process

        manager.restart_worker()
        qtbot.waitUntil(lambda: manager.is_running, timeout=5_000)

        assert manager._process is not first_process
        assert manager._parent_conn is not None
        assert manager._frame_queue is not None
        assert manager._result_queue is not None
    finally:
        manager.shutdown(timeout_s=3.0)


def test_worker_unloads_and_reloads_model_without_process_restart(qtbot) -> None:
    manager = InferenceProcessManager(stop_timeout_ms=3_000)
    try:
        manager.start_worker()
        qtbot.waitUntil(lambda: manager.is_running, timeout=5_000)
        process = manager._process

        _wait_for_ack(
            qtbot,
            manager,
            lambda: manager.load_model(
                MOCK_MODEL_ID,
                {"device": "cpu", "precision": "fp32", "latency_ms": 0.0},
            ),
            CommandType.LOAD_MODEL,
        )

        unload_ack = _wait_for_ack(
            qtbot,
            manager,
            manager.unload_model,
            CommandType.UNLOAD_MODEL,
        )
        assert unload_ack.metadata and unload_ack.metadata["released"] is True
        assert unload_ack.unload_time_ms is not None
        assert manager._process is process
        assert manager.is_running

        reload_ack = _wait_for_ack(
            qtbot,
            manager,
            lambda: manager.load_model(
                MOCK_MODEL_ID,
                {"device": "cpu", "precision": "fp32", "latency_ms": 0.0},
            ),
            CommandType.LOAD_MODEL,
        )
        assert reload_ack.model_id == MOCK_MODEL_ID
        assert manager._process is process
    finally:
        manager.shutdown(timeout_s=3.0)


def test_spawned_worker_classifies_mock_oom_and_remains_restartable(qtbot) -> None:
    manager = InferenceProcessManager(stop_timeout_ms=3_000)
    try:
        manager.start_worker()
        qtbot.waitUntil(lambda: manager.is_running, timeout=5_000)
        process = manager._process

        _wait_for_ack(
            qtbot,
            manager,
            lambda: manager.load_model(
                MOCK_MODEL_ID,
                {
                    "device": "cpu",
                    "precision": "fp32",
                    "latency_ms": 0.0,
                    "simulate_oom": True,
                },
            ),
            CommandType.LOAD_MODEL,
        )
        _wait_for_ack(qtbot, manager, manager.resume, CommandType.RESUME)

        image = np.zeros((12, 16, 3), dtype=np.uint8)
        with qtbot.waitSignal(manager.result_ready, timeout=10_000) as signal:
            assert manager.submit_frame(5, 0, image, point_budget=100)
        result = signal.args[0]

        assert result.depth is None
        assert result.failure is not None
        assert result.failure.kind is FailureKind.CUDA_OUT_OF_MEMORY
        assert result.failure.requires_worker_restart
        assert manager._process is process
        assert manager.is_running

        reload_ack = _wait_for_ack(
            qtbot,
            manager,
            lambda: manager.load_model(
                MOCK_MODEL_ID,
                {"device": "cpu", "precision": "fp32", "latency_ms": 0.0},
            ),
            CommandType.LOAD_MODEL,
        )
        assert reload_ack.success
        assert manager._process is process
    finally:
        manager.shutdown(timeout_s=3.0)


def test_spawned_worker_reports_structured_warmup_failure_and_can_retry(qtbot) -> None:
    manager = InferenceProcessManager(stop_timeout_ms=3_000)
    try:
        manager.start_worker()
        qtbot.waitUntil(lambda: manager.is_running, timeout=5_000)
        process = manager._process

        with qtbot.waitSignal(manager.ack_received, timeout=10_000) as signal:
            manager.load_model(
                MOCK_MODEL_ID,
                {
                    "device": "cpu",
                    "precision": "fp32",
                    "latency_ms": 0.0,
                    "simulate_warmup_error": "warm-up failed deliberately",
                },
            )
        ack = signal.args[0]
        assert not ack.success
        assert ack.failure is not None
        assert ack.failure.kind is FailureKind.WARMUP_FAILED
        assert ack.failure.stage.value == "warmup"
        assert manager._process is process
        assert manager.is_running

        retry_ack = _wait_for_ack(
            qtbot,
            manager,
            lambda: manager.load_model(
                MOCK_MODEL_ID,
                {"device": "cpu", "precision": "fp32", "latency_ms": 0.0},
            ),
            CommandType.LOAD_MODEL,
        )
        assert retry_ack.success
        assert manager._process is process
    finally:
        manager.shutdown(timeout_s=3.0)
