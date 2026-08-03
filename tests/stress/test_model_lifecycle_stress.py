"""Longer real-process lifecycle loops using only Development Mock."""

from __future__ import annotations

import numpy as np
import pytest

from core.inference.process_manager import InferenceProcessManager
from core.models.catalog import MOCK_MODEL_ID
from core.runtime.messages import CommandType
from core.runtime.stress import wait_for_process_exit

pytestmark = [pytest.mark.integration, pytest.mark.stress]


def _wait_ack(qtbot, manager, action, expected, timeout=10_000):
    with qtbot.waitSignal(manager.ack_received, timeout=timeout) as signal:
        action()
    ack = signal.args[0]
    assert ack.command_type is expected
    assert ack.success, ack.message
    return ack


def test_repeated_mock_load_infer_stop_unload_uses_one_worker(qtbot) -> None:
    manager = InferenceProcessManager(stop_timeout_ms=3_000)
    worker_pid = None
    try:
        manager.start_worker()
        qtbot.waitUntil(lambda: manager.is_running, timeout=5_000)
        worker_pid = manager.worker_pid
        generation = manager.generation
        image = np.zeros((24, 32, 3), dtype=np.uint8)

        for cycle in range(1, 6):
            _wait_ack(
                qtbot,
                manager,
                lambda: manager.load_model(
                    MOCK_MODEL_ID,
                    {"device": "cpu", "precision": "fp32", "latency_ms": 0.0},
                ),
                CommandType.LOAD_MODEL,
            )
            _wait_ack(qtbot, manager, manager.resume, CommandType.RESUME)
            with qtbot.waitSignal(manager.result_ready, timeout=10_000) as result_signal:
                assert manager.submit_frame(cycle, 0, image, point_budget=500)
            result = result_signal.args[0]
            assert result.session_id == cycle
            assert result.error is None
            _wait_ack(
                qtbot,
                manager,
                lambda: manager.stop(cycle),
                CommandType.STOP,
            )
            unload = _wait_ack(
                qtbot,
                manager,
                manager.unload_model,
                CommandType.UNLOAD_MODEL,
            )
            assert unload.metadata and unload.metadata["released"] is True
            assert manager.worker_pid == worker_pid
            assert manager.generation == generation
    finally:
        manager.shutdown(timeout_s=3.0)

    assert not manager.is_running
    assert wait_for_process_exit(worker_pid, timeout_s=3.0)


def test_repeated_worker_restart_reaps_each_previous_child(qtbot) -> None:
    manager = InferenceProcessManager()
    observed = []
    try:
        manager.start_worker()
        qtbot.waitUntil(lambda: manager.is_running, timeout=5_000)
        for expected_generation in range(1, 5):
            assert manager.generation == expected_generation
            old_process = manager._process
            old_pid = manager.worker_pid
            assert old_pid is not None and old_process is not None
            observed.append(old_pid)
            if expected_generation == 4:
                break
            manager.restart_worker()
            qtbot.waitUntil(
                lambda: manager.is_running
                and manager.generation == expected_generation + 1,
                timeout=5_000,
            )
            assert manager._process is not old_process
            assert not old_process.is_alive()
    finally:
        final_pid = manager.worker_pid
        manager.shutdown(timeout_s=3.0)

    assert manager.restart_count == 3
    assert wait_for_process_exit(final_pid, timeout_s=3.0)
