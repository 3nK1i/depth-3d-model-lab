"""UI-process IPC manager tests with local fake queues and pipes."""

from __future__ import annotations

import queue

import numpy as np

from core.inference.process_manager import InferenceProcessManager
from core.runtime.messages import (
    CommandType,
    FrameRequest,
    ModelStage,
    ModelStageEvent,
)


class _FakeConnection:
    def __init__(self) -> None:
        self.sent = []
        self.closed = False

    def send(self, item) -> None:
        self.sent.append(item)

    def close(self) -> None:
        self.closed = True


class _PollingConnection(_FakeConnection):
    def __init__(self, items) -> None:
        super().__init__()
        self.items = list(items)

    def poll(self, _timeout=0) -> bool:
        return bool(self.items)

    def recv(self):
        return self.items.pop(0)


class _ClosableQueue(queue.Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize)
        self.closed = False
        self.cancelled = False

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        self.cancelled = True


def test_submit_frame_replaces_obsolete_queued_frame(qapp) -> None:
    manager = InferenceProcessManager()
    manager._frame_queue = queue.Queue(maxsize=1)
    manager._frame_queue.put(FrameRequest(1, 1, np.zeros((1, 1, 3), np.uint8)))

    accepted = manager.submit_frame(
        2,
        9,
        np.ones((2, 3, 3), np.uint8),
        point_budget=123,
        submitted_at=4.0,
        source_timestamp=3.5,
    )

    assert accepted
    latest = manager._frame_queue.get_nowait()
    assert latest.session_id == 2
    assert latest.frame_id == 9
    assert latest.point_budget == 123
    assert latest.image.shape == (2, 3, 3)


def test_submit_frame_without_worker_queue_is_rejected(qapp) -> None:
    manager = InferenceProcessManager()
    assert not manager.submit_frame(1, 1, np.zeros((1, 1, 3), np.uint8))


def test_stop_sends_session_id_and_starts_watchdog(qapp) -> None:
    manager = InferenceProcessManager()
    connection = _FakeConnection()
    manager._parent_conn = connection

    manager.stop(session_id=77)

    assert connection.sent[-1].type is CommandType.STOP
    assert connection.sent[-1].session_id == 77
    assert manager._watchdog.is_watching
    manager._watchdog.cancel()


def test_dispose_ipc_closes_and_clears_resources(qapp) -> None:
    manager = InferenceProcessManager()
    connection = _FakeConnection()
    frame_queue = _ClosableQueue()
    result_queue = _ClosableQueue()
    manager._parent_conn = connection
    manager._frame_queue = frame_queue
    manager._result_queue = result_queue
    manager._process = object()  # type: ignore[assignment]

    manager._dispose_ipc()

    assert connection.closed
    assert frame_queue.closed and frame_queue.cancelled
    assert result_queue.closed and result_queue.cancelled
    assert manager._parent_conn is None
    assert manager._frame_queue is None
    assert manager._result_queue is None
    assert manager._process is None


def test_unload_sends_command_and_starts_watchdog(qapp) -> None:
    manager = InferenceProcessManager()
    connection = _FakeConnection()
    manager._parent_conn = connection

    manager.unload_model()

    assert connection.sent[-1].type is CommandType.UNLOAD_MODEL
    assert manager._watchdog.is_watching
    manager._watchdog.cancel()


def test_poll_routes_model_stage_without_treating_it_as_ack(qtbot) -> None:
    manager = InferenceProcessManager()
    event = ModelStageEvent(ModelStage.WARMING_UP, "moge-2", "Priming kernels")
    manager._parent_conn = _PollingConnection([event])

    with qtbot.waitSignal(manager.model_stage_changed, timeout=500) as signal:
        manager._poll()

    assert signal.args == [event]


def test_load_model_starts_long_command_watchdog(qapp) -> None:
    manager = InferenceProcessManager(model_load_timeout_ms=1234)
    connection = _FakeConnection()
    manager._parent_conn = connection

    manager.load_model("moge-2", {"device": "cpu"})

    assert connection.sent[-1].type is CommandType.LOAD_MODEL
    assert manager._watchdog.is_watching
    assert manager._watchdog._timer.interval() == 1234
    manager._watchdog.cancel()


def test_worker_crash_signal_is_emitted_only_once(qtbot) -> None:
    manager = InferenceProcessManager()
    emissions: list[bool] = []
    manager.worker_crashed.connect(lambda: emissions.append(True))

    manager._emit_worker_crashed_once()
    manager._emit_worker_crashed_once()

    assert emissions == [True]


def test_active_model_can_extend_generic_command_timeout(qapp) -> None:
    manager = InferenceProcessManager(stop_timeout_ms=5_000)
    manager.set_command_timeout_ms(120_000)
    assert manager._watchdog._timeout_ms == 120_000
