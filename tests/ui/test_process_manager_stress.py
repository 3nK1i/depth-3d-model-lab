"""Teardown and restart hardening tests for the Qt process manager."""

from __future__ import annotations

from core.inference.process_manager import InferenceProcessManager


class _Connection:
    def __init__(self) -> None:
        self.sent = []
        self.closed = False

    def send(self, item) -> None:
        self.sent.append(item)

    def close(self) -> None:
        self.closed = True


class _Queue:
    def __init__(self) -> None:
        self.closed = False
        self.cancelled = False

    def close(self) -> None:
        self.closed = True

    def cancel_join_thread(self) -> None:
        self.cancelled = True


class _StubbornProcess:
    def __init__(self, pid: int = 1234) -> None:
        self.pid = pid
        self.exitcode = None
        self.alive = True
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False
        self.exitcode = -9

    def join(self, timeout=None) -> None:
        self.join_calls.append(timeout)


class _GracefulProcess(_StubbornProcess):
    def join(self, timeout=None) -> None:
        super().join(timeout)
        self.alive = False
        self.exitcode = 0


class _BrokenResultQueue:
    def get_nowait(self):
        raise ValueError("queue is closed")


def _install_ipc(manager, process):
    connection = _Connection()
    frame_queue = _Queue()
    result_queue = _Queue()
    manager._process = process
    manager._parent_conn = connection
    manager._frame_queue = frame_queue
    manager._result_queue = result_queue
    return connection, frame_queue, result_queue


def test_restart_escalates_to_kill_and_clears_old_ipc(qapp, monkeypatch) -> None:
    manager = InferenceProcessManager()
    process = _StubbornProcess()
    connection, frame_queue, result_queue = _install_ipc(manager, process)
    starts: list[bool] = []
    monkeypatch.setattr(manager, "start_worker", lambda: starts.append(True))

    manager.restart_worker()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert starts == [True]
    assert manager.restart_count == 1
    assert connection.closed
    assert frame_queue.closed and frame_queue.cancelled
    assert result_queue.closed and result_queue.cancelled
    assert manager._process is None
    assert manager._parent_conn is None


def test_shutdown_is_idempotent_and_never_reports_intentional_crash(qapp) -> None:
    manager = InferenceProcessManager()
    process = _GracefulProcess()
    connection, frame_queue, result_queue = _install_ipc(manager, process)
    crashes: list[bool] = []
    manager.worker_crashed.connect(lambda: crashes.append(True))

    manager.shutdown(timeout_s=0.1)
    manager.shutdown(timeout_s=0.1)

    assert crashes == []
    assert connection.sent and connection.sent[-1].type.value == "shutdown"
    assert connection.closed
    assert frame_queue.closed and result_queue.closed
    assert manager._process is None


def test_closed_result_queue_emits_only_one_crash_signal(qapp) -> None:
    manager = InferenceProcessManager()
    manager._result_queue = _BrokenResultQueue()
    manager._process = _StubbornProcess()
    crashes: list[bool] = []
    manager.worker_crashed.connect(lambda: crashes.append(True))

    manager._poll()
    manager._poll()

    assert crashes == [True]
