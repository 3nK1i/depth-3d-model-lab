"""Owns the inference worker process and its IPC (spec §12, §13).

The UI thread never blocks on the worker: all polling happens on a
QTimer, all commands are fire-and-forget over the pipe (acks arrive
asynchronously via the same polling), and the frame queue is always
drained-then-filled so it can never carry a backlog. This is the class
:class:`~app.main_window.MainWindow`'s real (non-demo) runtime lifecycle
is built on.

Uses ``multiprocessing.get_context("spawn")`` explicitly rather than the
platform default — spawn is required for CUDA safety (a forked process
inherits a CUDA context in a broken state) and gives consistent behavior
across platforms.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
from multiprocessing.connection import Connection
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.inference.worker import worker_main
from core.runtime.messages import (
    Ack,
    Command,
    CommandType,
    FrameRequest,
    InferenceResult,
    ModelStageEvent,
)
from core.runtime.watchdog import DEFAULT_TIMEOUT_MS, CommandWatchdog

_POLL_INTERVAL_MS = 20
_FRAME_QUEUE_SIZE = 1
_RESULT_QUEUE_SIZE = 1


class InferenceProcessManager(QObject):
    """Manages the lifecycle of one inference worker process."""

    ack_received = pyqtSignal(object)  # Ack
    result_ready = pyqtSignal(object)  # InferenceResult
    model_stage_changed = pyqtSignal(object)  # ModelStageEvent
    worker_unresponsive = pyqtSignal(object)  # CommandType that timed out
    worker_crashed = pyqtSignal()

    def __init__(self, stop_timeout_ms: int = DEFAULT_TIMEOUT_MS, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._ctx = mp.get_context("spawn")
        self._process: mp.process.BaseProcess | None = None
        self._parent_conn: Connection | None = None
        self._frame_queue: Any = None
        self._result_queue: Any = None

        self._watchdog = CommandWatchdog(timeout_ms=stop_timeout_ms, parent=self)
        self._watchdog.timeout_exceeded.connect(self.worker_unresponsive.emit)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll)

    # -- Lifecycle -----------------------------------------------------------

    def start_worker(self) -> None:
        """Spawn a fresh worker process with fresh IPC. Safe to call
        again after :meth:`restart_worker` or an initial cold start."""
        if self._process is not None:
            return

        self._parent_conn, child_conn = self._ctx.Pipe()
        self._frame_queue = self._ctx.Queue(maxsize=_FRAME_QUEUE_SIZE)
        self._result_queue = self._ctx.Queue(maxsize=_RESULT_QUEUE_SIZE)

        self._process = self._ctx.Process(
            target=worker_main,
            args=(child_conn, self._frame_queue, self._result_queue),
            daemon=True,
        )
        self._process.start()
        # The parent must close its duplicate of the child endpoint. Keeping
        # both ends open can delay EOF detection and leak descriptors across
        # worker restarts.
        child_conn.close()
        self._poll_timer.start()

    def restart_worker(self) -> None:
        """Terminate ONLY the inference process and recreate it with
        fresh IPC (spec §13). The UI/main process is never affected."""
        self._poll_timer.stop()
        self._watchdog.cancel()
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._dispose_ipc()
        self.start_worker()

    def shutdown(self, timeout_s: float = 3.0) -> None:
        """Ask the worker to exit; terminate it if it doesn't in time."""
        if self._process is None:
            return
        try:
            if self._parent_conn is not None:
                self._parent_conn.send(Command(type=CommandType.SHUTDOWN))
        except (BrokenPipeError, OSError):
            pass
        self._process.join(timeout=timeout_s)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._poll_timer.stop()
        self._watchdog.cancel()
        self._dispose_ipc()

    # -- Commands (fire-and-forget; results arrive via signals) --------------

    def load_model(self, model_id: str, config: dict[str, Any] | None = None) -> None:
        self._send(Command(type=CommandType.LOAD_MODEL, model_id=model_id, config=config or {}))

    def unload_model(self) -> None:
        self._send(Command(type=CommandType.UNLOAD_MODEL), watch=True)

    def pause(self) -> None:
        self._send(Command(type=CommandType.PAUSE))

    def resume(self) -> None:
        self._send(Command(type=CommandType.RESUME))

    def stop(self, session_id: int = 0) -> None:
        """Send Stop and watch for the ack. The caller (application
        controller) is responsible for having already invalidated the
        session and stopped submitting frames — this just tells the
        worker to drain its queue and pause."""
        self._send(Command(type=CommandType.STOP, session_id=session_id), watch=True)

    def submit_frame(
        self,
        session_id: int,
        frame_id: int,
        image,
        *,
        point_budget: int | None = None,
        submitted_at: float | None = None,
        source_timestamp: float | None = None,
    ) -> bool:
        """Submit one frame without ever allowing a queue backlog.

        Returns ``True`` only when the frame was accepted. The UI-side
        controller also permits at most one request in flight; this bounded
        queue remains a second safety net for restarts or unusual races.
        """
        if self._frame_queue is None:
            return False
        try:
            self._frame_queue.get_nowait()
        except queue_module.Empty:
            pass
        try:
            self._frame_queue.put_nowait(
                FrameRequest(
                    session_id,
                    frame_id,
                    image,
                    point_budget=point_budget,
                    submitted_at=submitted_at,
                    source_timestamp=source_timestamp,
                )
            )
        except queue_module.Full:
            return False
        return True

    # -- Internals -------------------------------------------------------------

    def _send(self, command: Command, *, watch: bool = False) -> None:
        if self._parent_conn is None:
            return
        try:
            self._parent_conn.send(command)
        except (BrokenPipeError, OSError):
            self.worker_crashed.emit()
            return
        if watch:
            self._watchdog.watch(command.type)

    def _poll(self) -> None:
        if self._parent_conn is not None:
            try:
                while self._parent_conn.poll(0):
                    message = self._parent_conn.recv()
                    if isinstance(message, Ack):
                        self._watchdog.acknowledge(message.command_type)
                        self.ack_received.emit(message)
                    elif isinstance(message, ModelStageEvent):
                        self.model_stage_changed.emit(message)
            except (BrokenPipeError, OSError, EOFError):
                self.worker_crashed.emit()

        if self._result_queue is not None:
            try:
                while True:
                    result: InferenceResult = self._result_queue.get_nowait()
                    self.result_ready.emit(result)
            except queue_module.Empty:
                pass

        if self._process is not None and not self._process.is_alive():
            self.worker_crashed.emit()
            self._poll_timer.stop()

    def _dispose_ipc(self) -> None:
        """Close and clear process/pipe/queue resources after shutdown."""
        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except OSError:
                pass

        for q in (self._frame_queue, self._result_queue):
            if q is None:
                continue
            try:
                q.close()
                q.cancel_join_thread()
            except (OSError, ValueError):
                pass

        self._process = None
        self._parent_conn = None
        self._frame_queue = None
        self._result_queue = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()
