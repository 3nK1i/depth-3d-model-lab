"""Own the inference worker process and its IPC (spec §12, §13).

The UI thread never blocks on the worker: polling happens on a ``QTimer``,
commands are fire-and-forget over the control pipe, and both frame/result
queues are latest-only.  The worker is always created with the ``spawn``
context because CUDA state inherited through ``fork`` is unsafe.

Phase 5 Step 4.3 hardens repeated restart/shutdown behaviour.  Every worker
spawn receives a monotonically increasing generation, dead children are
reaped, stubborn children receive terminate/kill fallbacks, and late poll
callbacks are suppressed during intentional shutdown.
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
_MODEL_LOAD_TIMEOUT_MS = 180_000
_TERMINATE_TIMEOUT_S = 2.0
_KILL_TIMEOUT_S = 1.0
_QUEUE_TRANSPORT_ERRORS = (EOFError, OSError, TypeError, ValueError)


class InferenceProcessManager(QObject):
    """Manage one isolated inference worker and its IPC resources."""

    ack_received = pyqtSignal(object)  # Ack
    result_ready = pyqtSignal(object)  # InferenceResult
    model_stage_changed = pyqtSignal(object)  # ModelStageEvent
    worker_unresponsive = pyqtSignal(object)  # CommandType that timed out
    worker_crashed = pyqtSignal()

    def __init__(
        self,
        stop_timeout_ms: int = DEFAULT_TIMEOUT_MS,
        model_load_timeout_ms: int = _MODEL_LOAD_TIMEOUT_MS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._ctx = mp.get_context("spawn")
        self._model_load_timeout_ms = max(1, int(model_load_timeout_ms))
        self._process: mp.process.BaseProcess | None = None
        self._parent_conn: Connection | None = None
        self._frame_queue: Any = None
        self._result_queue: Any = None
        self._crash_reported = False
        self._last_exitcode: int | None = None
        self._generation = 0
        self._restart_count = 0
        self._shutting_down = False

        self._watchdog = CommandWatchdog(timeout_ms=stop_timeout_ms, parent=self)
        self._watchdog.timeout_exceeded.connect(self.worker_unresponsive.emit)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll)

    # -- Lifecycle -----------------------------------------------------------

    def start_worker(self) -> None:
        """Spawn a fresh worker with fresh queues and a fresh control pipe.

        Calling this while a live worker exists is a no-op.  A stale/dead
        ``Process`` object is disposed first so repeated recovery cycles never
        accumulate pipe descriptors or zombie children.
        """
        if self._process is not None:
            if self._process.is_alive():
                return
            self._reap_process(self._process)
            self._dispose_ipc()

        self._shutting_down = False
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._frame_queue = self._ctx.Queue(maxsize=_FRAME_QUEUE_SIZE)
        self._result_queue = self._ctx.Queue(maxsize=_RESULT_QUEUE_SIZE)

        process = self._ctx.Process(
            target=worker_main,
            args=(child_conn, self._frame_queue, self._result_queue),
            daemon=True,
        )
        self._process = process
        try:
            process.start()
        except Exception:
            # The parent owns all resources created above; ensure a failed spawn
            # leaves the manager reusable rather than half-initialised.
            child_conn.close()
            self._dispose_ipc()
            raise

        self._generation += 1
        self._crash_reported = False
        self._last_exitcode = None
        # The parent must close its duplicate child endpoint. Keeping it open can
        # delay EOF detection and leak descriptors across restarts.
        child_conn.close()
        self._poll_timer.start()

    def restart_worker(self) -> None:
        """Replace only the inference process and all of its IPC resources."""
        self._restart_count += 1
        self._poll_timer.stop()
        self._watchdog.cancel()
        self._shutting_down = True
        process = self._process
        if process is not None:
            self._terminate_process(process)
        self._dispose_ipc()
        self._shutting_down = False
        self.start_worker()

    def shutdown(self, timeout_s: float = 3.0) -> None:
        """Request a clean worker exit, then terminate/kill as a fallback.

        The method is idempotent and the same manager may be started again after
        a source disconnect.  Intentional shutdown never emits ``worker_crashed``.
        """
        process = self._process
        if process is None:
            self._poll_timer.stop()
            self._watchdog.cancel()
            return

        self._shutting_down = True
        self._poll_timer.stop()
        self._watchdog.cancel()
        try:
            if self._parent_conn is not None and process.is_alive():
                self._parent_conn.send(Command(type=CommandType.SHUTDOWN))
        except (BrokenPipeError, EOFError, OSError, TypeError, ValueError):
            pass

        self._wait_for_exit_or_terminate(process, timeout_s=max(0.0, float(timeout_s)))
        self._dispose_ipc()
        self._shutting_down = False

    # -- Commands (fire-and-forget; results arrive via signals) --------------

    def load_model(self, model_id: str, config: dict[str, Any] | None = None) -> None:
        self._send(
            Command(type=CommandType.LOAD_MODEL, model_id=model_id, config=config or {}),
            watch=True,
            timeout_ms=self._model_load_timeout_ms,
        )

    def unload_model(self) -> None:
        self._send(Command(type=CommandType.UNLOAD_MODEL), watch=True)

    def set_command_timeout_ms(self, timeout_ms: int) -> None:
        """Set the generic watched-command timeout for the active model."""

        self._watchdog.set_timeout_ms(max(1, int(timeout_ms)))

    def pause(self) -> None:
        self._send(Command(type=CommandType.PAUSE))

    def resume(self) -> None:
        self._send(Command(type=CommandType.RESUME))

    def stop(self, session_id: int = 0) -> None:
        """Send Stop and watch for its acknowledgement."""
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
        """Submit one frame while retaining at most the newest queued frame."""
        if self._frame_queue is None or self._shutting_down:
            return False
        try:
            self._frame_queue.get_nowait()
        except queue_module.Empty:
            pass
        except _QUEUE_TRANSPORT_ERRORS:
            self._emit_worker_crashed_once()
            return False
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
        except _QUEUE_TRANSPORT_ERRORS:
            self._emit_worker_crashed_once()
            return False
        return True

    # -- Internals -----------------------------------------------------------

    def _send(
        self,
        command: Command,
        *,
        watch: bool = False,
        timeout_ms: int | None = None,
    ) -> None:
        if self._parent_conn is None or self._shutting_down:
            return
        try:
            self._parent_conn.send(command)
        except (BrokenPipeError, EOFError, OSError, TypeError, ValueError):
            self._emit_worker_crashed_once()
            return
        if watch:
            self._watchdog.watch(command.type, timeout_ms=timeout_ms)

    def _poll(self) -> None:
        if self._shutting_down:
            return

        connection = self._parent_conn
        if connection is not None:
            try:
                while connection.poll(0):
                    message = connection.recv()
                    if isinstance(message, Ack):
                        self._watchdog.acknowledge(message.command_type)
                        self.ack_received.emit(message)
                    elif isinstance(message, ModelStageEvent):
                        self.model_stage_changed.emit(message)
            except (BrokenPipeError, EOFError, OSError, TypeError, ValueError):
                self._emit_worker_crashed_once()

        result_queue = self._result_queue
        if result_queue is not None and not self._shutting_down:
            try:
                while True:
                    result: InferenceResult = result_queue.get_nowait()
                    self.result_ready.emit(result)
            except queue_module.Empty:
                pass
            except _QUEUE_TRANSPORT_ERRORS:
                self._emit_worker_crashed_once()

        process = self._process
        if process is not None and not process.is_alive():
            self._last_exitcode = process.exitcode
            self._reap_process(process)
            self._emit_worker_crashed_once()
            self._poll_timer.stop()

    def _emit_worker_crashed_once(self) -> None:
        if self._shutting_down or self._crash_reported:
            return
        self._crash_reported = True
        self._watchdog.cancel()
        self.worker_crashed.emit()

    def _wait_for_exit_or_terminate(
        self,
        process: mp.process.BaseProcess,
        *,
        timeout_s: float,
    ) -> None:
        try:
            process.join(timeout=timeout_s)
        except (AssertionError, OSError, ValueError):
            pass
        if process.is_alive():
            self._terminate_process(process)
        else:
            self._reap_process(process)

    def _terminate_process(self, process: mp.process.BaseProcess) -> None:
        """Terminate a worker and escalate to kill if it refuses to exit."""
        if process.is_alive():
            try:
                process.terminate()
            except (AttributeError, OSError, ValueError):
                pass
            try:
                process.join(timeout=_TERMINATE_TIMEOUT_S)
            except (AssertionError, OSError, ValueError):
                pass

        if process.is_alive():
            kill = getattr(process, "kill", None)
            if callable(kill):
                try:
                    kill()
                except (OSError, ValueError):
                    pass
                try:
                    process.join(timeout=_KILL_TIMEOUT_S)
                except (AssertionError, OSError, ValueError):
                    pass

        self._reap_process(process)

    @staticmethod
    def _reap_process(process: mp.process.BaseProcess) -> None:
        """Best-effort non-blocking join for an already-dead child."""
        if process.is_alive():
            return
        try:
            process.join(timeout=0.0)
        except (AssertionError, OSError, ValueError):
            pass

    def _dispose_ipc(self) -> None:
        """Close and clear process/pipe/queue resources after a lifecycle end."""
        connection = self._parent_conn
        frame_queue = self._frame_queue
        result_queue = self._result_queue

        # Clear references before closing. A queued Qt callback that runs during
        # teardown therefore observes an inert manager rather than stale IPC.
        self._process = None
        self._parent_conn = None
        self._frame_queue = None
        self._result_queue = None

        if connection is not None:
            try:
                connection.close()
            except (OSError, TypeError, ValueError):
                pass

        for ipc_queue in (frame_queue, result_queue):
            if ipc_queue is None:
                continue
            try:
                ipc_queue.close()
            except (OSError, TypeError, ValueError):
                pass
            try:
                ipc_queue.cancel_join_thread()
            except (AttributeError, OSError, TypeError, ValueError):
                pass

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def last_exitcode(self) -> int | None:
        return self._last_exitcode

    @property
    def worker_pid(self) -> int | None:
        process = self._process
        return int(process.pid) if process is not None and process.pid is not None else None

    @property
    def generation(self) -> int:
        """Monotonic worker-spawn generation for diagnostics and stress tests."""
        return self._generation

    @property
    def restart_count(self) -> int:
        return self._restart_count
