"""Abstract base for all capture sources (image/video/camera/RTSP).

Every source runs its acquisition loop on its own QThread (never the UI
thread) and emits frames through a Qt signal, which is thread-safe by
construction — no manual locking needed to get a frame from the capture
thread to the UI.

The "latest-frame-only" rule (spec §5/§12: never let a backlog build up)
is enforced structurally: sources emit ``frame_ready`` directly rather
than pushing into any queue, so there is nothing to accumulate. A slow
consumer simply misses intermediate frames rather than falling behind on
an unbounded queue.
"""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.capture.frame import SourceFrame


class SourceStatus(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"


class BaseSource(QObject):
    """Common lifecycle + signals for every capture source.

    Subclasses implement :meth:`_run` (the acquisition loop, executed on
    a worker thread) and call :meth:`_emit_frame` for each captured
    frame. The public API (`start`/`stop`) is what
    :class:`~core.capture.source_controller.SourceController` calls; it
    never touches thread internals directly. ``_run`` is a plain method
    (not ``abstractmethod``) since ``ABCMeta`` conflicts with QObject's
    metaclass in PyQt6 — see the note on ``_run`` below.
    """

    frame_ready = pyqtSignal(object)  # SourceFrame
    status_changed = pyqtSignal(object)  # SourceStatus
    error_occurred = pyqtSignal(str)
    fps_measured = pyqtSignal(float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._status = SourceStatus.IDLE
        self._frame_id = 0
        self._stop_requested = False

    # -- Public lifecycle ---------------------------------------------------

    def start(self) -> None:
        """Start acquisition on a dedicated worker thread."""
        if self._thread is not None:
            return  # already running
        self._stop_requested = False
        self._frame_id = 0
        self._set_status(SourceStatus.CONNECTING)

        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()

    def stop(self) -> None:
        """Request a cooperative stop and wait briefly for the thread to
        exit. Never uses ``QThread.terminate()`` (spec §13)."""
        self._stop_requested = True
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
        self._set_status(SourceStatus.STOPPED)

    @property
    def status(self) -> SourceStatus:
        return self._status

    # -- Subclass contract ---------------------------------------------------

    @staticmethod
    def _not_implemented() -> None:
        raise NotImplementedError

    def _run(self) -> None:
        """The acquisition loop. Runs on the worker thread. Must check
        ``self._stop_requested`` regularly and return promptly when set.

        Not an ``abstractmethod`` — QObject's metaclass conflicts with
        ``ABCMeta`` in PyQt6, so this is enforced at call time instead of
        at class-definition time. Every concrete subclass overrides it.
        """
        self._not_implemented()

    # -- Helpers for subclasses -----------------------------------------------

    def _emit_frame(self, frame_image) -> None:
        frame = SourceFrame(image=frame_image, frame_id=self._frame_id)
        self._frame_id += 1
        self.frame_ready.emit(frame)

    def _set_status(self, status: SourceStatus) -> None:
        self._status = status
        self.status_changed.emit(status)

    def _emit_error(self, message: str) -> None:
        self._set_status(SourceStatus.ERROR)
        self.error_occurred.emit(message)