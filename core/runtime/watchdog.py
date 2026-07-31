"""Timeout detector for commands sent to the inference worker (spec §13).

If the worker doesn't acknowledge a command (e.g. Stop) within the
configured timeout, this fires ``timeout_exceeded`` so the UI can show
"Inference worker is not responding" with Continue Waiting / Restart
Worker options — never an unbounded wait.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.runtime.messages import CommandType

DEFAULT_TIMEOUT_MS = 5000


class CommandWatchdog(QObject):
    """Tracks one in-flight command at a time and fires a timeout signal
    if it isn't acknowledged in time."""

    timeout_exceeded = pyqtSignal(object)  # CommandType that timed out

    def __init__(self, timeout_ms: int = DEFAULT_TIMEOUT_MS, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timeout_ms = timeout_ms
        self._pending_command: CommandType | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

    def watch(self, command_type: CommandType) -> None:
        """Start watching for an ack of ``command_type``. Replaces any
        previously-pending watch (only one in-flight command at a time is
        tracked, matching how commands are sent serially)."""
        self._pending_command = command_type
        self._timer.start(self._timeout_ms)

    def acknowledge(self, command_type: CommandType) -> None:
        """Call when the matching Ack arrives — cancels the timeout."""
        if self._pending_command == command_type:
            self._pending_command = None
            self._timer.stop()

    def cancel(self) -> None:
        self._pending_command = None
        self._timer.stop()

    def _on_timeout(self) -> None:
        if self._pending_command is not None:
            command_type = self._pending_command
            self.timeout_exceeded.emit(command_type)

    def set_timeout_ms(self, timeout_ms: int) -> None:
        self._timeout_ms = timeout_ms

    @property
    def is_watching(self) -> bool:
        return self._pending_command is not None