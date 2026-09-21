"""I6.8D: passive Qt relay for latest-only per-job diagnostic sidecars.

Only the active controller creates/starts this timer. Read/parse is bounded,
no nested job is started, no result/admission path consumes observations.
The main-page presentation is reserved for I6.8E.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.reconstruction.reconstruction_job_telemetry_transport import (
    JobTelemetrySidecarTransport,
)

TELEMETRY_POLL_INTERVAL_MS = 1500


class JobTelemetryQtRelay(QObject):
    observation_received = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._transport: JobTelemetrySidecarTransport | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(TELEMETRY_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self.poll)

    def begin(self, job_kind: str, source_session_id: str,
              dense_run_id: str | None, directory: Path) -> None:
        self.close()
        try:
            self._transport = JobTelemetrySidecarTransport(
                job_kind, source_session_id, dense_run_id, directory)
            self._timer.start()
        except (OSError, ValueError, TypeError):
            self._transport = None  # telemetry never controls job launch

    def poll(self) -> None:
        transport = self._transport
        if transport is None or transport.closed:
            return
        try:
            for event in transport.poll():
                # A synchronous UI listener may start a new run during emit.
                # Never emit the remaining old-run observations after a reset.
                if transport is not self._transport or transport.closed:
                    break
                self.observation_received.emit(event)
        except (OSError, ValueError, TypeError):
            # Fail-soft: no telemetry parse/sensor issue can alter SfM or dense.
            return

    def close(self) -> None:
        self._timer.stop()
        if self._transport is not None:
            self._transport.close()

    def latest_state(self, producer: str) -> str:
        if self._transport is None:
            return 'unavailable'
        return self._transport.latest_state(producer)

    def latest(self, producer: str):
        return self._transport.latest(producer) if self._transport is not None else None

    @property
    def history(self) -> tuple:
        return self._transport.history if self._transport is not None else ()


__all__ = ['JobTelemetryQtRelay', 'TELEMETRY_POLL_INTERVAL_MS']
