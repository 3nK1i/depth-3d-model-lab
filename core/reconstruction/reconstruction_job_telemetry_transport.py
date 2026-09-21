"""I6.8D bounded Qt-free transport for the two existing latest-only worker sidecars.

Only passive observations are accepted; this is not completion, admission,
recovery, or persistent scene state.  The Qt controller owns the polling timer.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import os
from pathlib import Path
import stat
from typing import Callable

from core.reconstruction.reconstruction_job_telemetry_contract import (
    JOB_TELEMETRY_MAX_LINE_BYTES, JobTelemetryObservation,
    JobTelemetryStreamGate, TelemetryValidationError, parse_job_telemetry_line,
)

TELEMETRY_HISTORY_LIMIT = 32
TELEMETRY_FRESH_SECONDS = 15.0
TELEMETRY_ACCEPT_MAX_AGE_SECONDS = 30.0
TELEMETRY_FUTURE_SKEW_SECONDS = 5.0
_SIDECARS = ('job_progress_observation.txt', 'job_resource_observation.txt')


class JobTelemetrySidecarTransport:
    """At most two small sidecar reads per poll, one bounded in-memory history.

    The directory is bound by its owning Qt controller. A sidecar's embedded
    session/run, per-producer sequence, and source timestamp are checked; even
    an old file copied into a new run is not accepted as that run's telemetry.
    """

    def __init__(self, job_kind: str, source_session_id: str,
                 dense_run_id: str | None, directory: Path,
                 *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.reset(job_kind, source_session_id, dense_run_id, directory)

    def reset(self, job_kind: str, source_session_id: str,
              dense_run_id: str | None, directory: Path) -> None:
        self._gate = JobTelemetryStreamGate(job_kind, source_session_id, dense_run_id)
        self._directory = Path(directory)
        self._history: deque[JobTelemetryObservation] = deque(maxlen=TELEMETRY_HISTORY_LIMIT)
        self._latest: dict[str, JobTelemetryObservation] = {}
        self.malformed_count = 0
        self.stale_count = 0

    @property
    def history(self) -> tuple[JobTelemetryObservation, ...]:
        return tuple(self._history)

    @property
    def closed(self) -> bool:
        return self._gate.closed

    def close(self) -> None:
        self._gate.close()

    def latest_state(self, producer: str) -> str:
        if self.closed:
            return 'closed'
        event = self._latest.get(producer)
        if event is None:
            return 'unavailable'
        age = (self._now() - self._instant(event)).total_seconds()
        return 'current' if -TELEMETRY_FUTURE_SKEW_SECONDS <= age <= TELEMETRY_FRESH_SECONDS else 'stale'

    def latest(self, producer: str) -> JobTelemetryObservation | None:
        return self._latest.get(producer) if self.latest_state(producer) == 'current' else None

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError('telemetry clock must return an aware datetime')
        return now.astimezone(timezone.utc)

    @staticmethod
    def _instant(event: JobTelemetryObservation) -> datetime:
        return datetime.fromisoformat(event.observed_at_utc.replace('Z', '+00:00'))

    def _read(self, path: Path) -> JobTelemetryObservation | None:
        """Never follow a symlink; cap bytes even if a file changes under us."""
        fd = None
        try:
            flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
            fd = os.open(path, flags)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > JOB_TELEMETRY_MAX_LINE_BYTES + 1:
                self.malformed_count += 1
                return None
            raw = os.read(fd, JOB_TELEMETRY_MAX_LINE_BYTES + 2)
            if len(raw) > JOB_TELEMETRY_MAX_LINE_BYTES + 1 or not raw.endswith(b'\n'):
                self.malformed_count += 1
                return None
            event = parse_job_telemetry_line(raw[:-1].decode('utf-8'))
            if event is None:
                self.malformed_count += 1
            return event
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, ValueError, TelemetryValidationError):
            self.malformed_count += 1
            return None
        finally:
            if fd is not None:
                os.close(fd)

    def poll(self) -> tuple[JobTelemetryObservation, ...]:
        if self.closed:
            return ()
        incoming: list[JobTelemetryObservation] = []
        now = self._now()
        for name in _SIDECARS:
            event = self._read(self._directory / name)
            if event is None:
                continue
            age = (now - self._instant(event)).total_seconds()
            if not -TELEMETRY_FUTURE_SKEW_SECONDS <= age <= TELEMETRY_ACCEPT_MAX_AGE_SECONDS:
                self.stale_count += 1
                continue
            if not self._gate.offer(event):
                continue
            self._latest[event.producer] = event
            self._history.append(event)
            incoming.append(event)
        return tuple(incoming)


__all__ = ['JobTelemetrySidecarTransport', 'TELEMETRY_HISTORY_LIMIT',
           'TELEMETRY_FRESH_SECONDS', 'TELEMETRY_ACCEPT_MAX_AGE_SECONDS']
