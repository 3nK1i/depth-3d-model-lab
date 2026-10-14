"""Standalone bounded process manager for the separate perception pipeline."""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any

from core.perception.messages import (
    PerceptionPipelineRequest,
    PerceptionShutdownRequest,
    PerceptionWorkerConfiguration,
)
from core.perception.worker import perception_worker_main

_QUEUE_TRANSPORT_ERRORS = (EOFError, OSError, TypeError, ValueError)


class PerceptionProcessManager:
    """Own one perception worker without depending on Qt or the depth runtime."""

    def __init__(self, *, context_name: str = "spawn") -> None:
        self._ctx = mp.get_context(context_name)
        self._process: mp.process.BaseProcess | None = None
        self._input: Any = None
        self._output: Any = None
        self.dropped_requests = 0

    @property
    def is_alive(self) -> bool:
        return bool(self._process and self._process.is_alive())

    def start(self, configuration: PerceptionWorkerConfiguration) -> None:
        if self.is_alive:
            raise RuntimeError("Perception process is already running.")

        self._input = self._ctx.Queue(maxsize=1)
        self._output = self._ctx.Queue(maxsize=8)
        self._process = self._ctx.Process(
            target=perception_worker_main,
            args=(configuration, self._input, self._output),
            name="depth3d-perception-worker",
            daemon=True,
        )
        try:
            self._process.start()
        except Exception:
            self._close_queues()
            self._process = None
            raise

    def submit_latest(self, request: PerceptionPipelineRequest) -> bool:
        """Submit a frame request while retaining at most the newest queued item."""

        if not self.is_alive or self._input is None:
            raise RuntimeError("Perception process is not running.")

        self.dropped_requests += _drain(self._input)
        try:
            self._input.put_nowait(request)
            return True
        except queue.Full:
            # A multiprocessing feeder-thread race may report Full immediately
            # after a drain. Retry once without ever blocking the caller.
            self.dropped_requests += _drain(self._input)
            try:
                self._input.put_nowait(request)
                return True
            except queue.Full:
                self.dropped_requests += 1
                return False
        except _QUEUE_TRANSPORT_ERRORS as exc:
            raise RuntimeError("Perception request queue is unavailable.") from exc

    def get(self, timeout: float | None = None):
        if self._output is None:
            raise RuntimeError("Perception process is not running.")
        return self._output.get(timeout=timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        if self._process is None:
            return

        if self._process.is_alive() and self._input is not None:
            _drain(self._input)
            try:
                self._input.put_nowait(PerceptionShutdownRequest())
            except (queue.Full, *_QUEUE_TRANSPORT_ERRORS):
                pass

        self._process.join(timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(2.0)
        self._close_queues()
        self._process = None

    def _close_queues(self) -> None:
        for value in (self._input, self._output):
            if value is None:
                continue
            try:
                value.close()
                value.join_thread()
            except Exception:
                pass
        self._input = None
        self._output = None


def _drain(value: Any) -> int:
    """Drain a multiprocessing queue without blocking and return the count."""

    count = 0
    while True:
        try:
            value.get_nowait()
            count += 1
        except queue.Empty:
            return count
        except _QUEUE_TRANSPORT_ERRORS:
            return count
