"""Shared Qt event-pump helpers for command-line worker stress tools."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from PyQt6.QtCore import QCoreApplication, QEventLoop

from core.inference.process_manager import InferenceProcessManager
from core.runtime.messages import Ack, CommandType, InferenceResult, ModelStageEvent

_T = TypeVar("_T")


class WorkerHarness:
    """Drive :class:`InferenceProcessManager` without a GUI window."""

    def __init__(self, *, command_timeout_ms: int = 5_000) -> None:
        self.app = QCoreApplication.instance() or QCoreApplication([])
        self.manager = InferenceProcessManager(stop_timeout_ms=command_timeout_ms)
        self.acks: list[Ack] = []
        self.results: list[InferenceResult] = []
        self.stages: list[ModelStageEvent] = []
        self.crashes = 0
        self.timeouts: list[CommandType] = []
        self.manager.ack_received.connect(self.acks.append)
        self.manager.result_ready.connect(self.results.append)
        self.manager.model_stage_changed.connect(self.stages.append)
        self.manager.worker_crashed.connect(self._record_crash)
        self.manager.worker_unresponsive.connect(self.timeouts.append)

    def _record_crash(self) -> None:
        self.crashes += 1

    def pump(self, duration_s: float = 0.01) -> None:
        deadline = time.monotonic() + max(0.0, float(duration_s))
        while True:
            self.app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
            if time.monotonic() >= deadline:
                return
            time.sleep(0.002)

    def wait_until(
        self,
        predicate: Callable[[], bool],
        *,
        timeout_s: float,
        description: str,
    ) -> None:
        deadline = time.monotonic() + max(0.01, float(timeout_s))
        while not predicate():
            self.pump(0.01)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {description}")

    def wait_for_ack(
        self,
        action: Callable[[], None],
        expected: CommandType,
        *,
        timeout_s: float = 30.0,
        require_success: bool = True,
    ) -> Ack:
        start = len(self.acks)
        action()
        self.wait_until(
            lambda: any(ack.command_type is expected for ack in self.acks[start:]),
            timeout_s=timeout_s,
            description=f"{expected.value} acknowledgement",
        )
        ack = next(ack for ack in self.acks[start:] if ack.command_type is expected)
        if require_success and not ack.success:
            details = ack.failure.technical_details if ack.failure else ack.message
            raise RuntimeError(f"{expected.value} failed: {details}")
        return ack

    def wait_for_result(
        self,
        action: Callable[[], bool],
        *,
        session_id: int,
        frame_id: int,
        timeout_s: float = 30.0,
    ) -> InferenceResult:
        start = len(self.results)
        if not action():
            raise RuntimeError("The worker rejected the frame submission")
        self.wait_until(
            lambda: any(
                result.session_id == session_id and result.frame_id == frame_id
                for result in self.results[start:]
            ),
            timeout_s=timeout_s,
            description=f"result session={session_id} frame={frame_id}",
        )
        result = next(
            result
            for result in self.results[start:]
            if result.session_id == session_id and result.frame_id == frame_id
        )
        if result.error:
            details = result.failure.technical_details if result.failure else result.error
            raise RuntimeError(f"inference failed: {details}")
        return result

    def close(self, *, timeout_s: float = 3.0) -> None:
        self.manager.shutdown(timeout_s=timeout_s)
        self.pump(0.02)
