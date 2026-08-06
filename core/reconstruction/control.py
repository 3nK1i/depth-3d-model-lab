"""Cooperative cancellation and pause controls for future long-running backends."""

from __future__ import annotations

from threading import Condition, Event, Lock

from core.reconstruction.errors import ReconstructionCancelledError


class ReconstructionOperationControl:
    """Thread-safe cooperative control shared with a future backend worker."""

    def __init__(self) -> None:
        self._cancel_event = Event()
        self._pause_condition = Condition(Lock())
        self._paused = False

    def request_cancel(self) -> bool:
        """Request cancellation and return whether this changed the state."""

        changed = not self._cancel_event.is_set()
        self._cancel_event.set()
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()
        return changed

    def request_pause(self) -> bool:
        with self._pause_condition:
            if self._cancel_event.is_set() or self._paused:
                return False
            self._paused = True
            return True

    def request_resume(self) -> bool:
        with self._pause_condition:
            if not self._paused:
                return False
            self._paused = False
            self._pause_condition.notify_all()
            return True

    def checkpoint(self, *, wait_timeout_s: float = 0.05) -> None:
        """Cooperative poll point for future backend loops."""

        if self._cancel_event.is_set():
            raise ReconstructionCancelledError("Reconstruction cancellation requested.")
        with self._pause_condition:
            while self._paused and not self._cancel_event.is_set():
                self._pause_condition.wait(timeout=wait_timeout_s)
        if self._cancel_event.is_set():
            raise ReconstructionCancelledError("Reconstruction cancellation requested.")

    @property
    def cancellation_requested(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def paused(self) -> bool:
        with self._pause_condition:
            return self._paused
