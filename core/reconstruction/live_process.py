"""Native live-SLAM worker process-boundary contracts for Phase 8.4A.

This module does not spawn stella_vslam yet. It fixes the lifecycle and endpoint
contract that Phase 8.4B will implement with a native executable, a Unix-domain
socket control channel, and the shared-memory frame ring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.contracts.validation import require_finite, require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.live_protocol import LIVE_SLAM_PROTOCOL_VERSION


class LiveSlamWorkerState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


_ALLOWED_WORKER_TRANSITIONS: dict[LiveSlamWorkerState, frozenset[LiveSlamWorkerState]] = {
    LiveSlamWorkerState.CREATED: frozenset(
        {LiveSlamWorkerState.STARTING, LiveSlamWorkerState.STOPPED, LiveSlamWorkerState.FAILED}
    ),
    LiveSlamWorkerState.STARTING: frozenset(
        {LiveSlamWorkerState.READY, LiveSlamWorkerState.STOPPING, LiveSlamWorkerState.FAILED}
    ),
    LiveSlamWorkerState.READY: frozenset(
        {LiveSlamWorkerState.RUNNING, LiveSlamWorkerState.STOPPING, LiveSlamWorkerState.FAILED}
    ),
    LiveSlamWorkerState.RUNNING: frozenset(
        {LiveSlamWorkerState.READY, LiveSlamWorkerState.STOPPING, LiveSlamWorkerState.FAILED}
    ),
    LiveSlamWorkerState.STOPPING: frozenset(
        {LiveSlamWorkerState.STOPPED, LiveSlamWorkerState.FAILED}
    ),
    LiveSlamWorkerState.STOPPED: frozenset(),
    LiveSlamWorkerState.FAILED: frozenset(
        {LiveSlamWorkerState.STOPPING, LiveSlamWorkerState.STOPPED}
    ),
}


@dataclass(frozen=True, slots=True)
class LiveSlamWorkerProcessSpec:
    backend_id: str
    executable_path: str
    unix_socket_path: str
    startup_timeout_s: float = 10.0
    shutdown_timeout_s: float = 5.0
    protocol_version: int = LIVE_SLAM_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend_id", require_nonempty(self.backend_id, field_name="backend_id")
        )
        executable = Path(require_nonempty(self.executable_path, field_name="executable_path"))
        socket_path = Path(require_nonempty(self.unix_socket_path, field_name="unix_socket_path"))
        if not executable.is_absolute() or not socket_path.is_absolute():
            raise ReconstructionValidationError(
                "Live SLAM native executable and Unix socket paths must be absolute."
            )
        if len(str(socket_path).encode("utf-8")) >= 100:
            raise ReconstructionValidationError(
                "Live SLAM Unix socket path must remain short enough for portable AF_UNIX use."
            )
        startup = require_finite(self.startup_timeout_s, field_name="startup_timeout_s")
        shutdown = require_finite(self.shutdown_timeout_s, field_name="shutdown_timeout_s")
        if startup <= 0.0 or shutdown <= 0.0:
            raise ReconstructionValidationError("Live SLAM process timeouts must be positive.")
        if self.protocol_version != LIVE_SLAM_PROTOCOL_VERSION:
            raise ReconstructionValidationError("Unsupported live SLAM worker protocol version.")
        object.__setattr__(self, "executable_path", str(executable))
        object.__setattr__(self, "unix_socket_path", str(socket_path))
        object.__setattr__(self, "startup_timeout_s", startup)
        object.__setattr__(self, "shutdown_timeout_s", shutdown)

    def as_dict(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "executable_path": self.executable_path,
            "unix_socket_path": self.unix_socket_path,
            "startup_timeout_s": self.startup_timeout_s,
            "shutdown_timeout_s": self.shutdown_timeout_s,
            "protocol_version": self.protocol_version,
        }


class LiveSlamWorkerLifecycle:
    """Small deterministic state machine independent from the app runtime state."""

    def __init__(self) -> None:
        self._state = LiveSlamWorkerState.CREATED
        self._failure_reason = ""

    @property
    def state(self) -> LiveSlamWorkerState:
        return self._state

    @property
    def failure_reason(self) -> str:
        return self._failure_reason

    def can_transition_to(self, target: LiveSlamWorkerState) -> bool:
        if not isinstance(target, LiveSlamWorkerState):
            return False
        return target in _ALLOWED_WORKER_TRANSITIONS[self._state]

    def transition_to(self, target: LiveSlamWorkerState) -> None:
        if not isinstance(target, LiveSlamWorkerState):
            raise TypeError("target must be LiveSlamWorkerState.")
        if target not in _ALLOWED_WORKER_TRANSITIONS[self._state]:
            raise ReconstructionValidationError(
                f"Cannot transition live SLAM worker from {self._state.value} to {target.value}."
            )
        self._state = target

    def fail(self, reason: str) -> None:
        reason = require_nonempty(reason, field_name="reason")
        if self._state is LiveSlamWorkerState.STOPPED:
            raise ReconstructionValidationError("A stopped live SLAM worker cannot fail.")
        if self._state is not LiveSlamWorkerState.FAILED:
            if LiveSlamWorkerState.FAILED not in _ALLOWED_WORKER_TRANSITIONS[self._state]:
                raise ReconstructionValidationError(
                    f"Live SLAM worker cannot fail from state {self._state.value}."
                )
            self._state = LiveSlamWorkerState.FAILED
        self._failure_reason = reason
