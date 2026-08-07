"""Native live-SLAM process and Unix-domain-socket boundary contracts.

Phase 8.4B keeps stella_vslam in a separate native process.  This module owns
only dependency-light process/socket plumbing; it never imports or dynamically
loads the native SLAM library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import socket
from time import monotonic

from core.contracts.validation import require_finite, require_nonempty
from core.reconstruction.errors import (
    LiveSlamProtocolError,
    LiveSlamWorkerExecutionError,
    ReconstructionValidationError,
)
from core.reconstruction.live_protocol import (
    LIVE_SLAM_MAX_MESSAGE_BYTES,
    LIVE_SLAM_PROTOCOL_VERSION,
    LiveSlamProtocolMessage,
    decode_live_slam_message,
    encode_live_slam_message,
)


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
    frame_timeout_s: float = 3.0
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
        frame = require_finite(self.frame_timeout_s, field_name="frame_timeout_s")
        shutdown = require_finite(self.shutdown_timeout_s, field_name="shutdown_timeout_s")
        if startup <= 0.0 or frame <= 0.0 or shutdown <= 0.0:
            raise ReconstructionValidationError("Live SLAM process timeouts must be positive.")
        if self.protocol_version != LIVE_SLAM_PROTOCOL_VERSION:
            raise ReconstructionValidationError("Unsupported live SLAM worker protocol version.")
        object.__setattr__(self, "executable_path", str(executable))
        object.__setattr__(self, "unix_socket_path", str(socket_path))
        object.__setattr__(self, "startup_timeout_s", startup)
        object.__setattr__(self, "frame_timeout_s", frame)
        object.__setattr__(self, "shutdown_timeout_s", shutdown)

    def as_dict(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "executable_path": self.executable_path,
            "unix_socket_path": self.unix_socket_path,
            "startup_timeout_s": self.startup_timeout_s,
            "frame_timeout_s": self.frame_timeout_s,
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


class LiveSlamJsonSocketClient:
    """Bounded JSON-lines client for the private native-worker AF_UNIX socket."""

    def __init__(self, sock: socket.socket) -> None:
        if sock.family != socket.AF_UNIX or sock.type != socket.SOCK_STREAM:
            raise LiveSlamProtocolError("Live SLAM control channel must be an AF_UNIX stream socket.")
        self._socket = sock
        self._buffer = bytearray()
        self._closed = False

    @classmethod
    def connect(
        cls,
        unix_socket_path: str | Path,
        *,
        timeout_s: float,
        poll_interval_s: float = 0.02,
    ) -> "LiveSlamJsonSocketClient":
        path = Path(unix_socket_path)
        timeout = require_finite(timeout_s, field_name="timeout_s")
        poll = require_finite(poll_interval_s, field_name="poll_interval_s")
        if timeout <= 0.0 or poll <= 0.0:
            raise ReconstructionValidationError("Socket connect timing values must be positive.")
        deadline = monotonic() + timeout
        last_error: OSError | None = None
        while monotonic() < deadline:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                remaining = max(0.01, deadline - monotonic())
                sock.settimeout(min(poll, remaining))
                sock.connect(str(path))
                sock.settimeout(None)
                return cls(sock)
            except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError) as exc:
                last_error = exc
                sock.close()
                # socket timeouts provide the bounded wait without an extra sleep.
                if not isinstance(exc, socket.timeout):
                    import time

                    time.sleep(min(poll, max(0.0, deadline - monotonic())))
        detail = "" if last_error is None else f": {last_error}"
        raise LiveSlamWorkerExecutionError(
            f"Timed out connecting to isolated live SLAM worker socket{detail}"
        )

    def send_message(self, message: LiveSlamProtocolMessage) -> None:
        self._ensure_open()
        payload = encode_live_slam_message(message)
        try:
            self._socket.sendall(payload)
        except OSError as exc:
            raise LiveSlamWorkerExecutionError(
                "Could not send a message to the isolated live SLAM worker."
            ) from exc

    def receive_message(self, *, timeout_s: float) -> LiveSlamProtocolMessage:
        self._ensure_open()
        timeout = require_finite(timeout_s, field_name="timeout_s")
        if timeout <= 0.0:
            raise ReconstructionValidationError("Live SLAM receive timeout must be positive.")
        deadline = monotonic() + timeout
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self._buffer[: newline + 1])
                del self._buffer[: newline + 1]
                return decode_live_slam_message(raw)
            if len(self._buffer) > LIVE_SLAM_MAX_MESSAGE_BYTES:
                raise LiveSlamProtocolError("Live SLAM worker sent an oversized JSON message.")
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                raise TimeoutError("Timed out waiting for live SLAM worker message.")
            self._socket.settimeout(remaining)
            try:
                chunk = self._socket.recv(min(65536, LIVE_SLAM_MAX_MESSAGE_BYTES + 1))
            except socket.timeout as exc:
                raise TimeoutError("Timed out waiting for live SLAM worker message.") from exc
            except OSError as exc:
                raise LiveSlamWorkerExecutionError(
                    "Live SLAM worker control socket failed while receiving."
                ) from exc
            if not chunk:
                raise LiveSlamWorkerExecutionError(
                    "Isolated live SLAM worker closed the control socket unexpectedly."
                )
            self._buffer.extend(chunk)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise LiveSlamWorkerExecutionError("Live SLAM control socket is closed.")

    def __enter__(self) -> "LiveSlamJsonSocketClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
