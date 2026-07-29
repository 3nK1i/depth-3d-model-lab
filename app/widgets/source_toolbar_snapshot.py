"""Structured preview/state model for the source + runtime toolbar.

Mirrors the pattern used for the performance footer
(:mod:`app.widgets.performance_snapshot`): one dataclass carries every
value the toolbar displays, so the widget can be driven by
:meth:`SourceRuntimeBar.update_snapshot` from either static Phase 1
preview data or a real source/inference controller later, without the
widget itself knowing which.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SourceType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    CAMERA = "camera"
    RTSP = "rtsp"


class ConnectionState(Enum):
    """States :class:`SourceConnectionBadge` supports (spec §8)."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class InferenceState(Enum):
    """Drives which runtime buttons are shown/enabled."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


_CONNECTION_LABELS: dict[ConnectionState, str] = {
    ConnectionState.DISCONNECTED: "DISCONNECTED",
    ConnectionState.CONNECTING: "CONNECTING",
    ConnectionState.CONNECTED: "CONNECTED",
    ConnectionState.RECONNECTING: "RECONNECTING",
    ConnectionState.ERROR: "ERROR",
}


def connection_label(state: ConnectionState) -> str:
    return _CONNECTION_LABELS[state]


@dataclass(frozen=True, slots=True)
class SourceToolbarSnapshot:
    """Everything the source/runtime toolbar displays, in one structure."""

    source_type: SourceType
    device_name: str | None
    resolution: tuple[int, int] | None
    fps: int | None
    connection_state: ConnectionState
    inference_state: InferenceState

    @classmethod
    def disconnected(cls) -> "SourceToolbarSnapshot":
        """Initial/idle state: camera selected, nothing connected yet."""
        return cls(
            source_type=SourceType.CAMERA,
            device_name=None,
            resolution=None,
            fps=None,
            connection_state=ConnectionState.DISCONNECTED,
            inference_state=InferenceState.STOPPED,
        )

    @classmethod
    def preview(cls) -> "SourceToolbarSnapshot":
        """Static, deterministic Phase-1 preview matching the target
        mockup (spec §17). Never randomized or animated."""
        return cls(
            source_type=SourceType.CAMERA,
            device_name="Camera 0",
            resolution=(1280, 720),
            fps=30,
            connection_state=ConnectionState.CONNECTED,
            inference_state=InferenceState.RUNNING,
        )