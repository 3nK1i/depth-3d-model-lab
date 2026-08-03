"""Source/runtime toolbar view-model tests."""

from __future__ import annotations

from app.widgets.source_toolbar_snapshot import (
    ConnectionState,
    InferenceState,
    SourceToolbarSnapshot,
    SourceType,
    connection_label,
)


def test_connection_labels_are_explicit() -> None:
    assert connection_label(ConnectionState.DISCONNECTED) == "DISCONNECTED"
    assert connection_label(ConnectionState.CONNECTING) == "CONNECTING"
    assert connection_label(ConnectionState.CONNECTED) == "CONNECTED"
    assert connection_label(ConnectionState.RECONNECTING) == "RECONNECTING"
    assert connection_label(ConnectionState.ERROR) == "ERROR"


def test_disconnected_snapshot_is_safe_initial_state() -> None:
    snapshot = SourceToolbarSnapshot.disconnected()
    assert snapshot.source_type is SourceType.CAMERA
    assert snapshot.connection_state is ConnectionState.DISCONNECTED
    assert snapshot.inference_state is InferenceState.STOPPED
    assert snapshot.device_name is None
    assert snapshot.resolution is None


def test_preview_snapshot_is_deterministic() -> None:
    snapshot = SourceToolbarSnapshot.preview()
    assert snapshot.device_name == "Camera 0"
    assert snapshot.resolution == (1280, 720)
    assert snapshot.fps == 30
    assert snapshot.inference_state is InferenceState.RUNNING
