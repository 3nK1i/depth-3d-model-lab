from __future__ import annotations

import pytest

from core.reconstruction import (
    LIVE_SLAM_PROTOCOL_VERSION,
    LiveSlamWorkerLifecycle,
    LiveSlamWorkerProcessSpec,
    LiveSlamWorkerState,
    ReconstructionValidationError,
)


def test_worker_process_spec_is_absolute_and_versioned() -> None:
    spec = LiveSlamWorkerProcessSpec(
        backend_id="stella-vslam",
        executable_path="/opt/eia/live-slam-worker",
        unix_socket_path="/tmp/eia-live-slam.sock",
    )
    assert spec.protocol_version == LIVE_SLAM_PROTOCOL_VERSION
    assert spec.as_dict()["unix_socket_path"] == "/tmp/eia-live-slam.sock"


def test_worker_process_spec_rejects_relative_paths() -> None:
    with pytest.raises(ReconstructionValidationError, match="absolute"):
        LiveSlamWorkerProcessSpec(
            backend_id="stella-vslam",
            executable_path="relative-worker",
            unix_socket_path="/tmp/eia.sock",
        )


def test_worker_lifecycle_requires_safe_order() -> None:
    lifecycle = LiveSlamWorkerLifecycle()
    lifecycle.transition_to(LiveSlamWorkerState.STARTING)
    lifecycle.transition_to(LiveSlamWorkerState.READY)
    lifecycle.transition_to(LiveSlamWorkerState.RUNNING)
    lifecycle.transition_to(LiveSlamWorkerState.READY)
    lifecycle.transition_to(LiveSlamWorkerState.STOPPING)
    lifecycle.transition_to(LiveSlamWorkerState.STOPPED)
    assert lifecycle.state is LiveSlamWorkerState.STOPPED
    with pytest.raises(ReconstructionValidationError):
        lifecycle.transition_to(LiveSlamWorkerState.RUNNING)


def test_worker_lifecycle_records_failure() -> None:
    lifecycle = LiveSlamWorkerLifecycle()
    lifecycle.transition_to(LiveSlamWorkerState.STARTING)
    lifecycle.fail("backend handshake timeout")
    assert lifecycle.state is LiveSlamWorkerState.FAILED
    assert lifecycle.failure_reason == "backend handshake timeout"
