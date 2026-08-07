from __future__ import annotations

from time import monotonic, sleep

import numpy as np
import pytest

from core.capture.frame import SourceFrame
from core.reconstruction import (
    DeterministicMockLiveSlamBackend,
    LiveCaptureRuntimeState,
    LiveCaptureSlamRuntime,
    LiveFrameStaleError,
    LiveSlamRecoveryPolicy,
    LiveSlamWorkerExecutionError,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request


def _frame(frame_id: int) -> SourceFrame:
    return SourceFrame(
        np.zeros((LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3), dtype=np.uint8),
        frame_id=frame_id,
        timestamp=monotonic() + frame_id * 0.001,
    )


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        if predicate():
            return True
        sleep(0.01)
    return bool(predicate())


def test_recovery_policy_is_bounded_and_serializable() -> None:
    policy = LiveSlamRecoveryPolicy(
        max_worker_restarts=3,
        restart_backoff_s=0.05,
        max_consecutive_stale_frames=4,
    )
    assert policy.as_dict() == {
        "max_worker_restarts": 3,
        "restart_backoff_s": 0.05,
        "max_consecutive_stale_frames": 4,
    }
    with pytest.raises(ReconstructionValidationError):
        LiveSlamRecoveryPolicy(max_worker_restarts=-1)
    with pytest.raises(ReconstructionValidationError):
        LiveSlamRecoveryPolicy(restart_backoff_s=-0.1)


def test_worker_execution_failure_restarts_backend_without_restarting_capture_runtime() -> None:
    created = 0
    states: list[LiveCaptureRuntimeState] = []

    class CrashFirstBackend(DeterministicMockLiveSlamBackend):
        def __init__(self, should_crash: bool) -> None:
            super().__init__(initialization_frames=1)
            self._should_crash = should_crash

        def track_live_frame(self, frame, context):
            if self._should_crash:
                self._should_crash = False
                raise LiveSlamWorkerExecutionError("synthetic worker crash")
            return super().track_live_frame(frame, context)

    def factory(_spec):
        nonlocal created
        created += 1
        return CrashFirstBackend(should_crash=created == 1)

    observations = []
    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=factory,
        recovery_policy=LiveSlamRecoveryPolicy(max_worker_restarts=2, restart_backoff_s=0.0),
        observation_callback=observations.append,
        state_callback=states.append,
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
    runtime.offer_source_frame(_frame(0))
    assert _wait_until(lambda: runtime.stats.worker_restarts == 1)
    assert _wait_until(lambda: runtime.state is LiveCaptureRuntimeState.RUNNING)
    runtime.offer_source_frame(_frame(1))
    runtime.offer_source_frame(_frame(2))
    assert _wait_until(lambda: bool(observations))
    assert created == 2
    assert LiveCaptureRuntimeState.RECOVERING in states
    assert runtime.stats.worker_restart_failures == 0
    assert runtime.stop(timeout_s=2.0)
    assert runtime.stats.cleanup_complete is True


def test_worker_recovery_budget_exhaustion_fails_closed() -> None:
    class AlwaysCrashBackend(DeterministicMockLiveSlamBackend):
        def track_live_frame(self, frame, context):
            raise LiveSlamWorkerExecutionError("always crashes")

    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: AlwaysCrashBackend(),
        recovery_policy=LiveSlamRecoveryPolicy(max_worker_restarts=1, restart_backoff_s=0.0),
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
    runtime.offer_source_frame(_frame(0))
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RECOVERING, timeout_s=2.0) or runtime.stats.worker_restarts == 1
    runtime.offer_source_frame(_frame(1))
    assert runtime.wait_for_state(LiveCaptureRuntimeState.FAILED, timeout_s=2.0)
    assert runtime.stats.worker_restarts == 1
    assert "recovery budget exhausted" in runtime.stats.last_error
    assert _wait_until(lambda: runtime.stats.cleanup_complete)


def test_stale_frames_are_dropped_with_a_bounded_budget() -> None:
    class StaleThenTrackBackend(DeterministicMockLiveSlamBackend):
        def __init__(self) -> None:
            super().__init__(initialization_frames=1)
            self._stale_remaining = 2

        def track_live_frame(self, frame, context):
            if self._stale_remaining:
                self._stale_remaining -= 1
                raise LiveFrameStaleError("synthetic stale frame")
            return super().track_live_frame(frame, context)

    backend = StaleThenTrackBackend()
    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: backend,
        recovery_policy=LiveSlamRecoveryPolicy(
            max_worker_restarts=0,
            restart_backoff_s=0.0,
            max_consecutive_stale_frames=2,
        ),
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
    for index in range(4):
        runtime.offer_source_frame(_frame(index))
        sleep(0.03)
    assert _wait_until(lambda: runtime.stats.processed_frames >= 1)
    assert runtime.stats.stale_frames == 2
    assert runtime.stats.worker_restarts == 0
    assert runtime.stop(timeout_s=2.0)


def test_tracking_loss_recovery_and_latency_telemetry_are_counted() -> None:
    backend = DeterministicMockLiveSlamBackend(
        initialization_frames=1,
        degraded_sequences=(3,),
        lost_sequences=(2,),
    )
    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: backend,
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
    for index in range(5):
        runtime.offer_source_frame(_frame(index))
        sleep(0.025)
    assert _wait_until(lambda: runtime.stats.observations >= 5)
    stats = runtime.stats
    assert stats.tracking_loss_events == 1
    assert stats.tracking_recoveries == 1
    assert stats.last_processing_latency_ns is not None
    assert stats.max_processing_latency_ns >= stats.last_processing_latency_ns
    assert stats.mean_processing_latency_ns is not None
    telemetry = runtime.hardening_telemetry.as_dict()
    assert telemetry["tracking_loss_events"] == 1
    assert telemetry["tracking_recoveries"] == 1
    assert runtime.stop(timeout_s=2.0)
    assert runtime.hardening_telemetry.cleanup_complete is True
