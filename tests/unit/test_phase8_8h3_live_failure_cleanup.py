from __future__ import annotations

from threading import Event
from time import monotonic, sleep

import numpy as np

from core.capture.frame import SourceFrame
from core.reconstruction import (
    DeterministicMockLiveSlamBackend,
    LiveCaptureRuntimeState,
    LiveCaptureSlamRuntime,
    LiveSlamRecoveryPolicy,
)
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


def test_h3_live_failure_keeps_resource_ownership_until_cleanup_finishes() -> None:
    shutdown_entered = Event()
    allow_shutdown = Event()
    states: list[LiveCaptureRuntimeState] = []
    errors: list[str] = []

    class SlowCleanupFailureBackend(DeterministicMockLiveSlamBackend):
        def track_live_frame(self, frame, context):
            raise RuntimeError("synthetic terminal backend failure")

        def shutdown(self) -> None:
            shutdown_entered.set()
            allow_shutdown.wait(timeout=2.0)
            super().shutdown()

    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: SlowCleanupFailureBackend(),
        recovery_policy=LiveSlamRecoveryPolicy(max_worker_restarts=0, restart_backoff_s=0.0),
        state_callback=states.append,
        error_callback=errors.append,
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
    runtime.offer_source_frame(_frame(0))

    assert shutdown_entered.wait(timeout=2.0)
    assert runtime.state is LiveCaptureRuntimeState.STOPPING
    assert runtime.is_active is True
    assert runtime.stats.cleanup_complete is False
    assert LiveCaptureRuntimeState.FAILED not in states
    assert errors and "synthetic terminal backend failure" in errors[-1]

    allow_shutdown.set()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.FAILED, timeout_s=2.0)
    assert runtime.stats.cleanup_complete is True
    assert runtime.is_active is False
    assert states[-1] is LiveCaptureRuntimeState.FAILED
    assert LiveCaptureRuntimeState.STOPPING in states


def test_h3_failed_state_is_published_only_after_cleanup_flag_is_true() -> None:
    cleanup_at_failed: list[bool] = []
    runtime_box: list[LiveCaptureSlamRuntime] = []

    class FailureBackend(DeterministicMockLiveSlamBackend):
        def track_live_frame(self, frame, context):
            raise RuntimeError("h3 fail closed")

    def on_state(state: LiveCaptureRuntimeState) -> None:
        if state is LiveCaptureRuntimeState.FAILED:
            cleanup_at_failed.append(runtime_box[0].stats.cleanup_complete)

    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: FailureBackend(),
        recovery_policy=LiveSlamRecoveryPolicy(max_worker_restarts=0, restart_backoff_s=0.0),
        state_callback=on_state,
    )
    runtime_box.append(runtime)
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
    runtime.offer_source_frame(_frame(0))
    assert runtime.wait_for_state(LiveCaptureRuntimeState.FAILED, timeout_s=2.0)

    assert cleanup_at_failed == [True]
    assert runtime.stats.cleanup_complete is True


def test_h3_normal_stop_still_finishes_stopped_with_cleanup_complete() -> None:
    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: DeterministicMockLiveSlamBackend(),
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
    assert runtime.stop(timeout_s=2.0)
    assert _wait_until(lambda: runtime.state is LiveCaptureRuntimeState.STOPPED)
    assert runtime.stats.cleanup_complete is True
    assert runtime.is_active is False
