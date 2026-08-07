from __future__ import annotations

from multiprocessing import shared_memory
from time import monotonic, sleep

import numpy as np

from core.capture.frame import SourceFrame
from core.reconstruction import (
    DeterministicMockLiveSlamBackend,
    LiveCaptureRuntimeState,
    LiveCaptureSlamRuntime,
    LiveFrameTransportError,
    LiveSharedMemoryFrameRing,
    LiveSlamRecoveryPolicy,
)
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request


def _frame(frame_id: int) -> SourceFrame:
    return SourceFrame(
        np.full(
            (LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3),
            frame_id % 255,
            dtype=np.uint8,
        ),
        frame_id=frame_id,
        timestamp=monotonic() + frame_id * 0.0005,
    )


def test_repeated_runtime_start_stop_unlinks_every_owned_shared_memory_ring() -> None:
    ring_names: list[str] = []

    def factory(spec):
        ring_names.append(spec.shared_memory_name)
        return DeterministicMockLiveSlamBackend(initialization_frames=1)

    for cycle in range(20):
        runtime = LiveCaptureSlamRuntime(
            request=make_live_request(),
            backend_factory=factory,
            recovery_policy=LiveSlamRecoveryPolicy(max_worker_restarts=1, restart_backoff_s=0.0),
        )
        runtime.start()
        assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
        runtime.offer_source_frame(_frame(cycle))
        sleep(0.01)
        assert runtime.stop(timeout_s=2.0)
        assert runtime.stats.cleanup_complete is True

    assert len(ring_names) == 20
    for name in ring_names:
        try:
            leaked = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            continue
        else:
            leaked.close()
            raise AssertionError(f"live SLAM shared memory leaked after stop: {name}")


def test_latest_frame_backpressure_remains_bounded_during_long_burst() -> None:
    class SlowBackend(DeterministicMockLiveSlamBackend):
        def track_live_frame(self, frame, context):
            sleep(0.003)
            return super().track_live_frame(frame, context)

    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: SlowBackend(initialization_frames=1),
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
    for index in range(1000):
        runtime.offer_source_frame(_frame(index))
    sleep(0.5)
    stats = runtime.stats
    assert stats.offered_frames == 1000
    assert stats.replaced_frames > 0
    assert stats.processed_frames < stats.offered_frames
    assert stats.state is LiveCaptureRuntimeState.RUNNING
    assert runtime.stop(timeout_s=2.0)
    assert runtime.stats.cleanup_complete is True
