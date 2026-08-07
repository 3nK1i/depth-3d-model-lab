from __future__ import annotations

from time import monotonic, sleep

import numpy as np

from core.capture.frame import SourceFrame
from core.reconstruction import (
    DeterministicMockLiveSlamBackend,
    LiveCaptureRuntimeState,
    LiveCaptureSlamRuntime,
)
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request


def _frame(frame_id: int, timestamp: float) -> SourceFrame:
    image = np.full(
        (LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3),
        frame_id % 255,
        dtype=np.uint8,
    )
    return SourceFrame(image=image, frame_id=frame_id, timestamp=timestamp)


def test_runtime_tracks_frames_off_caller_thread_and_stops_cleanly() -> None:
    observations = []
    request = make_live_request()
    runtime = LiveCaptureSlamRuntime(
        request=request,
        backend_factory=lambda _spec: DeterministicMockLiveSlamBackend(),
        observation_callback=observations.append,
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)

    base = monotonic()
    for index in range(6):
        runtime.offer_source_frame(_frame(index, base + index * 0.01))
        sleep(0.015)

    deadline = monotonic() + 2.0
    while len(observations) < 6 and monotonic() < deadline:
        sleep(0.01)

    assert len(observations) == 6
    assert observations[-1].sequence_index == 5
    assert runtime.stats.processed_frames == 6
    assert runtime.stats.observations == 6
    assert runtime.stop(timeout_s=2.0)
    assert runtime.state is LiveCaptureRuntimeState.STOPPED


def test_runtime_pause_resume_gates_capture_without_backend_pause_dependency() -> None:
    request = make_live_request()
    runtime = LiveCaptureSlamRuntime(
        request=request,
        backend_factory=lambda _spec: DeterministicMockLiveSlamBackend(),
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
    assert runtime.pause()
    assert runtime.state is LiveCaptureRuntimeState.PAUSED

    base = monotonic()
    assert runtime.offer_source_frame(_frame(0, base)) is False
    assert runtime.stats.rejected_frames == 1

    assert runtime.resume()
    runtime.offer_source_frame(_frame(1, base + 0.01))
    deadline = monotonic() + 2.0
    while runtime.stats.processed_frames < 1 and monotonic() < deadline:
        sleep(0.01)
    assert runtime.stats.processed_frames == 1
    assert runtime.stop(timeout_s=2.0)


def test_runtime_latest_frame_mailbox_is_bounded() -> None:
    class SlowMock(DeterministicMockLiveSlamBackend):
        def track_live_frame(self, frame, context):
            sleep(0.025)
            return super().track_live_frame(frame, context)

    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: SlowMock(),
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
    base = monotonic()
    for index in range(30):
        runtime.offer_source_frame(_frame(index, base + index * 0.001))
    sleep(0.25)
    stats = runtime.stats
    assert stats.offered_frames == 30
    assert stats.replaced_frames > 0
    assert stats.processed_frames < stats.offered_frames
    assert runtime.stop(timeout_s=2.0)
