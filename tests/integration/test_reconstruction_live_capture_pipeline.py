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


class _SlowMockLiveSlamBackend(DeterministicMockLiveSlamBackend):
    """Force deterministic producer/consumer overlap for replacement testing."""

    def track_live_frame(self, frame, context):
        sleep(0.02)
        return super().track_live_frame(frame, context)


def _frame(*, frame_id: int, timestamp: float) -> SourceFrame:
    return SourceFrame(
        np.zeros((LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3), dtype=np.uint8),
        frame_id=frame_id,
        timestamp=timestamp,
    )


def test_sourceframe_to_live_observation_pipeline_survives_frame_replacement() -> None:
    observations = []
    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: _SlowMockLiveSlamBackend(),
        observation_callback=observations.append,
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)

    base = monotonic()
    for index in range(40):
        runtime.offer_source_frame(_frame(frame_id=index, timestamp=base + index * 0.002))

    # The burst must exercise latest-frame replacement, but replacement means the
    # backend may legitimately see only one or two initialization frames. Feed a
    # few later frames at a paced rate so tracking can advance deterministically.
    for index in range(40, 46):
        runtime.offer_source_frame(_frame(frame_id=index, timestamp=base + index * 0.002))
        sleep(0.03)
        if observations and observations[-1].pose is not None:
            break

    assert observations
    assert any(observation.pose is not None for observation in observations)
    pose_observation = next(
        observation for observation in reversed(observations) if observation.pose is not None
    )
    assert pose_observation.pose is not None
    assert pose_observation.pose.gauge.unit_to_metre is None

    stats = runtime.stats
    assert stats.offered_frames >= 40
    assert stats.replaced_frames > 0
    assert stats.processed_frames < stats.offered_frames
    assert runtime.stop(timeout_s=2.0)
