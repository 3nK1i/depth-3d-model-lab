from __future__ import annotations

from time import monotonic, sleep

import numpy as np

from core.capture.frame import SourceFrame
from core.reconstruction import DeterministicMockLiveSlamBackend, LiveCaptureRuntimeState, LiveCaptureSlamRuntime
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request


def test_live_capture_mailbox_stays_bounded_under_burst() -> None:
    runtime = LiveCaptureSlamRuntime(
        request=make_live_request(),
        backend_factory=lambda _spec: DeterministicMockLiveSlamBackend(),
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=2.0)
    base = monotonic()
    image = np.zeros((LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3), dtype=np.uint8)
    for index in range(1000):
        runtime.offer_source_frame(SourceFrame(image, index, base + index * 0.0001))
    sleep(0.25)
    stats = runtime.stats
    assert stats.offered_frames == 1000
    assert stats.replaced_frames > 0
    assert stats.processed_frames < 1000
    assert runtime.stop(timeout_s=2.0)
