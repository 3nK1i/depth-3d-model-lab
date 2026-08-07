from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction import LiveFrameStaleError, LiveSharedMemoryFrameRing


def test_live_shared_memory_ring_remains_bounded_across_repeated_frames() -> None:
    ring = LiveSharedMemoryFrameRing.create(slot_count=3, max_payload_bytes=64 * 48 * 3)
    try:
        first = None
        latest = None
        for index in range(2_000):
            image = np.full((48, 64, 3), index % 251, dtype=np.uint8)
            reference = ring.publish(image)
            if first is None:
                first = reference
            latest = reference
        assert latest is not None
        assert ring.latest_generation == 2_000
        restored = ring.read(latest)
        assert np.all(restored == (1_999 % 251))
        assert first is not None
        with pytest.raises(LiveFrameStaleError):
            ring.read(first)
        assert ring.spec.slot_stride_bytes % 64 == 0
        assert ring.spec.total_bytes == 3 * ring.spec.slot_stride_bytes
    finally:
        ring.shutdown()


def test_live_mailbox_keeps_only_newest_frame_under_burst_load() -> None:
    from core.capture.frame import SourceFrame
    from core.reconstruction import LatestLiveFrameMailbox, LiveSourceFrameAdapter
    from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request

    request = make_live_request()
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
    try:
        adapter = LiveSourceFrameAdapter(
            session_id=request.session_id,
            source=request.source,
            camera=request.camera,
            ring=ring,
        )
        mailbox = LatestLiveFrameMailbox()
        for index in range(1_000):
            frame = adapter.publish(
                SourceFrame(
                    np.full((240, 320, 3), index % 251, dtype=np.uint8),
                    index,
                    10.0 + index * 0.001,
                )
            )
            mailbox.offer(frame)
        latest = mailbox.take_latest()
        assert latest is not None
        assert latest.sequence_index == 999
        assert mailbox.stats.replaced_frames == 999
        assert np.all(ring.read(latest.buffer) == (999 % 251))
    finally:
        ring.shutdown()
