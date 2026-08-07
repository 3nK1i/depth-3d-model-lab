from __future__ import annotations

import numpy as np

from core.capture.frame import SourceFrame
from core.reconstruction import (
    LatestLiveFrameMailbox,
    LiveSharedMemoryFrameRing,
    LiveSourceFrameAdapter,
)
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request


def test_latest_frame_mailbox_replaces_pending_frame_without_backlog() -> None:
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
        first = adapter.publish(
            SourceFrame(np.zeros((240, 320, 3), dtype=np.uint8), 0, 1.0)
        )
        second = adapter.publish(
            SourceFrame(np.ones((240, 320, 3), dtype=np.uint8), 1, 1.1)
        )
        assert mailbox.offer(first) is False
        assert mailbox.offer(second) is True
        delivered = mailbox.take_latest()
        assert delivered is not None
        assert delivered.sequence_index == 1
        assert mailbox.take_latest() is None
        stats = mailbox.stats
        assert stats.offered_frames == 2
        assert stats.delivered_frames == 1
        assert stats.replaced_frames == 1
        assert stats.pending is False
    finally:
        ring.shutdown()
