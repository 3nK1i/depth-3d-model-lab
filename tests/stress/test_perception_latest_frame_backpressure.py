from __future__ import annotations

import queue
import time

import numpy as np

from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.perception.messages import (
    FramePerceptionRequest,
    PerceptionPipelineRequest,
    PerceptionWorkerConfiguration,
    PerceptionWorkerReady,
    PerceptionWorkerResponse,
)
from core.perception.metadata import PerceptionTask
from core.perception.process_manager import PerceptionProcessManager


def _request(request_id: int) -> PerceptionPipelineRequest:
    frame = FrameReference(1, request_id, float(request_id), ImageSize(16, 12))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    return PerceptionPipelineRequest(
        request_id,
        FramePerceptionRequest(frame, rgb, space, (PerceptionTask.DETECTION,)),
    )


def test_latest_frame_backpressure_preserves_newest_queued_request() -> None:
    manager = PerceptionProcessManager()
    manager.start(
        PerceptionWorkerConfiguration(
            "development-perception-mock",
            (("delay_ms", 50.0),),
        )
    )
    try:
        assert isinstance(manager.get(timeout=10), PerceptionWorkerReady)
        assert manager.submit_latest(_request(1))
        time.sleep(0.02)
        for request_id in range(2, 50):
            manager.submit_latest(_request(request_id))

        accepted = False
        acceptance_deadline = time.monotonic() + 2.0
        while time.monotonic() < acceptance_deadline and not accepted:
            accepted = manager.submit_latest(_request(50))
            if not accepted:
                time.sleep(0.005)
        assert accepted

        received: list[int] = []
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and 50 not in received:
            try:
                message = manager.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(message, PerceptionWorkerResponse):
                received.append(message.request_id)

        assert 50 in received
        assert manager.dropped_requests > 0
    finally:
        manager.shutdown()
