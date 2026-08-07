from __future__ import annotations

import numpy as np

from core.capture.frame import SourceFrame
from core.reconstruction import (
    BackendOperationContext,
    ExternalStellaLiveSlamBackend,
    LiveSharedMemoryFrameRing,
    LiveSourceFrameAdapter,
    ReconstructionOperationControl,
    ReconstructionStage,
    STELLA_LIVE_BACKEND_ID,
)
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request
from tests.reconstruction_live_worker_helpers import make_fake_stella_worker_layout


def test_external_worker_protocol_stays_bounded_across_many_frames(tmp_path) -> None:
    prefix, worker, vocab = make_fake_stella_worker_layout(tmp_path)
    request = make_live_request(backend_id=STELLA_LIVE_BACKEND_ID)
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
    context = BackendOperationContext(
        request.session_id,
        ReconstructionStage.LIVE_TRACKING,
        request.profile,
        ReconstructionOperationControl(),
        lambda event: None,
    )
    backend = ExternalStellaLiveSlamBackend(
        install_prefix=prefix,
        worker_executable=worker,
        vocabulary_path=vocab,
        frame_ring_spec=ring.spec,
        require_qualified_build=False,
    )
    adapter = LiveSourceFrameAdapter(
        session_id=request.session_id,
        source=request.source,
        camera=request.camera,
        ring=ring,
    )
    try:
        backend.start_live_session(request, context)
        for index in range(250):
            frame = adapter.publish(
                SourceFrame(
                    image=np.full(
                        (LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3),
                        index % 251,
                        dtype=np.uint8,
                    ),
                    frame_id=index,
                    timestamp=10.0 + index / 30.0,
                )
            )
            observation = backend.track_live_frame(frame, context)
            assert observation.sequence_index == index
            assert observation.pose is not None
        backend.stop_live_session(context)
    finally:
        backend.shutdown()
        ring.shutdown()
