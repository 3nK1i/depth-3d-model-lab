from __future__ import annotations

import numpy as np

from core.capture.frame import SourceFrame
from core.reconstruction import (
    BackendOperationContext,
    ExternalStellaLiveSlamBackend,
    LiveSharedMemoryFrameRing,
    LiveSlamWorkerState,
    LiveSourceFrameAdapter,
    LiveTrackingState,
    ReconstructionOperationControl,
    ReconstructionStage,
    STELLA_LIVE_BACKEND_ID,
)
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request
from tests.reconstruction_live_worker_helpers import make_fake_stella_worker_layout


def _context(request, control=None):
    return BackendOperationContext(
        session_id=request.session_id,
        stage=ReconstructionStage.LIVE_TRACKING,
        profile=request.profile,
        control=control or ReconstructionOperationControl(),
        emit_event=lambda event: None,
    )


def test_external_stella_backend_crosses_real_process_socket_boundary_without_native_python_linkage(tmp_path) -> None:
    prefix, worker, vocab = make_fake_stella_worker_layout(tmp_path)
    request = make_live_request(backend_id=STELLA_LIVE_BACKEND_ID)
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
    backend = ExternalStellaLiveSlamBackend(
        install_prefix=prefix,
        worker_executable=worker,
        vocabulary_path=vocab,
        frame_ring_spec=ring.spec,
        require_qualified_build=False,
    )
    try:
        context = _context(request)
        backend.prepare(context)
        backend.start_live_session(request, context)
        assert backend.worker_state is LiveSlamWorkerState.RUNNING
        adapter = LiveSourceFrameAdapter(
            session_id=request.session_id,
            source=request.source,
            camera=request.camera,
            ring=ring,
        )
        frame = adapter.publish(
            SourceFrame(
                image=np.zeros((LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3), dtype=np.uint8),
                frame_id=0,
                timestamp=2.0,
            )
        )
        observation = backend.track_live_frame(frame, context)
        assert observation.tracking_state is LiveTrackingState.TRACKING
        assert observation.pose is not None
        assert observation.pose.gauge.unit_to_metre is None
        assert observation.is_keyframe is True
        assert observation.map_snapshot is not None
        assert observation.map_snapshot.keyframe_count == 1
        backend.stop_live_session(context)
        assert backend.worker_state is LiveSlamWorkerState.READY
    finally:
        backend.shutdown()
        ring.shutdown()
    assert backend.worker_state is LiveSlamWorkerState.STOPPED
