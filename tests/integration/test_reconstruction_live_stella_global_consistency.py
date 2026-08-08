from __future__ import annotations

from dataclasses import replace

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


def _context(request):
    return BackendOperationContext(
        session_id=request.session_id,
        stage=ReconstructionStage.LIVE_TRACKING,
        profile=request.profile,
        control=ReconstructionOperationControl(),
        emit_event=lambda event: None,
    )


def test_external_worker_exposes_native_global_consistency_without_overwriting_raw_pose(tmp_path) -> None:
    prefix, worker, vocab = make_fake_stella_worker_layout(tmp_path)
    request = replace(
        make_live_request(backend_id=STELLA_LIVE_BACKEND_ID),
        enable_loop_closure=True,
        enable_global_optimization=True,
    )
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
        raw_observation = backend.track_live_frame(frame, context)
        assert raw_observation.pose is not None
        raw_matrix = raw_observation.pose.T_reconstruction_from_camera.copy()

        before = backend.get_native_global_consistency_status(context)
        assert before.loop_detector_enabled is True
        assert before.global_optimization_enabled is True
        assert before.accepted_loop_closure_requests == 0

        result = backend.request_native_loop_closure(
            "stella-keyframe-0", "stella-keyframe-9", context
        )
        assert result.accepted is True
        assert result.accepted_request_count == 1

        after = backend.get_native_global_consistency_status(context)
        assert after.accepted_loop_closure_requests == 1
        assert after.optimized_pose_revision > before.optimized_pose_revision
        assert after.optimized_keyframe_poses
        assert after.optimized_keyframe_poses[0].as_dict()["pose_layer"] == "optimized"
        assert raw_observation.pose.as_dict()["pose_convention"] == "T_reconstruction_from_camera"
        assert np.array_equal(raw_observation.pose.T_reconstruction_from_camera, raw_matrix)
        assert after.gauge.unit_to_metre is None
        backend.stop_live_session(context)
    finally:
        backend.shutdown()
        ring.shutdown()
