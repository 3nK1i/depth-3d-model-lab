from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.capture.frame import SourceFrame
from core.reconstruction import (
    BackendOperationContext,
    ExternalStellaLiveSlamBackend,
    LiveOptimizedTrajectory,
    LiveSharedMemoryFrameRing,
    LiveSourceFrameAdapter,
    LiveTrajectoryPoseLayer,
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


def test_native_optimized_revision_propagates_without_mutating_raw_tracking(tmp_path) -> None:
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
    trajectory = LiveOptimizedTrajectory(session_id=request.session_id)
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
                timestamp=1.0,
            )
        )
        observation = backend.track_live_frame(frame, context)
        assert observation.pose is not None
        raw_matrix = observation.pose.T_reconstruction_from_camera.copy()
        assert observation.keyframe_id == "stella-keyframe-0"
        trajectory.ingest_observation(observation)

        before = backend.get_native_global_consistency_status(context)
        first_update = trajectory.apply_native_status(before)
        first_best = trajectory.best_pose_for_frame(observation.frame_id)
        assert first_update.exact_optimized_keyframes == 1
        assert first_best is not None
        assert first_best.layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME
        assert np.allclose(first_best.T_reconstruction_from_camera, raw_matrix)

        result = backend.request_native_loop_closure(
            "stella-keyframe-0", "stella-keyframe-9", context
        )
        assert result.accepted is True
        after = backend.get_native_global_consistency_status(context)
        second_update = trajectory.apply_native_status(after)
        corrected = trajectory.best_pose_for_frame(observation.frame_id)
        raw_after = trajectory.raw_pose_for_frame(observation.frame_id)

        assert second_update.optimization_revision > first_update.optimization_revision
        assert second_update.corrections
        assert corrected is not None and raw_after is not None
        assert corrected.layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME
        assert np.isclose(corrected.T_reconstruction_from_camera[0, 3], 0.09)
        assert np.array_equal(raw_after.T_reconstruction_from_camera, raw_matrix)
        assert raw_after.layer is LiveTrajectoryPoseLayer.TRACKING
        assert corrected.raw_pose_id == observation.pose.pose_id
        assert corrected.gauge.unit_to_metre is None
        backend.stop_live_session(context)
    finally:
        backend.shutdown()
        ring.shutdown()
