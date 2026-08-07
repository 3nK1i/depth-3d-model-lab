from __future__ import annotations

import numpy as np

from core.capture.frame import SourceFrame
from core.reconstruction import (
    BackendOperationContext,
    DeterministicMockLiveSlamBackend,
    LiveSharedMemoryFrameRing,
    LiveSlamBackend,
    LiveSourceFrameAdapter,
    LiveTrackingState,
    ReconstructionEventRecorder,
    ReconstructionQualityProfile,
    ReconstructionStage,
)
from tests.reconstruction_helpers import SESSION_ID
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request


def _context():
    from core.reconstruction import ReconstructionOperationControl

    recorder = ReconstructionEventRecorder(SESSION_ID)
    return BackendOperationContext(
        session_id=SESSION_ID,
        stage=ReconstructionStage.LIVE_TRACKING,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=ReconstructionOperationControl(),
        emit_event=lambda event: None,
    ), recorder


def test_mock_live_backend_is_deterministic_and_non_metric() -> None:
    request = make_live_request()
    backend = DeterministicMockLiveSlamBackend(
        initialization_frames=2,
        keyframe_interval=3,
        degraded_sequences=(4,),
        lost_sequences=(6,),
    )
    context, _ = _context()
    assert isinstance(backend, LiveSlamBackend)
    backend.prepare(context)
    backend.start_live_session(request, context)
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
    try:
        adapter = LiveSourceFrameAdapter(
            session_id=SESSION_ID,
            source=request.source,
            camera=request.camera,
            ring=ring,
        )
        states = []
        keyframes = []
        for index in range(8):
            source_frame = SourceFrame(
                image=np.full((240, 320, 3), index, dtype=np.uint8),
                frame_id=index,
                timestamp=100.0 + index * 0.033,
            )
            frame = adapter.publish(source_frame)
            observation = backend.track_live_frame(frame, context)
            states.append(observation.tracking_state)
            if observation.is_keyframe:
                keyframes.append(observation.keyframe_id)
            if observation.pose is not None:
                assert observation.pose.gauge.unit_to_metre is None
                assert observation.pose.as_dict()["pose_convention"] == "T_reconstruction_from_camera"
        assert states[:2] == [LiveTrackingState.INITIALIZING] * 2
        assert states[4] is LiveTrackingState.DEGRADED
        assert states[6] is LiveTrackingState.LOST
        assert keyframes == ["live-keyframe-0", "live-keyframe-1"]
    finally:
        ring.shutdown()
        backend.stop_live_session(context)
        backend.shutdown()
