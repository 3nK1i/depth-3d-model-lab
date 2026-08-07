from __future__ import annotations

import numpy as np

from core.capture.frame import SourceFrame
from core.reconstruction import (
    BackendOperationContext,
    DeterministicMockLiveSlamBackend,
    KeyframeSelectionReason,
    LiveSharedMemoryFrameRing,
    LiveSourceFrameAdapter,
    ReconstructionFrame,
    ReconstructionKeyframe,
    ReconstructionQualityProfile,
    ReconstructionSession,
    ReconstructionStage,
)
from tests.reconstruction_helpers import SESSION_ID
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request


def test_live_foundation_lifecycle_from_capture_adapter_to_session_keyframes() -> None:
    request = make_live_request()
    session = ReconstructionSession(
        request.source,
        ReconstructionQualityProfile.EFFICIENT,
        session_id=SESSION_ID,
        created_at_utc="2026-08-07T14:00:00+00:00",
    )
    session.prepare()
    session.start()

    backend = DeterministicMockLiveSlamBackend(initialization_frames=2, keyframe_interval=3)
    context = BackendOperationContext(
        session_id=SESSION_ID,
        stage=ReconstructionStage.LIVE_TRACKING,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=session.control,
        emit_event=lambda event: None,
    )
    backend.prepare(context)
    backend.start_live_session(request, context)

    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
    adapter = LiveSourceFrameAdapter(
        session_id=SESSION_ID,
        source=request.source,
        camera=request.camera,
        ring=ring,
    )

    observations = []
    try:
        for index in range(9):
            source_frame = SourceFrame(
                image=np.full((240, 320, 3), index, dtype=np.uint8),
                frame_id=index,
                timestamp=1_000.0 + index * 0.033,
            )
            live_frame = adapter.publish(source_frame)
            session.add_frame(
                ReconstructionFrame(
                    frame_id=live_frame.frame_id,
                    session_id=live_frame.session_id,
                    source_id=live_frame.source_id,
                    sequence_index=live_frame.sequence_index,
                    timestamp_ns=live_frame.capture_timestamp_ns,
                    image_size=live_frame.image_size,
                )
            )
            observation = backend.track_live_frame(live_frame, context)
            observations.append(observation)
            if observation.is_keyframe:
                session.add_keyframe(
                    ReconstructionKeyframe(
                        keyframe_id=observation.keyframe_id,
                        session_id=SESSION_ID,
                        frame_id=observation.frame_id,
                        selection_sequence=len(session.snapshot().keyframes),
                        reason=KeyframeSelectionReason.BACKEND_SELECTED,
                        score=observation.quality.tracking_confidence,
                        pose_id=observation.pose.pose_id if observation.pose else "",
                    )
                )

        snapshot = session.snapshot()
        assert len(snapshot.frames) == 9
        assert len(snapshot.keyframes) == 3
        assert observations[-1].pose is not None
        assert observations[-1].pose.gauge.unit_to_metre is None
        assert observations[-1].map_snapshot is not None
        assert observations[-1].map_snapshot.keyframe_count == 3
    finally:
        backend.stop_live_session(context)
        backend.shutdown()
        ring.shutdown()
        session.request_cancel()
        session.acknowledge_cancelled()

    assert session.state.value == "cancelled"


def _read_shared_frame_in_child(spec, reference, output_queue) -> None:
    from core.reconstruction import LiveSharedMemoryFrameRing

    ring = LiveSharedMemoryFrameRing.open_existing(spec)
    try:
        image = ring.read(reference)
        output_queue.put((image.shape, int(image.sum())))
    finally:
        ring.close()


def test_live_frame_shared_memory_crosses_spawn_process_boundary() -> None:
    import multiprocessing as mp

    ring = LiveSharedMemoryFrameRing.create(slot_count=3, max_payload_bytes=32 * 24 * 3)
    ctx = mp.get_context("spawn")
    output = ctx.Queue(maxsize=1)
    process = None
    try:
        image = np.full((24, 32, 3), 7, dtype=np.uint8)
        reference = ring.publish(image)
        process = ctx.Process(
            target=_read_shared_frame_in_child,
            args=(ring.spec, reference, output),
        )
        process.start()
        process.join(10.0)
        assert not process.is_alive()
        assert process.exitcode == 0
        shape, total = output.get(timeout=2.0)
        assert tuple(shape) == (24, 32, 3)
        assert total == int(image.sum())
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(2.0)
        output.close()
        output.join_thread()
        ring.shutdown()
