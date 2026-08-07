#!/usr/bin/env python3
"""Non-UI Phase 8.4A smoke for capture adaptation, SHM transport and mock SLAM."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.capture.frame import SourceFrame
from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageSize,
)
from core.models.metadata import IntrinsicsSource
from core.reconstruction import (
    BackendOperationContext,
    CameraDistortionModel,
    DeterministicMockLiveSlamBackend,
    LiveFrameStaleError,
    LiveSharedMemoryFrameRing,
    LiveSlamRequest,
    LiveSourceFrameAdapter,
    LiveTrackingState,
    ReconstructionCamera,
    ReconstructionEventRecorder,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    make_usb_source_metadata,
    new_reconstruction_session_id,
)


def main() -> int:
    session_id = new_reconstruction_session_id()
    image_size = ImageSize(320, 240)
    source = make_usb_source_metadata(
        0,
        width=image_size.width,
        height=image_size.height,
        frame_rate_hz=30.0,
    )
    intrinsics = CameraIntrinsics(
        np.array(
            [[250.0, 0.0, 160.0], [0.0, 251.0, 120.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "live-rgb",
        IntrinsicsSource.CALIBRATED,
        CalibrationStatus.CALIBRATED,
        calibration_record_id="phase8-4a-smoke-calibration",
    )
    camera = ReconstructionCamera(
        camera_id="phase8-4a-smoke-camera",
        image_size=image_size,
        intrinsics=intrinsics,
        camera_frame=CoordinateFrame(
            "phase8-4a-smoke-camera-frame",
            CoordinateFrameKind.CAMERA_GENERIC,
            AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
            parent_frame_id="phase8-4a-live-reconstruction",
        ),
        distortion_model=CameraDistortionModel.NONE,
    )
    request = LiveSlamRequest(
        session_id=session_id,
        source=source,
        camera=camera,
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id="development-live-slam-mock",
    )
    recorder = ReconstructionEventRecorder(session_id)
    context = BackendOperationContext(
        session_id=session_id,
        stage=ReconstructionStage.LIVE_TRACKING,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=ReconstructionOperationControl(),
        emit_event=lambda event: None,
    )
    backend = DeterministicMockLiveSlamBackend(
        initialization_frames=2,
        keyframe_interval=4,
        degraded_sequences=(7,),
    )
    backend.prepare(context)
    backend.start_live_session(request, context)

    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=image_size.width * image_size.height * 3,
    )
    adapter = LiveSourceFrameAdapter(
        session_id=session_id,
        source=source,
        camera=camera,
        ring=ring,
    )
    observations = []
    first_reference = None
    try:
        for index in range(12):
            image = np.full((240, 320, 3), index, dtype=np.uint8)
            frame = adapter.publish(
                SourceFrame(
                    image=image,
                    frame_id=index,
                    timestamp=5_000.0 + index * 0.033,
                )
            )
            if first_reference is None:
                first_reference = frame.buffer
            restored = ring.read(frame.buffer)
            if not np.array_equal(restored, image):
                raise RuntimeError("Shared-memory frame round-trip mismatch.")
            observations.append(backend.track_live_frame(frame, context))

        stale_rejected = False
        try:
            assert first_reference is not None
            ring.read(first_reference)
        except LiveFrameStaleError:
            stale_rejected = True

        states = {state: 0 for state in LiveTrackingState}
        for observation in observations:
            states[observation.tracking_state] += 1
        keyframes = [item for item in observations if item.is_keyframe]
        posed = [item for item in observations if item.pose is not None]
        if not posed:
            raise RuntimeError("Mock SLAM produced no poses.")
        last_pose = posed[-1].pose
        assert last_pose is not None
        if last_pose.gauge.unit_to_metre is not None:
            raise RuntimeError("Phase 8.4A smoke unexpectedly claimed metric scale.")
        if not stale_rejected:
            raise RuntimeError("Bounded ring failed to reject an overwritten frame reference.")

        print("Phase 8.4A live SLAM foundation smoke: PASS")
        print(f"source_kind={source.kind.value} frames={len(observations)}")
        print(
            "tracking_states="
            + ",".join(
                f"{state.value}:{states[state]}" for state in LiveTrackingState if states[state]
            )
        )
        print(f"keyframes={len(keyframes)} latest_generation={ring.latest_generation}")
        print(f"stale_reference_rejected={stale_rejected}")
        print("pose_convention=T_reconstruction_from_camera")
        print(
            f"gauge={last_pose.gauge.scale_status.value} "
            f"unit_to_metre={last_pose.gauge.unit_to_metre}"
        )
        print("loop_closure_enabled=False global_optimization_enabled=False")
        print("real_stella_executed=False")
        print(f"event_recorder_ready={len(recorder.events) == 0}")
        return 0
    finally:
        backend.stop_live_session(context)
        backend.shutdown()
        ring.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
