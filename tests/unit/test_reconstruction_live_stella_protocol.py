from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.capture.frame import SourceFrame
from core.reconstruction import (
    LIVE_SLAM_RECONSTRUCTION_FRAME,
    LiveSlamMessageType,
    LiveSlamProtocolError,
    LiveSlamProtocolMessage,
    LiveSharedMemoryFrameRing,
    LiveSourceFrameAdapter,
    STELLA_LIVE_BACKEND_ID,
    make_live_observation_message,
    parse_live_observation_message,
)
from core.reconstruction.live_contracts import (
    LiveMapSnapshot,
    LiveSlamObservation,
    LiveSlamPose,
    LiveTrackingQuality,
    LiveTrackingState,
)
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request


def _frame_and_observation():
    request = make_live_request(backend_id=STELLA_LIVE_BACKEND_ID)
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
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
            timestamp=1.25,
        )
    )
    pose = LiveSlamPose(
        pose_id="stella-live-pose-0",
        camera_id=request.camera.camera_id,
        frame_id=frame.frame_id,
        timestamp_ns=frame.capture_timestamp_ns,
        camera_frame_id=request.camera.camera_frame.frame_id,
        T_reconstruction_from_camera=np.eye(4, dtype=np.float64),
    )
    observation = LiveSlamObservation(
        session_id=request.session_id,
        frame_id=frame.frame_id,
        sequence_index=frame.sequence_index,
        tracking_state=LiveTrackingState.TRACKING,
        quality=LiveTrackingQuality(0, 0, 2_000_000, reason="stella_pose_valid"),
        pose=pose,
        is_keyframe=True,
        keyframe_id="stella-keyframe-0",
        map_snapshot=LiveMapSnapshot(1, 1, 85),
    )
    return request, ring, frame, observation


def test_native_observation_parser_preserves_arbitrary_scale_pose_contract() -> None:
    request, ring, frame, observation = _frame_and_observation()
    try:
        message = make_live_observation_message(observation, sequence_number=3)
        restored = parse_live_observation_message(
            message,
            expected_session_id=request.session_id,
            expected_frame=frame,
        )
        assert restored.tracking_state is LiveTrackingState.TRACKING
        assert restored.pose is not None
        assert restored.pose.reconstruction_frame_id == LIVE_SLAM_RECONSTRUCTION_FRAME
        assert restored.pose.gauge.unit_to_metre is None
        assert np.allclose(restored.pose.T_reconstruction_from_camera, np.eye(4))
    finally:
        ring.shutdown()


def test_native_observation_parser_rejects_world_pose_convention() -> None:
    request, ring, frame, observation = _frame_and_observation()
    try:
        payload = observation.as_dict()
        assert payload["pose"] is not None
        payload["pose"]["pose_convention"] = "T_world_from_camera"
        message = LiveSlamProtocolMessage(
            message_type=LiveSlamMessageType.OBSERVATION,
            session_id=request.session_id,
            sequence_number=3,
            payload={"observation": payload},
        )
        with pytest.raises(LiveSlamProtocolError, match="T_reconstruction_from_camera"):
            parse_live_observation_message(
                message,
                expected_session_id=request.session_id,
                expected_frame=frame,
            )
    finally:
        ring.shutdown()


def test_native_observation_parser_rejects_metric_gauge_claim() -> None:
    request, ring, frame, observation = _frame_and_observation()
    try:
        payload = observation.as_dict()
        assert payload["pose"] is not None
        payload["pose"]["gauge"] = {
            "scale_status": "estimated_metric",
            "alignment_status": "arbitrary",
            "unit_to_metre": 1.0,
            "scale_source": "worker",
            "relative_scale_uncertainty": 0.1,
        }
        message = LiveSlamProtocolMessage(
            message_type=LiveSlamMessageType.OBSERVATION,
            session_id=request.session_id,
            sequence_number=3,
            payload={"observation": payload},
        )
        with pytest.raises((LiveSlamProtocolError, ValueError), match="arbitrary"):
            parse_live_observation_message(
                message,
                expected_session_id=request.session_id,
                expected_frame=frame,
            )
    finally:
        ring.shutdown()


def test_native_observation_parser_rejects_wrong_frame_identity() -> None:
    request, ring, frame, observation = _frame_and_observation()
    try:
        message = make_live_observation_message(
            replace(observation, frame_id="another-frame", pose=None, tracking_state=LiveTrackingState.LOST, is_keyframe=False, keyframe_id=""),
            sequence_number=3,
        )
        with pytest.raises(LiveSlamProtocolError, match="submitted frame"):
            parse_live_observation_message(
                message,
                expected_session_id=request.session_id,
                expected_frame=frame,
            )
    finally:
        ring.shutdown()
