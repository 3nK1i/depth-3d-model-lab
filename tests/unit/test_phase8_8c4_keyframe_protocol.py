from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from core.reconstruction import LiveMapGeometrySnapshot, LiveMapKeyframeSample
from core.reconstruction.live_contracts import LiveSlamObservation
from core.reconstruction.live_protocol import (
    LiveSlamMessageType,
    LiveSlamProtocolError,
    LiveSlamProtocolMessage,
    decode_live_slam_message,
    encode_live_slam_message,
    parse_live_observation_message,
)
from tests.unit.test_reconstruction_live_stella_protocol import _frame_and_observation


def _keyframe_payload(index: int, x: float) -> dict[str, object]:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return {
        "keyframe_id": f"stella-keyframe-{index}",
        "timestamp_ns": 2_000_000_000 + index,
        "pose_convention": "T_reconstruction_from_camera",
        "T_reconstruction_from_camera": pose.tolist(),
    }


def test_c4_protocol_round_trips_optional_keyframe_pose_samples() -> None:
    request, ring, frame, observation = _frame_and_observation()
    try:
        payload = observation.as_dict()
        payload["map_snapshot"]["keyframe_count"] = 4
        payload["map_snapshot"]["keyframes_reconstruction"] = [
            _keyframe_payload(1, 0.1),
            _keyframe_payload(3, 0.4),
        ]
        message = LiveSlamProtocolMessage(
            message_type=LiveSlamMessageType.OBSERVATION,
            session_id=request.session_id,
            sequence_number=7,
            payload={"observation": payload},
        )
        restored = parse_live_observation_message(
            decode_live_slam_message(encode_live_slam_message(message)),
            expected_session_id=request.session_id,
            expected_frame=frame,
        )
        assert isinstance(restored.map_snapshot, LiveMapGeometrySnapshot)
        assert len(restored.map_snapshot.keyframes_reconstruction) == 2
        first = restored.map_snapshot.keyframes_reconstruction[0]
        assert isinstance(first, LiveMapKeyframeSample)
        assert first.keyframe_id == "stella-keyframe-1"
        assert np.isclose(first.T_reconstruction_from_camera[0, 3], 0.1)
    finally:
        ring.shutdown()


def test_c4_protocol_accepts_empty_keyframe_array_after_wire_round_trip() -> None:
    request, ring, frame, observation = _frame_and_observation()
    try:
        payload = observation.as_dict()
        payload["map_snapshot"]["keyframes_reconstruction"] = []
        message = LiveSlamProtocolMessage(
            message_type=LiveSlamMessageType.OBSERVATION,
            session_id=request.session_id,
            sequence_number=8,
            payload={"observation": payload},
        )
        restored = parse_live_observation_message(
            decode_live_slam_message(encode_live_slam_message(message)),
            expected_session_id=request.session_id,
            expected_frame=frame,
        )
        assert isinstance(restored.map_snapshot, LiveMapGeometrySnapshot)
        assert restored.map_snapshot.keyframes_reconstruction == ()
    finally:
        ring.shutdown()


def test_c4_protocol_rejects_wrong_keyframe_pose_convention() -> None:
    request, ring, frame, observation = _frame_and_observation()
    try:
        payload = observation.as_dict()
        bad = _keyframe_payload(1, 0.1)
        bad["pose_convention"] = "T_camera_from_reconstruction"
        payload["map_snapshot"]["keyframes_reconstruction"] = [bad]
        message = LiveSlamProtocolMessage(
            message_type=LiveSlamMessageType.OBSERVATION,
            session_id=request.session_id,
            sequence_number=9,
            payload={"observation": payload},
        )
        with pytest.raises(LiveSlamProtocolError, match="pose convention"):
            parse_live_observation_message(
                message,
                expected_session_id=request.session_id,
                expected_frame=frame,
            )
    finally:
        ring.shutdown()


def test_c4_does_not_rewrite_live_observation_shape() -> None:
    assert [field.name for field in fields(LiveSlamObservation)] == [
        "session_id",
        "frame_id",
        "sequence_index",
        "tracking_state",
        "quality",
        "pose",
        "is_keyframe",
        "keyframe_id",
        "map_snapshot",
    ]
