from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from core.reconstruction import LiveMapGeometrySnapshot
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


def test_c3_protocol_round_trips_optional_sparse_landmark_samples() -> None:
    request, ring, frame, observation = _frame_and_observation()
    try:
        payload = observation.as_dict()
        assert payload["map_snapshot"] is not None
        payload["map_snapshot"]["landmarks_reconstruction"] = [
            [0.1, 0.2, 0.3],
            [-0.5, 1.0, 2.0],
        ]
        message = LiveSlamProtocolMessage(
            message_type=LiveSlamMessageType.OBSERVATION,
            session_id=request.session_id,
            sequence_number=5,
            payload={"observation": payload},
        )
        restored = parse_live_observation_message(
            message,
            expected_session_id=request.session_id,
            expected_frame=frame,
        )
        assert isinstance(restored.map_snapshot, LiveMapGeometrySnapshot)
        assert restored.map_snapshot.landmark_count == 85
        assert restored.map_snapshot.landmarks_reconstruction == (
            (0.1, 0.2, 0.3),
            (-0.5, 1.0, 2.0),
        )
    finally:
        ring.shutdown()


def test_c3_protocol_accepts_empty_sparse_landmark_array_after_wire_round_trip() -> None:
    request, ring, frame, observation = _frame_and_observation()
    try:
        payload = observation.as_dict()
        assert payload["map_snapshot"] is not None
        payload["map_snapshot"]["landmarks_reconstruction"] = []
        message = LiveSlamProtocolMessage(
            message_type=LiveSlamMessageType.OBSERVATION,
            session_id=request.session_id,
            sequence_number=5,
            payload={"observation": payload},
        )
        wire_message = decode_live_slam_message(encode_live_slam_message(message))
        restored = parse_live_observation_message(
            wire_message,
            expected_session_id=request.session_id,
            expected_frame=frame,
        )
        assert isinstance(restored.map_snapshot, LiveMapGeometrySnapshot)
        assert restored.map_snapshot.landmarks_reconstruction == ()
    finally:
        ring.shutdown()


def test_c3_protocol_rejects_invalid_sparse_landmark_payload() -> None:
    request, ring, frame, observation = _frame_and_observation()
    try:
        payload = observation.as_dict()
        payload["map_snapshot"]["landmarks_reconstruction"] = [[0.0, 1.0]]
        message = LiveSlamProtocolMessage(
            message_type=LiveSlamMessageType.OBSERVATION,
            session_id=request.session_id,
            sequence_number=5,
            payload={"observation": payload},
        )
        with pytest.raises(LiveSlamProtocolError, match="exactly three"):
            parse_live_observation_message(
                message,
                expected_session_id=request.session_id,
                expected_frame=frame,
            )
    finally:
        ring.shutdown()


def test_c3_does_not_rewrite_live_observation_dataclass_shape() -> None:
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
