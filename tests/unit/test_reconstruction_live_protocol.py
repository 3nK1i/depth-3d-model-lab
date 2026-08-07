from __future__ import annotations

import json

import pytest

from core.reconstruction import (
    LIVE_SLAM_PROTOCOL_VERSION,
    LiveSlamMessageType,
    LiveSlamProtocolError,
    LiveSlamProtocolMessage,
    decode_live_slam_message,
    encode_live_slam_message,
    make_live_frame_available_message,
    make_live_start_message,
    LiveSharedMemoryFrameRing,
    LiveSourceFrameAdapter,
)
from tests.reconstruction_helpers import SESSION_ID


def test_live_protocol_json_round_trip_is_versioned_and_deterministic() -> None:
    message = LiveSlamProtocolMessage(
        message_type=LiveSlamMessageType.FRAME_AVAILABLE,
        session_id=SESSION_ID,
        sequence_number=9,
        payload={"generation": 10, "slot": 0},
    )
    encoded = encode_live_slam_message(message)
    assert encoded.endswith(b"\n")
    restored = decode_live_slam_message(encoded)
    assert restored == message
    assert json.loads(encoded)["protocol_version"] == LIVE_SLAM_PROTOCOL_VERSION


def test_live_protocol_rejects_unknown_version() -> None:
    raw = json.dumps(
        {
            "protocol_version": 99,
            "message_type": "hello",
            "session_id": SESSION_ID,
            "sequence_number": 0,
            "payload": {},
        }
    )
    with pytest.raises(LiveSlamProtocolError, match="version"):
        decode_live_slam_message(raw)


def test_live_protocol_rejects_non_object_payload() -> None:
    raw = json.dumps(
        {
            "protocol_version": 1,
            "message_type": "hello",
            "session_id": SESSION_ID,
            "sequence_number": 0,
            "payload": [],
        }
    )
    with pytest.raises(LiveSlamProtocolError, match="payload"):
        decode_live_slam_message(raw)


def test_frame_available_protocol_carries_reference_not_pixels() -> None:
    import numpy as np
    from core.capture.frame import SourceFrame
    from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request

    request = make_live_request()
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
    try:
        adapter = LiveSourceFrameAdapter(
            session_id=request.session_id,
            source=request.source,
            camera=request.camera,
            ring=ring,
        )
        frame = adapter.publish(
            SourceFrame(
                image=np.zeros((240, 320, 3), dtype=np.uint8),
                frame_id=0,
                timestamp=1.0,
            )
        )
        message = make_live_frame_available_message(frame, sequence_number=1)
        encoded = encode_live_slam_message(message)
        text = encoded.decode("utf-8")
        assert "shared_memory_name" in text
        assert '"image"' not in text
        assert len(encoded) < 2_000

        start = make_live_start_message(request, ring.spec, sequence_number=0)
        assert start.payload
    finally:
        ring.shutdown()
