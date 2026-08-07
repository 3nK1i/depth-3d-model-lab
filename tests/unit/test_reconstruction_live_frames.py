from __future__ import annotations

import numpy as np
import pytest

from core.capture.frame import SourceFrame
from core.reconstruction import (
    LiveSharedMemoryFrameRing,
    LiveSourceFrameAdapter,
    ReconstructionValidationError,
)
from tests.reconstruction_helpers import SESSION_ID
from tests.reconstruction_live_helpers import (
    LIVE_IMAGE_SIZE,
    make_calibrated_live_camera,
    make_live_request,
)


def test_source_frame_adapter_preserves_capture_identity_without_modifying_sourceframe() -> None:
    request = make_live_request()
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
        image = np.zeros((LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3), dtype=np.uint8)
        frame = SourceFrame(image=image, frame_id=7, timestamp=10.0)
        live = adapter.publish(frame)
        assert live.source_frame_id == 7
        assert live.sequence_index == 0
        assert live.capture_timestamp_ns == 10_000_000_000
        assert live.camera_id == request.camera.camera_id
        assert np.array_equal(ring.read(live.buffer), image)
        assert set(SourceFrame.__dataclass_fields__) == {"image", "frame_id", "timestamp"}
    finally:
        ring.shutdown()


def test_source_frame_adapter_rejects_non_monotonic_input() -> None:
    request = make_live_request()
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=2,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
    try:
        adapter = LiveSourceFrameAdapter(
            session_id=SESSION_ID,
            source=request.source,
            camera=request.camera,
            ring=ring,
        )
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        adapter.publish(SourceFrame(image=image, frame_id=1, timestamp=2.0))
        with pytest.raises(ReconstructionValidationError, match="frame IDs"):
            adapter.publish(SourceFrame(image=image, frame_id=1, timestamp=3.0))
    finally:
        ring.shutdown()


def test_source_frame_adapter_rejects_offline_image_sequence_source() -> None:
    from core.reconstruction import InputSourceKind, InputSourceMetadata, TimestampOrigin

    camera = make_live_request().camera
    source = InputSourceMetadata(
        source_id="images",
        kind=InputSourceKind.IMAGE_SEQUENCE,
        display_name="Images",
        source_uri="sequence:///tmp/images",
        expected_image_size=LIVE_IMAGE_SIZE,
        timestamp_origin=TimestampOrigin.SYNTHETIC,
    )
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=2,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
    try:
        with pytest.raises(ReconstructionValidationError, match="USB, RTSP"):
            LiveSourceFrameAdapter(
                session_id=SESSION_ID,
                source=source,
                camera=camera,
                ring=ring,
            )
    finally:
        ring.shutdown()


def test_source_frame_adapter_rejects_mislabeled_timestamp_origin() -> None:
    from core.reconstruction import InputSourceKind, InputSourceMetadata, TimestampOrigin

    source = InputSourceMetadata(
        source_id="recorded-live",
        kind=InputSourceKind.RECORDED_VIDEO,
        display_name="Recorded live simulation",
        source_uri="video:///tmp/test.avi",
        expected_image_size=LIVE_IMAGE_SIZE,
        expected_frame_rate_hz=20.0,
        timestamp_origin=TimestampOrigin.PRESENTATION_TIMESTAMP,
    )
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
    try:
        with pytest.raises(ReconstructionValidationError, match="CAPTURE_CLOCK"):
            LiveSourceFrameAdapter(
                session_id=SESSION_ID,
                source=source,
                camera=make_calibrated_live_camera(),
                ring=ring,
            )
    finally:
        ring.shutdown()
