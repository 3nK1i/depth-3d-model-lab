from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction import (
    LiveFrameStaleError,
    LiveFrameTransportError,
    LiveSharedMemoryFrameRing,
)


def test_shared_memory_ring_round_trip_and_cross_open() -> None:
    ring = LiveSharedMemoryFrameRing.create(slot_count=3, max_payload_bytes=64 * 48 * 3)
    reader = None
    try:
        image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        reference = ring.publish(image)
        reader = LiveSharedMemoryFrameRing.open_existing(ring.spec)
        restored = reader.read(reference)
        assert np.array_equal(restored, image)
        assert restored is not image
    finally:
        if reader is not None:
            reader.close()
        ring.shutdown()


def test_overwritten_reference_is_rejected_not_misread() -> None:
    ring = LiveSharedMemoryFrameRing.create(slot_count=2, max_payload_bytes=12)
    try:
        old = ring.publish(np.zeros((2, 2, 3), dtype=np.uint8))
        ring.publish(np.ones((2, 2, 3), dtype=np.uint8))
        newest = ring.publish(np.full((2, 2, 3), 2, dtype=np.uint8))
        with pytest.raises(LiveFrameStaleError, match="overwritten"):
            ring.read(old)
        assert np.all(ring.read(newest) == 2)
    finally:
        ring.shutdown()


def test_ring_rejects_oversized_and_non_bgr_frames() -> None:
    ring = LiveSharedMemoryFrameRing.create(slot_count=2, max_payload_bytes=12)
    try:
        with pytest.raises(LiveFrameTransportError, match="exceeds"):
            ring.publish(np.zeros((3, 3, 3), dtype=np.uint8))
        with pytest.raises(LiveFrameTransportError, match="HxWx3"):
            ring.publish(np.zeros((2, 2), dtype=np.uint8))
    finally:
        ring.shutdown()


def test_opened_reader_cannot_publish_into_owner_ring() -> None:
    ring = LiveSharedMemoryFrameRing.create(slot_count=2, max_payload_bytes=12)
    reader = None
    try:
        reader = LiveSharedMemoryFrameRing.open_existing(ring.spec)
        with pytest.raises(LiveFrameTransportError, match="owner"):
            reader.publish(np.zeros((2, 2, 3), dtype=np.uint8))
    finally:
        if reader is not None:
            reader.close()
        ring.shutdown()
