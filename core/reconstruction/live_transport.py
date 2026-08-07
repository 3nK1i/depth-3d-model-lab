"""Bounded POSIX shared-memory transport for high-rate live SLAM frames."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from struct import Struct
from typing import Any

import numpy as np

from core.contracts import ImageSize
from core.contracts.validation import require_nonempty
from core.reconstruction.errors import LiveFrameStaleError, LiveFrameTransportError
from core.reconstruction.live_contracts import (
    LIVE_SLAM_SCHEMA_VERSION,
    LiveFrameBufferReference,
    LiveFramePixelFormat,
)
from core.reconstruction.validation import require_positive_int

_TOKEN = Struct("<Q")
_HEADER_BYTES = _TOKEN.size
_SLOT_ALIGNMENT_BYTES = 64


def _aligned_slot_stride(max_payload_bytes: int) -> int:
    unaligned = _HEADER_BYTES + max_payload_bytes
    return (
        (unaligned + _SLOT_ALIGNMENT_BYTES - 1)
        // _SLOT_ALIGNMENT_BYTES
        * _SLOT_ALIGNMENT_BYTES
    )


@dataclass(frozen=True, slots=True)
class LiveSharedMemoryRingSpec:
    shared_memory_name: str
    slot_count: int
    max_payload_bytes: int
    schema_version: int = LIVE_SLAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shared_memory_name",
            require_nonempty(self.shared_memory_name, field_name="shared_memory_name"),
        )
        object.__setattr__(
            self, "slot_count", require_positive_int(self.slot_count, field_name="slot_count")
        )
        object.__setattr__(
            self,
            "max_payload_bytes",
            require_positive_int(self.max_payload_bytes, field_name="max_payload_bytes"),
        )
        if self.schema_version != LIVE_SLAM_SCHEMA_VERSION:
            raise LiveFrameTransportError("Unsupported live shared-memory schema version.")

    @property
    def slot_stride_bytes(self) -> int:
        return _aligned_slot_stride(self.max_payload_bytes)

    @property
    def total_bytes(self) -> int:
        return self.slot_stride_bytes * self.slot_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shared_memory_name": self.shared_memory_name,
            "slot_count": self.slot_count,
            "max_payload_bytes": self.max_payload_bytes,
            "slot_stride_bytes": self.slot_stride_bytes,
            "total_bytes": self.total_bytes,
        }


class LiveSharedMemoryFrameRing:
    """Tiny overwrite ring with generation validation and no frame backlog.

    Writers mark a slot as being modified with an odd token, copy the payload,
    then publish an even generation token. Readers verify the token before and
    after copying; an overwritten reference is therefore rejected rather than
    silently reading a torn/newer frame.
    """

    def __init__(
        self,
        memory: shared_memory.SharedMemory,
        spec: LiveSharedMemoryRingSpec,
        *,
        owner: bool,
    ) -> None:
        if memory.name != spec.shared_memory_name:
            raise LiveFrameTransportError("Shared-memory object does not match ring spec.")
        if memory.size < spec.total_bytes:
            raise LiveFrameTransportError("Shared-memory object is smaller than ring spec.")
        self._memory = memory
        self._spec = spec
        self._owner = owner
        self._generation = 0
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        slot_count: int = 3,
        max_payload_bytes: int,
        name: str | None = None,
    ) -> "LiveSharedMemoryFrameRing":
        slots = require_positive_int(slot_count, field_name="slot_count")
        payload = require_positive_int(max_payload_bytes, field_name="max_payload_bytes")
        size = slots * _aligned_slot_stride(payload)
        try:
            memory = shared_memory.SharedMemory(name=name, create=True, size=size)
        except (FileExistsError, OSError, ValueError) as exc:
            raise LiveFrameTransportError("Could not create live SLAM shared memory.") from exc
        spec = LiveSharedMemoryRingSpec(memory.name, slots, payload)
        ring = cls(memory, spec, owner=True)
        for slot_index in range(slots):
            _TOKEN.pack_into(
                ring._memory.buf,
                slot_index * spec.slot_stride_bytes,
                0,
            )
        return ring

    @classmethod
    def open_existing(cls, spec: LiveSharedMemoryRingSpec) -> "LiveSharedMemoryFrameRing":
        if not isinstance(spec, LiveSharedMemoryRingSpec):
            raise TypeError("spec must be a LiveSharedMemoryRingSpec.")
        try:
            memory = shared_memory.SharedMemory(name=spec.shared_memory_name, create=False)
        except (FileNotFoundError, OSError) as exc:
            raise LiveFrameTransportError("Live SLAM shared memory is unavailable.") from exc
        return cls(memory, spec, owner=False)

    @property
    def spec(self) -> LiveSharedMemoryRingSpec:
        return self._spec

    @property
    def latest_generation(self) -> int:
        return self._generation

    def publish(self, image: np.ndarray) -> LiveFrameBufferReference:
        self._ensure_open()
        if not self._owner:
            raise LiveFrameTransportError(
                "Only the shared-memory owner may publish live SLAM frames."
            )
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise LiveFrameTransportError("Live SLAM frame must be HxWx3 uint8 BGR.")
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            raise LiveFrameTransportError("Live SLAM frame dimensions must be positive.")
        contiguous = np.ascontiguousarray(array)
        payload = contiguous.reshape(-1)
        byte_length = int(payload.nbytes)
        if byte_length > self._spec.max_payload_bytes:
            raise LiveFrameTransportError(
                "Live SLAM frame exceeds the configured shared-memory slot capacity."
            )

        self._generation += 1
        generation = self._generation
        slot_index = (generation - 1) % self._spec.slot_count
        header_offset = slot_index * self._spec.slot_stride_bytes
        payload_offset = header_offset + _HEADER_BYTES
        writing_token = (generation << 1) | 1
        stable_token = generation << 1

        _TOKEN.pack_into(self._memory.buf, header_offset, writing_token)
        self._memory.buf[payload_offset : payload_offset + byte_length] = payload.tobytes()
        _TOKEN.pack_into(self._memory.buf, header_offset, stable_token)

        return LiveFrameBufferReference(
            shared_memory_name=self._spec.shared_memory_name,
            slot_index=slot_index,
            generation=generation,
            byte_length=byte_length,
            image_size=ImageSize(int(array.shape[1]), int(array.shape[0])),
            pixel_format=LiveFramePixelFormat.BGR8,
        )

    def read(self, reference: LiveFrameBufferReference) -> np.ndarray:
        self._ensure_open()
        if not isinstance(reference, LiveFrameBufferReference):
            raise TypeError("reference must be a LiveFrameBufferReference.")
        if reference.shared_memory_name != self._spec.shared_memory_name:
            raise LiveFrameTransportError("Frame reference points to another shared-memory ring.")
        if reference.slot_index >= self._spec.slot_count:
            raise LiveFrameTransportError("Frame reference slot is outside this ring.")
        if reference.byte_length > self._spec.max_payload_bytes:
            raise LiveFrameTransportError("Frame reference exceeds this ring's slot capacity.")

        header_offset = reference.slot_index * self._spec.slot_stride_bytes
        payload_offset = header_offset + _HEADER_BYTES
        expected_token = reference.generation << 1
        before = _TOKEN.unpack_from(self._memory.buf, header_offset)[0]
        if before != expected_token or before & 1:
            raise LiveFrameStaleError("Live frame reference has already been overwritten.")

        payload = bytes(
            self._memory.buf[payload_offset : payload_offset + reference.byte_length]
        )
        after = _TOKEN.unpack_from(self._memory.buf, header_offset)[0]
        if after != expected_token or after & 1:
            raise LiveFrameStaleError("Live frame changed while it was being read.")

        height = reference.image_size.height
        width = reference.image_size.width
        expected = height * width * reference.channels
        if len(payload) != expected:
            raise LiveFrameTransportError("Live frame payload size does not match its metadata.")
        return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, reference.channels).copy()

    def close(self) -> None:
        if self._closed:
            return
        self._memory.close()
        self._closed = True

    def unlink(self) -> None:
        if not self._owner:
            raise LiveFrameTransportError("Only the shared-memory owner may unlink the ring.")
        try:
            self._memory.unlink()
        except FileNotFoundError:
            pass

    def shutdown(self) -> None:
        """Close and unlink an owner ring; safe to call more than once."""
        if self._owner:
            try:
                self.unlink()
            finally:
                self.close()
        else:
            self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise LiveFrameTransportError("Live shared-memory ring is closed.")

    def __enter__(self) -> "LiveSharedMemoryFrameRing":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown() if self._owner else self.close()
