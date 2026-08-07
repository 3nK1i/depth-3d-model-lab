"""Adapter from stable capture SourceFrame objects into live SLAM envelopes."""

from __future__ import annotations

from core.capture.frame import SourceFrame
from core.reconstruction.camera import ReconstructionCamera
from core.reconstruction.contracts import InputSourceKind, InputSourceMetadata, TimestampOrigin
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.live_contracts import LiveSlamFrame
from core.reconstruction.live_transport import LiveSharedMemoryFrameRing
from core.reconstruction.validation import require_uuid


class LiveSourceFrameAdapter:
    """Publish stable capture frames without modifying the Phase 2 SourceFrame API."""

    def __init__(
        self,
        *,
        session_id: str,
        source: InputSourceMetadata,
        camera: ReconstructionCamera,
        ring: LiveSharedMemoryFrameRing,
    ) -> None:
        self._session_id = require_uuid(session_id, field_name="session_id")
        if not isinstance(source, InputSourceMetadata):
            raise TypeError("source must be InputSourceMetadata.")
        if not isinstance(camera, ReconstructionCamera):
            raise TypeError("camera must be ReconstructionCamera.")
        if not isinstance(ring, LiveSharedMemoryFrameRing):
            raise TypeError("ring must be LiveSharedMemoryFrameRing.")
        if source.kind not in {
            InputSourceKind.USB_CAMERA,
            InputSourceKind.RTSP_STREAM,
            InputSourceKind.RECORDED_VIDEO,
        }:
            raise ReconstructionValidationError(
                "Live frame adapter accepts only USB, RTSP, or recorded-video sources."
            )
        if source.timestamp_origin is not TimestampOrigin.CAPTURE_CLOCK:
            raise ReconstructionValidationError(
                "SourceFrame.timestamp uses the capture monotonic clock; live frame adapter "
                "metadata must therefore declare CAPTURE_CLOCK."
            )
        if (
            source.expected_image_size is not None
            and source.expected_image_size != camera.image_size
        ):
            raise ReconstructionValidationError(
                "Live frame adapter source size must match camera calibration size."
            )
        self._source = source
        self._camera = camera
        self._ring = ring
        self._next_sequence = 0
        self._last_timestamp_ns: int | None = None
        self._last_source_frame_id: int | None = None

    def publish(self, frame: SourceFrame) -> LiveSlamFrame:
        if not isinstance(frame, SourceFrame):
            raise TypeError("frame must be SourceFrame.")
        if frame.frame_id < 0:
            raise ReconstructionValidationError("SourceFrame.frame_id must be non-negative.")
        if (
            self._last_source_frame_id is not None
            and frame.frame_id <= self._last_source_frame_id
        ):
            raise ReconstructionValidationError(
                "Live source frame IDs must be strictly increasing within a connection."
            )
        if (
            frame.width != self._camera.image_size.width
            or frame.height != self._camera.image_size.height
        ):
            raise ReconstructionValidationError(
                "Captured frame dimensions do not match the live SLAM camera calibration."
            )
        timestamp_ns = int(round(float(frame.timestamp) * 1_000_000_000.0))
        if timestamp_ns < 0:
            raise ReconstructionValidationError("Live capture timestamp must be non-negative.")
        if self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            raise ReconstructionValidationError(
                "Live capture timestamps must be strictly increasing."
            )
        reference = self._ring.publish(frame.image)
        sequence = self._next_sequence
        envelope = LiveSlamFrame(
            session_id=self._session_id,
            source_id=self._source.source_id,
            camera_id=self._camera.camera_id,
            frame_id=f"live-frame-{sequence}",
            source_frame_id=int(frame.frame_id),
            sequence_index=sequence,
            capture_timestamp_ns=timestamp_ns,
            timestamp_origin=self._source.timestamp_origin,
            buffer=reference,
        )
        self._next_sequence += 1
        self._last_timestamp_ns = timestamp_ns
        self._last_source_frame_id = int(frame.frame_id)
        return envelope
