"""Shared immutable reconstruction identifiers, inputs, frames, and keyframes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import uuid4

from core.contracts import ImageSize
from core.contracts.validation import (
    JsonValue,
    metadata_as_dict,
    normalize_metadata,
    require_finite,
    require_nonempty,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.validation import (
    require_nonnegative_int,
    require_probability,
    require_timestamp_ns,
    require_uuid,
)


class ReconstructionPath(str, Enum):
    OFFLINE = "offline"
    LIVE = "live"


class InputSourceKind(str, Enum):
    IMAGE_SEQUENCE = "image_sequence"
    RECORDED_VIDEO = "recorded_video"
    USB_CAMERA = "usb_camera"
    RTSP_STREAM = "rtsp_stream"


class TimestampOrigin(str, Enum):
    PRESENTATION_TIMESTAMP = "presentation_timestamp"
    CAPTURE_CLOCK = "capture_clock"
    FILE_METADATA = "file_metadata"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class FrameValidity(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


class FrameInvalidReason(str, Enum):
    DECODE_FAILED = "decode_failed"
    TIMESTAMP_INVALID = "timestamp_invalid"
    DIMENSION_MISMATCH = "dimension_mismatch"
    UNSUPPORTED_FORMAT = "unsupported_format"
    DUPLICATE = "duplicate"
    INSUFFICIENT_QUALITY = "insufficient_quality"
    CANCELLED = "cancelled"
    OTHER = "other"


class KeyframeSelectionReason(str, Enum):
    FIRST_FRAME = "first_frame"
    TEMPORAL_SPACING = "temporal_spacing"
    VISUAL_CHANGE = "visual_change"
    MOTION = "motion"
    TRACKING_QUALITY = "tracking_quality"
    LOOP_CANDIDATE = "loop_candidate"
    USER_SELECTED = "user_selected"
    BACKEND_SELECTED = "backend_selected"


def new_reconstruction_session_id() -> str:
    return str(uuid4())


@dataclass(frozen=True, slots=True)
class InputSourceMetadata:
    source_id: str
    kind: InputSourceKind
    display_name: str
    source_uri: str
    expected_image_size: ImageSize | None = None
    expected_frame_rate_hz: float | None = None
    timestamp_origin: TimestampOrigin = TimestampOrigin.UNKNOWN
    fingerprint: str = ""
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", require_nonempty(self.source_id, field_name="source_id"))
        object.__setattr__(
            self, "display_name", require_nonempty(self.display_name, field_name="display_name")
        )
        object.__setattr__(
            self, "source_uri", require_nonempty(self.source_uri, field_name="source_uri")
        )
        if not isinstance(self.kind, InputSourceKind):
            raise TypeError("kind must be an InputSourceKind.")
        if self.expected_image_size is not None and not isinstance(
            self.expected_image_size, ImageSize
        ):
            raise TypeError("expected_image_size must be an ImageSize or None.")
        if self.expected_frame_rate_hz is not None:
            rate = require_finite(
                self.expected_frame_rate_hz,
                field_name="expected_frame_rate_hz",
            )
            if rate <= 0.0:
                raise ReconstructionValidationError(
                    "expected_frame_rate_hz must be positive when provided."
                )
            object.__setattr__(self, "expected_frame_rate_hz", rate)
        if not isinstance(self.timestamp_origin, TimestampOrigin):
            raise TypeError("timestamp_origin must be a TimestampOrigin.")
        object.__setattr__(self, "fingerprint", str(self.fingerprint).strip().lower())
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))

    @property
    def path(self) -> ReconstructionPath:
        if self.kind in {InputSourceKind.IMAGE_SEQUENCE, InputSourceKind.RECORDED_VIDEO}:
            return ReconstructionPath.OFFLINE
        return ReconstructionPath.LIVE

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind.value,
            "path": self.path.value,
            "display_name": self.display_name,
            "source_uri": self.source_uri,
            "expected_image_size": (
                self.expected_image_size.as_dict() if self.expected_image_size else None
            ),
            "expected_frame_rate_hz": self.expected_frame_rate_hz,
            "timestamp_origin": self.timestamp_origin.value,
            "fingerprint": self.fingerprint,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }


@dataclass(frozen=True, slots=True)
class ReconstructionFrame:
    frame_id: str
    session_id: str
    source_id: str
    sequence_index: int
    timestamp_ns: int
    image_size: ImageSize
    validity: FrameValidity = FrameValidity.VALID
    invalid_reasons: tuple[FrameInvalidReason, ...] = ()
    content_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        object.__setattr__(
            self, "session_id", require_uuid(self.session_id, field_name="session_id")
        )
        object.__setattr__(self, "source_id", require_nonempty(self.source_id, field_name="source_id"))
        object.__setattr__(
            self,
            "sequence_index",
            require_nonnegative_int(self.sequence_index, field_name="sequence_index"),
        )
        object.__setattr__(self, "timestamp_ns", require_timestamp_ns(self.timestamp_ns))
        if not isinstance(self.image_size, ImageSize):
            raise TypeError("image_size must be an ImageSize.")
        if not isinstance(self.validity, FrameValidity):
            raise TypeError("validity must be a FrameValidity.")
        reasons = tuple(self.invalid_reasons)
        if any(not isinstance(reason, FrameInvalidReason) for reason in reasons):
            raise TypeError("invalid_reasons must contain FrameInvalidReason values.")
        if self.validity is FrameValidity.VALID and reasons:
            raise ReconstructionValidationError("Valid frames cannot contain invalid reasons.")
        if self.validity is FrameValidity.INVALID and not reasons:
            raise ReconstructionValidationError("Invalid frames require at least one reason.")
        if len(set(reasons)) != len(reasons):
            raise ReconstructionValidationError("Frame invalid reasons must be unique.")
        digest = str(self.content_sha256).strip().lower()
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ReconstructionValidationError(
                "content_sha256 must be a 64-character hexadecimal digest."
            )
        object.__setattr__(self, "invalid_reasons", reasons)
        object.__setattr__(self, "content_sha256", digest)

    @property
    def is_valid(self) -> bool:
        return self.validity is FrameValidity.VALID

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "session_id": self.session_id,
            "source_id": self.source_id,
            "sequence_index": self.sequence_index,
            "timestamp_ns": self.timestamp_ns,
            "image_size": self.image_size.as_dict(),
            "validity": self.validity.value,
            "invalid_reasons": [reason.value for reason in self.invalid_reasons],
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionKeyframe:
    keyframe_id: str
    session_id: str
    frame_id: str
    selection_sequence: int
    reason: KeyframeSelectionReason
    score: float | None = None
    pose_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "keyframe_id", require_nonempty(self.keyframe_id, field_name="keyframe_id")
        )
        object.__setattr__(
            self, "session_id", require_uuid(self.session_id, field_name="session_id")
        )
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        object.__setattr__(
            self,
            "selection_sequence",
            require_nonnegative_int(self.selection_sequence, field_name="selection_sequence"),
        )
        if not isinstance(self.reason, KeyframeSelectionReason):
            raise TypeError("reason must be a KeyframeSelectionReason.")
        if self.score is not None:
            object.__setattr__(self, "score", require_probability(self.score, field_name="score"))
        object.__setattr__(self, "pose_id", str(self.pose_id).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "selection_sequence": self.selection_sequence,
            "reason": self.reason.value,
            "score": self.score,
            "pose_id": self.pose_id,
        }
