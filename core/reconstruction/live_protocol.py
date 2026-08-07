"""Versioned JSON control/event protocol for the future native live SLAM worker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata
from core.reconstruction.errors import LiveSlamProtocolError
from core.reconstruction.validation import require_nonnegative_int, require_uuid

if TYPE_CHECKING:
    from core.reconstruction.live_contracts import LiveSlamFrame, LiveSlamObservation, LiveSlamRequest
    from core.reconstruction.live_transport import LiveSharedMemoryRingSpec

LIVE_SLAM_PROTOCOL_VERSION = 1
LIVE_SLAM_MAX_MESSAGE_BYTES = 1024 * 1024


class LiveSlamMessageType(str, Enum):
    HELLO = "hello"
    READY = "ready"
    START = "start"
    STARTED = "started"
    FRAME_AVAILABLE = "frame_available"
    OBSERVATION = "observation"
    STOP = "stop"
    STOPPED = "stopped"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class LiveSlamProtocolMessage:
    message_type: LiveSlamMessageType
    session_id: str
    sequence_number: int
    payload: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(
        default_factory=tuple
    )
    protocol_version: int = LIVE_SLAM_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.message_type, LiveSlamMessageType):
            raise TypeError("message_type must be LiveSlamMessageType.")
        object.__setattr__(
            self, "session_id", require_uuid(self.session_id, field_name="session_id")
        )
        object.__setattr__(
            self,
            "sequence_number",
            require_nonnegative_int(self.sequence_number, field_name="sequence_number"),
        )
        if self.protocol_version != LIVE_SLAM_PROTOCOL_VERSION:
            raise LiveSlamProtocolError("Unsupported live SLAM protocol version.")
        object.__setattr__(self, "payload", normalize_metadata(self.payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "message_type": self.message_type.value,
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "payload": metadata_as_dict(self.payload),  # type: ignore[arg-type]
        }


def encode_live_slam_message(message: LiveSlamProtocolMessage) -> bytes:
    if not isinstance(message, LiveSlamProtocolMessage):
        raise TypeError("message must be LiveSlamProtocolMessage.")
    encoded = json.dumps(
        message.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    if len(encoded) > LIVE_SLAM_MAX_MESSAGE_BYTES:
        raise LiveSlamProtocolError("Live SLAM protocol message exceeds size limit.")
    return encoded


def decode_live_slam_message(value: bytes | str) -> LiveSlamProtocolMessage:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raise TypeError("value must be bytes or str.")
    if len(raw) > LIVE_SLAM_MAX_MESSAGE_BYTES:
        raise LiveSlamProtocolError("Live SLAM protocol message exceeds size limit.")
    try:
        parsed = json.loads(raw.decode("utf-8").strip())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveSlamProtocolError("Malformed live SLAM JSON message.") from exc
    if not isinstance(parsed, dict):
        raise LiveSlamProtocolError("Live SLAM message root must be an object.")
    try:
        protocol_version = parsed["protocol_version"]
        message_type = LiveSlamMessageType(parsed["message_type"])
        session_id = parsed["session_id"]
        sequence_number = parsed["sequence_number"]
        payload = parsed.get("payload", {})
    except (KeyError, ValueError, TypeError) as exc:
        raise LiveSlamProtocolError("Invalid live SLAM message fields.") from exc
    if not isinstance(payload, dict):
        raise LiveSlamProtocolError("Live SLAM message payload must be an object.")
    try:
        return LiveSlamProtocolMessage(
            message_type=message_type,
            session_id=session_id,
            sequence_number=sequence_number,
            payload=payload,
            protocol_version=protocol_version,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, LiveSlamProtocolError):
            raise
        raise LiveSlamProtocolError("Invalid live SLAM protocol contract.") from exc


def make_live_start_message(
    request: "LiveSlamRequest",
    ring_spec: "LiveSharedMemoryRingSpec",
    *,
    sequence_number: int = 0,
) -> LiveSlamProtocolMessage:
    from core.reconstruction.live_contracts import LiveSlamRequest
    from core.reconstruction.live_transport import LiveSharedMemoryRingSpec

    if not isinstance(request, LiveSlamRequest):
        raise TypeError("request must be LiveSlamRequest.")
    if not isinstance(ring_spec, LiveSharedMemoryRingSpec):
        raise TypeError("ring_spec must be LiveSharedMemoryRingSpec.")
    return LiveSlamProtocolMessage(
        message_type=LiveSlamMessageType.START,
        session_id=request.session_id,
        sequence_number=sequence_number,
        payload={"request": request.as_dict(), "frame_ring": ring_spec.as_dict()},
    )


def make_live_frame_available_message(
    frame: "LiveSlamFrame",
    *,
    sequence_number: int,
) -> LiveSlamProtocolMessage:
    from core.reconstruction.live_contracts import LiveSlamFrame

    if not isinstance(frame, LiveSlamFrame):
        raise TypeError("frame must be LiveSlamFrame.")
    return LiveSlamProtocolMessage(
        message_type=LiveSlamMessageType.FRAME_AVAILABLE,
        session_id=frame.session_id,
        sequence_number=sequence_number,
        payload={"frame": frame.as_dict()},
    )


def make_live_observation_message(
    observation: "LiveSlamObservation",
    *,
    sequence_number: int,
) -> LiveSlamProtocolMessage:
    from core.reconstruction.live_contracts import LiveSlamObservation

    if not isinstance(observation, LiveSlamObservation):
        raise TypeError("observation must be LiveSlamObservation.")
    return LiveSlamProtocolMessage(
        message_type=LiveSlamMessageType.OBSERVATION,
        session_id=observation.session_id,
        sequence_number=sequence_number,
        payload={"observation": observation.as_dict()},
    )
