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


def make_live_control_message(
    message_type: LiveSlamMessageType,
    session_id: str,
    *,
    sequence_number: int,
    payload: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = (),
) -> LiveSlamProtocolMessage:
    if message_type not in {
        LiveSlamMessageType.READY,
        LiveSlamMessageType.STOP,
        LiveSlamMessageType.SHUTDOWN,
        LiveSlamMessageType.HEARTBEAT,
    }:
        raise LiveSlamProtocolError(
            "make_live_control_message accepts READY/STOP/SHUTDOWN/HEARTBEAT only."
        )
    return LiveSlamProtocolMessage(
        message_type=message_type,
        session_id=session_id,
        sequence_number=sequence_number,
        payload=payload,
    )


def _payload_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LiveSlamProtocolError(f"{field_name} must be a JSON object.")
    return value


def _payload_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiveSlamProtocolError(f"{field_name} must be an integer.")
    return value


def _payload_optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveSlamProtocolError(f"{field_name} must be numeric or null.")
    return float(value)


def parse_live_observation_message(
    message: LiveSlamProtocolMessage,
    *,
    expected_session_id: str,
    expected_frame: "LiveSlamFrame | None" = None,
) -> "LiveSlamObservation":
    """Convert a native worker observation into dependency-light EIA contracts."""

    import numpy as np

    from core.reconstruction.live_contracts import (
        LIVE_SLAM_RECONSTRUCTION_FRAME,
        LiveMapSnapshot,
        LiveSlamFrame,
        LiveSlamObservation,
        LiveSlamPose,
        LiveTrackingQuality,
        LiveTrackingState,
    )
    from core.reconstruction.sfm_contracts import (
        ReconstructionGauge,
        SfmScaleStatus,
        SfmWorldAlignmentStatus,
    )

    if not isinstance(message, LiveSlamProtocolMessage):
        raise TypeError("message must be LiveSlamProtocolMessage.")
    if message.message_type is not LiveSlamMessageType.OBSERVATION:
        raise LiveSlamProtocolError("Expected a live SLAM observation message.")
    if message.session_id != expected_session_id:
        raise LiveSlamProtocolError("Live SLAM observation session_id mismatch.")
    if expected_frame is not None and not isinstance(expected_frame, LiveSlamFrame):
        raise TypeError("expected_frame must be a LiveSlamFrame or None.")

    payload = _payload_mapping(metadata_as_dict(message.payload), "payload")
    raw = _payload_mapping(payload.get("observation"), "payload.observation")
    try:
        session_id = str(raw["session_id"])
        frame_id = str(raw["frame_id"])
        sequence_index = _payload_int(raw["sequence_index"], "sequence_index")
        tracking_state = LiveTrackingState(str(raw["tracking_state"]))
    except (KeyError, ValueError) as exc:
        raise LiveSlamProtocolError("Invalid live SLAM observation identity/state.") from exc
    if session_id != expected_session_id:
        raise LiveSlamProtocolError("Native observation session_id does not match the request.")
    if expected_frame is not None:
        if frame_id != expected_frame.frame_id or sequence_index != expected_frame.sequence_index:
            raise LiveSlamProtocolError("Native observation does not correspond to the submitted frame.")

    quality_raw = _payload_mapping(raw.get("quality"), "observation.quality")
    quality = LiveTrackingQuality(
        tracked_features=_payload_int(quality_raw.get("tracked_features"), "tracked_features"),
        inlier_features=_payload_int(quality_raw.get("inlier_features"), "inlier_features"),
        processing_latency_ns=_payload_int(
            quality_raw.get("processing_latency_ns"), "processing_latency_ns"
        ),
        tracking_confidence=_payload_optional_float(
            quality_raw.get("tracking_confidence"), "tracking_confidence"
        ),
        inlier_ratio=_payload_optional_float(quality_raw.get("inlier_ratio"), "inlier_ratio"),
        reason=str(quality_raw.get("reason", "")),
    )

    pose = None
    raw_pose = raw.get("pose")
    if raw_pose is not None:
        pose_payload = _payload_mapping(raw_pose, "observation.pose")
        if pose_payload.get("pose_convention") != "T_reconstruction_from_camera":
            raise LiveSlamProtocolError(
                "Native stella pose must use T_reconstruction_from_camera at the EIA boundary."
            )
        if pose_payload.get("quaternion_order") not in {None, "wxyz"}:
            raise LiveSlamProtocolError("Native live SLAM quaternion order must be wxyz.")
        gauge_payload = _payload_mapping(pose_payload.get("gauge"), "pose.gauge")
        try:
            gauge = ReconstructionGauge(
                scale_status=SfmScaleStatus(str(gauge_payload.get("scale_status"))),
                alignment_status=SfmWorldAlignmentStatus(
                    str(gauge_payload.get("alignment_status"))
                ),
                unit_to_metre=gauge_payload.get("unit_to_metre"),
                scale_source=str(gauge_payload.get("scale_source", "")),
                relative_scale_uncertainty=_payload_optional_float(
                    gauge_payload.get("relative_scale_uncertainty"),
                    "relative_scale_uncertainty",
                ),
            )
        except (TypeError, ValueError) as exc:
            raise LiveSlamProtocolError("Invalid native live SLAM gauge metadata.") from exc
        if (
            gauge.scale_status is not SfmScaleStatus.ARBITRARY
            or gauge.alignment_status is not SfmWorldAlignmentStatus.ARBITRARY
            or gauge.unit_to_metre is not None
        ):
            raise LiveSlamProtocolError(
                "Phase 8.4B native monocular SLAM must remain arbitrary-scale."
            )
        reconstruction_frame_id = str(
            pose_payload.get("reconstruction_frame_id", LIVE_SLAM_RECONSTRUCTION_FRAME)
        )
        if reconstruction_frame_id != LIVE_SLAM_RECONSTRUCTION_FRAME:
            raise LiveSlamProtocolError("Unexpected native live reconstruction frame identifier.")
        try:
            pose = LiveSlamPose(
                pose_id=str(pose_payload["pose_id"]),
                camera_id=str(pose_payload["camera_id"]),
                frame_id=str(pose_payload["frame_id"]),
                timestamp_ns=_payload_int(pose_payload["timestamp_ns"], "pose.timestamp_ns"),
                camera_frame_id=str(pose_payload["camera_frame_id"]),
                reconstruction_frame_id=reconstruction_frame_id,
                T_reconstruction_from_camera=np.asarray(
                    pose_payload["T_reconstruction_from_camera"], dtype=np.float64
                ),
                gauge=gauge,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, LiveSlamProtocolError):
                raise
            raise LiveSlamProtocolError("Invalid native live SLAM pose payload.") from exc

    map_snapshot = None
    raw_map = raw.get("map_snapshot")
    if raw_map is not None:
        map_payload = _payload_mapping(raw_map, "observation.map_snapshot")
        map_snapshot = LiveMapSnapshot(
            revision=_payload_int(map_payload.get("revision"), "map.revision"),
            keyframe_count=_payload_int(
                map_payload.get("keyframe_count"), "map.keyframe_count"
            ),
            landmark_count=_payload_int(
                map_payload.get("landmark_count"), "map.landmark_count"
            ),
        )

    is_keyframe = raw.get("is_keyframe", False)
    if not isinstance(is_keyframe, bool):
        raise LiveSlamProtocolError("observation.is_keyframe must be bool.")
    try:
        return LiveSlamObservation(
            session_id=session_id,
            frame_id=frame_id,
            sequence_index=sequence_index,
            tracking_state=tracking_state,
            quality=quality,
            pose=pose,
            is_keyframe=is_keyframe,
            keyframe_id=str(raw.get("keyframe_id", "")),
            map_snapshot=map_snapshot,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, LiveSlamProtocolError):
            raise
        raise LiveSlamProtocolError("Native live SLAM observation violates EIA contracts.") from exc


def parse_live_worker_error(message: LiveSlamProtocolMessage) -> dict[str, object]:
    if not isinstance(message, LiveSlamProtocolMessage):
        raise TypeError("message must be LiveSlamProtocolMessage.")
    if message.message_type is not LiveSlamMessageType.ERROR:
        raise LiveSlamProtocolError("Expected a live SLAM ERROR message.")
    payload = _payload_mapping(metadata_as_dict(message.payload), "payload")
    code = str(payload.get("code", "worker_error")).strip() or "worker_error"
    text = str(payload.get("message", "Live SLAM worker failed.")).strip()
    if not text:
        text = "Live SLAM worker failed."
    retryable = payload.get("retryable", False)
    if not isinstance(retryable, bool):
        raise LiveSlamProtocolError("Worker error retryable field must be bool.")
    return {
        "code": code,
        "message": text,
        "retryable": retryable,
        "frame_id": str(payload.get("frame_id", "")),
        "sequence_index": payload.get("sequence_index"),
    }
