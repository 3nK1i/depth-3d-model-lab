"""Dependency-light contracts for Phase 8.4 live visual SLAM.

Live monocular poses deliberately use ``reconstruction`` naming. They are not
metric world poses until a later explicit scale/world-alignment stage converts
them. Phase 8.4A therefore forbids metric scale claims and loop/global
optimization enablement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np

from core.contracts import CalibrationStatus, ImageSize
from core.contracts.validation import (
    JsonValue,
    metadata_as_dict,
    normalize_metadata,
    require_nonempty,
)
from core.reconstruction.camera import ReconstructionCamera
from core.reconstruction.contracts import InputSourceKind, InputSourceMetadata, TimestampOrigin
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.poses import rotation_matrix_to_quaternion_wxyz
from core.reconstruction.profiles import ReconstructionQualityProfile
from core.reconstruction.sfm_contracts import (
    ReconstructionGauge,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
)
from core.reconstruction.sfm_coordinates import validate_rigid_matrix
from core.reconstruction.validation import (
    require_nonnegative_int,
    require_positive_int,
    require_probability,
    require_timestamp_ns,
    require_uuid,
)

LIVE_SLAM_SCHEMA_VERSION = 1
LIVE_SLAM_RECONSTRUCTION_FRAME = "arbitrary_gauge_live_reconstruction"


class LiveTrackingState(str, Enum):
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    DEGRADED = "degraded"
    LOST = "lost"
    RECOVERING = "recovering"
    STOPPED = "stopped"


class LiveFramePixelFormat(str, Enum):
    BGR8 = "bgr8"


@dataclass(frozen=True, slots=True)
class LiveFrameBufferReference:
    """Serializable reference to one immutable generation in shared memory."""

    shared_memory_name: str
    slot_index: int
    generation: int
    byte_length: int
    image_size: ImageSize
    pixel_format: LiveFramePixelFormat = LiveFramePixelFormat.BGR8
    dtype: str = "uint8"
    channels: int = 3
    schema_version: int = LIVE_SLAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shared_memory_name",
            require_nonempty(self.shared_memory_name, field_name="shared_memory_name"),
        )
        object.__setattr__(
            self, "slot_index", require_nonnegative_int(self.slot_index, field_name="slot_index")
        )
        object.__setattr__(
            self, "generation", require_positive_int(self.generation, field_name="generation")
        )
        object.__setattr__(
            self, "byte_length", require_positive_int(self.byte_length, field_name="byte_length")
        )
        if not isinstance(self.image_size, ImageSize):
            raise TypeError("image_size must be an ImageSize.")
        if not isinstance(self.pixel_format, LiveFramePixelFormat):
            raise TypeError("pixel_format must be a LiveFramePixelFormat.")
        if self.pixel_format is not LiveFramePixelFormat.BGR8:
            raise ReconstructionValidationError("Phase 8.4A supports BGR8 live frames only.")
        if self.dtype != "uint8" or self.channels != 3:
            raise ReconstructionValidationError("BGR8 buffers require uint8 with 3 channels.")
        expected_bytes = self.image_size.width * self.image_size.height * self.channels
        if self.byte_length != expected_bytes:
            raise ReconstructionValidationError(
                "Live frame byte_length must exactly match image_size * channels."
            )
        if self.schema_version != LIVE_SLAM_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported live frame reference schema version.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shared_memory_name": self.shared_memory_name,
            "slot_index": self.slot_index,
            "generation": self.generation,
            "byte_length": self.byte_length,
            "image_size": self.image_size.as_dict(),
            "pixel_format": self.pixel_format.value,
            "dtype": self.dtype,
            "channels": self.channels,
        }


@dataclass(frozen=True, slots=True)
class LiveSlamFrame:
    """One live frame envelope; pixel bytes remain outside control IPC."""

    session_id: str
    source_id: str
    camera_id: str
    frame_id: str
    source_frame_id: int
    sequence_index: int
    capture_timestamp_ns: int
    timestamp_origin: TimestampOrigin
    buffer: LiveFrameBufferReference

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_id", require_uuid(self.session_id, field_name="session_id")
        )
        object.__setattr__(
            self, "source_id", require_nonempty(self.source_id, field_name="source_id")
        )
        object.__setattr__(
            self, "camera_id", require_nonempty(self.camera_id, field_name="camera_id")
        )
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        object.__setattr__(
            self,
            "source_frame_id",
            require_nonnegative_int(self.source_frame_id, field_name="source_frame_id"),
        )
        object.__setattr__(
            self,
            "sequence_index",
            require_nonnegative_int(self.sequence_index, field_name="sequence_index"),
        )
        object.__setattr__(
            self,
            "capture_timestamp_ns",
            require_timestamp_ns(self.capture_timestamp_ns, field_name="capture_timestamp_ns"),
        )
        if not isinstance(self.timestamp_origin, TimestampOrigin):
            raise TypeError("timestamp_origin must be a TimestampOrigin.")
        if self.timestamp_origin is TimestampOrigin.UNKNOWN:
            raise ReconstructionValidationError("Live SLAM frames require a known timestamp origin.")
        if not isinstance(self.buffer, LiveFrameBufferReference):
            raise TypeError("buffer must be a LiveFrameBufferReference.")

    @property
    def image_size(self) -> ImageSize:
        return self.buffer.image_size

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source_id": self.source_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "source_frame_id": self.source_frame_id,
            "sequence_index": self.sequence_index,
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "timestamp_origin": self.timestamp_origin.value,
            "buffer": self.buffer.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class LiveSlamPose:
    """Camera-to-reconstruction pose in arbitrary monocular SLAM units."""

    pose_id: str
    camera_id: str
    frame_id: str
    timestamp_ns: int
    camera_frame_id: str
    T_reconstruction_from_camera: np.ndarray
    reconstruction_frame_id: str = LIVE_SLAM_RECONSTRUCTION_FRAME
    gauge: ReconstructionGauge = field(default_factory=ReconstructionGauge)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose_id", require_nonempty(self.pose_id, field_name="pose_id"))
        object.__setattr__(
            self, "camera_id", require_nonempty(self.camera_id, field_name="camera_id")
        )
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        object.__setattr__(self, "timestamp_ns", require_timestamp_ns(self.timestamp_ns))
        object.__setattr__(
            self,
            "camera_frame_id",
            require_nonempty(self.camera_frame_id, field_name="camera_frame_id"),
        )
        object.__setattr__(
            self,
            "reconstruction_frame_id",
            require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id"),
        )
        if self.camera_frame_id == self.reconstruction_frame_id:
            raise ReconstructionValidationError(
                "Live SLAM camera and reconstruction frame IDs must differ."
            )
        object.__setattr__(
            self,
            "T_reconstruction_from_camera",
            validate_rigid_matrix(
                self.T_reconstruction_from_camera,
                field_name="T_reconstruction_from_camera",
            ),
        )
        if not isinstance(self.gauge, ReconstructionGauge):
            raise TypeError("gauge must be a ReconstructionGauge.")
        if (
            self.gauge.scale_status is not SfmScaleStatus.ARBITRARY
            or self.gauge.alignment_status is not SfmWorldAlignmentStatus.ARBITRARY
            or self.gauge.unit_to_metre is not None
        ):
            raise ReconstructionValidationError(
                "Phase 8.4 live monocular poses must remain arbitrary-scale and unaligned."
            )

    @property
    def quaternion_wxyz(self) -> np.ndarray:
        return rotation_matrix_to_quaternion_wxyz(
            self.T_reconstruction_from_camera[:3, :3]
        )

    @property
    def position_reconstruction(self) -> np.ndarray:
        value = np.array(self.T_reconstruction_from_camera[:3, 3], copy=True)
        value.setflags(write=False)
        return value

    def as_dict(self) -> dict[str, Any]:
        return {
            "pose_id": self.pose_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "camera_frame_id": self.camera_frame_id,
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "pose_convention": "T_reconstruction_from_camera",
            "T_reconstruction_from_camera": self.T_reconstruction_from_camera.tolist(),
            "quaternion_order": "wxyz",
            "quaternion_wxyz": self.quaternion_wxyz.tolist(),
            "position_reconstruction": self.position_reconstruction.tolist(),
            "gauge": self.gauge.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class LiveTrackingQuality:
    tracked_features: int
    inlier_features: int
    processing_latency_ns: int
    tracking_confidence: float | None = None
    inlier_ratio: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        tracked = require_nonnegative_int(self.tracked_features, field_name="tracked_features")
        inliers = require_nonnegative_int(self.inlier_features, field_name="inlier_features")
        if inliers > tracked:
            raise ReconstructionValidationError("inlier_features cannot exceed tracked_features.")
        object.__setattr__(self, "tracked_features", tracked)
        object.__setattr__(self, "inlier_features", inliers)
        object.__setattr__(
            self,
            "processing_latency_ns",
            require_nonnegative_int(self.processing_latency_ns, field_name="processing_latency_ns"),
        )
        if self.tracking_confidence is not None:
            object.__setattr__(
                self,
                "tracking_confidence",
                require_probability(self.tracking_confidence, field_name="tracking_confidence"),
            )
        ratio = self.inlier_ratio
        if tracked == 0 and ratio is not None:
            raise ReconstructionValidationError(
                "inlier_ratio must be None when tracked_features is zero."
            )
        if ratio is None and tracked > 0:
            ratio = inliers / tracked
        if ratio is not None:
            ratio = require_probability(ratio, field_name="inlier_ratio")
            if abs(ratio - inliers / tracked) > 1e-9:
                raise ReconstructionValidationError(
                    "inlier_ratio must agree with inlier_features / tracked_features."
                )
            object.__setattr__(self, "inlier_ratio", ratio)
        object.__setattr__(self, "reason", str(self.reason).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "tracked_features": self.tracked_features,
            "inlier_features": self.inlier_features,
            "inlier_ratio": self.inlier_ratio,
            "tracking_confidence": self.tracking_confidence,
            "processing_latency_ns": self.processing_latency_ns,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class LiveMapSnapshot:
    revision: int
    keyframe_count: int
    landmark_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "revision", require_nonnegative_int(self.revision, field_name="revision")
        )
        object.__setattr__(
            self,
            "keyframe_count",
            require_nonnegative_int(self.keyframe_count, field_name="keyframe_count"),
        )
        object.__setattr__(
            self,
            "landmark_count",
            require_nonnegative_int(self.landmark_count, field_name="landmark_count"),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "revision": self.revision,
            "keyframe_count": self.keyframe_count,
            "landmark_count": self.landmark_count,
        }


@dataclass(frozen=True, slots=True)
class LiveSlamObservation:
    session_id: str
    frame_id: str
    sequence_index: int
    tracking_state: LiveTrackingState
    quality: LiveTrackingQuality
    pose: LiveSlamPose | None = None
    is_keyframe: bool = False
    keyframe_id: str = ""
    map_snapshot: LiveMapSnapshot | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_id", require_uuid(self.session_id, field_name="session_id")
        )
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        object.__setattr__(
            self,
            "sequence_index",
            require_nonnegative_int(self.sequence_index, field_name="sequence_index"),
        )
        if not isinstance(self.tracking_state, LiveTrackingState):
            raise TypeError("tracking_state must be a LiveTrackingState.")
        if not isinstance(self.quality, LiveTrackingQuality):
            raise TypeError("quality must be a LiveTrackingQuality.")
        if self.pose is not None:
            if not isinstance(self.pose, LiveSlamPose):
                raise TypeError("pose must be a LiveSlamPose or None.")
            if self.pose.frame_id != self.frame_id:
                raise ReconstructionValidationError("Observation pose frame_id must match frame_id.")
            if self.pose.timestamp_ns < 0:
                raise ReconstructionValidationError("Observation pose timestamp must be non-negative.")
        if self.tracking_state is LiveTrackingState.TRACKING and self.pose is None:
            raise ReconstructionValidationError("TRACKING observations require a live SLAM pose.")
        if self.tracking_state in {
            LiveTrackingState.INITIALIZING,
            LiveTrackingState.LOST,
            LiveTrackingState.STOPPED,
        } and self.pose is not None:
            raise ReconstructionValidationError(
                f"{self.tracking_state.value} observations must not claim a current pose."
            )
        if not isinstance(self.is_keyframe, bool):
            raise TypeError("is_keyframe must be a bool.")
        keyframe_id = str(self.keyframe_id).strip()
        if self.is_keyframe:
            if self.pose is None:
                raise ReconstructionValidationError("Keyframe observations require a pose.")
            keyframe_id = require_nonempty(keyframe_id, field_name="keyframe_id")
        elif keyframe_id:
            raise ReconstructionValidationError(
                "keyframe_id may be set only when is_keyframe is True."
            )
        object.__setattr__(self, "keyframe_id", keyframe_id)
        if self.map_snapshot is not None and not isinstance(self.map_snapshot, LiveMapSnapshot):
            raise TypeError("map_snapshot must be a LiveMapSnapshot or None.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "sequence_index": self.sequence_index,
            "tracking_state": self.tracking_state.value,
            "quality": self.quality.as_dict(),
            "pose": None if self.pose is None else self.pose.as_dict(),
            "is_keyframe": self.is_keyframe,
            "keyframe_id": self.keyframe_id,
            "map_snapshot": None if self.map_snapshot is None else self.map_snapshot.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class LiveSlamRequest:
    session_id: str
    source: InputSourceMetadata
    camera: ReconstructionCamera
    profile: ReconstructionQualityProfile
    backend_id: str
    simulated_live: bool = False
    enable_mapping: bool = True
    enable_loop_closure: bool = False
    enable_global_optimization: bool = False
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_id", require_uuid(self.session_id, field_name="session_id")
        )
        if not isinstance(self.source, InputSourceMetadata):
            raise TypeError("source must be InputSourceMetadata.")
        if not isinstance(self.camera, ReconstructionCamera):
            raise TypeError("camera must be ReconstructionCamera.")
        if not isinstance(self.profile, ReconstructionQualityProfile):
            raise TypeError("profile must be ReconstructionQualityProfile.")
        object.__setattr__(
            self, "backend_id", require_nonempty(self.backend_id, field_name="backend_id")
        )
        if not isinstance(self.simulated_live, bool):
            raise TypeError("simulated_live must be a bool.")
        if not isinstance(self.enable_mapping, bool):
            raise TypeError("enable_mapping must be a bool.")
        if not isinstance(self.enable_loop_closure, bool) or not isinstance(
            self.enable_global_optimization, bool
        ):
            raise TypeError("Loop/global optimization enablement fields must be bool values.")
        if self.enable_loop_closure or self.enable_global_optimization:
            raise ReconstructionValidationError(
                "Loop closure and global optimization are deferred to Phase 8.5."
            )

        live_kinds = {InputSourceKind.USB_CAMERA, InputSourceKind.RTSP_STREAM}
        if self.source.kind not in live_kinds:
            if not (
                self.simulated_live and self.source.kind is InputSourceKind.RECORDED_VIDEO
            ):
                raise ReconstructionValidationError(
                    "Live SLAM accepts USB/RTSP sources, or recorded video only in simulated-live mode."
                )
        if self.source.timestamp_origin is TimestampOrigin.UNKNOWN:
            raise ReconstructionValidationError("Live SLAM requires a known source timestamp origin.")
        if self.source.kind is InputSourceKind.RTSP_STREAM:
            password = urlsplit(self.source.source_uri).password
            if password not in {None, "***"}:
                raise ReconstructionValidationError(
                    "Live SLAM RTSP metadata must not contain plaintext credentials."
                )
        if self.source.expected_image_size is not None and (
            self.source.expected_image_size != self.camera.image_size
        ):
            raise ReconstructionValidationError(
                "Live source image size must match the calibrated camera image size."
            )
        if self.camera.intrinsics.calibration_status is not CalibrationStatus.CALIBRATED:
            raise ReconstructionValidationError(
                "Production live SLAM requires explicitly calibrated camera intrinsics."
            )
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LIVE_SLAM_SCHEMA_VERSION,
            "session_id": self.session_id,
            "source": self.source.as_dict(),
            "camera": self.camera.as_dict(),
            "profile": self.profile.value,
            "backend_id": self.backend_id,
            "simulated_live": self.simulated_live,
            "enable_mapping": self.enable_mapping,
            "enable_loop_closure": self.enable_loop_closure,
            "enable_global_optimization": self.enable_global_optimization,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }
