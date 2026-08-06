"""Stable JSON serialization for Phase 8 reconstruction metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import numpy as np

from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageSize,
    LengthUnit,
    RigidTransform3D,
)
from core.models.metadata import IntrinsicsSource
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera
from core.reconstruction.checkpoint import (
    CheckpointArtifact,
    CheckpointMetadata,
    RECONSTRUCTION_SCHEMA_VERSION,
)
from core.reconstruction.contracts import (
    FrameInvalidReason,
    FrameValidity,
    InputSourceKind,
    InputSourceMetadata,
    KeyframeSelectionReason,
    ReconstructionFrame,
    ReconstructionKeyframe,
    ReconstructionPath,
    TimestampOrigin,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.events import ReconstructionStage
from core.reconstruction.poses import CameraPose
from core.reconstruction.profiles import ReconstructionQualityProfile
from core.reconstruction.provenance import BackendProvenance, ReconstructionProvenance
from core.reconstruction.session import ReconstructionSessionSnapshot, ReconstructionSessionState
from core.reconstruction.sparse import SparseLandmark, SparseObservation
from core.reconstruction.trajectory import ReconstructionTrajectory


@dataclass(frozen=True, slots=True)
class ReconstructionMetadataBundle:
    session: ReconstructionSessionSnapshot
    cameras: tuple[ReconstructionCamera, ...]
    trajectory: ReconstructionTrajectory
    sparse_landmarks: tuple[SparseLandmark, ...]
    checkpoint: CheckpointMetadata
    provenance: ReconstructionProvenance
    schema_version: int = RECONSTRUCTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != RECONSTRUCTION_SCHEMA_VERSION:
            raise ReconstructionValidationError(
                f"Unsupported reconstruction schema version: {self.schema_version}."
            )
        if self.checkpoint.session_id != self.session.session_id:
            raise ReconstructionValidationError(
                "Checkpoint and session IDs must match in a metadata bundle."
            )
        cameras = tuple(self.cameras)
        landmarks = tuple(self.sparse_landmarks)
        if any(not isinstance(item, ReconstructionCamera) for item in cameras):
            raise TypeError("cameras must contain ReconstructionCamera values.")
        if any(not isinstance(item, SparseLandmark) for item in landmarks):
            raise TypeError("sparse_landmarks must contain SparseLandmark values.")
        camera_ids = [item.camera_id for item in cameras]
        landmark_ids = [item.landmark_id for item in landmarks]
        if len(set(camera_ids)) != len(camera_ids):
            raise ReconstructionValidationError("Camera IDs must be unique.")
        if len(set(landmark_ids)) != len(landmark_ids):
            raise ReconstructionValidationError("Sparse landmark IDs must be unique.")
        keyframe_ids = {item.keyframe_id for item in self.session.keyframes}
        for sample in self.trajectory.samples:
            if sample.camera_id not in camera_ids:
                raise ReconstructionValidationError(
                    "Trajectory samples must reference a declared camera."
                )
            if sample.frame_id not in {item.frame_id for item in self.session.frames}:
                raise ReconstructionValidationError(
                    "Trajectory samples must reference declared session frames."
                )
        for landmark in landmarks:
            if any(item.keyframe_id not in keyframe_ids for item in landmark.observations):
                raise ReconstructionValidationError(
                    "Sparse observations must reference declared keyframes."
                )
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "sparse_landmarks", landmarks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session": self.session.as_dict(),
            "cameras": [camera.as_dict() for camera in self.cameras],
            "trajectory": self.trajectory.as_dict(),
            "sparse_landmarks": [landmark.as_dict() for landmark in self.sparse_landmarks],
            "checkpoint": self.checkpoint.as_dict(),
            "provenance": self.provenance.as_dict(),
        }


def metadata_bundle_to_json(
    bundle: ReconstructionMetadataBundle,
    *,
    indent: int | None = 2,
) -> str:
    if not isinstance(bundle, ReconstructionMetadataBundle):
        raise TypeError("bundle must be a ReconstructionMetadataBundle.")
    if indent is not None and (
        isinstance(indent, bool) or int(indent) != indent or int(indent) < 0
    ):
        raise ReconstructionValidationError("indent must be a non-negative integer or None.")
    return json.dumps(
        bundle.as_dict(),
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
    )


def metadata_bundle_from_json(payload: str | bytes | bytearray) -> ReconstructionMetadataBundle:
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ReconstructionValidationError("Reconstruction metadata must be valid JSON.") from exc
    if not isinstance(raw, Mapping):
        raise ReconstructionValidationError("Reconstruction metadata root must be an object.")
    try:
        schema_version = int(raw["schema_version"])
        if schema_version != RECONSTRUCTION_SCHEMA_VERSION:
            raise ReconstructionValidationError(
                f"Unsupported reconstruction schema version: {schema_version}."
            )
        return ReconstructionMetadataBundle(
            session=_parse_session(_mapping(raw["session"], "session")),
            cameras=tuple(
                _parse_camera(_mapping(item, "camera"))
                for item in _sequence(raw["cameras"], "cameras")
            ),
            trajectory=_parse_trajectory(_mapping(raw["trajectory"], "trajectory")),
            sparse_landmarks=tuple(
                _parse_landmark(_mapping(item, "sparse_landmark"))
                for item in _sequence(raw["sparse_landmarks"], "sparse_landmarks")
            ),
            checkpoint=_parse_checkpoint(_mapping(raw["checkpoint"], "checkpoint")),
            provenance=_parse_provenance(_mapping(raw["provenance"], "provenance")),
            schema_version=schema_version,
        )
    except KeyError as exc:
        raise ReconstructionValidationError(
            f"Reconstruction metadata is missing required field: {exc.args[0]}."
        ) from exc
    except ReconstructionValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(
            f"Reconstruction metadata contains an invalid value: {exc}."
        ) from exc


def _parse_session(raw: Mapping[str, Any]) -> ReconstructionSessionSnapshot:
    source = _parse_source(_mapping(raw["source"], "source"))
    frames = tuple(_parse_frame(_mapping(item, "frame")) for item in _sequence(raw["frames"], "frames"))
    keyframes = tuple(
        _parse_keyframe(_mapping(item, "keyframe"))
        for item in _sequence(raw["keyframes"], "keyframes")
    )
    return ReconstructionSessionSnapshot(
        session_id=str(raw["session_id"]),
        path=ReconstructionPath(str(raw["path"])),
        source=source,
        profile=ReconstructionQualityProfile(str(raw["profile"])),
        state=ReconstructionSessionState(str(raw["state"])),
        frames=frames,
        keyframes=keyframes,
        created_at_utc=str(raw["created_at_utc"]),
        updated_at_utc=str(raw["updated_at_utc"]),
        failure_code=str(raw.get("failure_code", "")),
        failure_message=str(raw.get("failure_message", "")),
    )


def _parse_source(raw: Mapping[str, Any]) -> InputSourceMetadata:
    size_raw = raw.get("expected_image_size")
    size = _parse_image_size(_mapping(size_raw, "expected_image_size")) if size_raw else None
    return InputSourceMetadata(
        source_id=str(raw["source_id"]),
        kind=InputSourceKind(str(raw["kind"])),
        display_name=str(raw["display_name"]),
        source_uri=str(raw["source_uri"]),
        expected_image_size=size,
        expected_frame_rate_hz=(
            None if raw.get("expected_frame_rate_hz") is None else float(raw["expected_frame_rate_hz"])
        ),
        timestamp_origin=TimestampOrigin(str(raw["timestamp_origin"])),
        fingerprint=str(raw.get("fingerprint", "")),
        metadata=_mapping(raw.get("metadata", {}), "source metadata"),
    )


def _parse_frame(raw: Mapping[str, Any]) -> ReconstructionFrame:
    return ReconstructionFrame(
        frame_id=str(raw["frame_id"]),
        session_id=str(raw["session_id"]),
        source_id=str(raw["source_id"]),
        sequence_index=int(raw["sequence_index"]),
        timestamp_ns=int(raw["timestamp_ns"]),
        image_size=_parse_image_size(_mapping(raw["image_size"], "image_size")),
        validity=FrameValidity(str(raw["validity"])),
        invalid_reasons=tuple(
            FrameInvalidReason(str(item))
            for item in _sequence(raw.get("invalid_reasons", []), "invalid_reasons")
        ),
        content_sha256=str(raw.get("content_sha256", "")),
    )


def _parse_keyframe(raw: Mapping[str, Any]) -> ReconstructionKeyframe:
    return ReconstructionKeyframe(
        keyframe_id=str(raw["keyframe_id"]),
        session_id=str(raw["session_id"]),
        frame_id=str(raw["frame_id"]),
        selection_sequence=int(raw["selection_sequence"]),
        reason=KeyframeSelectionReason(str(raw["reason"])),
        score=None if raw.get("score") is None else float(raw["score"]),
        pose_id=str(raw.get("pose_id", "")),
    )


def _parse_camera(raw: Mapping[str, Any]) -> ReconstructionCamera:
    intrinsics_raw = _mapping(raw["intrinsics"], "intrinsics")
    frame_raw = _mapping(raw["camera_frame"], "camera_frame")
    intrinsics = CameraIntrinsics(
        matrix=np.asarray(intrinsics_raw["matrix"], dtype=np.float64),
        pixel_space_id=str(intrinsics_raw["pixel_space_id"]),
        source=IntrinsicsSource(str(intrinsics_raw["source"])),
        calibration_status=CalibrationStatus(str(intrinsics_raw["calibration_status"])),
        calibration_record_id=str(intrinsics_raw.get("calibration_record_id", "")),
    )
    frame = CoordinateFrame(
        frame_id=str(frame_raw["frame_id"]),
        kind=CoordinateFrameKind(str(frame_raw["kind"])),
        axis_convention=AxisConvention(str(frame_raw["axis_convention"])),
        parent_frame_id=str(frame_raw.get("parent_frame_id", "")),
        length_unit=LengthUnit(str(frame_raw.get("length_unit", "m"))),
    )
    return ReconstructionCamera(
        camera_id=str(raw["camera_id"]),
        image_size=_parse_image_size(_mapping(raw["image_size"], "image_size")),
        intrinsics=intrinsics,
        camera_frame=frame,
        distortion_model=CameraDistortionModel(str(raw["distortion_model"])),
        distortion_coefficients=tuple(float(value) for value in raw.get("distortion_coefficients", [])),
        backend_distortion_name=str(raw.get("backend_distortion_name", "")),
    )


def _parse_trajectory(raw: Mapping[str, Any]) -> ReconstructionTrajectory:
    samples = []
    for item in _sequence(raw["samples"], "trajectory samples"):
        sample = _mapping(item, "trajectory sample")
        transform = RigidTransform3D(
            source_frame_id=str(sample["source_frame_id"]),
            destination_frame_id=str(sample["destination_frame_id"]),
            matrix=np.asarray(sample["matrix"], dtype=np.float64),
            method="metadata_restore",
        )
        samples.append(
            CameraPose(
                pose_id=str(sample["pose_id"]),
                camera_id=str(sample["camera_id"]),
                frame_id=str(sample["frame_id"]),
                timestamp_ns=int(sample["timestamp_ns"]),
                T_world_from_camera=transform,
                tracking_confidence=(
                    None
                    if sample.get("tracking_confidence") is None
                    else float(sample["tracking_confidence"])
                ),
            )
        )
    return ReconstructionTrajectory(tuple(samples))


def _parse_landmark(raw: Mapping[str, Any]) -> SparseLandmark:
    observations = tuple(
        SparseObservation(
            observation_id=str(observation["observation_id"]),
            landmark_id=str(observation["landmark_id"]),
            keyframe_id=str(observation["keyframe_id"]),
            feature_xy=tuple(float(value) for value in observation["feature_xy"]),
            feature_index=(
                None if observation.get("feature_index") is None else int(observation["feature_index"])
            ),
            reprojection_error_px=(
                None
                if observation.get("reprojection_error_px") is None
                else float(observation["reprojection_error_px"])
            ),
            confidence=(
                None if observation.get("confidence") is None else float(observation["confidence"])
            ),
        )
        for observation in (
            _mapping(item, "sparse observation")
            for item in _sequence(raw["observations"], "observations")
        )
    )
    color_raw = raw.get("color_rgb")
    color = None if color_raw is None else tuple(int(value) for value in color_raw)
    return SparseLandmark(
        landmark_id=str(raw["landmark_id"]),
        track_id=str(raw["track_id"]),
        position_world_m=np.asarray(raw["position_world_m"], dtype=np.float64),
        observations=observations,
        color_rgb=color,
    )


def _parse_checkpoint(raw: Mapping[str, Any]) -> CheckpointMetadata:
    backend_versions = tuple(
        (str(item["backend_id"]), str(item["backend_version"]))
        for item in (
            _mapping(value, "backend version")
            for value in _sequence(raw.get("backend_versions", []), "backend_versions")
        )
    )
    artifacts = tuple(
        CheckpointArtifact(
            role=str(item["role"]),
            relative_path=str(item["relative_path"]),
            sha256=str(item.get("sha256", "")),
            required_for_resume=bool(item.get("required_for_resume", True)),
        )
        for item in (
            _mapping(value, "checkpoint artifact")
            for value in _sequence(raw.get("artifacts", []), "artifacts")
        )
    )
    return CheckpointMetadata(
        checkpoint_id=str(raw["checkpoint_id"]),
        session_id=str(raw["session_id"]),
        application_version=str(raw["application_version"]),
        session_state=ReconstructionSessionState(str(raw["session_state"])),
        completed_stage=ReconstructionStage(str(raw["completed_stage"])),
        profile=ReconstructionQualityProfile(str(raw["profile"])),
        source_fingerprint=str(raw.get("source_fingerprint", "")),
        backend_versions=backend_versions,
        artifacts=artifacts,
        last_frame_id=str(raw.get("last_frame_id", "")),
        last_keyframe_id=str(raw.get("last_keyframe_id", "")),
        resume_allowed=bool(raw.get("resume_allowed", True)),
        schema_version=int(raw["schema_version"]),
        minimum_reader_version=int(raw["minimum_reader_version"]),
        coordinate_convention_version=int(raw["coordinate_convention_version"]),
        created_at_utc=str(raw["created_at_utc"]),
    )


def _parse_provenance(raw: Mapping[str, Any]) -> ReconstructionProvenance:
    backends = tuple(
        BackendProvenance(
            backend_id=str(item["backend_id"]),
            backend_version=str(item["backend_version"]),
            adapter_version=str(item.get("adapter_version", "")),
        )
        for item in (
            _mapping(value, "backend provenance")
            for value in _sequence(raw.get("backends", []), "backends")
        )
    )
    return ReconstructionProvenance(
        application_version=str(raw["application_version"]),
        source_fingerprint=str(raw.get("source_fingerprint", "")),
        hardware_label=str(raw["hardware_label"]),
        backends=backends,
        coordinate_convention_version=int(raw["coordinate_convention_version"]),
        created_at_utc=str(raw["created_at_utc"]),
        metadata=_mapping(raw.get("metadata", {}), "provenance metadata"),
    )


def _parse_image_size(raw: Mapping[str, Any]) -> ImageSize:
    return ImageSize(width=int(raw["width"]), height=int(raw["height"]))


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{field_name} must be an object.")
    return value


def _sequence(value: Any, field_name: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ReconstructionValidationError(f"{field_name} must be an array.")
    return value
