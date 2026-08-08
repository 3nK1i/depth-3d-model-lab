"""Phase 8.5C live optimized-trajectory propagation.

This module keeps raw live-tracking poses immutable and stores global
optimization as a separate pose layer.  Exact optimized keyframe poses are
matched back to their original live observations through the native keyframe
ID.  Non-keyframe consumers may opt into a causal propagated correction based
on the latest preceding optimized keyframe; such poses are explicitly tagged
as propagated estimates rather than exact optimizer outputs.

All poses remain ``T_reconstruction_from_camera`` in arbitrary monocular gauge.
No metric/world alignment is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.live_contracts import LiveSlamObservation, LiveSlamPose
from core.reconstruction.live_global_consistency import NativeGlobalConsistencyStatus
from core.reconstruction.sfm_contracts import (
    ReconstructionGauge,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
)
from core.reconstruction.sfm_coordinates import validate_rigid_matrix
from core.reconstruction.validation import require_nonnegative_int, require_positive_int, require_uuid

LIVE_TRAJECTORY_SCHEMA_VERSION = 2


class LiveTrajectoryPoseLayer(str, Enum):
    TRACKING = "tracking"
    OPTIMIZED_KEYFRAME = "optimized_keyframe"
    OPTIMIZED_PROPAGATED = "optimized_propagated"


def _require_arbitrary_gauge(gauge: ReconstructionGauge) -> ReconstructionGauge:
    if not isinstance(gauge, ReconstructionGauge):
        raise TypeError("gauge must be ReconstructionGauge.")
    if (
        gauge.scale_status is not SfmScaleStatus.ARBITRARY
        or gauge.alignment_status is not SfmWorldAlignmentStatus.ARBITRARY
        or gauge.unit_to_metre is not None
    ):
        raise ReconstructionValidationError(
            "Phase 8.5C live trajectory must remain arbitrary-scale and unaligned."
        )
    return gauge


def _matrix_equal(lhs: np.ndarray, rhs: np.ndarray, *, atol: float = 1e-10) -> bool:
    return bool(np.allclose(lhs, rhs, rtol=0.0, atol=atol))


def _rotation_angle_deg(matrix: np.ndarray) -> float:
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


@dataclass(frozen=True, slots=True)
class LiveTrajectoryPose:
    session_id: str
    frame_id: str
    sequence_index: int
    camera_id: str
    timestamp_ns: int
    raw_pose_id: str
    layer: LiveTrajectoryPoseLayer
    T_reconstruction_from_camera: np.ndarray
    source_keyframe_id: str = ""
    optimization_revision: int | None = None
    optimizer_epoch: int | None = None
    native_optimized_source_pose_id: str = ""
    gauge: ReconstructionGauge = field(default_factory=ReconstructionGauge)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        for name in ("frame_id", "camera_id", "raw_pose_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ReconstructionValidationError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "sequence_index",
            require_nonnegative_int(self.sequence_index, field_name="sequence_index"),
        )
        object.__setattr__(
            self,
            "timestamp_ns",
            require_nonnegative_int(self.timestamp_ns, field_name="timestamp_ns"),
        )
        if not isinstance(self.layer, LiveTrajectoryPoseLayer):
            raise TypeError("layer must be LiveTrajectoryPoseLayer.")
        matrix = validate_rigid_matrix(
            self.T_reconstruction_from_camera,
            field_name="T_reconstruction_from_camera",
        )
        object.__setattr__(self, "T_reconstruction_from_camera", matrix)
        keyframe_id = str(self.source_keyframe_id).strip()
        native_source = str(self.native_optimized_source_pose_id).strip()
        revision = self.optimization_revision
        epoch = self.optimizer_epoch
        if self.layer is LiveTrajectoryPoseLayer.TRACKING:
            if keyframe_id or native_source or revision is not None or epoch is not None:
                raise ReconstructionValidationError(
                    "Raw tracking poses cannot claim optimized-pose provenance."
                )
        else:
            if not keyframe_id:
                raise ReconstructionValidationError(
                    "Optimized trajectory poses require source_keyframe_id."
                )
            if revision is None:
                raise ReconstructionValidationError(
                    "Optimized trajectory poses require optimization_revision."
                )
            revision = require_positive_int(revision, field_name="optimization_revision")
            if epoch is None:
                raise ReconstructionValidationError(
                    "Optimized trajectory poses require optimizer_epoch."
                )
            epoch = require_positive_int(epoch, field_name="optimizer_epoch")
            if not native_source:
                raise ReconstructionValidationError(
                    "Optimized trajectory poses require native optimizer provenance."
                )
        object.__setattr__(self, "source_keyframe_id", keyframe_id)
        object.__setattr__(self, "native_optimized_source_pose_id", native_source)
        object.__setattr__(self, "optimization_revision", revision)
        object.__setattr__(self, "optimizer_epoch", epoch)
        object.__setattr__(self, "gauge", _require_arbitrary_gauge(self.gauge))

    @property
    def is_exact_optimizer_pose(self) -> bool:
        return self.layer is LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME

    @property
    def is_propagated(self) -> bool:
        return self.layer is LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LIVE_TRAJECTORY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "sequence_index": self.sequence_index,
            "camera_id": self.camera_id,
            "timestamp_ns": self.timestamp_ns,
            "raw_pose_id": self.raw_pose_id,
            "pose_layer": self.layer.value,
            "pose_convention": "T_reconstruction_from_camera",
            "T_reconstruction_from_camera": self.T_reconstruction_from_camera.tolist(),
            "source_keyframe_id": self.source_keyframe_id,
            "optimization_revision": self.optimization_revision,
            "optimizer_epoch": self.optimizer_epoch,
            "native_optimized_source_pose_id": self.native_optimized_source_pose_id,
            "is_exact_optimizer_pose": self.is_exact_optimizer_pose,
            "is_propagated": self.is_propagated,
            "gauge": self.gauge.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class LiveTrajectoryCorrection:
    keyframe_id: str
    raw_frame_id: str
    raw_pose_id: str
    native_optimized_source_pose_id: str
    optimization_revision: int
    optimizer_epoch: int
    timestamp_ns: int
    T_optimized_reconstruction_from_tracking_reconstruction: np.ndarray
    translation_correction: float
    rotation_correction_deg: float

    def __post_init__(self) -> None:
        for name in (
            "keyframe_id",
            "raw_frame_id",
            "raw_pose_id",
            "native_optimized_source_pose_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ReconstructionValidationError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "optimization_revision",
            require_positive_int(self.optimization_revision, field_name="optimization_revision"),
        )
        object.__setattr__(
            self,
            "optimizer_epoch",
            require_positive_int(self.optimizer_epoch, field_name="optimizer_epoch"),
        )
        object.__setattr__(
            self,
            "timestamp_ns",
            require_nonnegative_int(self.timestamp_ns, field_name="timestamp_ns"),
        )
        matrix = validate_rigid_matrix(
            self.T_optimized_reconstruction_from_tracking_reconstruction,
            field_name="T_optimized_reconstruction_from_tracking_reconstruction",
        )
        object.__setattr__(
            self,
            "T_optimized_reconstruction_from_tracking_reconstruction",
            matrix,
        )
        translation = float(self.translation_correction)
        rotation = float(self.rotation_correction_deg)
        if not np.isfinite(translation) or translation < 0.0:
            raise ReconstructionValidationError("translation_correction must be finite and non-negative.")
        if not np.isfinite(rotation) or rotation < 0.0:
            raise ReconstructionValidationError("rotation_correction_deg must be finite and non-negative.")
        object.__setattr__(self, "translation_correction", translation)
        object.__setattr__(self, "rotation_correction_deg", rotation)

    def as_dict(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "raw_frame_id": self.raw_frame_id,
            "raw_pose_id": self.raw_pose_id,
            "native_optimized_source_pose_id": self.native_optimized_source_pose_id,
            "optimization_revision": self.optimization_revision,
            "optimizer_epoch": self.optimizer_epoch,
            "timestamp_ns": self.timestamp_ns,
            "correction_convention": (
                "T_optimized_reconstruction_from_tracking_reconstruction"
            ),
            "T_optimized_reconstruction_from_tracking_reconstruction": (
                self.T_optimized_reconstruction_from_tracking_reconstruction.tolist()
            ),
            "translation_correction": self.translation_correction,
            "rotation_correction_deg": self.rotation_correction_deg,
        }


@dataclass(frozen=True, slots=True)
class LiveTrajectoryRevisionUpdate:
    session_id: str
    optimizer_epoch: int
    map_revision: int
    previous_optimization_revision: int
    optimization_revision: int
    corrections: tuple[LiveTrajectoryCorrection, ...]
    exact_optimized_keyframes: int
    unmatched_optimized_keyframe_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        object.__setattr__(self, "optimizer_epoch", require_positive_int(self.optimizer_epoch, field_name="optimizer_epoch"))
        object.__setattr__(self, "map_revision", require_nonnegative_int(self.map_revision, field_name="map_revision"))
        previous = require_nonnegative_int(
            self.previous_optimization_revision,
            field_name="previous_optimization_revision",
        )
        current = require_positive_int(self.optimization_revision, field_name="optimization_revision")
        if current < previous:
            raise ReconstructionValidationError("optimization_revision cannot move backwards.")
        object.__setattr__(self, "previous_optimization_revision", previous)
        object.__setattr__(self, "optimization_revision", current)
        corrections = tuple(self.corrections)
        if any(not isinstance(item, LiveTrajectoryCorrection) for item in corrections):
            raise TypeError("corrections must contain LiveTrajectoryCorrection values.")
        object.__setattr__(self, "corrections", corrections)
        object.__setattr__(
            self,
            "exact_optimized_keyframes",
            require_nonnegative_int(self.exact_optimized_keyframes, field_name="exact_optimized_keyframes"),
        )
        unmatched = tuple(str(item).strip() for item in self.unmatched_optimized_keyframe_ids)
        if any(not item for item in unmatched) or len(set(unmatched)) != len(unmatched):
            raise ReconstructionValidationError("unmatched optimized keyframe IDs must be unique and non-empty.")
        object.__setattr__(self, "unmatched_optimized_keyframe_ids", unmatched)

    @property
    def changed(self) -> bool:
        return bool(self.corrections) or self.optimization_revision > self.previous_optimization_revision

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LIVE_TRAJECTORY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "optimizer_epoch": self.optimizer_epoch,
            "map_revision": self.map_revision,
            "previous_optimization_revision": self.previous_optimization_revision,
            "optimization_revision": self.optimization_revision,
            "corrections": [item.as_dict() for item in self.corrections],
            "exact_optimized_keyframes": self.exact_optimized_keyframes,
            "unmatched_optimized_keyframe_ids": list(self.unmatched_optimized_keyframe_ids),
            "changed": self.changed,
        }


@dataclass(frozen=True, slots=True)
class LiveTrajectorySnapshot:
    session_id: str
    optimizer_epoch: int
    map_revision: int
    optimization_revision: int
    raw_pose_count: int
    optimized_keyframe_count: int
    poses: tuple[LiveTrajectoryPose, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        object.__setattr__(self, "optimizer_epoch", require_positive_int(self.optimizer_epoch, field_name="optimizer_epoch"))
        object.__setattr__(self, "map_revision", require_nonnegative_int(self.map_revision, field_name="map_revision"))
        object.__setattr__(self, "optimization_revision", require_nonnegative_int(self.optimization_revision, field_name="optimization_revision"))
        object.__setattr__(self, "raw_pose_count", require_nonnegative_int(self.raw_pose_count, field_name="raw_pose_count"))
        object.__setattr__(self, "optimized_keyframe_count", require_nonnegative_int(self.optimized_keyframe_count, field_name="optimized_keyframe_count"))
        poses = tuple(self.poses)
        if any(not isinstance(item, LiveTrajectoryPose) for item in poses):
            raise TypeError("poses must contain LiveTrajectoryPose values.")
        if any(a.sequence_index >= b.sequence_index for a, b in zip(poses, poses[1:])):
            raise ReconstructionValidationError("Trajectory snapshot poses must be ordered by sequence_index.")
        object.__setattr__(self, "poses", poses)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LIVE_TRAJECTORY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "optimizer_epoch": self.optimizer_epoch,
            "map_revision": self.map_revision,
            "optimization_revision": self.optimization_revision,
            "raw_pose_count": self.raw_pose_count,
            "optimized_keyframe_count": self.optimized_keyframe_count,
            "poses": [item.as_dict() for item in self.poses],
        }


@dataclass(frozen=True, slots=True)
class LiveTrajectoryEpochTransition:
    session_id: str
    previous_optimizer_epoch: int
    optimizer_epoch: int
    reason: str
    raw_pose_count: int
    cleared_optimized_keyframes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        previous = require_positive_int(
            self.previous_optimizer_epoch,
            field_name="previous_optimizer_epoch",
        )
        current = require_positive_int(self.optimizer_epoch, field_name="optimizer_epoch")
        if current <= previous:
            raise ReconstructionValidationError("optimizer_epoch must increase across an epoch transition.")
        reason = str(self.reason).strip()
        if not reason:
            raise ReconstructionValidationError("Optimizer epoch transition reason must not be empty.")
        object.__setattr__(self, "previous_optimizer_epoch", previous)
        object.__setattr__(self, "optimizer_epoch", current)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "raw_pose_count", require_nonnegative_int(self.raw_pose_count, field_name="raw_pose_count"))
        object.__setattr__(
            self,
            "cleared_optimized_keyframes",
            require_nonnegative_int(
                self.cleared_optimized_keyframes,
                field_name="cleared_optimized_keyframes",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LIVE_TRAJECTORY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "previous_optimizer_epoch": self.previous_optimizer_epoch,
            "optimizer_epoch": self.optimizer_epoch,
            "reason": self.reason,
            "raw_pose_count": self.raw_pose_count,
            "cleared_optimized_keyframes": self.cleared_optimized_keyframes,
        }


@dataclass(frozen=True, slots=True)
class _RawPoseRecord:
    pose: LiveSlamPose
    sequence_index: int
    keyframe_id: str


class LiveOptimizedTrajectory:
    """Thread-safe raw/optimized trajectory store for reconstruction consumers."""

    def __init__(self, *, session_id: str, timestamp_match_tolerance_ns: int = 2_000) -> None:
        self._session_id = require_uuid(session_id, field_name="session_id")
        self._timestamp_match_tolerance_ns = require_nonnegative_int(
            timestamp_match_tolerance_ns,
            field_name="timestamp_match_tolerance_ns",
        )
        self._lock = RLock()
        self._raw_by_frame: dict[str, _RawPoseRecord] = {}
        self._raw_by_keyframe: dict[str, _RawPoseRecord] = {}
        self._optimized_by_keyframe: dict[str, object] = {}
        self._correction_by_keyframe: dict[str, LiveTrajectoryCorrection] = {}
        self._optimizer_epoch = 1
        self._map_revision = 0
        self._optimization_revision = 0
        self._revision_signature: tuple[tuple[str, bytes], ...] = ()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def optimizer_epoch(self) -> int:
        with self._lock:
            return self._optimizer_epoch

    def begin_optimizer_epoch(self, *, reason: str) -> LiveTrajectoryEpochTransition:
        """Invalidate optimizer-derived state while preserving immutable raw tracking poses.

        Native workers may restart with their map/keyframe IDs and optimization
        revisions reset to zero.  Starting a new optimizer epoch prevents stale
        corrections from the previous worker instance from leaking into the new
        trajectory while allowing raw observations to remain available.
        """

        reason_text = str(reason).strip()
        if not reason_text:
            raise ReconstructionValidationError("Optimizer epoch transition reason must not be empty.")
        with self._lock:
            previous_epoch = self._optimizer_epoch
            cleared = len(self._correction_by_keyframe)
            self._optimizer_epoch += 1
            self._raw_by_frame = {
                frame_id: _RawPoseRecord(
                    pose=record.pose,
                    sequence_index=record.sequence_index,
                    keyframe_id="",
                )
                for frame_id, record in self._raw_by_frame.items()
            }
            self._raw_by_keyframe.clear()
            self._optimized_by_keyframe.clear()
            self._correction_by_keyframe.clear()
            self._map_revision = 0
            self._optimization_revision = 0
            self._revision_signature = ()
            return LiveTrajectoryEpochTransition(
                session_id=self._session_id,
                previous_optimizer_epoch=previous_epoch,
                optimizer_epoch=self._optimizer_epoch,
                reason=reason_text,
                raw_pose_count=len(self._raw_by_frame),
                cleared_optimized_keyframes=cleared,
            )

    @property
    def optimization_revision(self) -> int:
        with self._lock:
            return self._optimization_revision

    @property
    def map_revision(self) -> int:
        with self._lock:
            return self._map_revision

    def ingest_observation(self, observation: LiveSlamObservation) -> bool:
        if not isinstance(observation, LiveSlamObservation):
            raise TypeError("observation must be LiveSlamObservation.")
        if observation.session_id != self._session_id:
            raise ReconstructionValidationError("Live trajectory observation session_id mismatch.")
        if observation.pose is None:
            return False
        record = _RawPoseRecord(
            pose=observation.pose,
            sequence_index=observation.sequence_index,
            keyframe_id=observation.keyframe_id if observation.is_keyframe else "",
        )
        with self._lock:
            existing = self._raw_by_frame.get(observation.frame_id)
            if existing is not None:
                if existing.sequence_index != observation.sequence_index or not _matrix_equal(
                    existing.pose.T_reconstruction_from_camera,
                    observation.pose.T_reconstruction_from_camera,
                ):
                    raise ReconstructionValidationError(
                        "Duplicate live trajectory frame_id has conflicting raw pose data."
                    )
                return False
            if any(item.sequence_index == observation.sequence_index for item in self._raw_by_frame.values()):
                raise ReconstructionValidationError(
                    "Duplicate live trajectory sequence_index is not allowed."
                )
            self._raw_by_frame[observation.frame_id] = record
            if record.keyframe_id:
                existing_keyframe = self._raw_by_keyframe.get(record.keyframe_id)
                if existing_keyframe is not None:
                    raise ReconstructionValidationError("Duplicate live keyframe_id is not allowed.")
                self._raw_by_keyframe[record.keyframe_id] = record
            return True

    def apply_native_status(
        self,
        status: NativeGlobalConsistencyStatus,
    ) -> LiveTrajectoryRevisionUpdate:
        if not isinstance(status, NativeGlobalConsistencyStatus):
            raise TypeError("status must be NativeGlobalConsistencyStatus.")
        if status.session_id != self._session_id:
            raise ReconstructionValidationError("Native global-consistency session_id mismatch.")

        with self._lock:
            if status.optimized_pose_revision < self._optimization_revision:
                raise ReconstructionValidationError(
                    "Stale native optimized trajectory revision was rejected."
                )

            previous_revision = self._optimization_revision
            same_revision = status.optimized_pose_revision == self._optimization_revision
            incoming = {item.keyframe_id: item for item in status.optimized_keyframe_poses}

            if same_revision and self._optimization_revision > 0:
                missing = sorted(set(self._optimized_by_keyframe) - set(incoming))
                if missing:
                    raise ReconstructionValidationError(
                        "Native optimized trajectory removed keyframes without advancing its revision."
                    )
                for keyframe_id, previous in self._optimized_by_keyframe.items():
                    current = incoming[keyframe_id]
                    if not _matrix_equal(
                        previous.T_reconstruction_from_camera,
                        current.T_reconstruction_from_camera,
                    ):
                        raise ReconstructionValidationError(
                            "Native optimized trajectory changed without advancing its revision."
                        )

            corrections: list[LiveTrajectoryCorrection] = []
            new_optimized: dict[str, object] = {}
            new_corrections: dict[str, LiveTrajectoryCorrection] = {}
            unmatched: list[str] = []

            for optimized in status.optimized_keyframe_poses:
                new_optimized[optimized.keyframe_id] = optimized
                raw = self._resolve_native_keyframe_raw_record(optimized)
                if raw is None:
                    unmatched.append(optimized.keyframe_id)
                    continue
                correction_matrix = (
                    optimized.T_reconstruction_from_camera
                    @ np.linalg.inv(raw.pose.T_reconstruction_from_camera)
                )
                correction_matrix = validate_rigid_matrix(
                    correction_matrix,
                    field_name="T_optimized_reconstruction_from_tracking_reconstruction",
                )
                correction = LiveTrajectoryCorrection(
                    keyframe_id=optimized.keyframe_id,
                    raw_frame_id=raw.pose.frame_id,
                    raw_pose_id=raw.pose.pose_id,
                    native_optimized_source_pose_id=optimized.source_pose_id,
                    optimization_revision=status.optimized_pose_revision,
                    optimizer_epoch=self._optimizer_epoch,
                    timestamp_ns=raw.pose.timestamp_ns,
                    T_optimized_reconstruction_from_tracking_reconstruction=correction_matrix,
                    translation_correction=float(np.linalg.norm(correction_matrix[:3, 3])),
                    rotation_correction_deg=_rotation_angle_deg(correction_matrix),
                )
                new_corrections[optimized.keyframe_id] = correction
                previous = self._correction_by_keyframe.get(optimized.keyframe_id)
                if previous is None or not _matrix_equal(
                    previous.T_optimized_reconstruction_from_tracking_reconstruction,
                    correction.T_optimized_reconstruction_from_tracking_reconstruction,
                ):
                    corrections.append(correction)

            self._optimized_by_keyframe = new_optimized
            self._correction_by_keyframe = new_corrections
            self._map_revision = max(self._map_revision, status.map_revision)
            self._optimization_revision = status.optimized_pose_revision
            self._revision_signature = tuple(
                (
                    item.keyframe_id,
                    np.asarray(item.T_reconstruction_from_camera, dtype=np.float64).tobytes(),
                )
                for item in status.optimized_keyframe_poses
            )
            corrections.sort(key=lambda item: (item.timestamp_ns, item.keyframe_id))
            unmatched.sort()
            return LiveTrajectoryRevisionUpdate(
                session_id=self._session_id,
                optimizer_epoch=self._optimizer_epoch,
                map_revision=self._map_revision,
                previous_optimization_revision=previous_revision,
                optimization_revision=self._optimization_revision,
                corrections=tuple(corrections),
                exact_optimized_keyframes=len(new_corrections),
                unmatched_optimized_keyframe_ids=tuple(unmatched),
            )


    def _resolve_native_keyframe_raw_record(self, optimized: object) -> _RawPoseRecord | None:
        """Resolve a native keyframe to its original raw live pose.

        stella's mapping thread may publish a newly-created keyframe after the
        frame that actually produced it.  The live observation therefore treats
        ``keyframe_id`` as an immediate hint, while the native keyframe timestamp
        is the authoritative association once global-consistency status is read.
        Raw pose matrices are never modified; only the internal keyframe-to-frame
        association is repaired when a unique timestamp match exists.
        """

        keyframe_id = str(getattr(optimized, "keyframe_id", "")).strip()
        timestamp_ns = require_nonnegative_int(
            getattr(optimized, "timestamp_ns", None),
            field_name="optimized.timestamp_ns",
        )
        hinted = self._raw_by_keyframe.get(keyframe_id)
        if (
            hinted is not None
            and abs(hinted.pose.timestamp_ns - timestamp_ns)
            <= self._timestamp_match_tolerance_ns
        ):
            return hinted

        matches = [
            record
            for record in self._raw_by_frame.values()
            if abs(record.pose.timestamp_ns - timestamp_ns) <= self._timestamp_match_tolerance_ns
        ]
        if len(matches) > 1:
            raise ReconstructionValidationError(
                "Native optimized keyframe timestamp matches multiple raw live poses."
            )
        if not matches:
            if hinted is not None:
                raise ReconstructionValidationError(
                    "Native optimized keyframe timestamp does not match any raw live pose."
                )
            return None

        matched = matches[0]
        if matched.keyframe_id and matched.keyframe_id != keyframe_id:
            raise ReconstructionValidationError(
                "Native optimized keyframe timestamp conflicts with another raw keyframe association."
            )

        # stella mapping can publish a new keyframe asynchronously.  If the
        # observation that first saw the new native ID was later than the actual
        # source frame, clear that provisional association and bind the native ID
        # to the unique raw pose carrying the native keyframe timestamp.
        if hinted is not None and hinted.pose.frame_id != matched.pose.frame_id:
            repaired_hint = _RawPoseRecord(
                pose=hinted.pose,
                sequence_index=hinted.sequence_index,
                keyframe_id="",
            )
            self._raw_by_frame[hinted.pose.frame_id] = repaired_hint

        repaired_match = _RawPoseRecord(
            pose=matched.pose,
            sequence_index=matched.sequence_index,
            keyframe_id=keyframe_id,
        )
        self._raw_by_frame[matched.pose.frame_id] = repaired_match
        self._raw_by_keyframe[keyframe_id] = repaired_match
        return repaired_match

    def raw_pose_for_frame(self, frame_id: str) -> LiveTrajectoryPose | None:
        frame_id = str(frame_id).strip()
        with self._lock:
            raw = self._raw_by_frame.get(frame_id)
            if raw is None:
                return None
            return self._make_tracking_pose(raw)

    def best_pose_for_frame(
        self,
        frame_id: str,
        *,
        allow_propagated: bool = True,
    ) -> LiveTrajectoryPose | None:
        frame_id = str(frame_id).strip()
        with self._lock:
            raw = self._raw_by_frame.get(frame_id)
            if raw is None:
                return None
            if raw.keyframe_id and raw.keyframe_id in self._correction_by_keyframe:
                optimized = self._optimized_by_keyframe[raw.keyframe_id]
                return LiveTrajectoryPose(
                    session_id=self._session_id,
                    frame_id=raw.pose.frame_id,
                    sequence_index=raw.sequence_index,
                    camera_id=raw.pose.camera_id,
                    timestamp_ns=raw.pose.timestamp_ns,
                    raw_pose_id=raw.pose.pose_id,
                    layer=LiveTrajectoryPoseLayer.OPTIMIZED_KEYFRAME,
                    T_reconstruction_from_camera=optimized.T_reconstruction_from_camera,
                    source_keyframe_id=raw.keyframe_id,
                    optimization_revision=self._optimization_revision,
                    optimizer_epoch=self._optimizer_epoch,
                    native_optimized_source_pose_id=optimized.source_pose_id,
                    gauge=raw.pose.gauge,
                )
            if allow_propagated:
                correction = self._latest_causal_correction(raw.pose.timestamp_ns)
                if correction is not None:
                    matrix = (
                        correction.T_optimized_reconstruction_from_tracking_reconstruction
                        @ raw.pose.T_reconstruction_from_camera
                    )
                    return LiveTrajectoryPose(
                        session_id=self._session_id,
                        frame_id=raw.pose.frame_id,
                        sequence_index=raw.sequence_index,
                        camera_id=raw.pose.camera_id,
                        timestamp_ns=raw.pose.timestamp_ns,
                        raw_pose_id=raw.pose.pose_id,
                        layer=LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED,
                        T_reconstruction_from_camera=matrix,
                        source_keyframe_id=correction.keyframe_id,
                        optimization_revision=correction.optimization_revision,
                        optimizer_epoch=correction.optimizer_epoch,
                        native_optimized_source_pose_id=correction.native_optimized_source_pose_id,
                        gauge=raw.pose.gauge,
                    )
            return self._make_tracking_pose(raw)

    def snapshot(self, *, allow_propagated: bool = True) -> LiveTrajectorySnapshot:
        with self._lock:
            ordered = sorted(self._raw_by_frame.values(), key=lambda item: item.sequence_index)
            poses = tuple(
                self.best_pose_for_frame(item.pose.frame_id, allow_propagated=allow_propagated)
                for item in ordered
            )
            return LiveTrajectorySnapshot(
                session_id=self._session_id,
                optimizer_epoch=self._optimizer_epoch,
                map_revision=self._map_revision,
                optimization_revision=self._optimization_revision,
                raw_pose_count=len(self._raw_by_frame),
                optimized_keyframe_count=len(self._correction_by_keyframe),
                poses=tuple(item for item in poses if item is not None),
            )

    def _latest_causal_correction(self, timestamp_ns: int) -> LiveTrajectoryCorrection | None:
        candidates = [
            item
            for item in self._correction_by_keyframe.values()
            if item.timestamp_ns <= timestamp_ns
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.timestamp_ns, item.keyframe_id))

    def _make_tracking_pose(self, raw: _RawPoseRecord) -> LiveTrajectoryPose:
        return LiveTrajectoryPose(
            session_id=self._session_id,
            frame_id=raw.pose.frame_id,
            sequence_index=raw.sequence_index,
            camera_id=raw.pose.camera_id,
            timestamp_ns=raw.pose.timestamp_ns,
            raw_pose_id=raw.pose.pose_id,
            layer=LiveTrajectoryPoseLayer.TRACKING,
            T_reconstruction_from_camera=raw.pose.T_reconstruction_from_camera,
            gauge=raw.pose.gauge,
        )
