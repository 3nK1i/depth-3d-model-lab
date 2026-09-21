"""I6.6A progressive sparse-SfM snapshot and authority contract.

This module is deliberately worker/controller/UI neutral.  It defines the
provisional sparse state that later I6.6 substeps may transport and render while
PyCOLMAP is still running, without weakening the existing final-result admission
boundary.

A progressive snapshot is never an admitted sparse result, never persistent
canonical scene authority, and never metric/metrology evidence.  Camera poses in
one snapshot are dependency-neutral samples of the current PyCOLMAP
reconstruction candidate in ``T_reconstruction_from_camera`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from core.contracts.validation import require_finite, require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_coordinates import validate_rigid_matrix
from core.reconstruction.validation import (
    readonly_vector3,
    require_iso8601_utc,
    require_nonnegative_int,
    require_positive_int,
    require_probability,
    require_sha256,
    require_uuid,
)

I66A_SCHEMA_VERSION = 1
I66A_PHASE = "I6.6A_PROGRESSIVE_SFM_SNAPSHOT_AND_AUTHORITY_CONTRACT"
I66A_PYCOLMAP_BACKEND_ID = "pycolmap-offline-sfm"
I66A_CAMERA_AUTHORITY_OWNER = "PyCOLMAP"
I66A_POSE_CONVENTION = "T_reconstruction_from_camera"
I66A_SCALE_STATUS = "arbitrary"
I66A_ALIGNMENT_STATUS = "arbitrary"
I66A_SNAPSHOT_AUTHORITY = "provisional_visualization_only"


class ProgressiveSfmStage(str, Enum):
    """Stages at which a bounded provisional sparse snapshot may be exposed."""

    MAPPING = "mapping"
    FINALIZING = "finalizing"


@dataclass(frozen=True, slots=True)
class ProgressiveSfmCameraSample:
    """Dependency-neutral sampled PyCOLMAP camera pose for progressive viewing."""

    backend_image_id: int
    source_image_name: str
    T_reconstruction_from_camera: np.ndarray
    backend_camera_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_image_id",
            require_nonnegative_int(self.backend_image_id, field_name="backend_image_id"),
        )
        if self.backend_camera_id is not None:
            object.__setattr__(
                self,
                "backend_camera_id",
                require_nonnegative_int(self.backend_camera_id, field_name="backend_camera_id"),
            )
        object.__setattr__(
            self,
            "source_image_name",
            require_nonempty(self.source_image_name, field_name="source_image_name"),
        )
        object.__setattr__(
            self,
            "T_reconstruction_from_camera",
            validate_rigid_matrix(
                self.T_reconstruction_from_camera,
                field_name="T_reconstruction_from_camera",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_image_id": self.backend_image_id,
            "backend_camera_id": self.backend_camera_id,
            "source_image_name": self.source_image_name,
            "T_reconstruction_from_camera": self.T_reconstruction_from_camera.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressiveSfmCameraSample":
        try:
            return cls(
                backend_image_id=payload["backend_image_id"],
                backend_camera_id=payload.get("backend_camera_id"),
                source_image_name=str(payload["source_image_name"]),
                T_reconstruction_from_camera=payload["T_reconstruction_from_camera"],
            )
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            raise ReconstructionValidationError(
                f"Invalid progressive SfM camera sample: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ProgressiveSfmLandmarkSample:
    """Bounded sparse-landmark sample in the arbitrary reconstruction frame."""

    backend_point3d_id: int
    position_reconstruction: np.ndarray
    color_rgb: tuple[int, int, int] | None = None
    track_length: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_point3d_id",
            require_nonnegative_int(
                self.backend_point3d_id, field_name="backend_point3d_id"
            ),
        )
        object.__setattr__(
            self,
            "position_reconstruction",
            readonly_vector3(
                self.position_reconstruction, field_name="position_reconstruction"
            ),
        )
        color = self.color_rgb
        if color is not None:
            if len(color) != 3 or any(
                isinstance(value, bool)
                or int(value) != value
                or not 0 <= int(value) <= 255
                for value in color
            ):
                raise ReconstructionValidationError(
                    "color_rgb must contain three integer values within [0, 255]."
                )
            color = tuple(int(value) for value in color)
            object.__setattr__(self, "color_rgb", color)
        if self.track_length is not None:
            track_length = require_nonnegative_int(
                self.track_length, field_name="track_length"
            )
            if track_length < 2:
                raise ReconstructionValidationError(
                    "Sparse progressive landmarks require track_length >= 2 when reported."
                )
            object.__setattr__(self, "track_length", track_length)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_point3d_id": self.backend_point3d_id,
            "position_reconstruction": self.position_reconstruction.tolist(),
            "color_rgb": list(self.color_rgb) if self.color_rgb is not None else None,
            "track_length": self.track_length,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressiveSfmLandmarkSample":
        try:
            color_value = payload.get("color_rgb")
            color = None if color_value is None else tuple(color_value)
            return cls(
                backend_point3d_id=payload["backend_point3d_id"],
                position_reconstruction=payload["position_reconstruction"],
                color_rgb=color,  # type: ignore[arg-type]
                track_length=payload.get("track_length"),
            )
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            raise ReconstructionValidationError(
                f"Invalid progressive SfM landmark sample: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ProgressiveSfmGeometryPreview:
    """Optional bounded geometry samples carried by one progressive snapshot.

    I6.6A deliberately does not choose operational sample limits.  I6.6B owns
    extraction/throttling policy.  This contract only guarantees that sample
    identities are unique and remain subordinate to the snapshot totals.
    """

    camera_samples: tuple[ProgressiveSfmCameraSample, ...] = ()
    landmark_samples: tuple[ProgressiveSfmLandmarkSample, ...] = ()
    sampling_profile: str = "bounded_preview_v1"

    def __post_init__(self) -> None:
        cameras = tuple(self.camera_samples)
        landmarks = tuple(self.landmark_samples)
        if any(not isinstance(item, ProgressiveSfmCameraSample) for item in cameras):
            raise TypeError("camera_samples must contain ProgressiveSfmCameraSample values.")
        if any(not isinstance(item, ProgressiveSfmLandmarkSample) for item in landmarks):
            raise TypeError("landmark_samples must contain ProgressiveSfmLandmarkSample values.")
        backend_image_ids = [item.backend_image_id for item in cameras]
        source_names = [item.source_image_name for item in cameras]
        point_ids = [item.backend_point3d_id for item in landmarks]
        if len(backend_image_ids) != len(set(backend_image_ids)):
            raise ReconstructionValidationError(
                "Progressive camera sample backend_image_id values must be unique."
            )
        if len(source_names) != len(set(source_names)):
            raise ReconstructionValidationError(
                "Progressive camera sample source_image_name values must be unique."
            )
        if len(point_ids) != len(set(point_ids)):
            raise ReconstructionValidationError(
                "Progressive landmark sample backend_point3d_id values must be unique."
            )
        object.__setattr__(self, "camera_samples", cameras)
        object.__setattr__(self, "landmark_samples", landmarks)
        object.__setattr__(
            self,
            "sampling_profile",
            require_nonempty(self.sampling_profile, field_name="sampling_profile"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sampling_profile": self.sampling_profile,
            "camera_samples": [item.as_dict() for item in self.camera_samples],
            "landmark_samples": [item.as_dict() for item in self.landmark_samples],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressiveSfmGeometryPreview":
        try:
            raw_cameras = payload.get("camera_samples", [])
            raw_landmarks = payload.get("landmark_samples", [])
            if not isinstance(raw_cameras, Sequence) or isinstance(raw_cameras, (str, bytes)):
                raise TypeError("camera_samples must be an array")
            if not isinstance(raw_landmarks, Sequence) or isinstance(raw_landmarks, (str, bytes)):
                raise TypeError("landmark_samples must be an array")
            if any(not isinstance(item, Mapping) for item in raw_cameras):
                raise TypeError("Every camera_samples item must be an object")
            if any(not isinstance(item, Mapping) for item in raw_landmarks):
                raise TypeError("Every landmark_samples item must be an object")
            return cls(
                camera_samples=tuple(
                    ProgressiveSfmCameraSample.from_payload(item) for item in raw_cameras
                ),
                landmark_samples=tuple(
                    ProgressiveSfmLandmarkSample.from_payload(item) for item in raw_landmarks
                ),
                sampling_profile=str(payload.get("sampling_profile", "bounded_preview_v1")),
            )
        except (TypeError, ValueError, ReconstructionValidationError) as exc:
            raise ReconstructionValidationError(
                f"Invalid progressive SfM geometry preview: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ProgressiveSfmStatistics:
    """Current sparse statistics reported for one provisional candidate model."""

    input_views: int
    registered_cameras: int
    sparse_landmarks: int
    observations: int
    mean_reprojection_error_px: float | None = None

    def __post_init__(self) -> None:
        input_views = require_nonnegative_int(self.input_views, field_name="input_views")
        if input_views < 3:
            raise ReconstructionValidationError(
                "Progressive offline SfM statistics require at least three input views."
            )
        registered = require_nonnegative_int(
            self.registered_cameras, field_name="registered_cameras"
        )
        landmarks = require_nonnegative_int(
            self.sparse_landmarks, field_name="sparse_landmarks"
        )
        observations = require_nonnegative_int(self.observations, field_name="observations")
        if registered > input_views:
            raise ReconstructionValidationError(
                "registered_cameras cannot exceed input_views."
            )
        if landmarks == 0 and observations != 0:
            raise ReconstructionValidationError(
                "observations must be zero when sparse_landmarks is zero."
            )
        error = self.mean_reprojection_error_px
        if error is not None:
            error = require_finite(error, field_name="mean_reprojection_error_px")
            if error < 0.0:
                raise ReconstructionValidationError(
                    "mean_reprojection_error_px must not be negative."
                )
        object.__setattr__(self, "input_views", input_views)
        object.__setattr__(self, "registered_cameras", registered)
        object.__setattr__(self, "sparse_landmarks", landmarks)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "mean_reprojection_error_px", error)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_views": self.input_views,
            "registered_cameras": self.registered_cameras,
            "sparse_landmarks": self.sparse_landmarks,
            "observations": self.observations,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressiveSfmStatistics":
        try:
            return cls(
                input_views=payload["input_views"],
                registered_cameras=payload["registered_cameras"],
                sparse_landmarks=payload["sparse_landmarks"],
                observations=payload["observations"],
                mean_reprojection_error_px=payload.get("mean_reprojection_error_px"),
            )
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            raise ReconstructionValidationError(
                f"Invalid progressive SfM statistics: {exc}"
            ) from exc


def _authority_bool(
    authority: Mapping[str, Any],
    payload: Mapping[str, Any],
    key: str,
    default: bool,
) -> bool:
    value = authority[key] if key in authority else payload.get(key, default)
    if not isinstance(value, bool):
        raise ReconstructionValidationError(f"{key} must be a boolean when provided.")
    return value


def _validate_authority_labels(authority: Mapping[str, Any]) -> None:
    expected = {
        "snapshot_authority": I66A_SNAPSHOT_AUTHORITY,
        "camera_pose_authority_owner": I66A_CAMERA_AUTHORITY_OWNER,
        "pose_convention": I66A_POSE_CONVENTION,
        "scale_status": I66A_SCALE_STATUS,
        "alignment_status": I66A_ALIGNMENT_STATUS,
    }
    for key, expected_value in expected.items():
        if key in authority and authority[key] != expected_value:
            raise ReconstructionValidationError(
                f"Progressive SfM authority label {key} must remain {expected_value!r}."
            )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ProgressiveSfmSnapshot:
    """One provisional PyCOLMAP sparse snapshot suitable for later live viewing."""

    session_id: str
    revision: int
    timestamp_utc: str
    elapsed_ns: int
    stage: ProgressiveSfmStage
    stage_detail: str
    backend_id: str
    backend_version: str
    request_fingerprint_sha256: str
    statistics: ProgressiveSfmStatistics
    progress_fraction: float | None = None
    candidate_model_id: str = ""
    candidate_model_count: int | None = None
    geometry: ProgressiveSfmGeometryPreview | None = None
    warnings: tuple[str, ...] = ()

    # Frozen authority boundary.  These are fields rather than derived values so
    # malformed/over-promoted payloads fail closed during parsing.
    provisional: bool = True
    final_result_admitted: bool = False
    persistent_canonical_scene: bool = False
    canonical_geometry_authoritative: bool = False
    metric_scale_claimed: bool = False
    measurement_authoritative: bool = False
    certified_metrology: bool = False
    schema_version: int = I66A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        object.__setattr__(self, "revision", require_positive_int(self.revision, field_name="revision"))
        object.__setattr__(
            self,
            "timestamp_utc",
            require_iso8601_utc(self.timestamp_utc, field_name="timestamp_utc"),
        )
        object.__setattr__(
            self, "elapsed_ns", require_nonnegative_int(self.elapsed_ns, field_name="elapsed_ns")
        )
        if not isinstance(self.stage, ProgressiveSfmStage):
            raise TypeError("stage must be ProgressiveSfmStage.")
        object.__setattr__(
            self, "stage_detail", require_nonempty(self.stage_detail, field_name="stage_detail")
        )
        backend_id = require_nonempty(self.backend_id, field_name="backend_id")
        if backend_id != I66A_PYCOLMAP_BACKEND_ID:
            raise ReconstructionValidationError(
                "I6.6A progressive SfM snapshots are qualified only for the canonical PyCOLMAP backend."
            )
        object.__setattr__(self, "backend_id", backend_id)
        object.__setattr__(
            self,
            "backend_version",
            require_nonempty(self.backend_version, field_name="backend_version"),
        )
        object.__setattr__(
            self,
            "request_fingerprint_sha256",
            require_sha256(
                self.request_fingerprint_sha256,
                field_name="request_fingerprint_sha256",
                allow_empty=False,
            ),
        )
        if not isinstance(self.statistics, ProgressiveSfmStatistics):
            raise TypeError("statistics must be ProgressiveSfmStatistics.")
        if self.progress_fraction is not None:
            progress = require_probability(
                self.progress_fraction, field_name="progress_fraction"
            )
            if progress >= 1.0:
                raise ReconstructionValidationError(
                    "A provisional progressive SfM snapshot cannot claim completed progress."
                )
            object.__setattr__(self, "progress_fraction", progress)
        candidate_id = str(self.candidate_model_id).strip()
        candidate_count = self.candidate_model_count
        if candidate_count is not None:
            candidate_count = require_nonnegative_int(
                candidate_count, field_name="candidate_model_count"
            )
            if candidate_count == 0 and candidate_id:
                raise ReconstructionValidationError(
                    "candidate_model_id cannot be present when candidate_model_count is zero."
                )
            if candidate_count > 0 and not candidate_id:
                raise ReconstructionValidationError(
                    "A positive candidate_model_count requires candidate_model_id."
                )
            object.__setattr__(self, "candidate_model_count", candidate_count)
        elif candidate_id:
            raise ReconstructionValidationError(
                "candidate_model_id requires candidate_model_count."
            )
        object.__setattr__(self, "candidate_model_id", candidate_id)

        if self.geometry is not None:
            if not isinstance(self.geometry, ProgressiveSfmGeometryPreview):
                raise TypeError("geometry must be ProgressiveSfmGeometryPreview or None.")
            if len(self.geometry.camera_samples) > self.statistics.registered_cameras:
                raise ReconstructionValidationError(
                    "Progressive camera sample count cannot exceed registered_cameras."
                )
            if len(self.geometry.landmark_samples) > self.statistics.sparse_landmarks:
                raise ReconstructionValidationError(
                    "Progressive landmark sample count cannot exceed sparse_landmarks."
                )

        warnings = tuple(require_nonempty(item, field_name="warning") for item in self.warnings)
        if len(warnings) != len(set(warnings)):
            raise ReconstructionValidationError("Progressive SfM warnings must be unique.")
        object.__setattr__(self, "warnings", warnings)

        if not self.provisional:
            raise ReconstructionValidationError("Progressive SfM snapshots must remain provisional.")
        if self.final_result_admitted:
            raise ReconstructionValidationError(
                "A progressive SfM snapshot cannot claim final-result admission."
            )
        if self.persistent_canonical_scene:
            raise ReconstructionValidationError(
                "A progressive SfM snapshot cannot claim a persistent canonical scene."
            )
        if self.canonical_geometry_authoritative:
            raise ReconstructionValidationError(
                "Progressive sparse preview geometry is not canonical geometry authority."
            )
        if self.metric_scale_claimed or self.measurement_authoritative or self.certified_metrology:
            raise ReconstructionValidationError(
                "Progressive SfM snapshots cannot establish metric/measurement/metrology authority."
            )
        if isinstance(self.schema_version, bool) or int(self.schema_version) != I66A_SCHEMA_VERSION:
            raise ReconstructionValidationError(
                f"Unsupported I6.6A progressive SfM schema: {self.schema_version}."
            )
        object.__setattr__(self, "schema_version", I66A_SCHEMA_VERSION)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": I66A_PHASE,
            "session_id": self.session_id,
            "revision": self.revision,
            "timestamp_utc": self.timestamp_utc,
            "elapsed_ns": self.elapsed_ns,
            "stage": self.stage.value,
            "stage_detail": self.stage_detail,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "request_fingerprint_sha256": self.request_fingerprint_sha256,
            "statistics": self.statistics.as_dict(),
            "progress_fraction": self.progress_fraction,
            "candidate_model_id": self.candidate_model_id,
            "candidate_model_count": self.candidate_model_count,
            "geometry": self.geometry.as_dict() if self.geometry is not None else None,
            "warnings": list(self.warnings),
            "authority": self.authority_as_dict(),
        }

    @property
    def snapshot_sha256(self) -> str:
        return _canonical_sha256(self._identity_payload())

    def authority_as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_authority": I66A_SNAPSHOT_AUTHORITY,
            "camera_pose_authority_owner": I66A_CAMERA_AUTHORITY_OWNER,
            "pose_convention": I66A_POSE_CONVENTION,
            "scale_status": I66A_SCALE_STATUS,
            "alignment_status": I66A_ALIGNMENT_STATUS,
            "provisional": self.provisional,
            "final_result_admitted": self.final_result_admitted,
            "persistent_canonical_scene": self.persistent_canonical_scene,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "metric_scale_claimed": self.metric_scale_claimed,
            "measurement_authoritative": self.measurement_authoritative,
            "certified_metrology": self.certified_metrology,
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["snapshot_sha256"] = self.snapshot_sha256
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressiveSfmSnapshot":
        try:
            statistics_raw = payload["statistics"]
            if not isinstance(statistics_raw, Mapping):
                raise TypeError("statistics must be an object")
            geometry_raw = payload.get("geometry")
            if geometry_raw is not None and not isinstance(geometry_raw, Mapping):
                raise TypeError("geometry must be an object or null")
            authority_raw = payload.get("authority", {})
            if not isinstance(authority_raw, Mapping):
                raise TypeError("authority must be an object")
            warnings_raw = payload.get("warnings", [])
            if not isinstance(warnings_raw, Sequence) or isinstance(warnings_raw, (str, bytes)):
                raise TypeError("warnings must be an array")
            _validate_authority_labels(authority_raw)
            if "phase" in payload and str(payload["phase"]) != I66A_PHASE:
                raise ReconstructionValidationError(
                    f"Unsupported progressive SfM phase identity: {payload['phase']!r}."
                )
            snapshot = cls(
                session_id=str(payload["session_id"]),
                revision=payload["revision"],
                timestamp_utc=str(payload["timestamp_utc"]),
                elapsed_ns=payload["elapsed_ns"],
                stage=ProgressiveSfmStage(str(payload["stage"])),
                stage_detail=str(payload["stage_detail"]),
                backend_id=str(payload["backend_id"]),
                backend_version=str(payload["backend_version"]),
                request_fingerprint_sha256=str(payload["request_fingerprint_sha256"]),
                statistics=ProgressiveSfmStatistics.from_payload(statistics_raw),
                progress_fraction=payload.get("progress_fraction"),
                candidate_model_id=str(payload.get("candidate_model_id", "")),
                candidate_model_count=payload.get("candidate_model_count"),
                geometry=(
                    None
                    if geometry_raw is None
                    else ProgressiveSfmGeometryPreview.from_payload(geometry_raw)
                ),
                warnings=tuple(str(item) for item in warnings_raw),
                provisional=_authority_bool(authority_raw, payload, "provisional", True),
                final_result_admitted=_authority_bool(
                    authority_raw, payload, "final_result_admitted", False
                ),
                persistent_canonical_scene=_authority_bool(
                    authority_raw, payload, "persistent_canonical_scene", False
                ),
                canonical_geometry_authoritative=_authority_bool(
                    authority_raw, payload, "canonical_geometry_authoritative", False
                ),
                metric_scale_claimed=_authority_bool(
                    authority_raw, payload, "metric_scale_claimed", False
                ),
                measurement_authoritative=_authority_bool(
                    authority_raw, payload, "measurement_authoritative", False
                ),
                certified_metrology=_authority_bool(
                    authority_raw, payload, "certified_metrology", False
                ),
                schema_version=int(payload.get("schema_version", 0)),
            )
            expected_digest = str(payload.get("snapshot_sha256", "")).strip().lower()
            if expected_digest and require_sha256(
                expected_digest, field_name="snapshot_sha256", allow_empty=False
            ) != snapshot.snapshot_sha256:
                raise ReconstructionValidationError(
                    "Progressive SfM snapshot_sha256 does not match payload contents."
                )
            return snapshot
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            if isinstance(exc, ReconstructionValidationError):
                raise
            raise ReconstructionValidationError(
                f"Invalid progressive SfM snapshot payload: {exc}"
            ) from exc


def validate_progressive_sfm_snapshot_transition(
    previous: ProgressiveSfmSnapshot,
    current: ProgressiveSfmSnapshot,
) -> ProgressiveSfmSnapshot:
    """Validate one sequence transition without requiring contiguous revisions.

    Revision gaps are legal because I6.6C may coalesce/backpressure updates.  The
    geometry counts themselves are not required to increase monotonically:
    multi-model mapping and bundle-adjustment cleanup can change the current
    visualization candidate.  Session/request identity and stage/revision/time
    ordering are the actual stale-update guards.
    """

    if not isinstance(previous, ProgressiveSfmSnapshot) or not isinstance(
        current, ProgressiveSfmSnapshot
    ):
        raise TypeError("previous/current must be ProgressiveSfmSnapshot values.")
    if current.session_id != previous.session_id:
        raise ReconstructionValidationError(
            "Progressive SfM snapshots cannot transition across sessions."
        )
    if current.backend_id != previous.backend_id or current.backend_version != previous.backend_version:
        raise ReconstructionValidationError(
            "Progressive SfM backend identity cannot change within one snapshot sequence."
        )
    if current.request_fingerprint_sha256 != previous.request_fingerprint_sha256:
        raise ReconstructionValidationError(
            "Progressive SfM request fingerprint cannot change within one snapshot sequence."
        )
    if current.statistics.input_views != previous.statistics.input_views:
        raise ReconstructionValidationError(
            "Progressive SfM input view count cannot change within one snapshot sequence."
        )
    if current.revision <= previous.revision:
        raise ReconstructionValidationError(
            "Progressive SfM revision must increase strictly; stale/duplicate updates fail closed."
        )
    if current.elapsed_ns < previous.elapsed_ns:
        raise ReconstructionValidationError(
            "Progressive SfM elapsed_ns cannot move backwards."
        )
    previous_time = datetime.fromisoformat(previous.timestamp_utc)
    current_time = datetime.fromisoformat(current.timestamp_utc)
    if current_time < previous_time:
        raise ReconstructionValidationError(
            "Progressive SfM timestamp_utc cannot move backwards."
        )
    stage_order = {
        ProgressiveSfmStage.MAPPING: 0,
        ProgressiveSfmStage.FINALIZING: 1,
    }
    if stage_order[current.stage] < stage_order[previous.stage]:
        raise ReconstructionValidationError(
            "Progressive SfM stage cannot regress from finalizing back to mapping."
        )
    return current
