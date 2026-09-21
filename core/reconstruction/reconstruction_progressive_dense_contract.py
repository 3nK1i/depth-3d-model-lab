"""I6.7A progressive dense snapshot and authority contract.

This module is deliberately provider-execution/controller/UI neutral.  It
extends the existing I4.3 dense-provider progress contract with one bounded,
inline dense-point preview suitable for later I6.7 transport and rendering.

The authority boundary is intentionally strict:

* a progressive dense snapshot is provisional visualization only;
* it is bound to one already-admitted sparse SfM result and one concrete dense
  provider execution;
* all preview geometry is expressed in the authoritative PyCOLMAP
  reconstruction frame;
* the dense provider never becomes camera/global-frame authority;
* a progressive snapshot is not an admitted final dense result and is not a
  persistent canonical dense scene;
* native metric-like depth does not establish physical scale, measurement
  authority, certified metrology, or a surface mesh.

I6.7A chooses no provider, starts no provider runtime, changes no production
threshold, and defines no operational preview-size limit.  I6.7B/C own
execution/emission and transport/memory policy respectively.
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
from core.reconstruction.dense_provider_architecture import (
    I43A_AUTHORITATIVE_FRAME_OWNER,
    DenseProviderFamily,
    DenseProviderImplementation,
    DenseProviderProgressStage,
    dense_provider_implementation_contracts,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_provider_selection_contract import (
    PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION,
)
from core.reconstruction.validation import (
    readonly_vector3,
    require_iso8601_utc,
    require_nonnegative_int,
    require_positive_int,
    require_probability,
    require_sha256,
    require_uuid,
)

I67A_SCHEMA_VERSION = 1
I67A_PHASE = "I6.7A_PROGRESSIVE_DENSE_CONTRACT_AND_AUTHORITY_BOUNDARY"
I67A_SNAPSHOT_AUTHORITY = "provisional_dense_visualization_only"
I67A_CAMERA_AUTHORITY_OWNER = I43A_AUTHORITATIVE_FRAME_OWNER
I67A_RECONSTRUCTION_FRAME_BINDING = "authoritative_sfm_frame"
I67A_SCALE_STATUS = "arbitrary"
I67A_REPRESENTATION_KIND = "dense_points_preview"
I67A_SAMPLING_PROFILE = "bounded_dense_preview_v1"

_IMPLEMENTATION_CONTRACT_BY_IMPL = {
    item.implementation: item for item in dense_provider_implementation_contracts()
}


class ProgressiveDenseStage(str, Enum):
    """Dense execution stages mirrored from the frozen I4.3 provider contract."""

    PREPARING = DenseProviderProgressStage.PREPARING.value
    VIEW_SELECTION = DenseProviderProgressStage.VIEW_SELECTION.value
    PROVIDER_EXECUTION = DenseProviderProgressStage.PROVIDER_EXECUTION.value
    ALIGNMENT = DenseProviderProgressStage.ALIGNMENT.value
    FUSION_FILTERING = DenseProviderProgressStage.FUSION_FILTERING.value
    FINALIZATION = DenseProviderProgressStage.FINALIZATION.value


_STAGE_ORDER = {stage: index for index, stage in enumerate(ProgressiveDenseStage)}


@dataclass(frozen=True, slots=True)
class ProgressiveDensePointSample:
    """One bounded dense-point sample in the arbitrary PyCOLMAP frame.

    ``sample_index`` is unique only within one preview.  It is not a persistent
    geometry identity and may change between revisions as filtering/downsampling
    evolves.
    """

    sample_index: int
    position_reconstruction: np.ndarray
    color_rgb: tuple[int, int, int] | None = None
    confidence: float | None = None
    view_support: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_index",
            require_nonnegative_int(self.sample_index, field_name="sample_index"),
        )
        object.__setattr__(
            self,
            "position_reconstruction",
            readonly_vector3(
                self.position_reconstruction,
                field_name="position_reconstruction",
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
            object.__setattr__(self, "color_rgb", tuple(int(value) for value in color))
        confidence = self.confidence
        if confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                require_probability(confidence, field_name="confidence"),
            )
        support = self.view_support
        if support is not None:
            object.__setattr__(
                self,
                "view_support",
                require_positive_int(support, field_name="view_support"),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "position_reconstruction": self.position_reconstruction.tolist(),
            "color_rgb": list(self.color_rgb) if self.color_rgb is not None else None,
            "confidence": self.confidence,
            "view_support": self.view_support,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressiveDensePointSample":
        try:
            color_raw = payload.get("color_rgb")
            color = None if color_raw is None else tuple(color_raw)
            return cls(
                sample_index=payload["sample_index"],
                position_reconstruction=payload["position_reconstruction"],
                color_rgb=color,  # type: ignore[arg-type]
                confidence=payload.get("confidence"),
                view_support=payload.get("view_support"),
            )
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            raise ReconstructionValidationError(
                f"Invalid progressive dense point sample: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ProgressiveDenseGeometryPreview:
    """Optional bounded dense RGB-point preview for one revision.

    I6.7A intentionally freezes no operational point limit.  Later emission and
    transport steps must define bounded hardware-aware policies without turning
    the preview count into a scientific reconstruction threshold.
    """

    point_samples: tuple[ProgressiveDensePointSample, ...] = ()
    sampling_profile: str = I67A_SAMPLING_PROFILE

    def __post_init__(self) -> None:
        points = tuple(self.point_samples)
        if any(not isinstance(item, ProgressiveDensePointSample) for item in points):
            raise TypeError(
                "point_samples must contain ProgressiveDensePointSample values."
            )
        indices = [item.sample_index for item in points]
        if len(indices) != len(set(indices)):
            raise ReconstructionValidationError(
                "Progressive dense point sample_index values must be unique within one preview."
            )
        object.__setattr__(self, "point_samples", points)
        object.__setattr__(
            self,
            "sampling_profile",
            require_nonempty(self.sampling_profile, field_name="sampling_profile"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "representation_kind": I67A_REPRESENTATION_KIND,
            "sampling_profile": self.sampling_profile,
            "point_samples": [item.as_dict() for item in self.point_samples],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressiveDenseGeometryPreview":
        try:
            if "representation_kind" in payload and payload["representation_kind"] != I67A_REPRESENTATION_KIND:
                raise ReconstructionValidationError(
                    "Progressive dense preview representation_kind drifted."
                )
            raw_points = payload.get("point_samples", [])
            if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
                raise TypeError("point_samples must be an array")
            if any(not isinstance(item, Mapping) for item in raw_points):
                raise TypeError("Every point_samples item must be an object")
            return cls(
                point_samples=tuple(
                    ProgressiveDensePointSample.from_payload(item) for item in raw_points
                ),
                sampling_profile=str(
                    payload.get("sampling_profile", I67A_SAMPLING_PROFILE)
                ),
            )
        except (TypeError, ValueError, ReconstructionValidationError) as exc:
            raise ReconstructionValidationError(
                f"Invalid progressive dense geometry preview: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ProgressiveDenseStatistics:
    """Current provider-neutral dense statistics for one provisional revision."""

    input_views: int
    selected_views: int
    processed_views: int
    current_dense_points: int
    mean_confidence: float | None = None
    mean_view_support: float | None = None

    def __post_init__(self) -> None:
        input_views = require_positive_int(self.input_views, field_name="input_views")
        if input_views < 3:
            raise ReconstructionValidationError(
                "Progressive dense reconstruction requires an admitted multi-view SfM source."
            )
        selected = require_positive_int(self.selected_views, field_name="selected_views")
        processed = require_nonnegative_int(self.processed_views, field_name="processed_views")
        points = require_nonnegative_int(
            self.current_dense_points, field_name="current_dense_points"
        )
        if selected > input_views:
            raise ReconstructionValidationError("selected_views cannot exceed input_views.")
        if processed > selected:
            raise ReconstructionValidationError("processed_views cannot exceed selected_views.")
        confidence = self.mean_confidence
        if confidence is not None:
            confidence = require_probability(confidence, field_name="mean_confidence")
        support = self.mean_view_support
        if support is not None:
            support = require_finite(support, field_name="mean_view_support")
            if support < 0.0:
                raise ReconstructionValidationError(
                    "mean_view_support must be non-negative when reported."
                )
        object.__setattr__(self, "input_views", input_views)
        object.__setattr__(self, "selected_views", selected)
        object.__setattr__(self, "processed_views", processed)
        object.__setattr__(self, "current_dense_points", points)
        object.__setattr__(self, "mean_confidence", confidence)
        object.__setattr__(self, "mean_view_support", support)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_views": self.input_views,
            "selected_views": self.selected_views,
            "processed_views": self.processed_views,
            "current_dense_points": self.current_dense_points,
            "mean_confidence": self.mean_confidence,
            "mean_view_support": self.mean_view_support,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressiveDenseStatistics":
        try:
            return cls(
                input_views=payload["input_views"],
                selected_views=payload["selected_views"],
                processed_views=payload["processed_views"],
                current_dense_points=payload["current_dense_points"],
                mean_confidence=payload.get("mean_confidence"),
                mean_view_support=payload.get("mean_view_support"),
            )
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            raise ReconstructionValidationError(
                f"Invalid progressive dense statistics: {exc}"
            ) from exc


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        "snapshot_authority": I67A_SNAPSHOT_AUTHORITY,
        "camera_pose_authority_owner": I67A_CAMERA_AUTHORITY_OWNER,
        "reconstruction_frame_binding": I67A_RECONSTRUCTION_FRAME_BINDING,
        "scale_status": I67A_SCALE_STATUS,
        "representation_kind": I67A_REPRESENTATION_KIND,
    }
    for key, expected_value in expected.items():
        if key in authority and authority[key] != expected_value:
            raise ReconstructionValidationError(
                f"Progressive dense authority label {key} must remain {expected_value!r}."
            )


@dataclass(frozen=True, slots=True)
class ProgressiveDenseSnapshot:
    """One provisional dense snapshot bound to an admitted sparse SfM result."""

    source_sfm_session_id: str
    dense_run_id: str
    revision: int
    timestamp_utc: str
    elapsed_ns: int
    provider_id: str
    implementation: DenseProviderImplementation
    stage: ProgressiveDenseStage
    stage_detail: str
    request_fingerprint_sha256: str
    source_sparse_result_sha256: str
    reconstruction_frame_id: str
    statistics: ProgressiveDenseStatistics
    progress_fraction: float | None = None
    geometry: ProgressiveDenseGeometryPreview | None = None
    warnings: tuple[str, ...] = ()

    # Explicit fail-closed authority boundary.
    provisional: bool = True
    final_dense_result_admitted: bool = False
    persistent_dense_scene: bool = False
    canonical_geometry_authoritative: bool = False
    camera_pose_authority_established: bool = False
    metric_scale_claimed: bool = False
    measurement_authoritative: bool = False
    certified_metrology: bool = False
    surface_mesh_generated: bool = False
    provider_promotion_performed: bool = False
    schema_version: int = I67A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_sfm_session_id",
            require_uuid(self.source_sfm_session_id, field_name="source_sfm_session_id"),
        )
        object.__setattr__(
            self,
            "dense_run_id",
            require_uuid(self.dense_run_id, field_name="dense_run_id"),
        )
        object.__setattr__(
            self, "revision", require_positive_int(self.revision, field_name="revision")
        )
        object.__setattr__(
            self,
            "timestamp_utc",
            require_iso8601_utc(self.timestamp_utc, field_name="timestamp_utc"),
        )
        object.__setattr__(
            self,
            "elapsed_ns",
            require_nonnegative_int(self.elapsed_ns, field_name="elapsed_ns"),
        )
        if not isinstance(self.implementation, DenseProviderImplementation):
            raise TypeError("implementation must be DenseProviderImplementation.")
        provider_id = require_nonempty(self.provider_id, field_name="provider_id")
        expected_provider = PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION[self.implementation]
        if provider_id != expected_provider:
            raise ReconstructionValidationError(
                "provider_id must match the canonical concrete dense-provider implementation; Auto is not executable."
            )
        object.__setattr__(self, "provider_id", provider_id)
        if not isinstance(self.stage, ProgressiveDenseStage):
            raise TypeError("stage must be ProgressiveDenseStage.")
        object.__setattr__(
            self,
            "stage_detail",
            require_nonempty(self.stage_detail, field_name="stage_detail"),
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
        object.__setattr__(
            self,
            "source_sparse_result_sha256",
            require_sha256(
                self.source_sparse_result_sha256,
                field_name="source_sparse_result_sha256",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "reconstruction_frame_id",
            require_nonempty(
                self.reconstruction_frame_id, field_name="reconstruction_frame_id"
            ),
        )
        if not isinstance(self.statistics, ProgressiveDenseStatistics):
            raise TypeError("statistics must be ProgressiveDenseStatistics.")
        if self.progress_fraction is not None:
            progress = require_probability(
                self.progress_fraction, field_name="progress_fraction"
            )
            if progress >= 1.0:
                raise ReconstructionValidationError(
                    "A provisional dense snapshot cannot claim completed progress."
                )
            object.__setattr__(self, "progress_fraction", progress)
        if self.geometry is not None:
            if not isinstance(self.geometry, ProgressiveDenseGeometryPreview):
                raise TypeError(
                    "geometry must be ProgressiveDenseGeometryPreview or None."
                )
            if len(self.geometry.point_samples) > self.statistics.current_dense_points:
                raise ReconstructionValidationError(
                    "Progressive dense preview point count cannot exceed current_dense_points."
                )
        warnings = tuple(
            require_nonempty(item, field_name="warning") for item in self.warnings
        )
        if len(warnings) != len(set(warnings)):
            raise ReconstructionValidationError(
                "Progressive dense warnings must be unique."
            )
        object.__setattr__(self, "warnings", warnings)

        if not self.provisional:
            raise ReconstructionValidationError(
                "Progressive dense snapshots must remain provisional."
            )
        forbidden = {
            "final_dense_result_admitted": self.final_dense_result_admitted,
            "persistent_dense_scene": self.persistent_dense_scene,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "camera_pose_authority_established": self.camera_pose_authority_established,
            "metric_scale_claimed": self.metric_scale_claimed,
            "measurement_authoritative": self.measurement_authoritative,
            "certified_metrology": self.certified_metrology,
            "surface_mesh_generated": self.surface_mesh_generated,
            "provider_promotion_performed": self.provider_promotion_performed,
        }
        claimed = [name for name, value in forbidden.items() if value]
        if claimed:
            raise ReconstructionValidationError(
                "Progressive dense snapshots cannot promote final/persistent/canonical/camera/metric/measurement/"
                f"metrology/mesh/provider authority: {', '.join(claimed)}."
            )
        if isinstance(self.schema_version, bool) or int(self.schema_version) != I67A_SCHEMA_VERSION:
            raise ReconstructionValidationError(
                f"Unsupported I6.7A progressive dense schema: {self.schema_version}."
            )
        object.__setattr__(self, "schema_version", I67A_SCHEMA_VERSION)

    @property
    def provider_family(self) -> DenseProviderFamily:
        return _IMPLEMENTATION_CONTRACT_BY_IMPL[self.implementation].family

    def authority_as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_authority": I67A_SNAPSHOT_AUTHORITY,
            "camera_pose_authority_owner": I67A_CAMERA_AUTHORITY_OWNER,
            "reconstruction_frame_binding": I67A_RECONSTRUCTION_FRAME_BINDING,
            "scale_status": I67A_SCALE_STATUS,
            "representation_kind": I67A_REPRESENTATION_KIND,
            "provisional": self.provisional,
            "final_dense_result_admitted": self.final_dense_result_admitted,
            "persistent_dense_scene": self.persistent_dense_scene,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "camera_pose_authority_established": self.camera_pose_authority_established,
            "metric_scale_claimed": self.metric_scale_claimed,
            "measurement_authoritative": self.measurement_authoritative,
            "certified_metrology": self.certified_metrology,
            "surface_mesh_generated": self.surface_mesh_generated,
            "provider_promotion_performed": self.provider_promotion_performed,
        }

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": I67A_PHASE,
            "source_sfm_session_id": self.source_sfm_session_id,
            "dense_run_id": self.dense_run_id,
            "revision": self.revision,
            "timestamp_utc": self.timestamp_utc,
            "elapsed_ns": self.elapsed_ns,
            "provider_id": self.provider_id,
            "implementation": self.implementation.value,
            "provider_family": self.provider_family.value,
            "stage": self.stage.value,
            "stage_detail": self.stage_detail,
            "request_fingerprint_sha256": self.request_fingerprint_sha256,
            "source_sparse_result_sha256": self.source_sparse_result_sha256,
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "statistics": self.statistics.as_dict(),
            "progress_fraction": self.progress_fraction,
            "geometry": self.geometry.as_dict() if self.geometry is not None else None,
            "warnings": list(self.warnings),
            "authority": self.authority_as_dict(),
        }

    @property
    def snapshot_sha256(self) -> str:
        return _canonical_sha256(self._identity_payload())

    def as_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["snapshot_sha256"] = self.snapshot_sha256
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgressiveDenseSnapshot":
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
            if not isinstance(warnings_raw, Sequence) or isinstance(
                warnings_raw, (str, bytes)
            ):
                raise TypeError("warnings must be an array")
            _validate_authority_labels(authority_raw)
            if "phase" in payload and str(payload["phase"]) != I67A_PHASE:
                raise ReconstructionValidationError(
                    f"Unsupported progressive dense phase identity: {payload['phase']!r}."
                )
            implementation = DenseProviderImplementation(str(payload["implementation"]))
            expected_family = _IMPLEMENTATION_CONTRACT_BY_IMPL[implementation].family.value
            if "provider_family" in payload and str(payload["provider_family"]) != expected_family:
                raise ReconstructionValidationError(
                    "provider_family does not match the concrete provider implementation."
                )
            snapshot = cls(
                source_sfm_session_id=str(payload["source_sfm_session_id"]),
                dense_run_id=str(payload["dense_run_id"]),
                revision=payload["revision"],
                timestamp_utc=str(payload["timestamp_utc"]),
                elapsed_ns=payload["elapsed_ns"],
                provider_id=str(payload["provider_id"]),
                implementation=implementation,
                stage=ProgressiveDenseStage(str(payload["stage"])),
                stage_detail=str(payload["stage_detail"]),
                request_fingerprint_sha256=str(payload["request_fingerprint_sha256"]),
                source_sparse_result_sha256=str(payload["source_sparse_result_sha256"]),
                reconstruction_frame_id=str(payload["reconstruction_frame_id"]),
                statistics=ProgressiveDenseStatistics.from_payload(statistics_raw),
                progress_fraction=payload.get("progress_fraction"),
                geometry=(
                    None
                    if geometry_raw is None
                    else ProgressiveDenseGeometryPreview.from_payload(geometry_raw)
                ),
                warnings=tuple(str(item) for item in warnings_raw),
                provisional=_authority_bool(authority_raw, payload, "provisional", True),
                final_dense_result_admitted=_authority_bool(
                    authority_raw, payload, "final_dense_result_admitted", False
                ),
                persistent_dense_scene=_authority_bool(
                    authority_raw, payload, "persistent_dense_scene", False
                ),
                canonical_geometry_authoritative=_authority_bool(
                    authority_raw, payload, "canonical_geometry_authoritative", False
                ),
                camera_pose_authority_established=_authority_bool(
                    authority_raw, payload, "camera_pose_authority_established", False
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
                surface_mesh_generated=_authority_bool(
                    authority_raw, payload, "surface_mesh_generated", False
                ),
                provider_promotion_performed=_authority_bool(
                    authority_raw, payload, "provider_promotion_performed", False
                ),
                schema_version=int(payload.get("schema_version", 0)),
            )
            expected_digest = str(payload.get("snapshot_sha256", "")).strip().lower()
            if expected_digest and require_sha256(
                expected_digest, field_name="snapshot_sha256", allow_empty=False
            ) != snapshot.snapshot_sha256:
                raise ReconstructionValidationError(
                    "Progressive dense snapshot_sha256 does not match payload contents."
                )
            return snapshot
        except (
            KeyError,
            TypeError,
            ValueError,
            ReconstructionValidationError,
        ) as exc:
            if isinstance(exc, ReconstructionValidationError):
                raise
            raise ReconstructionValidationError(
                f"Invalid progressive dense snapshot: {exc}"
            ) from exc


def validate_progressive_dense_snapshot_transition(
    previous: ProgressiveDenseSnapshot,
    current: ProgressiveDenseSnapshot,
) -> None:
    """Validate one same-run progressive transition.

    Revisions must strictly increase but may contain gaps so I6.7C can coalesce
    pending updates.  ``current_dense_points`` is deliberately *not* required
    to increase: alignment/fusion/filtering may remove provisional geometry.
    """

    if not isinstance(previous, ProgressiveDenseSnapshot) or not isinstance(
        current, ProgressiveDenseSnapshot
    ):
        raise TypeError("previous/current must be ProgressiveDenseSnapshot values.")
    identity_fields = (
        "source_sfm_session_id",
        "dense_run_id",
        "provider_id",
        "implementation",
        "request_fingerprint_sha256",
        "source_sparse_result_sha256",
        "reconstruction_frame_id",
    )
    for field_name in identity_fields:
        if getattr(previous, field_name) != getattr(current, field_name):
            raise ReconstructionValidationError(
                f"Progressive dense transition changed {field_name}."
            )
    if current.revision <= previous.revision:
        raise ReconstructionValidationError(
            "Progressive dense revisions must strictly increase; gaps are allowed."
        )
    if current.elapsed_ns < previous.elapsed_ns:
        raise ReconstructionValidationError(
            "Progressive dense elapsed_ns cannot move backwards."
        )
    previous_time = datetime.fromisoformat(previous.timestamp_utc)
    current_time = datetime.fromisoformat(current.timestamp_utc)
    if current_time < previous_time:
        raise ReconstructionValidationError(
            "Progressive dense timestamp_utc cannot move backwards."
        )
    if _STAGE_ORDER[current.stage] < _STAGE_ORDER[previous.stage]:
        raise ReconstructionValidationError(
            "Progressive dense execution stage cannot regress within one run."
        )
