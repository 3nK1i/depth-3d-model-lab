"""Unified Digital Twin workspace persistence for Phase 9.6F.

The existing ``PersistentReconstructionScene`` remains the authoritative scene
manifest.  This module adds a scene-bound, checksum-protected *workspace sidecar*
for presentation/session state that intentionally must not mutate canonical
geometry, metric-scale, measurement, QC, or metrology authority.

The sidecar persists the Phase 9.6E0 estimated ruler provenance, the active
ruler record, the selected neural-appearance artifact, and the last terminal E1
build-job summary.  Active heavyweight jobs are never resumable from this file.

A small state-export bundle is also provided.  It packages the persistent-scene
manifest/checksum, optional authority sidecar, and workspace sidecar for audit
and same-environment restore.  Heavy referenced artifact payloads are *not*
embedded; the bundle says so explicitly and never pretends to be a portable
asset archive or a product geometry/QC export.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from app.reconstruction_estimated_metric_measurement import (
    EstimatedMetricScaleHint,
    ReconstructionRulerMetricRecord,
)
from app.reconstruction_processing_jobs import (
    DigitalTwinProcessingKind,
    DigitalTwinProcessingLifecycle,
    DigitalTwinProcessingSnapshot,
)
from core.measurement.interactive_distance import (
    InteractiveDistanceEstimate,
    InteractiveDistanceState,
)
from core.reconstruction import (
    PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM,
    PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST,
    PERSISTENT_SCENE_AUTHORITY_CHECKSUM,
    PERSISTENT_SCENE_AUTHORITY_MANIFEST,
    PersistentReconstructionScene,
    ReconstructionArtifactAuthority,
    read_optional_persistent_scene_authority_snapshot,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.validation import require_iso8601_utc, require_sha256, require_uuid


DIGITAL_TWIN_WORKSPACE_SCHEMA_VERSION = 1
DIGITAL_TWIN_WORKSPACE_PHASE = "9.6F"
DIGITAL_TWIN_WORKSPACE_MANIFEST = "digital_twin_workspace_state.json"
DIGITAL_TWIN_WORKSPACE_CHECKSUM = "digital_twin_workspace_state.json.sha256"
DIGITAL_TWIN_STATE_EXPORT_SCHEMA = "eia.digital-twin.state-export.v1"
DIGITAL_TWIN_STATE_EXPORT_MANIFEST = "digital_twin_state_export.json"
DIGITAL_TWIN_STATE_EXPORT_CHECKSUM = "digital_twin_state_export.sha256"


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_absolute_file(value: str | Path, *, field_name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ReconstructionValidationError(f"{field_name} must be absolute.")
    path = path.resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"{field_name} does not exist: {path}")
    return path


def _estimate_as_dict(value: InteractiveDistanceEstimate) -> dict[str, Any]:
    return {
        "state": value.state.value,
        "source_kind": value.source_kind,
        "point_a_source": None if value.point_a_source is None else list(value.point_a_source),
        "point_b_source": None if value.point_b_source is None else list(value.point_b_source),
        "distance_source_units": value.distance_source_units,
        "distance_m": value.distance_m,
        "expanded_uncertainty_m": value.expanded_uncertainty_m,
        "uncertainty_basis": value.uncertainty_basis,
        "calibration_status": value.calibration_status,
        "scale_status": value.scale_status,
        "unit_to_metre": value.unit_to_metre,
        "reasons": list(value.reasons),
        "certified_metrology": False,
    }


def _point_or_none(value: Any, field_name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ReconstructionValidationError(f"{field_name} must be a 3-vector or null.")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(f"{field_name} contains invalid coordinates.") from exc


def _estimate_from_dict(raw: Mapping[str, Any]) -> InteractiveDistanceEstimate:
    try:
        return InteractiveDistanceEstimate(
            state=InteractiveDistanceState(str(raw["state"])),
            source_kind=str(raw["source_kind"]),
            point_a_source=_point_or_none(raw.get("point_a_source"), "point_a_source"),
            point_b_source=_point_or_none(raw.get("point_b_source"), "point_b_source"),
            distance_source_units=raw.get("distance_source_units"),
            distance_m=raw.get("distance_m"),
            expanded_uncertainty_m=raw.get("expanded_uncertainty_m"),
            uncertainty_basis=str(raw.get("uncertainty_basis", "")),
            calibration_status=str(raw.get("calibration_status", "")),
            scale_status=str(raw.get("scale_status", "")),
            unit_to_metre=raw.get("unit_to_metre"),
            reasons=tuple(str(item) for item in raw.get("reasons", [])),
            certified_metrology=bool(raw.get("certified_metrology", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionValidationError(f"Invalid persisted ruler estimate: {exc}") from exc


def _hint_as_dict(value: EstimatedMetricScaleHint) -> dict[str, Any]:
    return {
        "scene_id": value.scene_id,
        "source_kind": value.source_kind,
        "unit_to_metre_estimate": value.unit_to_metre_estimate,
        "relative_scale_uncertainty": value.relative_scale_uncertainty,
        "provenance": value.provenance,
        "reference_distance_reconstruction_units": value.reference_distance_reconstruction_units,
        "reference_distance_m": value.reference_distance_m,
        "reference_relative_uncertainty": value.reference_relative_uncertainty,
        "calibrated": False,
        "metric_scale_authoritative": False,
        "certified_metrology": False,
    }


def _hint_from_dict(raw: Mapping[str, Any]) -> EstimatedMetricScaleHint:
    try:
        return EstimatedMetricScaleHint(
            scene_id=str(raw["scene_id"]),
            source_kind=str(raw["source_kind"]),
            unit_to_metre_estimate=float(raw["unit_to_metre_estimate"]),
            relative_scale_uncertainty=float(raw["relative_scale_uncertainty"]),
            provenance=str(raw["provenance"]),
            reference_distance_reconstruction_units=raw.get("reference_distance_reconstruction_units"),
            reference_distance_m=raw.get("reference_distance_m"),
            reference_relative_uncertainty=raw.get("reference_relative_uncertainty"),
            calibrated=bool(raw.get("calibrated", False)),
            metric_scale_authoritative=bool(raw.get("metric_scale_authoritative", False)),
            certified_metrology=bool(raw.get("certified_metrology", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionValidationError(f"Invalid persisted estimated-scale hint: {exc}") from exc


def _ruler_as_dict(value: ReconstructionRulerMetricRecord) -> dict[str, Any]:
    return {
        "scene_id": value.scene_id,
        "point_a_reconstruction": list(value.point_a_reconstruction),
        "point_b_reconstruction": list(value.point_b_reconstruction),
        "current": _estimate_as_dict(value.current),
        "current_scale_source": value.current_scale_source,
        "current_scale_provenance": value.current_scale_provenance,
        "original_estimate": None if value.original_estimate is None else _estimate_as_dict(value.original_estimate),
        "original_scale_source": value.original_scale_source,
        "original_scale_provenance": value.original_scale_provenance,
        "certified_metrology": False,
    }


def _ruler_from_dict(raw: Mapping[str, Any]) -> ReconstructionRulerMetricRecord:
    try:
        current = raw["current"]
        original = raw.get("original_estimate")
        if not isinstance(current, Mapping):
            raise ReconstructionValidationError("Persisted ruler current estimate must be an object.")
        if original is not None and not isinstance(original, Mapping):
            raise ReconstructionValidationError("Persisted ruler original estimate must be an object or null.")
        point_a = _point_or_none(raw.get("point_a_reconstruction"), "point_a_reconstruction")
        point_b = _point_or_none(raw.get("point_b_reconstruction"), "point_b_reconstruction")
        if point_a is None or point_b is None:
            raise ReconstructionValidationError("Persisted ruler endpoints are required.")
        return ReconstructionRulerMetricRecord(
            scene_id=str(raw["scene_id"]),
            point_a_reconstruction=point_a,
            point_b_reconstruction=point_b,
            current=_estimate_from_dict(current),
            current_scale_source=str(raw.get("current_scale_source", "")),
            current_scale_provenance=str(raw.get("current_scale_provenance", "")),
            original_estimate=None if original is None else _estimate_from_dict(original),
            original_scale_source=str(raw.get("original_scale_source", "")),
            original_scale_provenance=str(raw.get("original_scale_provenance", "")),
            certified_metrology=bool(raw.get("certified_metrology", False)),
        )
    except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
        if isinstance(exc, ReconstructionValidationError):
            raise
        raise ReconstructionValidationError(f"Invalid persisted ruler record: {exc}") from exc


@dataclass(frozen=True, slots=True)
class DigitalTwinProcessingHistoryRecord:
    """Terminal E1 job history only; this record is never resumable."""

    job_id: str
    kind: DigitalTwinProcessingKind
    title: str
    backend_label: str
    runtime_label: str
    lifecycle: DigitalTwinProcessingLifecycle
    stage: str
    message: str
    profile_label: str = ""
    source_label: str = ""
    scene_id: str = ""
    result_path: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0
    completed_units: int | None = None
    total_units: int | None = None
    progress_fraction: float | None = None
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        require_uuid(self.job_id, field_name="job_id")
        if not isinstance(self.kind, DigitalTwinProcessingKind):
            raise TypeError("kind must be DigitalTwinProcessingKind.")
        if not isinstance(self.lifecycle, DigitalTwinProcessingLifecycle) or not self.lifecycle.is_terminal:
            raise ReconstructionValidationError("Persisted build history must be terminal and non-resumable.")
        if self.certified_metrology:
            raise ReconstructionValidationError("Build history cannot certify metrology.")
        object.__setattr__(self, "title", str(self.title).strip())
        object.__setattr__(self, "backend_label", str(self.backend_label).strip())
        object.__setattr__(self, "runtime_label", str(self.runtime_label).strip())
        object.__setattr__(self, "stage", str(self.stage).strip())
        object.__setattr__(self, "message", str(self.message).strip())
        object.__setattr__(self, "profile_label", str(self.profile_label).strip())
        object.__setattr__(self, "source_label", str(self.source_label).strip())
        object.__setattr__(self, "scene_id", str(self.scene_id).strip())
        object.__setattr__(self, "result_path", str(self.result_path).strip())
        object.__setattr__(self, "error", str(self.error).strip())
        object.__setattr__(self, "elapsed_seconds", max(0.0, float(self.elapsed_seconds)))
        if self.completed_units is not None:
            object.__setattr__(self, "completed_units", int(self.completed_units))
        if self.total_units is not None:
            object.__setattr__(self, "total_units", int(self.total_units))
        if self.progress_fraction is not None:
            fraction = float(self.progress_fraction)
            if not 0.0 <= fraction <= 1.0:
                raise ReconstructionValidationError("Persisted build progress must be within [0, 1].")
            object.__setattr__(self, "progress_fraction", fraction)
        object.__setattr__(self, "certified_metrology", False)

    @classmethod
    def from_snapshot(cls, snapshot: DigitalTwinProcessingSnapshot) -> "DigitalTwinProcessingHistoryRecord | None":
        if not isinstance(snapshot, DigitalTwinProcessingSnapshot):
            raise TypeError("snapshot must be DigitalTwinProcessingSnapshot.")
        if snapshot.spec is None or snapshot.lifecycle is DigitalTwinProcessingLifecycle.IDLE:
            return None
        if snapshot.lifecycle.is_active:
            raise ReconstructionValidationError("Active heavyweight build jobs cannot be persisted or resumed by Phase 9.6F.")
        spec = snapshot.spec
        return cls(
            job_id=spec.job_id,
            kind=spec.kind,
            title=spec.title,
            backend_label=spec.backend_label,
            runtime_label=spec.runtime_label,
            lifecycle=snapshot.lifecycle,
            stage=snapshot.stage,
            message=snapshot.message,
            profile_label=spec.profile_label,
            source_label=spec.source_label,
            scene_id=spec.scene_id,
            result_path=snapshot.result_path,
            error=snapshot.error,
            elapsed_seconds=snapshot.elapsed_seconds,
            completed_units=snapshot.completed_units,
            total_units=snapshot.total_units,
            progress_fraction=snapshot.progress_fraction,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind.value,
            "title": self.title,
            "backend_label": self.backend_label,
            "runtime_label": self.runtime_label,
            "lifecycle": self.lifecycle.value,
            "stage": self.stage,
            "message": self.message,
            "profile_label": self.profile_label,
            "source_label": self.source_label,
            "scene_id": self.scene_id,
            "result_path": self.result_path,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "progress_fraction": self.progress_fraction,
            "resumable": False,
            "canonical_geometry_authoritative": False,
            "metric_scale_authoritative": False,
            "measurement_authoritative": False,
            "qc_authoritative": False,
            "certified_metrology": False,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DigitalTwinProcessingHistoryRecord":
        try:
            if bool(raw.get("resumable", False)):
                raise ReconstructionValidationError("Persisted build history must never claim resumability.")
            forbidden = (
                "canonical_geometry_authoritative",
                "metric_scale_authoritative",
                "measurement_authoritative",
                "qc_authoritative",
                "certified_metrology",
            )
            if any(bool(raw.get(key, False)) for key in forbidden):
                raise ReconstructionValidationError("Persisted build history contains a forbidden authority claim.")
            return cls(
                job_id=str(raw["job_id"]),
                kind=DigitalTwinProcessingKind(str(raw["kind"])),
                title=str(raw["title"]),
                backend_label=str(raw["backend_label"]),
                runtime_label=str(raw["runtime_label"]),
                lifecycle=DigitalTwinProcessingLifecycle(str(raw["lifecycle"])),
                stage=str(raw.get("stage", "")),
                message=str(raw.get("message", "")),
                profile_label=str(raw.get("profile_label", "")),
                source_label=str(raw.get("source_label", "")),
                scene_id=str(raw.get("scene_id", "")),
                result_path=str(raw.get("result_path", "")),
                error=str(raw.get("error", "")),
                elapsed_seconds=float(raw.get("elapsed_seconds", 0.0)),
                completed_units=raw.get("completed_units"),
                total_units=raw.get("total_units"),
                progress_fraction=raw.get("progress_fraction"),
                certified_metrology=bool(raw.get("certified_metrology", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReconstructionValidationError(f"Invalid persisted build history: {exc}") from exc


def _state_payload(
    *,
    scene_id: str,
    scene_fingerprint_sha256: str,
    session_id: str,
    estimated_metric_scale_hint: EstimatedMetricScaleHint | None,
    ruler_metric_record: ReconstructionRulerMetricRecord | None,
    selected_appearance_artifact_id: str,
    processing_history: DigitalTwinProcessingHistoryRecord | None,
    created_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": DIGITAL_TWIN_WORKSPACE_SCHEMA_VERSION,
        "phase": DIGITAL_TWIN_WORKSPACE_PHASE,
        "scene_id": scene_id,
        "scene_fingerprint_sha256": scene_fingerprint_sha256,
        "session_id": session_id,
        "estimated_metric_scale_hint": None if estimated_metric_scale_hint is None else _hint_as_dict(estimated_metric_scale_hint),
        "ruler_metric_record": None if ruler_metric_record is None else _ruler_as_dict(ruler_metric_record),
        "selected_appearance_artifact_id": selected_appearance_artifact_id,
        "processing_history": None if processing_history is None else processing_history.as_dict(),
        "created_at_utc": created_at_utc,
        "authority_role": "presentation_state_only",
        "canonical_geometry_authoritative": False,
        "metric_scale_authoritative": False,
        "measurement_authoritative": False,
        "qc_authoritative": False,
        "certified_metrology": False,
    }


@dataclass(frozen=True, slots=True)
class DigitalTwinWorkspaceState:
    scene_id: str
    scene_fingerprint_sha256: str
    session_id: str
    estimated_metric_scale_hint: EstimatedMetricScaleHint | None
    ruler_metric_record: ReconstructionRulerMetricRecord | None
    selected_appearance_artifact_id: str
    processing_history: DigitalTwinProcessingHistoryRecord | None
    created_at_utc: str
    workspace_fingerprint_sha256: str
    schema_version: int = DIGITAL_TWIN_WORKSPACE_SCHEMA_VERSION
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        if not scene_id:
            raise ReconstructionValidationError("Digital Twin workspace scene_id is required.")
        scene_fingerprint = require_sha256(
            self.scene_fingerprint_sha256,
            field_name="scene_fingerprint_sha256",
            allow_empty=False,
        )
        session_id = require_uuid(self.session_id, field_name="session_id")
        created = require_iso8601_utc(self.created_at_utc, field_name="created_at_utc")
        if int(self.schema_version) != DIGITAL_TWIN_WORKSPACE_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported Digital Twin workspace schema version.")
        if self.certified_metrology:
            raise ReconstructionValidationError("Workspace presentation state cannot certify metrology.")
        hint = self.estimated_metric_scale_hint
        ruler = self.ruler_metric_record
        if hint is not None and hint.scene_id != scene_id:
            raise ReconstructionValidationError("Estimated-scale hint does not match workspace scene identity.")
        if ruler is not None and ruler.scene_id != scene_id:
            raise ReconstructionValidationError("Ruler provenance does not match workspace scene identity.")
        selected = str(self.selected_appearance_artifact_id).strip()
        fingerprint = require_sha256(
            self.workspace_fingerprint_sha256,
            field_name="workspace_fingerprint_sha256",
            allow_empty=False,
        )
        expected = hashlib.sha256(
            _canonical_json(
                _state_payload(
                    scene_id=scene_id,
                    scene_fingerprint_sha256=scene_fingerprint,
                    session_id=session_id,
                    estimated_metric_scale_hint=hint,
                    ruler_metric_record=ruler,
                    selected_appearance_artifact_id=selected,
                    processing_history=self.processing_history,
                    created_at_utc=created,
                )
            )
        ).hexdigest()
        if fingerprint != expected:
            raise ReconstructionValidationError("Digital Twin workspace fingerprint does not match normalized state.")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "scene_fingerprint_sha256", scene_fingerprint)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "selected_appearance_artifact_id", selected)
        object.__setattr__(self, "created_at_utc", created)
        object.__setattr__(self, "workspace_fingerprint_sha256", fingerprint)
        object.__setattr__(self, "schema_version", DIGITAL_TWIN_WORKSPACE_SCHEMA_VERSION)
        object.__setattr__(self, "certified_metrology", False)

    def validate_against_scene(self, scene: PersistentReconstructionScene) -> "DigitalTwinWorkspaceState":
        if not isinstance(scene, PersistentReconstructionScene):
            raise TypeError("scene must be PersistentReconstructionScene.")
        if self.scene_id != scene.scene_id:
            raise ReconstructionValidationError("Digital Twin workspace scene identity mismatch.")
        if self.scene_fingerprint_sha256 != scene.scene_fingerprint_sha256:
            raise ReconstructionValidationError("Digital Twin workspace scene fingerprint mismatch.")
        if self.session_id != scene.session_id:
            raise ReconstructionValidationError("Digital Twin workspace session identity mismatch.")
        if self.selected_appearance_artifact_id:
            artifact = next(
                (item for item in scene.artifacts if item.artifact_id == self.selected_appearance_artifact_id),
                None,
            )
            if artifact is None:
                raise ReconstructionValidationError("Selected appearance artifact is not present in the persistent scene.")
            if artifact.authority is not ReconstructionArtifactAuthority.APPEARANCE_ONLY:
                raise ReconstructionValidationError("Selected workspace appearance must remain appearance-only.")
        return self

    def as_dict(self) -> dict[str, Any]:
        payload = _state_payload(
            scene_id=self.scene_id,
            scene_fingerprint_sha256=self.scene_fingerprint_sha256,
            session_id=self.session_id,
            estimated_metric_scale_hint=self.estimated_metric_scale_hint,
            ruler_metric_record=self.ruler_metric_record,
            selected_appearance_artifact_id=self.selected_appearance_artifact_id,
            processing_history=self.processing_history,
            created_at_utc=self.created_at_utc,
        )
        payload["workspace_fingerprint_sha256"] = self.workspace_fingerprint_sha256
        return payload


def build_digital_twin_workspace_state(
    scene: PersistentReconstructionScene,
    *,
    estimated_metric_scale_hint: EstimatedMetricScaleHint | None = None,
    ruler_metric_record: ReconstructionRulerMetricRecord | None = None,
    selected_appearance_artifact_id: str = "",
    processing_history: DigitalTwinProcessingHistoryRecord | None = None,
    processing_snapshot: DigitalTwinProcessingSnapshot | None = None,
    created_at_utc: str | None = None,
) -> DigitalTwinWorkspaceState:
    if not isinstance(scene, PersistentReconstructionScene):
        raise TypeError("scene must be PersistentReconstructionScene.")
    if processing_snapshot is not None:
        snapshot_history = DigitalTwinProcessingHistoryRecord.from_snapshot(processing_snapshot)
        if snapshot_history is not None:
            processing_history = snapshot_history
    created = created_at_utc or datetime.now(timezone.utc).isoformat()
    selected = str(selected_appearance_artifact_id).strip()
    payload = _state_payload(
        scene_id=scene.scene_id,
        scene_fingerprint_sha256=scene.scene_fingerprint_sha256,
        session_id=scene.session_id,
        estimated_metric_scale_hint=estimated_metric_scale_hint,
        ruler_metric_record=ruler_metric_record,
        selected_appearance_artifact_id=selected,
        processing_history=processing_history,
        created_at_utc=created,
    )
    fingerprint = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return DigitalTwinWorkspaceState(
        scene_id=scene.scene_id,
        scene_fingerprint_sha256=scene.scene_fingerprint_sha256,
        session_id=scene.session_id,
        estimated_metric_scale_hint=estimated_metric_scale_hint,
        ruler_metric_record=ruler_metric_record,
        selected_appearance_artifact_id=selected,
        processing_history=processing_history,
        created_at_utc=created,
        workspace_fingerprint_sha256=fingerprint,
    ).validate_against_scene(scene)


def _state_from_mapping(raw: Mapping[str, Any]) -> DigitalTwinWorkspaceState:
    try:
        forbidden = (
            "canonical_geometry_authoritative",
            "metric_scale_authoritative",
            "measurement_authoritative",
            "qc_authoritative",
            "certified_metrology",
        )
        if any(bool(raw.get(key, False)) for key in forbidden):
            raise ReconstructionValidationError("Digital Twin workspace sidecar contains a forbidden authority claim.")
        if str(raw.get("authority_role", "")) != "presentation_state_only":
            raise ReconstructionValidationError("Digital Twin workspace sidecar has an invalid authority role.")
        hint_raw = raw.get("estimated_metric_scale_hint")
        ruler_raw = raw.get("ruler_metric_record")
        history_raw = raw.get("processing_history")
        if hint_raw is not None and not isinstance(hint_raw, Mapping):
            raise ReconstructionValidationError("estimated_metric_scale_hint must be an object or null.")
        if ruler_raw is not None and not isinstance(ruler_raw, Mapping):
            raise ReconstructionValidationError("ruler_metric_record must be an object or null.")
        if history_raw is not None and not isinstance(history_raw, Mapping):
            raise ReconstructionValidationError("processing_history must be an object or null.")
        return DigitalTwinWorkspaceState(
            scene_id=str(raw["scene_id"]),
            scene_fingerprint_sha256=str(raw["scene_fingerprint_sha256"]),
            session_id=str(raw["session_id"]),
            estimated_metric_scale_hint=None if hint_raw is None else _hint_from_dict(hint_raw),
            ruler_metric_record=None if ruler_raw is None else _ruler_from_dict(ruler_raw),
            selected_appearance_artifact_id=str(raw.get("selected_appearance_artifact_id", "")),
            processing_history=None if history_raw is None else DigitalTwinProcessingHistoryRecord.from_dict(history_raw),
            created_at_utc=str(raw["created_at_utc"]),
            workspace_fingerprint_sha256=str(raw["workspace_fingerprint_sha256"]),
            schema_version=int(raw.get("schema_version", 0)),
            certified_metrology=bool(raw.get("certified_metrology", False)),
        )
    except KeyError as exc:
        raise ReconstructionValidationError(f"Digital Twin workspace sidecar is missing field: {exc.args[0]}.") from exc
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(f"Digital Twin workspace sidecar contains an invalid value: {exc}.") from exc


def write_digital_twin_workspace_state(
    state: DigitalTwinWorkspaceState,
    output_dir: str | Path,
) -> Path:
    if not isinstance(state, DigitalTwinWorkspaceState):
        raise TypeError("state must be DigitalTwinWorkspaceState.")
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        raise ReconstructionValidationError("Digital Twin workspace output_dir must be absolute.")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / DIGITAL_TWIN_WORKSPACE_MANIFEST
    checksum = root / DIGITAL_TWIN_WORKSPACE_CHECKSUM
    payload = json.dumps(state.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    manifest_tmp = root / f".{DIGITAL_TWIN_WORKSPACE_MANIFEST}.{os.getpid()}.tmp"
    checksum_tmp = root / f".{DIGITAL_TWIN_WORKSPACE_CHECKSUM}.{os.getpid()}.tmp"
    try:
        manifest_tmp.write_text(payload, encoding="utf-8")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        checksum_tmp.write_text(f"{digest}  {DIGITAL_TWIN_WORKSPACE_MANIFEST}\n", encoding="utf-8")
        os.replace(manifest_tmp, manifest)
        os.replace(checksum_tmp, checksum)
    finally:
        manifest_tmp.unlink(missing_ok=True)
        checksum_tmp.unlink(missing_ok=True)
    return manifest.resolve()


def read_optional_digital_twin_workspace_state(
    scene_manifest_path: str | Path,
    *,
    expected_scene: PersistentReconstructionScene,
) -> tuple[DigitalTwinWorkspaceState | None, Path | None]:
    manifest = _require_absolute_file(scene_manifest_path, field_name="scene_manifest_path")
    root = manifest.parent
    state_path = root / DIGITAL_TWIN_WORKSPACE_MANIFEST
    checksum_path = root / DIGITAL_TWIN_WORKSPACE_CHECKSUM
    if not state_path.exists() and not checksum_path.exists():
        return None, None
    if not state_path.is_file() or not checksum_path.is_file():
        raise ReconstructionValidationError("Digital Twin workspace sidecar/checksum pair is incomplete.")
    parts = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2 or parts[1] != DIGITAL_TWIN_WORKSPACE_MANIFEST:
        raise ReconstructionValidationError("Digital Twin workspace checksum sidecar is malformed.")
    expected = require_sha256(parts[0], field_name="workspace_manifest_sha256", allow_empty=False)
    if _sha256_file(state_path) != expected:
        raise ReconstructionValidationError("Digital Twin workspace manifest checksum mismatch.")
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReconstructionValidationError("Digital Twin workspace manifest is not valid JSON.") from exc
    if not isinstance(raw, Mapping):
        raise ReconstructionValidationError("Digital Twin workspace manifest root must be an object.")
    state = _state_from_mapping(raw).validate_against_scene(expected_scene)
    return state, state_path.resolve()


def export_digital_twin_state_bundle(
    *,
    scene_manifest_path: str | Path,
    workspace_state: DigitalTwinWorkspaceState,
    output_dir: str | Path,
    authority_manifest_path: str | Path | None = None,
) -> Path:
    """Export integrity manifests plus workspace state, never heavy payloads.

    The existing Phase 8.9 product export remains the route for materialized
    geometry/QC payloads.  This bundle is a state/audit companion only.
    """

    scene_manifest = _require_absolute_file(scene_manifest_path, field_name="scene_manifest_path")
    scene = read_persistent_reconstruction_scene(scene_manifest)
    verify_persistent_scene_file_references(scene)
    workspace_state.validate_against_scene(scene)
    target = Path(output_dir).expanduser()
    if not target.is_absolute():
        raise ReconstructionValidationError("Digital Twin state export output_dir must be absolute.")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ReconstructionValidationError(f"Digital Twin state export destination already exists: {target}")
    authority_manifest: Path | None = None
    if authority_manifest_path is not None and str(authority_manifest_path).strip():
        authority_manifest = _require_absolute_file(authority_manifest_path, field_name="authority_manifest_path")
        if authority_manifest.name != PERSISTENT_SCENE_AUTHORITY_MANIFEST:
            raise ReconstructionValidationError("Authority sidecar manifest has an unexpected filename.")
        if authority_manifest.parent != scene_manifest.parent:
            raise ReconstructionValidationError("Authority sidecar must belong to the active persistent-scene directory.")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    try:
        files: list[dict[str, Any]] = []
        for source in (
            scene_manifest,
            scene_manifest.parent / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM,
        ):
            if not source.is_file():
                raise ReconstructionValidationError(f"Required scene integrity file is missing: {source}")
            destination = temporary / source.name
            shutil.copy2(source, destination)
            files.append({"role": "persistent_scene", "path": source.name, "size_bytes": destination.stat().st_size, "sha256": _sha256_file(destination)})

        workspace_manifest = write_digital_twin_workspace_state(workspace_state, temporary)
        workspace_checksum = temporary / DIGITAL_TWIN_WORKSPACE_CHECKSUM
        for path in (workspace_manifest, workspace_checksum):
            files.append({"role": "workspace_state", "path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})

        if authority_manifest is not None:
            authority_checksum = authority_manifest.parent / PERSISTENT_SCENE_AUTHORITY_CHECKSUM
            if not authority_checksum.is_file():
                raise ReconstructionValidationError("Persistent authority checksum sidecar is missing.")
            for source in (authority_manifest, authority_checksum):
                destination = temporary / source.name
                shutil.copy2(source, destination)
                files.append({"role": "authority_sidecar", "path": source.name, "size_bytes": destination.stat().st_size, "sha256": _sha256_file(destination)})

        files.sort(key=lambda item: (item["role"], item["path"]))
        export_manifest = {
            "schema": DIGITAL_TWIN_STATE_EXPORT_SCHEMA,
            "phase": DIGITAL_TWIN_WORKSPACE_PHASE,
            "scene_id": scene.scene_id,
            "scene_fingerprint_sha256": scene.scene_fingerprint_sha256,
            "session_id": scene.session_id,
            "workspace_fingerprint_sha256": workspace_state.workspace_fingerprint_sha256,
            "artifact_payloads_embedded": False,
            "portable_payload_bundle": False,
            "restore_requires_referenced_files": True,
            "canonical_geometry_authoritative": False,
            "metric_scale_authoritative": False,
            "measurement_authoritative": False,
            "qc_authoritative": False,
            "certified_metrology": False,
            "referenced_payload_count": len(scene.file_references),
            "referenced_payloads": [item.as_dict() for item in scene.file_references],
            "files": files,
        }
        export_path = temporary / DIGITAL_TWIN_STATE_EXPORT_MANIFEST
        export_path.write_text(json.dumps(export_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        export_sha = _sha256_file(export_path)
        (temporary / DIGITAL_TWIN_STATE_EXPORT_CHECKSUM).write_text(
            f"{export_sha}  {DIGITAL_TWIN_STATE_EXPORT_MANIFEST}\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return (target / DIGITAL_TWIN_STATE_EXPORT_MANIFEST).resolve()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_digital_twin_state_bundle(path: str | Path) -> Mapping[str, Any]:
    root = Path(path).expanduser()
    if not root.is_absolute():
        raise ReconstructionValidationError("Digital Twin state export path must be absolute.")
    root = root.resolve()
    manifest = root / DIGITAL_TWIN_STATE_EXPORT_MANIFEST
    checksum = root / DIGITAL_TWIN_STATE_EXPORT_CHECKSUM
    if not manifest.is_file() or not checksum.is_file():
        raise ReconstructionValidationError("Digital Twin state export manifest/checksum is missing.")
    parts = checksum.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2 or parts[1] != DIGITAL_TWIN_STATE_EXPORT_MANIFEST:
        raise ReconstructionValidationError("Digital Twin state export checksum sidecar is malformed.")
    expected = require_sha256(parts[0], field_name="state_export_sha256", allow_empty=False)
    if _sha256_file(manifest) != expected:
        raise ReconstructionValidationError("Digital Twin state export manifest checksum mismatch.")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema") != DIGITAL_TWIN_STATE_EXPORT_SCHEMA:
        raise ReconstructionValidationError("Unsupported Digital Twin state export schema.")
    if raw.get("artifact_payloads_embedded") is not False or raw.get("portable_payload_bundle") is not False:
        raise ReconstructionValidationError("Digital Twin state export must not claim embedded/portable payloads.")
    if any(bool(raw.get(key, False)) for key in (
        "canonical_geometry_authoritative",
        "metric_scale_authoritative",
        "measurement_authoritative",
        "qc_authoritative",
        "certified_metrology",
    )):
        raise ReconstructionValidationError("Digital Twin state export contains a forbidden authority claim.")
    files = raw.get("files", [])
    if not isinstance(files, list) or not files:
        raise ReconstructionValidationError("Digital Twin state export contains no integrity files.")
    for item in files:
        if not isinstance(item, Mapping):
            raise ReconstructionValidationError("Digital Twin state export file records must be objects.")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ReconstructionValidationError("Digital Twin state export file path escapes the bundle.")
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ReconstructionValidationError("Digital Twin state export file escapes the bundle.") from exc
        if not target.is_file():
            raise ReconstructionValidationError(f"Digital Twin state export file is missing: {target}")
        if target.stat().st_size != int(item.get("size_bytes", -1)):
            raise ReconstructionValidationError(f"Digital Twin state export file size changed: {target}")
        if _sha256_file(target) != str(item.get("sha256", "")):
            raise ReconstructionValidationError(f"Digital Twin state export file checksum changed: {target}")
    scene = read_persistent_reconstruction_scene(root / PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST)
    authority_manifest = root / PERSISTENT_SCENE_AUTHORITY_MANIFEST
    if authority_manifest.is_file():
        authority, _authority_path = read_optional_persistent_scene_authority_snapshot(
            root / PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST,
            expected_scene=scene,
        )
        if authority is None:
            raise ReconstructionValidationError("Digital Twin state export lost its authority sidecar.")
    state, _state_path = read_optional_digital_twin_workspace_state(
        root / PERSISTENT_RECONSTRUCTION_SCENE_MANIFEST,
        expected_scene=scene,
    )
    if state is None:
        raise ReconstructionValidationError("Digital Twin state export lost its workspace sidecar.")
    if state.workspace_fingerprint_sha256 != str(raw.get("workspace_fingerprint_sha256", "")):
        raise ReconstructionValidationError("Digital Twin state export workspace fingerprint mismatch.")
    return dict(raw)


__all__ = [
    "DIGITAL_TWIN_WORKSPACE_SCHEMA_VERSION",
    "DIGITAL_TWIN_WORKSPACE_PHASE",
    "DIGITAL_TWIN_WORKSPACE_MANIFEST",
    "DIGITAL_TWIN_WORKSPACE_CHECKSUM",
    "DIGITAL_TWIN_STATE_EXPORT_SCHEMA",
    "DIGITAL_TWIN_STATE_EXPORT_MANIFEST",
    "DIGITAL_TWIN_STATE_EXPORT_CHECKSUM",
    "DigitalTwinProcessingHistoryRecord",
    "DigitalTwinWorkspaceState",
    "build_digital_twin_workspace_state",
    "write_digital_twin_workspace_state",
    "read_optional_digital_twin_workspace_state",
    "export_digital_twin_state_bundle",
    "verify_digital_twin_state_bundle",
]
