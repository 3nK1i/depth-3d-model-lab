"""I6.1A dependency-light offline reconstruction workspace contract.

This module is a descriptive product-layer composition only.  It does not own
runtime state, start/cancel reconstruction, persist scenes, resume checkpoints,
select providers, activate Auto, or change reconstruction authority.

The contract deliberately composes the already-qualified application session
model with passive runtime/result/progress evidence so later I6 product steps
can bind to one stable workspace snapshot without duplicating the existing
controller, persistence, provider, or viewer architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

from app.reconstruction_session_model import (
    ReconstructionSessionIdentityStatus,
    ReconstructionSessionModel,
)

CANONICAL_CAMERA_AUTHORITY_OWNER = "pycolmap_sfm"
PRODUCTION_AUTO_POLICY_SCOPE = "offline_shadow_only"
_SUPPORTED_OFFLINE_WORKFLOWS = frozenset({"image_sequence", "recorded_video"})


class OfflineWorkspaceEvidenceIdentityStatus(str, Enum):
    NONE = "none"
    CONSISTENT = "consistent"
    CONFLICT = "conflict"


class OfflineWorkspaceScaleStatus(str, Enum):
    UNQUALIFIED = "unqualified"
    ARBITRARY = "arbitrary"
    METRIC = "metric"
    CALIBRATED_METRIC = "calibrated_metric"


@dataclass(frozen=True, slots=True)
class OfflineReconstructionWorkspaceSnapshot:
    """Immutable passive view of the current offline reconstruction workspace."""

    session_model: ReconstructionSessionModel
    session_id: str
    result_session_id: str
    evidence_identity_status: OfflineWorkspaceEvidenceIdentityStatus
    identity_details: str
    workflow: str
    offline_workflow_supported: bool
    source_ready: bool
    lifecycle: str
    offline_run_state: str
    output_dir: str
    result_path: str
    result_status: str
    current_stage: str
    progress_fraction: float | None
    progress_message: str
    scene_availability: str
    partial_result_available: bool
    persistent_scene_available: bool
    scene_manifest: str
    checkpoint_available: bool
    checkpoint_manifest: str
    checkpoint_resume_allowed: bool
    checkpoint_next_stage: str
    camera_authority_owner: str
    camera_pose_available: bool
    canonical_geometry_authoritative: bool
    scale_status: OfflineWorkspaceScaleStatus
    metric_coordinates_allowed: bool
    calibrated_metric_scale: bool
    certified_metrology: bool
    production_auto_policy_scope: str
    production_auto_qualified: bool
    production_auto_authorized: bool

    def __post_init__(self) -> None:
        if not isinstance(self.session_model, ReconstructionSessionModel):
            raise TypeError("session_model must be ReconstructionSessionModel.")
        if not isinstance(self.evidence_identity_status, OfflineWorkspaceEvidenceIdentityStatus):
            raise TypeError(
                "evidence_identity_status must be OfflineWorkspaceEvidenceIdentityStatus."
            )
        if not isinstance(self.scale_status, OfflineWorkspaceScaleStatus):
            raise TypeError("scale_status must be OfflineWorkspaceScaleStatus.")

        session_id = _uuid_or_empty(self.session_id, "session_id")
        result_session_id = _uuid_or_empty(self.result_session_id, "result_session_id")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "result_session_id", result_session_id)

        if self.camera_authority_owner != CANONICAL_CAMERA_AUTHORITY_OWNER:
            raise ValueError("I6.1A must preserve PyCOLMAP as canonical camera authority.")
        if bool(self.certified_metrology):
            raise ValueError("I6.1A cannot promote reconstruction state to certified metrology.")
        if self.production_auto_policy_scope != PRODUCTION_AUTO_POLICY_SCOPE:
            raise ValueError("I6.1A must preserve the frozen offline/shadow Auto policy scope.")
        if bool(self.production_auto_qualified) or bool(self.production_auto_authorized):
            raise ValueError("I6.1A cannot qualify or authorize production Auto.")

        progress = self.progress_fraction
        if progress is not None:
            progress = float(progress)
            if not isfinite(progress) or progress < 0.0 or progress > 1.0:
                raise ValueError("progress_fraction must be finite and within [0, 1].")
        object.__setattr__(self, "progress_fraction", progress)

        metric = bool(self.metric_coordinates_allowed)
        calibrated = bool(self.calibrated_metric_scale)
        if calibrated and not metric:
            raise ValueError("calibrated metric scale requires metric-coordinate authority.")
        expected_scale = (
            OfflineWorkspaceScaleStatus.CALIBRATED_METRIC
            if calibrated
            else OfflineWorkspaceScaleStatus.METRIC
            if metric
            else self.scale_status
        )
        if metric and self.scale_status not in {
            OfflineWorkspaceScaleStatus.METRIC,
            OfflineWorkspaceScaleStatus.CALIBRATED_METRIC,
        }:
            raise ValueError("metric-coordinate authority must be reflected in scale_status.")
        if not metric and self.scale_status in {
            OfflineWorkspaceScaleStatus.METRIC,
            OfflineWorkspaceScaleStatus.CALIBRATED_METRIC,
        }:
            raise ValueError("metric scale status requires explicit metric-coordinate authority.")
        if calibrated and self.scale_status is not OfflineWorkspaceScaleStatus.CALIBRATED_METRIC:
            raise ValueError("calibrated metric authority must be reflected in scale_status.")
        if expected_scale is not self.scale_status:
            raise ValueError("workspace scale qualification is internally inconsistent.")

        object.__setattr__(self, "identity_details", str(self.identity_details).strip())
        object.__setattr__(self, "workflow", _value(self.workflow) or "none")
        object.__setattr__(self, "offline_workflow_supported", bool(self.offline_workflow_supported))
        object.__setattr__(self, "source_ready", bool(self.source_ready))
        object.__setattr__(self, "lifecycle", _value(self.lifecycle) or "idle")
        object.__setattr__(self, "offline_run_state", _value(self.offline_run_state))
        object.__setattr__(self, "output_dir", _normalized_path_text(self.output_dir))
        object.__setattr__(self, "result_path", _normalized_path_text(self.result_path))
        object.__setattr__(self, "result_status", _value(self.result_status))
        object.__setattr__(self, "current_stage", _value(self.current_stage))
        object.__setattr__(self, "progress_message", str(self.progress_message).strip())
        object.__setattr__(self, "scene_availability", _value(self.scene_availability) or "none")
        object.__setattr__(self, "partial_result_available", bool(self.partial_result_available))
        object.__setattr__(self, "persistent_scene_available", bool(self.persistent_scene_available))
        object.__setattr__(self, "scene_manifest", _normalized_path_text(self.scene_manifest))
        object.__setattr__(self, "checkpoint_available", bool(self.checkpoint_available))
        object.__setattr__(self, "checkpoint_manifest", _normalized_path_text(self.checkpoint_manifest))
        object.__setattr__(self, "checkpoint_resume_allowed", bool(self.checkpoint_resume_allowed))
        object.__setattr__(self, "checkpoint_next_stage", _value(self.checkpoint_next_stage))
        object.__setattr__(self, "camera_pose_available", bool(self.camera_pose_available))
        object.__setattr__(
            self,
            "canonical_geometry_authoritative",
            bool(self.canonical_geometry_authoritative),
        )
        object.__setattr__(self, "metric_coordinates_allowed", metric)
        object.__setattr__(self, "calibrated_metric_scale", calibrated)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "production_auto_qualified", False)
        object.__setattr__(self, "production_auto_authorized", False)

    @property
    def identity_conflict(self) -> bool:
        return self.evidence_identity_status is OfflineWorkspaceEvidenceIdentityStatus.CONFLICT

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity": {
                "session_id": self.session_id,
                "result_session_id": self.result_session_id,
                "status": self.evidence_identity_status.value,
                "details": self.identity_details,
            },
            "source": {
                "workflow": self.workflow,
                "offline_workflow_supported": self.offline_workflow_supported,
                "ready": self.source_ready,
            },
            "runtime": {
                "lifecycle": self.lifecycle,
                "offline_run_state": self.offline_run_state,
                "output_dir": self.output_dir,
                "stage": self.current_stage,
                "progress_fraction": self.progress_fraction,
                "message": self.progress_message,
            },
            "result": {
                "path": self.result_path,
                "status": self.result_status,
                "scene_availability": self.scene_availability,
                "partial_available": self.partial_result_available,
                "persistent_scene_available": self.persistent_scene_available,
                "scene_manifest": self.scene_manifest,
            },
            "checkpoint": {
                "available": self.checkpoint_available,
                "manifest": self.checkpoint_manifest,
                "resume_allowed": self.checkpoint_resume_allowed,
                "next_stage": self.checkpoint_next_stage,
            },
            "authority": {
                "camera_authority_owner": self.camera_authority_owner,
                "camera_pose_available": self.camera_pose_available,
                "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
                "scale_status": self.scale_status.value,
                "metric_coordinates_allowed": self.metric_coordinates_allowed,
                "calibrated_metric_scale": self.calibrated_metric_scale,
                "certified_metrology": False,
            },
            "production_auto": {
                "policy_scope": self.production_auto_policy_scope,
                "qualified": False,
                "authorized": False,
            },
        }


def build_offline_reconstruction_workspace_snapshot(
    *,
    session_model: ReconstructionSessionModel,
    source_selection: Any = None,
    offline_state: Any = None,
    offline_output_dir: str | Path | None = None,
    offline_result: Mapping[str, object] | None = None,
    offline_result_path: str | Path | None = None,
    latest_event: Mapping[str, object] | None = None,
    scene_availability: Any = None,
    persistent_authority_snapshot: Any = None,
) -> OfflineReconstructionWorkspaceSnapshot:
    """Compose a passive I6 workspace snapshot from existing authoritative state."""

    if not isinstance(session_model, ReconstructionSessionModel):
        raise TypeError("session_model must be ReconstructionSessionModel.")
    if offline_result is not None and not isinstance(offline_result, Mapping):
        raise TypeError("offline_result must be a mapping or None.")
    if latest_event is not None and not isinstance(latest_event, Mapping):
        raise TypeError("latest_event must be a mapping or None.")

    workflow = session_model.workflow
    supported = workflow in _SUPPORTED_OFFLINE_WORKFLOWS
    source_ready = bool(getattr(source_selection, "ready", False)) and supported

    result = dict(offline_result or {})
    if bool(result.get("certified_metrology", False)):
        raise ValueError("I6.1A cannot consume an offline result that claims certified metrology.")
    result_session_id = _uuid_or_empty(result.get("session_id", ""), "offline_result.session_id")
    evidence_status, identity_details = _resolve_identity(
        session_model=session_model,
        result_session_id=result_session_id,
    )

    event = dict(latest_event or {})
    progress_fraction = event.get("progress_fraction")
    if progress_fraction is not None:
        try:
            progress_fraction = float(progress_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("progress_fraction must be numeric when present.") from exc
        if not isfinite(progress_fraction) or progress_fraction < 0.0 or progress_fraction > 1.0:
            raise ValueError("progress_fraction must be finite and within [0, 1].")

    availability = _value(scene_availability) or session_model.scene_availability or "none"
    persistent_available = bool(session_model.scene_id) and availability in {"ready", "degraded"}

    camera_pose_available = bool(result.get("camera_pose_available", False)) or any(
        artifact.stage == "camera_pose" for artifact in session_model.artifacts
    )
    result_status = _value(result.get("status", ""))
    partial_available = not persistent_available and (
        availability == "partial"
        or (
            result_status == "completed"
            and camera_pose_available
            and not bool(result.get("persistent_scene_ready", False))
        )
    )

    metric_coordinates_allowed = False
    calibrated_metric_scale = False
    if persistent_authority_snapshot is not None:
        if bool(getattr(persistent_authority_snapshot, "certified_metrology", False)):
            raise ValueError("I6.1A cannot consume authority that claims certified metrology.")
        metric_coordinates_allowed = bool(
            getattr(persistent_authority_snapshot, "metric_coordinates_allowed", False)
        )
        calibrated_metric_scale = bool(
            getattr(persistent_authority_snapshot, "calibrated_metric_scale", False)
        )
        if calibrated_metric_scale and not metric_coordinates_allowed:
            raise ValueError(
                "Persistent authority cannot authorize calibrated scale without metric coordinates."
            )

    scale_status = _scale_status(
        result=result,
        metric_coordinates_allowed=metric_coordinates_allowed,
        calibrated_metric_scale=calibrated_metric_scale,
    )

    return OfflineReconstructionWorkspaceSnapshot(
        session_model=session_model,
        session_id=session_model.session_id,
        result_session_id=result_session_id,
        evidence_identity_status=evidence_status,
        identity_details=identity_details,
        workflow=workflow,
        offline_workflow_supported=supported,
        source_ready=source_ready,
        lifecycle=session_model.lifecycle.value,
        offline_run_state=_value(offline_state),
        output_dir=str(offline_output_dir or ""),
        result_path=str(offline_result_path or ""),
        result_status=result_status,
        current_stage=_value(event.get("stage", "")),
        progress_fraction=progress_fraction,
        progress_message=str(event.get("message", "") or "").strip(),
        scene_availability=availability,
        partial_result_available=partial_available,
        persistent_scene_available=persistent_available,
        scene_manifest=session_model.scene_manifest,
        checkpoint_available=bool(session_model.checkpoint_id),
        checkpoint_manifest=session_model.checkpoint_manifest,
        checkpoint_resume_allowed=session_model.checkpoint_resume_allowed,
        checkpoint_next_stage=session_model.checkpoint_next_stage,
        camera_authority_owner=CANONICAL_CAMERA_AUTHORITY_OWNER,
        camera_pose_available=camera_pose_available,
        canonical_geometry_authoritative=session_model.canonical_geometry_authoritative,
        scale_status=scale_status,
        metric_coordinates_allowed=metric_coordinates_allowed,
        calibrated_metric_scale=calibrated_metric_scale,
        certified_metrology=False,
        production_auto_policy_scope=PRODUCTION_AUTO_POLICY_SCOPE,
        production_auto_qualified=False,
        production_auto_authorized=False,
    )


def _resolve_identity(
    *,
    session_model: ReconstructionSessionModel,
    result_session_id: str,
) -> tuple[OfflineWorkspaceEvidenceIdentityStatus, str]:
    details: list[str] = []
    if session_model.identity_details:
        details.append(session_model.identity_details)
    if result_session_id:
        details.append(f"offline_result={result_session_id}")

    if session_model.identity_status is ReconstructionSessionIdentityStatus.CONFLICT:
        return OfflineWorkspaceEvidenceIdentityStatus.CONFLICT, " • ".join(details)

    identities = {value for value in (session_model.session_id, result_session_id) if value}
    if len(identities) > 1:
        return OfflineWorkspaceEvidenceIdentityStatus.CONFLICT, " • ".join(details)
    if identities:
        return OfflineWorkspaceEvidenceIdentityStatus.CONSISTENT, " • ".join(details)
    return (
        OfflineWorkspaceEvidenceIdentityStatus.NONE,
        " • ".join(details) or "No offline reconstruction evidence identity is currently focused.",
    )


def _scale_status(
    *,
    result: Mapping[str, object],
    metric_coordinates_allowed: bool,
    calibrated_metric_scale: bool,
) -> OfflineWorkspaceScaleStatus:
    if calibrated_metric_scale:
        return OfflineWorkspaceScaleStatus.CALIBRATED_METRIC
    if metric_coordinates_allowed:
        return OfflineWorkspaceScaleStatus.METRIC

    sfm_summary = result.get("sfm_summary")
    if isinstance(sfm_summary, Mapping):
        gauge = sfm_summary.get("gauge")
        if isinstance(gauge, Mapping) and _value(gauge.get("scale_status", "")) == "arbitrary":
            return OfflineWorkspaceScaleStatus.ARBITRARY
    return OfflineWorkspaceScaleStatus.UNQUALIFIED


def _uuid_or_empty(value: Any, field_name: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return str(UUID(raw))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a valid UUID when present.") from exc


def _normalized_path_text(value: str | Path | None) -> str:
    raw = str(value or "").strip()
    return str(Path(raw).expanduser().resolve()) if raw else ""


def _value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip()


__all__ = [
    "CANONICAL_CAMERA_AUTHORITY_OWNER",
    "PRODUCTION_AUTO_POLICY_SCOPE",
    "OfflineReconstructionWorkspaceSnapshot",
    "OfflineWorkspaceEvidenceIdentityStatus",
    "OfflineWorkspaceScaleStatus",
    "build_offline_reconstruction_workspace_snapshot",
]
