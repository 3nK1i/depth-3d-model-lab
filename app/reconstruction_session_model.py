"""Phase 8.8G1 application-level reconstruction session composition.

G1 does not replace the Phase 8 core ``ReconstructionSession`` lifecycle and it
introduces no second persistence format.  It composes the application state that
already exists around runtime session IDs, persistent scenes, checkpoints,
artifacts, providers and selected workflow context into one immutable,
use-case-agnostic model for the later G2-G6 UX steps.

The model is deliberately descriptive.  It does not save/reload a session,
resume a checkpoint, recover a failed worker, launch a backend, or change any
geometry/measurement/inspection/reference authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID


class ReconstructionSessionFocus(str, Enum):
    NONE = "none"
    RUNTIME = "runtime"
    PERSISTENT_SCENE = "persistent_scene"
    CHECKPOINT = "checkpoint"
    LAST_RUNTIME = "last_runtime"


class ReconstructionSessionIdentityStatus(str, Enum):
    NONE = "none"
    CONSISTENT = "consistent"
    CONFLICT = "conflict"


class ReconstructionSessionLifecycle(str, Enum):
    IDLE = "idle"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    CANCELLED = "cancelled"
    FAILED = "failed"
    DEGRADED = "degraded"
    CHECKPOINTED = "checkpointed"


@dataclass(frozen=True, slots=True)
class ReconstructionSessionArtifactSummary:
    artifact_id: str
    stage: str
    kind: str
    authority: str
    provider_id: str

    def __post_init__(self) -> None:
        for field_name in ("artifact_id", "stage", "kind", "authority", "provider_id"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty.")
            object.__setattr__(self, field_name, value)

    def as_dict(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "stage": self.stage,
            "kind": self.kind,
            "authority": self.authority,
            "provider_id": self.provider_id,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionSessionProviderSummary:
    provider_id: str
    provider_version: str
    execution: str
    stages: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("provider_id", "provider_version", "execution"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty.")
            object.__setattr__(self, field_name, value)
        stages = tuple(str(value).strip() for value in self.stages if str(value).strip())
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("provider stages must be unique and non-empty.")
        object.__setattr__(self, "stages", stages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "execution": self.execution,
            "stages": list(self.stages),
        }


@dataclass(frozen=True, slots=True)
class ReconstructionSessionModel:
    """One immutable application view of the currently focused reconstruction session."""

    session_id: str
    focus: ReconstructionSessionFocus
    identity_status: ReconstructionSessionIdentityStatus
    lifecycle: ReconstructionSessionLifecycle
    workflow: str
    selected_source_label: str
    selected_source_detail: str
    selected_source_path: str
    scene_id: str
    scene_manifest: str
    scene_state: str
    scene_availability: str
    checkpoint_id: str
    checkpoint_manifest: str
    checkpoint_next_stage: str
    checkpoint_resume_allowed: bool
    artifacts: tuple[ReconstructionSessionArtifactSummary, ...]
    providers: tuple[ReconstructionSessionProviderSummary, ...]
    appearance_representation_kinds: tuple[str, ...]
    scene_created_at_utc: str
    checkpoint_created_at_utc: str
    canonical_geometry_authoritative: bool
    certified_metrology: bool
    identity_details: str

    def __post_init__(self) -> None:
        session_id = str(self.session_id).strip()
        if session_id:
            try:
                session_id = str(UUID(session_id))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError("session_id must be a valid UUID when present.") from exc
        object.__setattr__(self, "session_id", session_id)
        if not isinstance(self.focus, ReconstructionSessionFocus):
            raise TypeError("focus must be ReconstructionSessionFocus.")
        if not isinstance(self.identity_status, ReconstructionSessionIdentityStatus):
            raise TypeError("identity_status must be ReconstructionSessionIdentityStatus.")
        if not isinstance(self.lifecycle, ReconstructionSessionLifecycle):
            raise TypeError("lifecycle must be ReconstructionSessionLifecycle.")
        artifacts = tuple(self.artifacts)
        providers = tuple(self.providers)
        representations = tuple(str(value).strip() for value in self.appearance_representation_kinds)
        if any(not isinstance(item, ReconstructionSessionArtifactSummary) for item in artifacts):
            raise TypeError("artifacts must contain ReconstructionSessionArtifactSummary values.")
        if any(not isinstance(item, ReconstructionSessionProviderSummary) for item in providers):
            raise TypeError("providers must contain ReconstructionSessionProviderSummary values.")
        if len({item.artifact_id for item in artifacts}) != len(artifacts):
            raise ValueError("session artifact IDs must be unique.")
        if len({item.provider_id for item in providers}) != len(providers):
            raise ValueError("session provider IDs must be unique.")
        if any(not value for value in representations) or len(set(representations)) != len(representations):
            raise ValueError("appearance representation kinds must be unique and non-empty.")
        if self.identity_status is ReconstructionSessionIdentityStatus.NONE and session_id:
            raise ValueError("identity_status NONE cannot carry a session_id.")
        if self.identity_status is not ReconstructionSessionIdentityStatus.NONE and not session_id:
            raise ValueError("a non-empty identity status requires session_id.")
        if bool(self.certified_metrology):
            raise ValueError("G1 cannot promote reconstruction session state to certified metrology.")
        object.__setattr__(self, "workflow", _value(self.workflow))
        object.__setattr__(self, "selected_source_label", str(self.selected_source_label).strip())
        object.__setattr__(self, "selected_source_detail", str(self.selected_source_detail).strip())
        object.__setattr__(self, "selected_source_path", str(self.selected_source_path).strip())
        object.__setattr__(self, "scene_id", str(self.scene_id).strip())
        object.__setattr__(self, "scene_manifest", str(self.scene_manifest).strip())
        object.__setattr__(self, "scene_state", _value(self.scene_state))
        object.__setattr__(self, "scene_availability", _value(self.scene_availability))
        object.__setattr__(self, "checkpoint_id", str(self.checkpoint_id).strip())
        object.__setattr__(self, "checkpoint_manifest", str(self.checkpoint_manifest).strip())
        object.__setattr__(self, "checkpoint_next_stage", _value(self.checkpoint_next_stage))
        object.__setattr__(self, "checkpoint_resume_allowed", bool(self.checkpoint_resume_allowed))
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "providers", providers)
        object.__setattr__(self, "appearance_representation_kinds", representations)
        object.__setattr__(self, "scene_created_at_utc", str(self.scene_created_at_utc).strip())
        object.__setattr__(self, "checkpoint_created_at_utc", str(self.checkpoint_created_at_utc).strip())
        object.__setattr__(self, "canonical_geometry_authoritative", bool(self.canonical_geometry_authoritative))
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "identity_details", str(self.identity_details).strip())

    @property
    def has_session(self) -> bool:
        return bool(self.session_id)

    @property
    def short_session_id(self) -> str:
        return self.session_id[:8] if self.session_id else ""

    @property
    def context_text(self) -> str:
        if self.identity_status is ReconstructionSessionIdentityStatus.CONFLICT:
            return "Session identity conflict"
        focus = {
            ReconstructionSessionFocus.NONE: "No session context",
            ReconstructionSessionFocus.RUNTIME: "Active runtime",
            ReconstructionSessionFocus.PERSISTENT_SCENE: "Persistent scene",
            ReconstructionSessionFocus.CHECKPOINT: "Checkpoint",
            ReconstructionSessionFocus.LAST_RUNTIME: "Last runtime",
        }[self.focus]
        return f"{focus} • {_humanize(self.lifecycle.value)}"

    @property
    def evidence_text(self) -> str:
        parts: list[str] = []
        if self.scene_id:
            parts.append("Scene")
        if self.checkpoint_id:
            parts.append("checkpoint")
        if self.artifacts:
            parts.append(f"{len(self.artifacts)} artifacts")
        if self.providers:
            parts.append(f"{len(self.providers)} providers")
        if self.appearance_representation_kinds:
            parts.append(f"{len(self.appearance_representation_kinds)} appearance")
        return " • ".join(parts) if parts else "Runtime/session metadata only"

    @property
    def evidence_details(self) -> str:
        details: list[str] = []
        if self.scene_id:
            details.append(f"Scene: {self.scene_id}")
        if self.scene_manifest:
            details.append(f"Scene manifest: {self.scene_manifest}")
        if self.checkpoint_id:
            details.append(f"Checkpoint: {self.checkpoint_id}")
        if self.checkpoint_manifest:
            details.append(f"Checkpoint manifest: {self.checkpoint_manifest}")
        if self.artifacts:
            details.append("Artifact kinds: " + ", ".join(item.kind for item in self.artifacts))
        if self.providers:
            details.append("Providers: " + ", ".join(item.provider_id for item in self.providers))
        if self.appearance_representation_kinds:
            details.append(
                "Appearance representations: " + ", ".join(self.appearance_representation_kinds)
            )
        return "\n".join(details) or "No persistent scene/checkpoint evidence is attached."

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "focus": self.focus.value,
            "identity_status": self.identity_status.value,
            "lifecycle": self.lifecycle.value,
            "workflow": self.workflow,
            "selected_source": {
                "label": self.selected_source_label,
                "detail": self.selected_source_detail,
                "path": self.selected_source_path,
            },
            "scene": {
                "scene_id": self.scene_id,
                "manifest": self.scene_manifest,
                "state": self.scene_state,
                "availability": self.scene_availability,
                "created_at_utc": self.scene_created_at_utc,
            },
            "checkpoint": {
                "checkpoint_id": self.checkpoint_id,
                "manifest": self.checkpoint_manifest,
                "next_stage": self.checkpoint_next_stage,
                "resume_allowed": self.checkpoint_resume_allowed,
                "created_at_utc": self.checkpoint_created_at_utc,
            },
            "artifacts": [item.as_dict() for item in self.artifacts],
            "providers": [item.as_dict() for item in self.providers],
            "appearance_representation_kinds": list(self.appearance_representation_kinds),
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "certified_metrology": False,
            "identity_details": self.identity_details,
        }


def build_reconstruction_session_model(
    *,
    source_selection: Any = None,
    live_session_id: str | None = None,
    offline_session_id: str | None = None,
    live_state: Any = None,
    offline_state: Any = None,
    scene_availability: Any = None,
    scene: Any = None,
    scene_manifest: str | Path | None = None,
    checkpoint: Any = None,
    checkpoint_manifest: str | Path | None = None,
) -> ReconstructionSessionModel:
    """Compose existing application/core session evidence without mutating it."""

    workflow = _value(getattr(source_selection, "workflow", "none")) or "none"
    live_state_value = _value(live_state)
    offline_state_value = _value(offline_state)
    live_id = _uuid_or_empty(live_session_id, "live_session_id")
    offline_id = _uuid_or_empty(offline_session_id, "offline_session_id")
    scene_id = str(getattr(scene, "scene_id", "") or "").strip()
    scene_session_id = _uuid_or_empty(getattr(scene, "session_id", ""), "scene.session_id")
    checkpoint_id = str(getattr(checkpoint, "checkpoint_id", "") or "").strip()
    checkpoint_session_id = _uuid_or_empty(
        getattr(checkpoint, "session_id", ""), "checkpoint.session_id"
    )

    active_live = live_state_value in {"starting", "running", "paused", "recovering", "stopping"}
    active_offline = offline_state_value in {"starting", "running", "cancelling"}

    if active_live:
        focus = ReconstructionSessionFocus.RUNTIME
        anchor = live_id
    elif active_offline:
        focus = ReconstructionSessionFocus.RUNTIME
        anchor = offline_id
    elif scene_session_id:
        focus = ReconstructionSessionFocus.PERSISTENT_SCENE
        anchor = scene_session_id
    elif checkpoint_session_id:
        focus = ReconstructionSessionFocus.CHECKPOINT
        anchor = checkpoint_session_id
    elif workflow in {"image_sequence", "recorded_video"} and offline_id:
        focus = ReconstructionSessionFocus.LAST_RUNTIME
        anchor = offline_id
    elif workflow == "live_source" and live_id:
        focus = ReconstructionSessionFocus.LAST_RUNTIME
        anchor = live_id
    elif offline_id or live_id:
        focus = ReconstructionSessionFocus.LAST_RUNTIME
        anchor = offline_id or live_id
    else:
        focus = ReconstructionSessionFocus.NONE
        anchor = ""

    evidence_ids: list[tuple[str, str]] = []
    if focus is ReconstructionSessionFocus.RUNTIME and anchor:
        evidence_ids.append(("runtime", anchor))
    if scene_session_id:
        evidence_ids.append(("scene", scene_session_id))
    if checkpoint_session_id:
        evidence_ids.append(("checkpoint", checkpoint_session_id))
    if not evidence_ids and anchor:
        evidence_ids.append(("last_runtime", anchor))

    distinct_ids = {value for _, value in evidence_ids}
    if not anchor:
        identity_status = ReconstructionSessionIdentityStatus.NONE
        identity_details = "No reconstruction session identity is currently focused."
    elif len(distinct_ids) <= 1:
        identity_status = ReconstructionSessionIdentityStatus.CONSISTENT
        identity_details = " • ".join(f"{label}={value}" for label, value in evidence_ids)
    else:
        identity_status = ReconstructionSessionIdentityStatus.CONFLICT
        identity_details = "Session evidence belongs to different identities: " + "; ".join(
            f"{label}={value}" for label, value in evidence_ids
        )

    lifecycle = _resolve_lifecycle(
        live_state=live_state_value,
        offline_state=offline_state_value,
        active_live=active_live,
        active_offline=active_offline,
        focus=focus,
        scene=scene,
        checkpoint=checkpoint,
        source_ready=bool(getattr(source_selection, "ready", False)),
    )

    artifacts = tuple(_artifact_summary(item) for item in tuple(getattr(scene, "artifacts", ()) or ()))
    providers = _provider_summaries(scene=scene, checkpoint=checkpoint)
    appearance_kinds = tuple(
        item.kind for item in artifacts if item.authority == "appearance_only"
    )

    source_path = getattr(source_selection, "path", None)
    scene_authority = bool(getattr(scene, "canonical_geometry_authoritative", False)) if scene is not None else False
    certified_metrology = bool(getattr(scene, "certified_metrology", False)) if scene is not None else False

    return ReconstructionSessionModel(
        session_id=anchor,
        focus=focus,
        identity_status=identity_status,
        lifecycle=lifecycle,
        workflow=workflow,
        selected_source_label=str(getattr(source_selection, "display_name", "") or "").strip(),
        selected_source_detail=str(getattr(source_selection, "detail", "") or "").strip(),
        selected_source_path=str(Path(source_path).expanduser().resolve()) if source_path else "",
        scene_id=scene_id,
        scene_manifest=str(Path(scene_manifest).expanduser().resolve()) if scene_manifest else "",
        scene_state=_value(getattr(scene, "state", "")),
        scene_availability=_value(scene_availability),
        checkpoint_id=checkpoint_id,
        checkpoint_manifest=str(Path(checkpoint_manifest).expanduser().resolve()) if checkpoint_manifest else "",
        checkpoint_next_stage=_value(getattr(checkpoint, "next_stage", "")),
        checkpoint_resume_allowed=bool(getattr(checkpoint, "resume_allowed", False)),
        artifacts=artifacts,
        providers=providers,
        appearance_representation_kinds=appearance_kinds,
        scene_created_at_utc=str(getattr(scene, "created_at_utc", "") or "").strip(),
        checkpoint_created_at_utc=str(getattr(checkpoint, "created_at_utc", "") or "").strip(),
        canonical_geometry_authoritative=scene_authority,
        certified_metrology=certified_metrology,
        identity_details=identity_details,
    )


def _resolve_lifecycle(
    *,
    live_state: str,
    offline_state: str,
    active_live: bool,
    active_offline: bool,
    focus: ReconstructionSessionFocus,
    scene: Any,
    checkpoint: Any,
    source_ready: bool,
) -> ReconstructionSessionLifecycle:
    if active_live:
        return {
            "starting": ReconstructionSessionLifecycle.STARTING,
            "running": ReconstructionSessionLifecycle.RUNNING,
            "paused": ReconstructionSessionLifecycle.PAUSED,
            "recovering": ReconstructionSessionLifecycle.RECOVERING,
            "stopping": ReconstructionSessionLifecycle.STOPPING,
        }[live_state]
    if active_offline:
        return {
            "starting": ReconstructionSessionLifecycle.STARTING,
            "running": ReconstructionSessionLifecycle.RUNNING,
            "cancelling": ReconstructionSessionLifecycle.STOPPING,
        }[offline_state]
    if scene is not None:
        return (
            ReconstructionSessionLifecycle.DEGRADED
            if _value(getattr(scene, "state", "")) == "degraded"
            else ReconstructionSessionLifecycle.READY
        )
    if checkpoint is not None:
        return ReconstructionSessionLifecycle.CHECKPOINTED
    if focus is ReconstructionSessionFocus.LAST_RUNTIME:
        if offline_state == "completed":
            return ReconstructionSessionLifecycle.COMPLETED
        if offline_state == "cancelled":
            return ReconstructionSessionLifecycle.CANCELLED
        if offline_state == "failed" or live_state == "failed":
            return ReconstructionSessionLifecycle.FAILED
        if live_state == "stopped":
            return ReconstructionSessionLifecycle.STOPPED
    if source_ready:
        return ReconstructionSessionLifecycle.READY
    return ReconstructionSessionLifecycle.IDLE


def _artifact_summary(item: Any) -> ReconstructionSessionArtifactSummary:
    return ReconstructionSessionArtifactSummary(
        artifact_id=str(getattr(item, "artifact_id", "") or ""),
        stage=_value(getattr(item, "stage", "")),
        kind=_value(getattr(item, "kind", "")),
        authority=_value(getattr(item, "authority", "")),
        provider_id=str(getattr(item, "provider_id", "") or ""),
    )


def _provider_summaries(*, scene: Any, checkpoint: Any) -> tuple[ReconstructionSessionProviderSummary, ...]:
    by_id: dict[str, ReconstructionSessionProviderSummary] = {}
    for item in tuple(getattr(scene, "provider_provenance", ()) or ()):
        provider_id = str(getattr(item, "provider_id", "") or "").strip()
        if not provider_id:
            continue
        by_id[provider_id] = ReconstructionSessionProviderSummary(
            provider_id=provider_id,
            provider_version=str(getattr(item, "provider_version", "") or "unknown"),
            execution=_value(getattr(item, "execution", "unknown")) or "unknown",
            stages=tuple(_value(stage) for stage in tuple(getattr(item, "stages", ()) or ())),
        )
    for raw in tuple(getattr(checkpoint, "provider_versions", ()) or ()):
        if not isinstance(raw, (tuple, list)) or len(raw) != 4:
            continue
        provider_id, version, _adapter, execution = (str(value).strip() for value in raw)
        if not provider_id or provider_id in by_id:
            continue
        stages = tuple(
            _value(getattr(result, "stage", ""))
            for result in tuple(getattr(checkpoint, "stage_results", ()) or ())
            if str(getattr(getattr(result, "artifact", None), "provider_id", "") or "").strip()
            == provider_id
        )
        if not stages:
            stages = ("checkpoint_provenance",)
        by_id[provider_id] = ReconstructionSessionProviderSummary(
            provider_id=provider_id,
            provider_version=version or "unknown",
            execution=execution or "unknown",
            stages=stages,
        )
    return tuple(by_id[key] for key in sorted(by_id))


def _uuid_or_empty(value: Any, field_name: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return str(UUID(raw))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a valid UUID when present.") from exc


def _value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip()


def _humanize(value: str) -> str:
    return str(value).replace("_", " ").strip().title() or "—"
