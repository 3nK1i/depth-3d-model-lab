"""I6.3C passive controller-facing Reconstruction Advisor state composition.

This module binds the already-frozen I6.3A contract and I6.3B pure reasoning
engine to existing product state without granting the Advisor any execution
authority.  It intentionally does not resolve a live scene into an I5 evidence
context, discover/activate providers, mutate settings, start reconstruction, or
change camera/geometry authority.

An explicit :class:`ReconstructionAdvisorEvidenceBinding` supplies only frozen
qualification evidence, descriptive resources, and concrete-provider
availability.  Current source/workspace/SfM facts remain owned by the existing
ReconstructionController and are composed read-only into one immutable live
Advisor state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app.reconstruction_offline_workspace import (
    OfflineReconstructionWorkspaceSnapshot,
    OfflineWorkspaceEvidenceIdentityStatus,
)
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorEvidenceObservation,
    AdvisorResourceObservation,
    ReconstructionAdvisorDecision,
    ReconstructionAdvisorRequest,
)
from core.reconstruction.reconstruction_advisor_engine import (
    evaluate_reconstruction_advisor,
)

I63C_SCHEMA_VERSION = 1
I63C_PHASE = "I6.3C"
I63C_COMPOSITION_PROFILE = "passive_explicit_frozen_evidence_binding"


def _unobserved_resources() -> AdvisorResourceObservation:
    return AdvisorResourceObservation(
        gpu_name="unobserved",
        cuda_ready=False,
        total_vram_bytes=0,
        free_vram_bytes=0,
        available_ram_bytes=0,
    )


def _unique_provider_ids(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    result = tuple(str(value or "").strip() for value in values)
    if any(not value for value in result):
        raise ValueError("available_provider_ids must not contain empty values.")
    if "auto" in result:
        raise ValueError("available_provider_ids must contain concrete providers only.")
    if len(set(result)) != len(result):
        raise ValueError("available_provider_ids must be unique.")
    return result


@dataclass(frozen=True, slots=True)
class ReconstructionAdvisorEvidenceBinding:
    """Explicit frozen evidence input; never a live-context resolver or selector."""

    evidence_context_resolution: AdvisorContextResolution
    evidence_context_id: str
    available_provider_ids: tuple[str, ...]
    resources: AdvisorResourceObservation
    provider_evidence: tuple[AdvisorEvidenceObservation, ...]
    runtime_integrity_clean: bool = True
    schema_version: int = I63C_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_context_resolution, AdvisorContextResolution):
            raise TypeError("evidence_context_resolution must be AdvisorContextResolution.")
        context_id = str(self.evidence_context_id or "").strip()
        if self.evidence_context_resolution is AdvisorContextResolution.UNRESOLVED:
            if context_id:
                raise ValueError("UNRESOLVED binding must not carry evidence_context_id.")
        elif not context_id:
            raise ValueError("EXPLICIT_FROZEN_TOKEN binding requires evidence_context_id.")
        object.__setattr__(self, "evidence_context_id", context_id)
        object.__setattr__(
            self,
            "available_provider_ids",
            _unique_provider_ids(self.available_provider_ids),
        )
        if not isinstance(self.resources, AdvisorResourceObservation):
            raise TypeError("resources must be AdvisorResourceObservation.")
        evidence = tuple(self.provider_evidence)
        if any(not isinstance(item, AdvisorEvidenceObservation) for item in evidence):
            raise TypeError("provider_evidence must contain AdvisorEvidenceObservation values.")
        identities = [(item.provider_id, item.evidence_context_id) for item in evidence]
        if len(set(identities)) != len(identities):
            raise ValueError("provider_evidence contains duplicate provider/evidence context identity.")
        object.__setattr__(self, "provider_evidence", evidence)
        if not isinstance(self.runtime_integrity_clean, bool):
            raise TypeError("runtime_integrity_clean must be bool.")
        if int(self.schema_version) != I63C_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.3C evidence-binding schema version.")
        object.__setattr__(self, "schema_version", I63C_SCHEMA_VERSION)

    @classmethod
    def unresolved(cls) -> "ReconstructionAdvisorEvidenceBinding":
        """Return the safe startup binding without inferring provider/context state."""

        return cls(
            evidence_context_resolution=AdvisorContextResolution.UNRESOLVED,
            evidence_context_id="",
            available_provider_ids=(),
            resources=_unobserved_resources(),
            provider_evidence=(),
            runtime_integrity_clean=True,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "composition_profile": I63C_COMPOSITION_PROFILE,
            "evidence_context_resolution": self.evidence_context_resolution.value,
            "evidence_context_id": self.evidence_context_id,
            "available_provider_ids": list(self.available_provider_ids),
            "resources": self.resources.as_dict(),
            "provider_evidence": [item.as_dict() for item in self.provider_evidence],
            "runtime_integrity_clean": self.runtime_integrity_clean,
            "live_context_resolver_qualified": False,
            "provider_activation_authorized": False,
            "production_auto_authorized": False,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionAdvisorLiveState:
    """Immutable passive request+decision snapshot exposed by the controller."""

    request: ReconstructionAdvisorRequest
    decision: ReconstructionAdvisorDecision
    evidence_binding: ReconstructionAdvisorEvidenceBinding
    source_matches_admitted_result: bool
    schema_version: int = I63C_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.request, ReconstructionAdvisorRequest):
            raise TypeError("request must be ReconstructionAdvisorRequest.")
        if not isinstance(self.decision, ReconstructionAdvisorDecision):
            raise TypeError("decision must be ReconstructionAdvisorDecision.")
        if not isinstance(self.evidence_binding, ReconstructionAdvisorEvidenceBinding):
            raise TypeError("evidence_binding must be ReconstructionAdvisorEvidenceBinding.")
        object.__setattr__(
            self,
            "source_matches_admitted_result",
            bool(self.source_matches_admitted_result),
        )
        if int(self.schema_version) != I63C_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.3C live-state schema version.")
        object.__setattr__(self, "schema_version", I63C_SCHEMA_VERSION)

    @property
    def production_effect(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": I63C_PHASE,
            "composition_profile": I63C_COMPOSITION_PROFILE,
            "source_matches_admitted_result": self.source_matches_admitted_result,
            "request": self.request.as_dict(),
            "decision": self.decision.as_dict(),
            "effects": {
                "provider_selection": False,
                "provider_activation": False,
                "settings_mutation": False,
                "start_action": False,
                "worker_execution": False,
                "production_auto": False,
            },
        }


def _value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def _selection_path(source_selection: Any) -> Path | None:
    raw = getattr(source_selection, "path", None)
    if raw is None:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except (TypeError, ValueError, OSError):
        return None


def _result_source_matches(
    source_selection: Any,
    offline_result: Mapping[str, object] | None,
) -> bool:
    if not isinstance(offline_result, Mapping):
        return False
    selected = _selection_path(source_selection)
    if selected is None:
        return False
    source = offline_result.get("source", {})
    if not isinstance(source, Mapping):
        return False
    uri = str(source.get("source_uri", "") or "").strip()
    if not uri:
        return False
    try:
        result_path = Path(uri).expanduser().resolve()
    except (TypeError, ValueError, OSError):
        return False
    return selected == result_path


def _source_fingerprint(
    *,
    source_matches_result: bool,
    offline_result: Mapping[str, object] | None,
) -> str:
    if not source_matches_result or not isinstance(offline_result, Mapping):
        return ""
    source = offline_result.get("source", {})
    if not isinstance(source, Mapping):
        return ""
    return str(source.get("fingerprint", "") or "").strip()


def _authoritative_sfm_usable(
    *,
    source_matches_result: bool,
    workspace: OfflineReconstructionWorkspaceSnapshot,
    offline_result: Mapping[str, object] | None,
) -> bool:
    """Use admitted exact-source result evidence only; never infer SfM usability."""

    if not source_matches_result or not isinstance(offline_result, Mapping):
        return False
    if workspace.evidence_identity_status is not OfflineWorkspaceEvidenceIdentityStatus.CONSISTENT:
        return False
    if str(offline_result.get("status", "")).strip().lower() != "completed":
        return False
    if not bool(offline_result.get("pass", False)) or not workspace.camera_pose_available:
        return False
    sfm_summary = offline_result.get("sfm_summary", {})
    if not isinstance(sfm_summary, Mapping):
        return False
    quality = sfm_summary.get("quality_assessment", {})
    if not isinstance(quality, Mapping):
        return False
    return bool(quality.get("usable", False))


def _workspace_state(workspace: OfflineReconstructionWorkspaceSnapshot) -> str:
    if workspace.persistent_scene_available:
        return "persistent_scene"
    if workspace.partial_result_available:
        return "partial_result"
    lifecycle = str(workspace.lifecycle or "").strip().lower()
    if lifecycle:
        return lifecycle
    return "ready" if workspace.source_ready else "idle"


def build_reconstruction_advisor_live_state(
    *,
    source_selection: Any,
    workspace: OfflineReconstructionWorkspaceSnapshot,
    offline_result: Mapping[str, object] | None,
    evidence_binding: ReconstructionAdvisorEvidenceBinding,
) -> ReconstructionAdvisorLiveState:
    """Compose and evaluate one passive controller-facing Advisor snapshot."""

    if not isinstance(workspace, OfflineReconstructionWorkspaceSnapshot):
        raise TypeError("workspace must be OfflineReconstructionWorkspaceSnapshot.")
    if offline_result is not None and not isinstance(offline_result, Mapping):
        raise TypeError("offline_result must be a mapping or None.")
    if not isinstance(evidence_binding, ReconstructionAdvisorEvidenceBinding):
        raise TypeError("evidence_binding must be ReconstructionAdvisorEvidenceBinding.")

    workflow = _value(getattr(source_selection, "workflow", "none")) or "none"
    source_ready = bool(getattr(source_selection, "ready", False)) and workspace.source_ready
    item_count_raw = getattr(source_selection, "item_count", 0)
    item_count = int(item_count_raw or 0)
    if not source_ready:
        item_count = max(item_count, 0)
    elif item_count <= 0:
        # The frozen I6.3A request contract requires a positive count for a ready
        # source.  Existing supported offline selections already carry this fact;
        # fail closed if a future source adapter violates that invariant.
        source_ready = False
        item_count = 0

    source_matches_result = _result_source_matches(source_selection, offline_result)
    request = ReconstructionAdvisorRequest(
        reconstruction_mode="offline_reconstruction",
        source_workflow=workflow,
        source_ready=source_ready,
        source_item_count=item_count,
        source_fingerprint=_source_fingerprint(
            source_matches_result=source_matches_result,
            offline_result=offline_result,
        ),
        workspace_state=_workspace_state(workspace),
        authoritative_sfm_usable=_authoritative_sfm_usable(
            source_matches_result=source_matches_result,
            workspace=workspace,
            offline_result=offline_result,
        ),
        runtime_integrity_clean=(
            evidence_binding.runtime_integrity_clean and not workspace.identity_conflict
        ),
        evidence_context_resolution=evidence_binding.evidence_context_resolution,
        evidence_context_id=evidence_binding.evidence_context_id,
        available_provider_ids=evidence_binding.available_provider_ids,
        resources=evidence_binding.resources,
        provider_evidence=evidence_binding.provider_evidence,
    )
    decision = evaluate_reconstruction_advisor(request)
    return ReconstructionAdvisorLiveState(
        request=request,
        decision=decision,
        evidence_binding=evidence_binding,
        source_matches_admitted_result=source_matches_result,
    )


__all__ = [
    "I63C_SCHEMA_VERSION",
    "I63C_PHASE",
    "I63C_COMPOSITION_PROFILE",
    "ReconstructionAdvisorEvidenceBinding",
    "ReconstructionAdvisorLiveState",
    "build_reconstruction_advisor_live_state",
]
