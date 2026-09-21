"""I6.4B passive manual provider-selection state composition.

This module composes the frozen I6.4A provider availability/qualification
contract with the already-qualified I6.3 live Advisor state.  It represents
*user intent only*.  It never activates or executes a dense provider, mutates
provider settings, starts reconstruction, enables production Auto, creates a
provider rank/score, or changes PyCOLMAP camera/global-frame authority.

The selectable catalog is derived only from explicit I6.3 evidence inputs:
concrete-provider availability and exact-context evidence.  No runtime provider
probe and no live evidence-context resolver is introduced here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from app.reconstruction_advisor_state import ReconstructionAdvisorLiveState
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorDisposition,
    AdvisorEvidenceOutcome,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    I64A_CAMERA_AUTHORITY_OWNER,
    I64A_PRODUCTION_AUTO_POLICY_ID,
    ProductionProviderAvailability,
    ProductionProviderCatalogEntry,
    ProductionProviderQualification,
    ProductionProviderSelectionMode,
    ProductionProviderSelectionSnapshot,
    canonical_production_provider_catalog,
)

I64B_SCHEMA_VERSION = 1
I64B_PHASE = "I6.4B"
I64B_COMPOSITION_PROFILE = "manual_provider_selection_intent_only"


class ManualProviderSelectionStatus(str, Enum):
    """Lifecycle of explicit user provider intent; never execution state."""

    NONE = "none"
    SELECTED = "selected"
    CLEARED_INVALID = "cleared_invalid"


def _unique_text(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    result = tuple(str(value or "").strip() for value in values)
    if any(not value for value in result):
        raise ValueError("reason codes must not contain empty values.")
    if len(set(result)) != len(result):
        raise ValueError("reason codes must be unique.")
    return result


def _ordered_unique_sha(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        value = str(raw or "").strip().lower()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _catalog_from_advisor_state(
    advisor_state: ReconstructionAdvisorLiveState,
) -> tuple[ProductionProviderCatalogEntry, ...]:
    """Compose I6.4A provider facts from explicit I6.3 evidence only."""

    if not isinstance(advisor_state, ReconstructionAdvisorLiveState):
        raise TypeError("advisor_state must be ReconstructionAdvisorLiveState.")

    request = advisor_state.request
    binding = advisor_state.evidence_binding
    available = set(binding.available_provider_ids)
    resolved_context = (
        request.evidence_context_resolution is AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN
        and bool(request.evidence_context_id)
    )
    evidence_by_provider = {
        item.provider_id: item
        for item in binding.provider_evidence
        if resolved_context and item.evidence_context_id == request.evidence_context_id
    }

    entries: list[ProductionProviderCatalogEntry] = []
    for base in canonical_production_provider_catalog():
        availability = (
            ProductionProviderAvailability.AVAILABLE
            if base.provider_id in available
            else ProductionProviderAvailability.UNKNOWN
        )
        qualification = ProductionProviderQualification.UNRESOLVED
        qualification_context_id = ""
        evidence_sha256: tuple[str, ...] = ()
        observation = evidence_by_provider.get(base.provider_id)
        if observation is not None:
            qualification = (
                ProductionProviderQualification.QUALIFIED_FOR_CONTEXT
                if observation.outcome is AdvisorEvidenceOutcome.PASS
                else ProductionProviderQualification.NOT_QUALIFIED_FOR_CONTEXT
            )
            qualification_context_id = request.evidence_context_id
            evidence_sha256 = _ordered_unique_sha(
                [item.sha256 for item in observation.bindings]
            )
        manual_allowed = (
            bool(request.source_ready)
            and availability is ProductionProviderAvailability.AVAILABLE
            and qualification is ProductionProviderQualification.QUALIFIED_FOR_CONTEXT
        )
        entries.append(
            replace(
                base,
                availability=availability,
                availability_reason_codes=(),
                qualification=qualification,
                qualification_context_id=qualification_context_id,
                qualification_evidence_sha256=evidence_sha256,
                manual_selection_allowed=manual_allowed,
            )
        )
    return tuple(entries)


def _selection_blockers(
    *,
    advisor_state: ReconstructionAdvisorLiveState,
    provider_id: str,
    provider: ProductionProviderCatalogEntry | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not advisor_state.request.source_ready:
        reasons.append("source_not_ready")
    if advisor_state.request.evidence_context_resolution is not AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN:
        reasons.append("selection_context_unresolved")
    if not advisor_state.request.evidence_context_id:
        reasons.append("selection_context_missing")
    if provider is None:
        reasons.append("selected_provider_not_in_catalog")
    else:
        if provider.availability is not ProductionProviderAvailability.AVAILABLE:
            reasons.append("selected_provider_not_available")
        if provider.qualification is ProductionProviderQualification.UNRESOLVED:
            reasons.append("selected_provider_qualification_unresolved")
        elif provider.qualification is ProductionProviderQualification.NOT_QUALIFIED_FOR_CONTEXT:
            reasons.append("selected_provider_not_qualified_for_context")
        if (
            provider.qualification_context_id
            and advisor_state.request.evidence_context_id
            and provider.qualification_context_id != advisor_state.request.evidence_context_id
        ):
            reasons.append("selected_provider_context_mismatch")
        if not provider.manual_selection_allowed:
            reasons.append("manual_selection_not_allowed")
    return _unique_text(reasons)


@dataclass(frozen=True, slots=True)
class ReconstructionProviderSelectionState:
    """Controller-facing manual selection intent with zero production effect."""

    snapshot: ProductionProviderSelectionSnapshot
    status: ManualProviderSelectionStatus = ManualProviderSelectionStatus.NONE
    reason_codes: tuple[str, ...] = ()
    evidence_context_id: str = ""
    source_ready: bool = False
    schema_version: int = I64B_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, ProductionProviderSelectionSnapshot):
            raise TypeError("snapshot must be ProductionProviderSelectionSnapshot.")
        if not isinstance(self.status, ManualProviderSelectionStatus):
            raise TypeError("status must be ManualProviderSelectionStatus.")
        reasons = _unique_text(self.reason_codes)
        if self.status is ManualProviderSelectionStatus.SELECTED:
            if self.snapshot.selection_mode is not ProductionProviderSelectionMode.MANUAL:
                raise ValueError("SELECTED state requires manual selection mode.")
            if not self.snapshot.selected_provider_id:
                raise ValueError("SELECTED state requires selected_provider_id.")
            if reasons:
                raise ValueError("SELECTED state cannot carry blocker reason codes.")
        elif self.snapshot.selection_mode is not ProductionProviderSelectionMode.NONE:
            raise ValueError("Non-selected state must use selection_mode=none.")
        if self.status is ManualProviderSelectionStatus.CLEARED_INVALID and not reasons:
            raise ValueError("CLEARED_INVALID state requires reason codes.")
        context_id = str(self.evidence_context_id or "").strip()
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "evidence_context_id", context_id)
        object.__setattr__(self, "source_ready", bool(self.source_ready))
        if int(self.schema_version) != I64B_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.4B provider-selection-state schema version.")
        object.__setattr__(self, "schema_version", I64B_SCHEMA_VERSION)

    @property
    def selected_provider_id(self) -> str:
        return self.snapshot.selected_provider_id

    @property
    def recommended_provider_id(self) -> str:
        return self.snapshot.recommended_provider_id

    @property
    def manual_selection_active(self) -> bool:
        return self.status is ManualProviderSelectionStatus.SELECTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": I64B_PHASE,
            "composition_profile": I64B_COMPOSITION_PROFILE,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "evidence_context_id": self.evidence_context_id,
            "source_ready": self.source_ready,
            "selection": self.snapshot.as_dict(),
            "effects": {
                "provider_selection_intent_only": True,
                "provider_activation": False,
                "provider_execution": False,
                "settings_mutation": False,
                "start_action": False,
                "production_auto": False,
            },
        }


def build_reconstruction_provider_selection_state(
    *,
    advisor_state: ReconstructionAdvisorLiveState,
    selected_provider_id: str = "",
    previous_status: ManualProviderSelectionStatus = ManualProviderSelectionStatus.NONE,
    previous_reason_codes: tuple[str, ...] = (),
) -> ReconstructionProviderSelectionState:
    """Compose one passive selection state and fail closed on stale intent."""

    if not isinstance(advisor_state, ReconstructionAdvisorLiveState):
        raise TypeError("advisor_state must be ReconstructionAdvisorLiveState.")
    if not isinstance(previous_status, ManualProviderSelectionStatus):
        raise TypeError("previous_status must be ManualProviderSelectionStatus.")
    previous_reasons = _unique_text(previous_reason_codes)
    requested = str(selected_provider_id or "").strip()
    if requested == I64A_PRODUCTION_AUTO_POLICY_ID:
        raise ValueError("Auto is not a concrete manual provider selection.")

    providers = _catalog_from_advisor_state(advisor_state)
    by_id = {item.provider_id: item for item in providers}

    recommended = ""
    decision = advisor_state.decision
    if decision.disposition is AdvisorDisposition.RECOMMEND:
        candidate = by_id.get(decision.recommended_provider_id)
        if candidate is not None and candidate.manual_selection_allowed:
            recommended = candidate.provider_id

    status = ManualProviderSelectionStatus.NONE
    reasons: tuple[str, ...] = ()
    selection_mode = ProductionProviderSelectionMode.NONE
    selected = ""
    if requested:
        blockers = _selection_blockers(
            advisor_state=advisor_state,
            provider_id=requested,
            provider=by_id.get(requested),
        )
        if blockers:
            status = ManualProviderSelectionStatus.CLEARED_INVALID
            reasons = blockers
        else:
            status = ManualProviderSelectionStatus.SELECTED
            selection_mode = ProductionProviderSelectionMode.MANUAL
            selected = requested
    elif previous_status is ManualProviderSelectionStatus.CLEARED_INVALID:
        status = ManualProviderSelectionStatus.CLEARED_INVALID
        reasons = previous_reasons or ("manual_selection_cleared",)

    snapshot = ProductionProviderSelectionSnapshot(
        providers=providers,
        recommended_provider_id=recommended,
        selection_mode=selection_mode,
        selected_provider_id=selected,
    )
    return ReconstructionProviderSelectionState(
        snapshot=snapshot,
        status=status,
        reason_codes=reasons,
        evidence_context_id=advisor_state.request.evidence_context_id,
        source_ready=advisor_state.request.source_ready,
    )


__all__ = [
    "I64B_SCHEMA_VERSION",
    "I64B_PHASE",
    "I64B_COMPOSITION_PROFILE",
    "ManualProviderSelectionStatus",
    "ReconstructionProviderSelectionState",
    "build_reconstruction_provider_selection_state",
]
