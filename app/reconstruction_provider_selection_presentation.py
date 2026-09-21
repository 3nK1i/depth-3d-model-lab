"""I6.4D presentation-only provider/manual-Auto UI model.

This module converts the already-qualified I6.4A-C contracts into compact UI
text and option metadata.  It does not probe runtimes, select/activate/execute a
provider, mutate settings, start reconstruction, evaluate the I5 Auto policy, or
change PyCOLMAP authority.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.reconstruction_provider_selection_state import (
    ManualProviderSelectionStatus,
    ReconstructionProviderSelectionState,
)
from core.reconstruction.reconstruction_production_auto_gate import (
    ProductionAutoQualificationBoundary,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    ProductionProviderAvailability,
    ProductionProviderCatalogEntry,
    ProductionProviderQualification,
)

I64D_PHASE = "I6.4D"
I64D_PRESENTATION_PROFILE = "professional_provider_auto_ui"


_REASON_TEXT = {
    "source_selection_changed": "The source changed; previous provider intent was cleared.",
    "source_not_ready": "Choose a ready offline source before selecting a provider.",
    "selection_context_unresolved": "Exact evidence context is unresolved.",
    "selection_context_missing": "No exact evidence context is bound.",
    "selected_provider_not_in_catalog": "The previous provider is not in the canonical catalog.",
    "selected_provider_not_available": "The previous provider is not explicitly available.",
    "selected_provider_qualification_unresolved": "Provider qualification is unresolved for this context.",
    "selected_provider_not_qualified_for_context": "Provider is not qualified for this exact context.",
    "selected_provider_context_mismatch": "Provider evidence belongs to a different context.",
    "manual_selection_not_allowed": "Manual selection is not allowed by the current evidence boundary.",
    "manual_selection_cleared": "Manual provider intent was cleared.",
}


@dataclass(frozen=True, slots=True)
class ProviderSelectionOptionPresentation:
    provider_id: str
    label: str
    selectable: bool
    recommended: bool
    selected: bool
    status_text: str
    tooltip: str


@dataclass(frozen=True, slots=True)
class ReconstructionProviderAutoPresentation:
    status_text: str
    status_kind: str
    selection_text: str
    selection_tooltip: str
    recommendation_text: str
    recommendation_tooltip: str
    eligibility_text: str
    eligibility_tooltip: str
    options: tuple[ProviderSelectionOptionPresentation, ...]
    combo_enabled: bool
    clear_enabled: bool
    auto_text: str
    auto_status_kind: str
    auto_tooltip: str
    boundary_text: str
    boundary_tooltip: str


def _entry_status(entry: ProductionProviderCatalogEntry) -> str:
    if entry.manual_selection_allowed:
        return "Selectable"
    if entry.availability is ProductionProviderAvailability.UNKNOWN:
        return "Availability unresolved"
    if entry.availability is ProductionProviderAvailability.UNAVAILABLE:
        return "Unavailable"
    if entry.qualification is ProductionProviderQualification.UNRESOLVED:
        return "Qualification unresolved"
    if entry.qualification is ProductionProviderQualification.NOT_QUALIFIED_FOR_CONTEXT:
        return "Not qualified for context"
    return "Not selectable"


def _entry_tooltip(entry: ProductionProviderCatalogEntry) -> str:
    availability = {
        ProductionProviderAvailability.UNKNOWN: "runtime availability unresolved",
        ProductionProviderAvailability.AVAILABLE: "runtime available",
        ProductionProviderAvailability.UNAVAILABLE: "runtime unavailable",
    }[entry.availability]
    qualification = {
        ProductionProviderQualification.UNRESOLVED: "qualification unresolved",
        ProductionProviderQualification.QUALIFIED_FOR_CONTEXT: "qualified for this exact evidence context",
        ProductionProviderQualification.NOT_QUALIFIED_FOR_CONTEXT: "not qualified for this exact evidence context",
    }[entry.qualification]
    evidence = (
        f" • evidence: {len(entry.qualification_evidence_sha256)} SHA-256 binding(s)"
        if entry.qualification_evidence_sha256
        else ""
    )
    reasons = (
        " • availability reasons: " + ", ".join(entry.availability_reason_codes)
        if entry.availability_reason_codes
        else ""
    )
    return (
        f"{entry.label} • {availability} • {qualification}{evidence}{reasons}. "
        "Selection records user intent only; it does not activate or execute the provider."
    )


def present_reconstruction_provider_auto(
    state: ReconstructionProviderSelectionState,
    auto_gate: ProductionAutoQualificationBoundary,
) -> ReconstructionProviderAutoPresentation:
    """Build compact professional UI text from I6.4A-C state only."""

    if not isinstance(state, ReconstructionProviderSelectionState):
        raise TypeError("state must be ReconstructionProviderSelectionState.")
    if not isinstance(auto_gate, ProductionAutoQualificationBoundary):
        raise TypeError("auto_gate must be ProductionAutoQualificationBoundary.")

    snapshot = state.snapshot
    by_id = {entry.provider_id: entry for entry in snapshot.providers}
    selected = by_id.get(snapshot.selected_provider_id)
    recommended = by_id.get(snapshot.recommended_provider_id)

    options = tuple(
        ProviderSelectionOptionPresentation(
            provider_id=entry.provider_id,
            label=(
                f"{entry.label} — Recommended"
                if entry.provider_id == snapshot.recommended_provider_id
                else entry.label
            ),
            selectable=entry.manual_selection_allowed,
            recommended=entry.provider_id == snapshot.recommended_provider_id,
            selected=entry.provider_id == snapshot.selected_provider_id,
            status_text=_entry_status(entry),
            tooltip=_entry_tooltip(entry),
        )
        for entry in snapshot.providers
    )
    selectable_count = sum(option.selectable for option in options)

    if state.status is ManualProviderSelectionStatus.SELECTED and selected is not None:
        status_text = "Manual selection recorded"
        status_kind = "ready"
        selection_text = selected.label
        selection_tooltip = (
            f"Manual intent: {selected.label}. This does not activate or execute the provider."
        )
    elif state.status is ManualProviderSelectionStatus.CLEARED_INVALID:
        status_text = "Selection cleared"
        status_kind = "warning"
        selection_text = "No manual provider"
        details = tuple(_REASON_TEXT.get(code, code.replace("_", " ")) for code in state.reason_codes)
        selection_tooltip = " ".join(details) or "Previous provider intent became invalid and was cleared."
    elif not state.source_ready:
        status_text = "Waiting for source"
        status_kind = "neutral"
        selection_text = "No manual provider"
        selection_tooltip = "Choose a ready offline source before recording provider intent."
    elif selectable_count:
        status_text = "Ready for manual choice"
        status_kind = "ready"
        selection_text = "No manual provider"
        selection_tooltip = "Choose one explicitly available provider qualified for the exact evidence context."
    else:
        status_text = "No provider selectable"
        status_kind = "warning"
        selection_text = "No manual provider"
        selection_tooltip = (
            "No concrete provider currently satisfies both explicit runtime availability and exact-context qualification."
        )

    recommendation_text = recommended.label if recommended is not None else "No recommendation"
    recommendation_tooltip = (
        f"Advisor recommends {recommended.label}; recommendation is not selection."
        if recommended is not None
        else "The Advisor has not produced one exact-context recommendation."
    )

    eligibility_text = f"{selectable_count} / {len(options)} selectable"
    eligibility_lines = [
        f"{option.label}: {option.status_text}" for option in options
    ]
    eligibility_tooltip = "\n".join(eligibility_lines)

    auto_text = "Not qualified"
    auto_status_kind = "warning"
    auto_tooltip = (
        "Offline/shadow Auto is qualified, but production Auto remains deferred, "
        "unauthorized and non-selectable. "
        f"Reason: {auto_gate.decision_reason}"
    )

    return ReconstructionProviderAutoPresentation(
        status_text=status_text,
        status_kind=status_kind,
        selection_text=selection_text,
        selection_tooltip=selection_tooltip,
        recommendation_text=recommendation_text,
        recommendation_tooltip=recommendation_tooltip,
        eligibility_text=eligibility_text,
        eligibility_tooltip=eligibility_tooltip,
        options=options,
        combo_enabled=bool(selectable_count),
        clear_enabled=state.manual_selection_active,
        auto_text=auto_text,
        auto_status_kind=auto_status_kind,
        auto_tooltip=auto_tooltip,
        boundary_text="Intent only • no provider activated or started",
        boundary_tooltip=(
            "Manual selection records explicit user intent only. Auto remains unavailable. "
            "No provider activation, execution, settings mutation or Start action is authorized by I6.4D."
        ),
    )


__all__ = [
    "I64D_PHASE",
    "I64D_PRESENTATION_PROFILE",
    "ProviderSelectionOptionPresentation",
    "ReconstructionProviderAutoPresentation",
    "present_reconstruction_provider_auto",
]
