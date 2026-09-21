"""I6.3D professional presentation adapter for Reconstruction Advisor state.

This module converts the immutable, passive I6.3C live Advisor state into a
compact product-facing presentation.  It is deliberately presentation-only:
no provider is selected or activated, no setting is changed, no runtime is
started, and no production-Auto authority is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.reconstruction_advisor_state import ReconstructionAdvisorLiveState
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorDisposition,
    AdvisorEvidenceSupport,
)

I63D_SCHEMA_VERSION = 1
I63D_PHASE = "I6.3D"
I63D_BOUNDARY_TEXT = "Advisory only • no settings changed"

_IMPLEMENTATION_LABELS = {
    DenseProviderImplementation.COLMAP_MVS: "COLMAP MVS",
    DenseProviderImplementation.DA3: "Depth Anything 3",
    DenseProviderImplementation.ROLLING_DEPTH: "Rolling Depth",
    DenseProviderImplementation.VIDEO_DEPTH_ANYTHING: "Video Depth Anything",
    DenseProviderImplementation.METRIC3D_V2: "Metric3D v2",
    DenseProviderImplementation.UNIDEPTH_V2: "UniDepth V2",
    DenseProviderImplementation.MOGE_2: "MoGe-2",
    DenseProviderImplementation.DEPTH_PRO: "Depth Pro",
    DenseProviderImplementation.DEPTH_ANYTHING_V2: "Depth Anything V2",
    DenseProviderImplementation.MARIGOLD: "Marigold",
}

_SUPPORT_LABELS = {
    AdvisorEvidenceSupport.EXACT_QUALIFIED: "Exact qualified evidence",
    AdvisorEvidenceSupport.BOUNDED: "Bounded evidence",
    AdvisorEvidenceSupport.INSUFFICIENT: "Insufficient evidence",
    AdvisorEvidenceSupport.CONFLICTING: "Conflicting evidence",
}


@dataclass(frozen=True, slots=True)
class ReconstructionAdvisorPresentation:
    """Compact immutable UI projection for one Advisor live-state snapshot."""

    status_text: str
    status_kind: str
    guidance_text: str
    guidance_tooltip: str
    evidence_text: str
    evidence_tooltip: str
    rationale_text: str
    rationale_tooltip: str
    boundary_text: str = I63D_BOUNDARY_TEXT
    boundary_tooltip: str = (
        "The Reconstruction Advisor is read-only guidance. It does not select or activate "
        "a provider, change settings, press Start, execute reconstruction, or authorize "
        "production Auto. PyCOLMAP remains the canonical camera/global-frame authority."
    )
    schema_version: int = I63D_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "status_text",
            "status_kind",
            "guidance_text",
            "guidance_tooltip",
            "evidence_text",
            "evidence_tooltip",
            "rationale_text",
            "rationale_tooltip",
            "boundary_text",
            "boundary_tooltip",
        ):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty.")
            object.__setattr__(self, field_name, value)
        if self.status_kind not in {"neutral", "ready", "warning"}:
            raise ValueError("Advisor presentation status_kind must be neutral, ready, or warning.")
        if int(self.schema_version) != I63D_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.3D presentation schema version.")
        object.__setattr__(self, "schema_version", I63D_SCHEMA_VERSION)



def _provider_label(state: ReconstructionAdvisorLiveState) -> str:
    implementation = state.decision.recommended_implementation
    if implementation is None:
        return "No recommendation"
    return _IMPLEMENTATION_LABELS.get(
        implementation,
        implementation.value.replace("_", " ").title(),
    )


def _guidance_for_abstention(reason_codes: tuple[str, ...]) -> tuple[str, str]:
    reasons = set(reason_codes)
    if "source_not_ready" in reasons:
        return "Waiting for source", "neutral"
    if "authoritative_sfm_not_usable" in reasons:
        return "Waiting for qualified sparse SfM", "neutral"
    if "evidence_context_unresolved" in reasons:
        return "Evidence context unresolved", "neutral"
    if "no_concrete_provider_available" in reasons:
        return "No concrete provider available", "warning"
    if "unique_exact_pass_provider_unavailable" in reasons:
        return "Supported provider unavailable", "warning"
    if "no_exact_context_evidence" in reasons:
        return "No exact evidence match", "warning"
    if "runtime_integrity_not_clean" in reasons:
        return "Guidance withheld", "warning"
    return "Advisor abstained", "warning"


def _evidence_tooltip(state: ReconstructionAdvisorLiveState) -> str:
    decision = state.decision
    request = state.request
    lines = [
        f"Support: {_SUPPORT_LABELS[decision.evidence_support]}.",
        (
            f"Evidence context: {request.evidence_context_id}."
            if request.evidence_context_id
            else "Evidence context: unresolved."
        ),
        f"Matched scenarios: {len(decision.matched_scenario_ids)}.",
        f"Evidence artifacts: {len(decision.evidence_bindings)}.",
    ]
    if decision.considered_provider_ids:
        lines.append("Considered providers: " + ", ".join(decision.considered_provider_ids) + ".")
    if decision.excluded_providers:
        lines.append(
            "Excluded providers: "
            + ", ".join(item.provider_id for item in decision.excluded_providers)
            + "."
        )
    lines.append("No global ranking or scalar provider score is created.")
    return "\n".join(lines)


def _rationale_tooltip(state: ReconstructionAdvisorLiveState) -> str:
    decision = state.decision
    lines = list(decision.explanations)
    if decision.limitations:
        lines.append("Limitations: " + "; ".join(decision.limitations) + ".")
    if decision.reason_codes:
        lines.append("Reason codes: " + ", ".join(decision.reason_codes) + ".")
    return "\n".join(lines)


def present_reconstruction_advisor(
    state: ReconstructionAdvisorLiveState,
) -> ReconstructionAdvisorPresentation:
    """Project one passive Advisor live state into professional UI text."""

    if not isinstance(state, ReconstructionAdvisorLiveState):
        raise TypeError("state must be ReconstructionAdvisorLiveState.")

    decision = state.decision
    support_text = _SUPPORT_LABELS[decision.evidence_support]
    rationale = decision.explanations[0]

    if decision.disposition is AdvisorDisposition.RECOMMEND:
        provider = _provider_label(state)
        return ReconstructionAdvisorPresentation(
            status_text="Recommendation available",
            status_kind="ready",
            guidance_text=provider,
            guidance_tooltip=(
                f"Advisor recommends {provider} ({decision.recommended_provider_id}) only for "
                "the exact frozen evidence context represented by this decision. The provider "
                "is not selected or activated automatically."
            ),
            evidence_text=support_text,
            evidence_tooltip=_evidence_tooltip(state),
            rationale_text=rationale,
            rationale_tooltip=_rationale_tooltip(state),
        )

    guidance, kind = _guidance_for_abstention(decision.reason_codes)
    return ReconstructionAdvisorPresentation(
        status_text="No recommendation",
        status_kind=kind,
        guidance_text=guidance,
        guidance_tooltip=(
            "The Advisor intentionally abstained. This is a valid product state, not a "
            "reconstruction runtime failure; no fallback provider is selected automatically."
        ),
        evidence_text=support_text,
        evidence_tooltip=_evidence_tooltip(state),
        rationale_text=rationale,
        rationale_tooltip=_rationale_tooltip(state),
    )


__all__ = [
    "I63D_SCHEMA_VERSION",
    "I63D_PHASE",
    "I63D_BOUNDARY_TEXT",
    "ReconstructionAdvisorPresentation",
    "present_reconstruction_advisor",
]
