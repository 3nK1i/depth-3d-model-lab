from __future__ import annotations

from dataclasses import replace

import pytest

from app.reconstruction_advisor_state import (
    ReconstructionAdvisorEvidenceBinding,
    ReconstructionAdvisorLiveState,
)
from app.reconstruction_provider_selection_state import (
    ManualProviderSelectionStatus,
    build_reconstruction_provider_selection_state,
)
from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
)
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorDisposition,
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorEvidenceSupport,
    AdvisorResourceObservation,
    ReconstructionAdvisorDecision,
    ReconstructionAdvisorRequest,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    ProductionProviderAvailability,
    ProductionProviderQualification,
    ProductionProviderSelectionMode,
)


def _resources() -> AdvisorResourceObservation:
    return AdvisorResourceObservation(
        gpu_name="RTX 2060",
        cuda_ready=True,
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=5 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )


def _evidence(
    provider_id: str,
    implementation: DenseProviderImplementation,
    family: DenseProviderFamily,
    *,
    outcome: AdvisorEvidenceOutcome = AdvisorEvidenceOutcome.PASS,
    context: str = "ctx-a",
    digest: str = "a" * 64,
) -> AdvisorEvidenceObservation:
    return AdvisorEvidenceObservation(
        provider_id=provider_id,
        provider_family=family,
        provider_implementation=implementation,
        outcome=outcome,
        evidence_context_id=context,
        matched_scenario_ids=("scenario",),
        reason_codes=("frozen_evidence",),
        limitations=(),
        bindings=(
            AdvisorEvidenceBinding(
                evidence_id=f"ev-{provider_id}",
                phase="I5",
                path=f"/evidence/{provider_id}.json",
                sha256=digest,
            ),
        ),
    )


def _state(
    *,
    context: str = "ctx-a",
    source_ready: bool = True,
    available: tuple[str, ...] = ("colmap-mvs-external",),
    evidence: tuple[AdvisorEvidenceObservation, ...] | None = None,
    recommend: str = "",
) -> ReconstructionAdvisorLiveState:
    observations = evidence or (
        _evidence(
            "colmap-mvs-external",
            DenseProviderImplementation.COLMAP_MVS,
            DenseProviderFamily.MVS,
        ),
    )
    resolution = (
        AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN
        if context
        else AdvisorContextResolution.UNRESOLVED
    )
    binding = ReconstructionAdvisorEvidenceBinding(
        evidence_context_resolution=resolution,
        evidence_context_id=context,
        available_provider_ids=available,
        resources=_resources(),
        provider_evidence=observations if context else (),
    )
    request = ReconstructionAdvisorRequest(
        reconstruction_mode="offline_reconstruction",
        source_workflow="image_sequence",
        source_ready=source_ready,
        source_item_count=4 if source_ready else 0,
        source_fingerprint="",
        workspace_state="ready" if source_ready else "idle",
        authoritative_sfm_usable=False,
        runtime_integrity_clean=True,
        evidence_context_resolution=resolution,
        evidence_context_id=context,
        available_provider_ids=available,
        resources=_resources(),
        provider_evidence=observations if context else (),
    )
    if recommend:
        observation = next(item for item in observations if item.provider_id == recommend)
        decision = ReconstructionAdvisorDecision(
            disposition=AdvisorDisposition.RECOMMEND,
            evidence_support=AdvisorEvidenceSupport.EXACT_QUALIFIED,
            reason_codes=("unique_exact_pass",),
            explanations=("exact evidence",),
            recommended_provider_id=recommend,
            recommended_family=observation.provider_family,
            recommended_implementation=observation.provider_implementation,
            matched_scenario_ids=observation.matched_scenario_ids,
            evidence_bindings=observation.bindings,
            considered_provider_ids=(recommend,),
        )
    else:
        decision = ReconstructionAdvisorDecision(
            disposition=AdvisorDisposition.ABSTAIN,
            evidence_support=AdvisorEvidenceSupport.INSUFFICIENT,
            reason_codes=("manual_test_abstention",),
            explanations=("no recommendation",),
        )
    return ReconstructionAdvisorLiveState(
        request=request,
        decision=decision,
        evidence_binding=binding,
        source_matches_admitted_result=False,
    )


def _provider(state, provider_id: str):
    return next(item for item in state.snapshot.providers if item.provider_id == provider_id)


def test_i6_4b_unresolved_state_is_fail_closed_and_auto_absent() -> None:
    state = build_reconstruction_provider_selection_state(advisor_state=_state(context="", available=()))
    assert state.status is ManualProviderSelectionStatus.NONE
    assert state.snapshot.selection_mode is ProductionProviderSelectionMode.NONE
    assert state.snapshot.selected_provider_id == ""
    assert all(item.manual_selection_allowed is False for item in state.snapshot.providers)
    assert "auto" not in {item.provider_id for item in state.snapshot.providers}
    assert state.snapshot.auto_gate.selectable is False


def test_i6_4b_explicit_available_pass_becomes_manually_selectable() -> None:
    state = build_reconstruction_provider_selection_state(advisor_state=_state())
    provider = _provider(state, "colmap-mvs-external")
    assert provider.availability is ProductionProviderAvailability.AVAILABLE
    assert provider.qualification is ProductionProviderQualification.QUALIFIED_FOR_CONTEXT
    assert provider.qualification_context_id == "ctx-a"
    assert provider.qualification_evidence_sha256 == ("a" * 64,)
    assert provider.manual_selection_allowed is True


def test_i6_4b_nonpass_evidence_is_preserved_as_not_qualified() -> None:
    evidence = (
        _evidence(
            "colmap-mvs-external",
            DenseProviderImplementation.COLMAP_MVS,
            DenseProviderFamily.MVS,
            outcome=AdvisorEvidenceOutcome.DEGRADED,
        ),
    )
    state = build_reconstruction_provider_selection_state(
        advisor_state=_state(evidence=evidence)
    )
    provider = _provider(state, "colmap-mvs-external")
    assert provider.qualification is ProductionProviderQualification.NOT_QUALIFIED_FOR_CONTEXT
    assert provider.manual_selection_allowed is False


def test_i6_4b_available_without_evidence_is_not_selectable() -> None:
    rolling = "rolling-depth-temporal"
    state = build_reconstruction_provider_selection_state(
        advisor_state=_state(available=(rolling,))
    )
    provider = _provider(state, rolling)
    assert provider.availability is ProductionProviderAvailability.AVAILABLE
    assert provider.qualification is ProductionProviderQualification.UNRESOLVED
    assert provider.manual_selection_allowed is False


def test_i6_4b_manual_selection_is_user_intent_only() -> None:
    state = build_reconstruction_provider_selection_state(
        advisor_state=_state(),
        selected_provider_id="colmap-mvs-external",
    )
    assert state.status is ManualProviderSelectionStatus.SELECTED
    assert state.snapshot.selection_mode is ProductionProviderSelectionMode.MANUAL
    assert state.selected_provider_id == "colmap-mvs-external"
    assert state.snapshot.provider_selection_effect is False
    assert state.snapshot.provider_activation_authorized is False
    assert state.snapshot.provider_execution_authorized is False
    assert state.snapshot.settings_mutation_authorized is False
    assert state.snapshot.start_action_authorized is False


def test_i6_4b_advisor_recommendation_never_becomes_selection() -> None:
    state = build_reconstruction_provider_selection_state(
        advisor_state=_state(recommend="colmap-mvs-external")
    )
    assert state.recommended_provider_id == "colmap-mvs-external"
    assert state.selected_provider_id == ""
    assert state.snapshot.selection_mode is ProductionProviderSelectionMode.NONE


def test_i6_4b_source_not_ready_clears_manual_intent() -> None:
    state = build_reconstruction_provider_selection_state(
        advisor_state=_state(source_ready=False),
        selected_provider_id="colmap-mvs-external",
    )
    assert state.status is ManualProviderSelectionStatus.CLEARED_INVALID
    assert "source_not_ready" in state.reason_codes
    assert state.selected_provider_id == ""


def test_i6_4b_unresolved_context_clears_manual_intent() -> None:
    state = build_reconstruction_provider_selection_state(
        advisor_state=_state(context="", available=()),
        selected_provider_id="colmap-mvs-external",
    )
    assert state.status is ManualProviderSelectionStatus.CLEARED_INVALID
    assert "selection_context_unresolved" in state.reason_codes
    assert state.selected_provider_id == ""


def test_i6_4b_unavailable_provider_clears_manual_intent() -> None:
    state = build_reconstruction_provider_selection_state(
        advisor_state=_state(available=()),
        selected_provider_id="colmap-mvs-external",
    )
    assert state.status is ManualProviderSelectionStatus.CLEARED_INVALID
    assert "selected_provider_not_available" in state.reason_codes


def test_i6_4b_unknown_provider_clears_manual_intent() -> None:
    state = build_reconstruction_provider_selection_state(
        advisor_state=_state(),
        selected_provider_id="unknown-provider",
    )
    assert state.status is ManualProviderSelectionStatus.CLEARED_INVALID
    assert "selected_provider_not_in_catalog" in state.reason_codes


def test_i6_4b_auto_cannot_be_used_as_manual_provider_id() -> None:
    with pytest.raises(ValueError, match="Auto is not a concrete"):
        build_reconstruction_provider_selection_state(
            advisor_state=_state(),
            selected_provider_id="auto",
        )


def test_i6_4b_cleared_reason_can_be_preserved_after_intent_is_removed() -> None:
    state = build_reconstruction_provider_selection_state(
        advisor_state=_state(),
        previous_status=ManualProviderSelectionStatus.CLEARED_INVALID,
        previous_reason_codes=("source_selection_changed",),
    )
    assert state.status is ManualProviderSelectionStatus.CLEARED_INVALID
    assert state.reason_codes == ("source_selection_changed",)


def test_i6_4b_state_is_immutable() -> None:
    state = build_reconstruction_provider_selection_state(advisor_state=_state())
    with pytest.raises(Exception):
        state.source_ready = False  # type: ignore[misc]
