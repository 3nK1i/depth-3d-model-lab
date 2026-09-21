from __future__ import annotations

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


def _live_state(*, available=("colmap-mvs-external",)) -> ReconstructionAdvisorLiveState:
    binding_item = AdvisorEvidenceBinding(
        evidence_id="stress-evidence",
        phase="I5.14B",
        path="/evidence/stress.json",
        sha256="b" * 64,
    )
    observation = AdvisorEvidenceObservation(
        provider_id="colmap-mvs-external",
        provider_family=DenseProviderFamily.MVS,
        provider_implementation=DenseProviderImplementation.COLMAP_MVS,
        outcome=AdvisorEvidenceOutcome.PASS,
        evidence_context_id="ctx-stress",
        matched_scenario_ids=("stress",),
        reason_codes=("pass",),
        limitations=(),
        bindings=(binding_item,),
    )
    resources = AdvisorResourceObservation(
        gpu_name="RTX",
        cuda_ready=True,
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=5 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )
    binding = ReconstructionAdvisorEvidenceBinding(
        evidence_context_resolution=AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
        evidence_context_id="ctx-stress",
        available_provider_ids=available,
        resources=resources,
        provider_evidence=(observation,),
    )
    request = ReconstructionAdvisorRequest(
        reconstruction_mode="offline_reconstruction",
        source_workflow="image_sequence",
        source_ready=True,
        source_item_count=100,
        source_fingerprint="",
        workspace_state="ready",
        authoritative_sfm_usable=False,
        runtime_integrity_clean=True,
        evidence_context_resolution=AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
        evidence_context_id="ctx-stress",
        available_provider_ids=available,
        resources=resources,
        provider_evidence=(observation,),
    )
    decision = ReconstructionAdvisorDecision(
        disposition=AdvisorDisposition.ABSTAIN,
        evidence_support=AdvisorEvidenceSupport.INSUFFICIENT,
        reason_codes=("stress_abstain",),
        explanations=("stress",),
    )
    return ReconstructionAdvisorLiveState(
        request=request,
        decision=decision,
        evidence_binding=binding,
        source_matches_admitted_result=False,
    )


def test_i6_4b_repeated_manual_state_composition_is_deterministic() -> None:
    live = _live_state()
    first = build_reconstruction_provider_selection_state(
        advisor_state=live,
        selected_provider_id="colmap-mvs-external",
    )
    for _ in range(250):
        assert build_reconstruction_provider_selection_state(
            advisor_state=live,
            selected_provider_id="colmap-mvs-external",
        ) == first


def test_i6_4b_provider_order_is_canonical_not_input_availability_order() -> None:
    forward = build_reconstruction_provider_selection_state(
        advisor_state=_live_state(available=("colmap-mvs-external", "rolling-depth-temporal"))
    )
    reverse = build_reconstruction_provider_selection_state(
        advisor_state=_live_state(available=("rolling-depth-temporal", "colmap-mvs-external"))
    )
    assert tuple(item.provider_id for item in forward.snapshot.providers) == tuple(
        item.provider_id for item in reverse.snapshot.providers
    )


def test_i6_4b_repeated_unavailable_transition_never_resurrects_manual_intent() -> None:
    selected = build_reconstruction_provider_selection_state(
        advisor_state=_live_state(),
        selected_provider_id="colmap-mvs-external",
    )
    assert selected.status is ManualProviderSelectionStatus.SELECTED
    cleared = build_reconstruction_provider_selection_state(
        advisor_state=_live_state(available=()),
        selected_provider_id=selected.selected_provider_id,
    )
    assert cleared.status is ManualProviderSelectionStatus.CLEARED_INVALID
    for _ in range(100):
        cleared = build_reconstruction_provider_selection_state(
            advisor_state=_live_state(available=()),
            previous_status=cleared.status,
            previous_reason_codes=cleared.reason_codes,
        )
        assert cleared.status is ManualProviderSelectionStatus.CLEARED_INVALID
        assert cleared.selected_provider_id == ""
