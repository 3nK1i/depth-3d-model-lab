from __future__ import annotations

from dataclasses import replace

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
    ReconstructionAdvisorRequest,
)
from core.reconstruction.reconstruction_advisor_engine import (
    I63B_ENGINE_PROFILE,
    evaluate_reconstruction_advisor,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _binding(tag: str, sha: str) -> AdvisorEvidenceBinding:
    return AdvisorEvidenceBinding(
        evidence_id=tag,
        phase="I5.14B",
        path=f"/frozen/{tag}.json",
        sha256=sha,
    )


def _resources(*, gpu: str = "NVIDIA GeForce RTX 2060", free_vram: int = 5 * 1024**3) -> AdvisorResourceObservation:
    return AdvisorResourceObservation(
        gpu_name=gpu,
        cuda_ready=True,
        total_vram_bytes=32 * 1024**3 if "5090" in gpu else 6 * 1024**3,
        free_vram_bytes=free_vram,
        available_ram_bytes=12 * 1024**3,
    )


def _obs(
    provider_id: str,
    implementation: DenseProviderImplementation,
    family: DenseProviderFamily,
    outcome: AdvisorEvidenceOutcome,
    *,
    context: str = "ctx-clean",
    scenario: str = "scenario-clean",
    sha: str = SHA_A,
) -> AdvisorEvidenceObservation:
    return AdvisorEvidenceObservation(
        provider_id=provider_id,
        provider_family=family,
        provider_implementation=implementation,
        outcome=outcome,
        evidence_context_id=context,
        matched_scenario_ids=(scenario,),
        reason_codes=(f"{outcome.value}_frozen_evidence",),
        limitations=("bounded_to_exact_frozen_context",),
        bindings=(_binding(f"{provider_id}-{context}", sha),),
    )


def _request(
    *,
    evidence: tuple[AdvisorEvidenceObservation, ...],
    available: tuple[str, ...],
    resolution: AdvisorContextResolution = AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
    context: str = "ctx-clean",
    source_ready: bool = True,
    sfm_usable: bool = True,
    runtime_clean: bool = True,
    resources: AdvisorResourceObservation | None = None,
) -> ReconstructionAdvisorRequest:
    return ReconstructionAdvisorRequest(
        reconstruction_mode="offline_reconstruction",
        source_workflow="image_sequence",
        source_ready=source_ready,
        source_item_count=128 if source_ready else 0,
        source_fingerprint="source-fingerprint" if source_ready else "",
        workspace_state="partial_result" if sfm_usable else "ready",
        authoritative_sfm_usable=sfm_usable,
        runtime_integrity_clean=runtime_clean,
        evidence_context_resolution=resolution,
        evidence_context_id=context if resolution is AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN else "",
        available_provider_ids=available,
        resources=resources or _resources(),
        provider_evidence=evidence,
    )


def _mvs_pass(*, context: str = "ctx-clean") -> AdvisorEvidenceObservation:
    return _obs(
        "colmap-mvs-external",
        DenseProviderImplementation.COLMAP_MVS,
        DenseProviderFamily.MVS,
        AdvisorEvidenceOutcome.PASS,
        context=context,
        sha=SHA_A,
    )


def _rolling(outcome: AdvisorEvidenceOutcome = AdvisorEvidenceOutcome.PASS, *, context: str = "ctx-clean") -> AdvisorEvidenceObservation:
    return _obs(
        "rolling-depth-temporal",
        DenseProviderImplementation.ROLLING_DEPTH,
        DenseProviderFamily.TEMPORAL_DEPTH,
        outcome,
        context=context,
        scenario="scenario-temporal",
        sha=SHA_B,
    )


def _vda(outcome: AdvisorEvidenceOutcome = AdvisorEvidenceOutcome.PASS, *, context: str = "ctx-clean") -> AdvisorEvidenceObservation:
    return _obs(
        "video-depth-anything-temporal",
        DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
        DenseProviderFamily.TEMPORAL_DEPTH,
        outcome,
        context=context,
        scenario="scenario-temporal",
        sha=SHA_C,
    )


def test_i6_3b_unique_exact_pass_is_recommended_advisory_only() -> None:
    request = _request(
        evidence=(_mvs_pass(),),
        available=("colmap-mvs-external",),
    )

    decision = evaluate_reconstruction_advisor(request)

    assert decision.disposition is AdvisorDisposition.RECOMMEND
    assert decision.evidence_support is AdvisorEvidenceSupport.EXACT_QUALIFIED
    assert decision.recommended_provider_id == "colmap-mvs-external"
    assert decision.recommended_family is DenseProviderFamily.MVS
    assert decision.recommended_implementation is DenseProviderImplementation.COLMAP_MVS
    assert decision.reason_codes == (
        "advisor_exact_unique_evidence_recommendation",
        "advisory_only_no_production_effect",
    )
    assert decision.production_effect is False
    assert decision.provider_selection_effect is False
    assert decision.provider_execution is False
    assert decision.settings_mutation is False
    assert decision.start_action is False
    assert decision.production_auto_qualified is False
    assert decision.production_auto_authorized is False


def test_i6_3b_degraded_and_rejected_exact_evidence_are_excluded_not_promoted() -> None:
    request = _request(
        evidence=(
            _rolling(AdvisorEvidenceOutcome.PASS),
            _vda(AdvisorEvidenceOutcome.DEGRADED),
            _obs(
                "depth-anything-3-external",
                DenseProviderImplementation.DA3,
                DenseProviderFamily.MULTI_VIEW_AI_DEPTH,
                AdvisorEvidenceOutcome.REJECTED,
                sha=SHA_C,
            ),
        ),
        available=(
            "rolling-depth-temporal",
            "video-depth-anything-temporal",
            "depth-anything-3-external",
        ),
    )

    decision = evaluate_reconstruction_advisor(request)

    assert decision.recommended_provider_id == "rolling-depth-temporal"
    excluded = {item.provider_id: item.reason_codes for item in decision.excluded_providers}
    assert excluded["video-depth-anything-temporal"] == ("evidence_status_degraded",)
    assert excluded["depth-anything-3-external"] == ("evidence_status_rejected",)


def test_i6_3b_multiple_exact_passes_abstain_without_availability_tiebreak() -> None:
    request = _request(
        evidence=(_rolling(), _vda()),
        available=("rolling-depth-temporal",),
    )

    decision = evaluate_reconstruction_advisor(request)

    assert decision.disposition is AdvisorDisposition.ABSTAIN
    assert decision.evidence_support is AdvisorEvidenceSupport.CONFLICTING
    assert decision.recommended_provider_id == ""
    assert "multiple_exact_pass_candidates_no_tiebreak" in decision.reason_codes
    excluded = {item.provider_id: item.reason_codes for item in decision.excluded_providers}
    assert "multiple_exact_pass_candidates_no_tiebreak" in excluded["rolling-depth-temporal"]
    assert "provider_unavailable" in excluded["video-depth-anything-temporal"]
    assert "multiple_exact_pass_candidates_no_tiebreak" in excluded["video-depth-anything-temporal"]


def test_i6_3b_gpu_and_ram_never_break_an_exact_evidence_tie() -> None:
    evidence = (_rolling(), _vda())
    low = _request(
        evidence=evidence,
        available=("rolling-depth-temporal", "video-depth-anything-temporal"),
        resources=_resources(),
    )
    high = _request(
        evidence=evidence,
        available=("rolling-depth-temporal", "video-depth-anything-temporal"),
        resources=_resources(gpu="NVIDIA GeForce RTX 5090", free_vram=30 * 1024**3),
    )

    assert evaluate_reconstruction_advisor(low).as_dict() == evaluate_reconstruction_advisor(high).as_dict()


def test_i6_3b_unresolved_context_abstains_without_scene_classification() -> None:
    request = _request(
        evidence=(_mvs_pass(),),
        available=("colmap-mvs-external",),
        resolution=AdvisorContextResolution.UNRESOLVED,
    )

    decision = evaluate_reconstruction_advisor(request)

    assert decision.disposition is AdvisorDisposition.ABSTAIN
    assert decision.evidence_support is AdvisorEvidenceSupport.INSUFFICIENT
    assert decision.reason_codes == (
        "advisor_abstain",
        "evidence_context_unresolved",
        "live_context_resolver_not_qualified",
    )


def test_i6_3b_nonmatching_context_evidence_is_not_generalized() -> None:
    request = _request(
        evidence=(_mvs_pass(context="other-context"),),
        available=("colmap-mvs-external",),
        context="ctx-clean",
    )

    decision = evaluate_reconstruction_advisor(request)

    assert decision.disposition is AdvisorDisposition.ABSTAIN
    assert decision.evidence_support is AdvisorEvidenceSupport.INSUFFICIENT
    assert "no_exact_context_evidence" in decision.reason_codes
    assert "cross_context_generalization_forbidden" in decision.reason_codes


def test_i6_3b_source_not_ready_is_fail_closed() -> None:
    request = _request(
        evidence=(_mvs_pass(),),
        available=("colmap-mvs-external",),
        source_ready=False,
    )
    decision = evaluate_reconstruction_advisor(request)
    assert decision.disposition is AdvisorDisposition.ABSTAIN
    assert decision.reason_codes == ("advisor_abstain", "source_not_ready")


def test_i6_3b_authoritative_sfm_not_usable_is_fail_closed() -> None:
    request = _request(
        evidence=(_mvs_pass(),),
        available=("colmap-mvs-external",),
        sfm_usable=False,
    )
    decision = evaluate_reconstruction_advisor(request)
    assert decision.disposition is AdvisorDisposition.ABSTAIN
    assert decision.reason_codes == ("advisor_abstain", "authoritative_sfm_not_usable")


def test_i6_3b_runtime_integrity_failure_is_fail_closed() -> None:
    request = _request(
        evidence=(_mvs_pass(),),
        available=("colmap-mvs-external",),
        runtime_clean=False,
    )
    decision = evaluate_reconstruction_advisor(request)
    assert decision.disposition is AdvisorDisposition.ABSTAIN
    assert decision.reason_codes == ("advisor_abstain", "runtime_integrity_not_clean")


def test_i6_3b_no_available_provider_is_fail_closed() -> None:
    request = _request(evidence=(_mvs_pass(),), available=())
    decision = evaluate_reconstruction_advisor(request)
    assert decision.disposition is AdvisorDisposition.ABSTAIN
    assert decision.reason_codes == ("advisor_abstain", "no_concrete_provider_available")


def test_i6_3b_unique_pass_provider_unavailable_abstains_instead_of_fallback() -> None:
    request = _request(
        evidence=(_mvs_pass(),),
        available=("rolling-depth-temporal",),
    )

    decision = evaluate_reconstruction_advisor(request)

    assert decision.disposition is AdvisorDisposition.ABSTAIN
    assert decision.evidence_support is AdvisorEvidenceSupport.BOUNDED
    assert "unique_exact_pass_provider_unavailable" in decision.reason_codes
    assert decision.recommended_provider_id == ""
    excluded = {item.provider_id: item.reason_codes for item in decision.excluded_providers}
    assert excluded["colmap-mvs-external"] == ("provider_unavailable",)
    assert excluded["rolling-depth-temporal"] == ("no_exact_context_evidence",)


def test_i6_3b_exact_degraded_only_evidence_abstains_bounded() -> None:
    request = _request(
        evidence=(_rolling(AdvisorEvidenceOutcome.DEGRADED),),
        available=("rolling-depth-temporal",),
    )

    decision = evaluate_reconstruction_advisor(request)

    assert decision.disposition is AdvisorDisposition.ABSTAIN
    assert decision.evidence_support is AdvisorEvidenceSupport.BOUNDED
    assert "no_exact_clean_pass_provider" in decision.reason_codes
    assert decision.excluded_providers[0].reason_codes == ("evidence_status_degraded",)


def test_i6_3b_exact_rejected_or_unqualified_evidence_remains_visible() -> None:
    rejected = _rolling(AdvisorEvidenceOutcome.REJECTED)
    unqualified = _vda(AdvisorEvidenceOutcome.UNQUALIFIED)
    request = _request(
        evidence=(rejected, unqualified),
        available=("rolling-depth-temporal", "video-depth-anything-temporal"),
    )

    decision = evaluate_reconstruction_advisor(request)

    assert decision.disposition is AdvisorDisposition.ABSTAIN
    assert decision.evidence_support is AdvisorEvidenceSupport.BOUNDED
    excluded = {item.provider_id: item.reason_codes for item in decision.excluded_providers}
    assert excluded["rolling-depth-temporal"] == ("evidence_status_rejected",)
    assert excluded["video-depth-anything-temporal"] == ("evidence_status_unqualified",)


def test_i6_3b_available_provider_without_exact_evidence_is_excluded_but_does_not_block_unique_pass() -> None:
    request = _request(
        evidence=(_mvs_pass(),),
        available=("colmap-mvs-external", "rolling-depth-temporal"),
    )

    decision = evaluate_reconstruction_advisor(request)

    assert decision.disposition is AdvisorDisposition.RECOMMEND
    excluded = {item.provider_id: item.reason_codes for item in decision.excluded_providers}
    assert excluded["rolling-depth-temporal"] == ("no_exact_context_evidence",)


def test_i6_3b_decision_is_order_invariant_and_deterministic() -> None:
    evidence_a = (
        _mvs_pass(),
        _rolling(AdvisorEvidenceOutcome.DEGRADED),
        _vda(AdvisorEvidenceOutcome.REJECTED),
    )
    evidence_b = tuple(reversed(evidence_a))
    request_a = _request(
        evidence=evidence_a,
        available=("video-depth-anything-temporal", "colmap-mvs-external", "rolling-depth-temporal"),
    )
    request_b = _request(
        evidence=evidence_b,
        available=("rolling-depth-temporal", "video-depth-anything-temporal", "colmap-mvs-external"),
    )

    first = evaluate_reconstruction_advisor(request_a).as_dict()
    second = evaluate_reconstruction_advisor(request_a).as_dict()
    reordered = evaluate_reconstruction_advisor(request_b).as_dict()

    assert first == second == reordered


def test_i6_3b_engine_profile_is_frozen_and_has_no_execution_effect() -> None:
    assert I63B_ENGINE_PROFILE == "evidence_bound_advisory_v1"
    decision = evaluate_reconstruction_advisor(
        _request(evidence=(_mvs_pass(),), available=("colmap-mvs-external",))
    )
    payload = decision.as_dict()
    assert payload["effects"] == {
        "production_effect": False,
        "provider_selection_effect": False,
        "provider_execution": False,
        "settings_mutation": False,
        "start_action": False,
    }
