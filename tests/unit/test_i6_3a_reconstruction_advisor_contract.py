from __future__ import annotations

import pytest

from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
)
from core.reconstruction.reconstruction_advisor_contract import (
    ADVISOR_CAMERA_AUTHORITY_OWNER,
    ADVISOR_PRODUCTION_AUTO_SCOPE,
    AdvisorContextResolution,
    AdvisorDisposition,
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorEvidenceSupport,
    AdvisorProviderExclusion,
    AdvisorResourceObservation,
    ReconstructionAdvisorDecision,
    ReconstructionAdvisorRequest,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _binding(*, evidence_id: str = "i5.14b", sha256: str = SHA_A) -> AdvisorEvidenceBinding:
    return AdvisorEvidenceBinding(
        evidence_id=evidence_id,
        phase="I5.14B",
        path=f"/evidence/{evidence_id}.json",
        sha256=sha256,
    )


def _resource() -> AdvisorResourceObservation:
    return AdvisorResourceObservation(
        gpu_name="NVIDIA GeForce RTX 2060",
        cuda_ready=True,
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=5 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )


def _provider_evidence(
    *,
    outcome: AdvisorEvidenceOutcome = AdvisorEvidenceOutcome.PASS,
    provider_id: str = "depth-anything-3-external",
) -> AdvisorEvidenceObservation:
    return AdvisorEvidenceObservation(
        provider_id=provider_id,
        provider_family=DenseProviderFamily.MULTI_VIEW_AI_DEPTH,
        provider_implementation=DenseProviderImplementation.DA3,
        outcome=outcome,
        evidence_context_id="clean_static_multiview",
        matched_scenario_ids=("i4.3g2-clean-static",),
        reason_codes=("exact_clean_static_evidence",),
        limitations=("bounded_to_frozen_clean_static_evidence",),
        bindings=(_binding(),),
    )


def _request(*, evidence=()) -> ReconstructionAdvisorRequest:
    return ReconstructionAdvisorRequest(
        reconstruction_mode="offline_reconstruction",
        source_workflow="image_sequence",
        source_ready=True,
        source_item_count=128,
        source_fingerprint="source-fingerprint",
        workspace_state="ready",
        authoritative_sfm_usable=True,
        runtime_integrity_clean=True,
        evidence_context_resolution=AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
        evidence_context_id="clean_static_multiview",
        available_provider_ids=("depth-anything-3-external",),
        resources=_resource(),
        provider_evidence=tuple(evidence),
        camera_authority_owner=ADVISOR_CAMERA_AUTHORITY_OWNER,
        production_auto_policy_scope=ADVISOR_PRODUCTION_AUTO_SCOPE,
    )


def test_i6_3a_request_is_immutable_and_preserves_authority_boundaries() -> None:
    request = _request(evidence=(_provider_evidence(),))

    assert request.reconstruction_mode == "offline_reconstruction"
    assert request.source_workflow == "image_sequence"
    assert request.source_item_count == 128
    assert request.camera_authority_owner == "pycolmap_sfm"
    assert request.production_auto_qualified is False
    assert request.production_auto_authorized is False
    assert request.provider_activation_authorized is False
    assert request.execution_authorized is False
    assert request.settings_mutation_authorized is False

    with pytest.raises((AttributeError, TypeError)):
        request.source_ready = False  # type: ignore[misc]


def test_i6_3a_evidence_binding_requires_content_addressed_provenance() -> None:
    with pytest.raises(ValueError, match="sha256"):
        _binding(sha256="not-a-sha")

    with pytest.raises(ValueError, match="path"):
        AdvisorEvidenceBinding(
            evidence_id="i5.14b",
            phase="I5.14B",
            path="",
            sha256=SHA_A,
        )


def test_i6_3a_resource_observation_is_descriptive_not_a_quality_score() -> None:
    resources = _resource()

    assert resources.gpu_name == "NVIDIA GeForce RTX 2060"
    assert resources.free_vram_bytes < resources.total_vram_bytes
    assert resources.as_dict()["is_quality_score"] is False

    with pytest.raises(ValueError, match="free_vram_bytes"):
        AdvisorResourceObservation(
            gpu_name="GPU",
            cuda_ready=True,
            total_vram_bytes=1024,
            free_vram_bytes=2048,
            available_ram_bytes=1024,
        )


def test_i6_3a_negative_and_degraded_evidence_are_preserved_not_promoted() -> None:
    degraded = _provider_evidence(outcome=AdvisorEvidenceOutcome.DEGRADED)
    rejected = _provider_evidence(outcome=AdvisorEvidenceOutcome.REJECTED, provider_id="other-provider")
    request = _request(evidence=(degraded, rejected))

    assert request.provider_evidence[0].outcome is AdvisorEvidenceOutcome.DEGRADED
    assert request.provider_evidence[1].outcome is AdvisorEvidenceOutcome.REJECTED
    assert request.provider_evidence[0].limitations
    assert request.provider_evidence[1].bindings


def test_i6_3a_exact_supported_recommendation_is_advisory_only() -> None:
    evidence = _provider_evidence()
    decision = ReconstructionAdvisorDecision(
        disposition=AdvisorDisposition.RECOMMEND,
        evidence_support=AdvisorEvidenceSupport.EXACT_QUALIFIED,
        recommended_provider_id=evidence.provider_id,
        recommended_family=evidence.provider_family,
        recommended_implementation=evidence.provider_implementation,
        reason_codes=("exact_unique_evidence_support",),
        explanations=("Frozen exact-context evidence supports this provider for advisory use.",),
        limitations=evidence.limitations,
        matched_scenario_ids=evidence.matched_scenario_ids,
        evidence_bindings=evidence.bindings,
        considered_provider_ids=(evidence.provider_id,),
        excluded_providers=(),
    )

    assert decision.recommends_provider is True
    assert decision.production_effect is False
    assert decision.provider_selection_effect is False
    assert decision.provider_execution is False
    assert decision.settings_mutation is False
    assert decision.start_action is False
    assert decision.production_auto_qualified is False
    assert decision.production_auto_authorized is False
    assert decision.global_provider_ranking_created is False
    assert decision.global_scalar_score_created is False


def test_i6_3a_recommendation_requires_exact_qualified_support_and_provenance() -> None:
    evidence = _provider_evidence()

    with pytest.raises(ValueError, match="EXACT_QUALIFIED"):
        ReconstructionAdvisorDecision(
            disposition=AdvisorDisposition.RECOMMEND,
            evidence_support=AdvisorEvidenceSupport.BOUNDED,
            recommended_provider_id=evidence.provider_id,
            recommended_family=evidence.provider_family,
            recommended_implementation=evidence.provider_implementation,
            reason_codes=("bounded_only",),
            explanations=("Bounded evidence only.",),
            limitations=evidence.limitations,
            matched_scenario_ids=evidence.matched_scenario_ids,
            evidence_bindings=evidence.bindings,
            considered_provider_ids=(evidence.provider_id,),
            excluded_providers=(),
        )

    with pytest.raises(ValueError, match="evidence bindings"):
        ReconstructionAdvisorDecision(
            disposition=AdvisorDisposition.RECOMMEND,
            evidence_support=AdvisorEvidenceSupport.EXACT_QUALIFIED,
            recommended_provider_id=evidence.provider_id,
            recommended_family=evidence.provider_family,
            recommended_implementation=evidence.provider_implementation,
            reason_codes=("exact_unique_evidence_support",),
            explanations=("Exact support.",),
            limitations=(),
            matched_scenario_ids=evidence.matched_scenario_ids,
            evidence_bindings=(),
            considered_provider_ids=(evidence.provider_id,),
            excluded_providers=(),
        )


def test_i6_3a_abstention_forbids_hidden_provider_choice() -> None:
    decision = ReconstructionAdvisorDecision(
        disposition=AdvisorDisposition.ABSTAIN,
        evidence_support=AdvisorEvidenceSupport.INSUFFICIENT,
        reason_codes=("no_exact_evidence_context",),
        explanations=("No exact frozen evidence context supports a defensible recommendation.",),
        limitations=("engineer_choice_required",),
        considered_provider_ids=("provider-a", "provider-b"),
        excluded_providers=(
            AdvisorProviderExclusion("provider-a", ("no_exact_pass",)),
            AdvisorProviderExclusion("provider-b", ("conflicting_evidence",)),
        ),
    )

    assert decision.recommends_provider is False
    assert decision.recommended_provider_id == ""
    assert decision.as_dict()["recommendation"]["provider_id"] == ""

    with pytest.raises(ValueError, match="must not carry a provider"):
        ReconstructionAdvisorDecision(
            disposition=AdvisorDisposition.ABSTAIN,
            evidence_support=AdvisorEvidenceSupport.CONFLICTING,
            recommended_provider_id="provider-a",
            recommended_family=DenseProviderFamily.MVS,
            recommended_implementation=DenseProviderImplementation.COLMAP_MVS,
            reason_codes=("conflicting_evidence",),
            explanations=("Evidence conflicts.",),
            considered_provider_ids=("provider-a", "provider-b"),
            excluded_providers=(),
        )


def test_i6_3a_request_rejects_duplicate_provider_evidence_identity() -> None:
    evidence = _provider_evidence()
    with pytest.raises(ValueError, match="duplicate provider/evidence context"):
        _request(evidence=(evidence, evidence))


def test_i6_3a_context_resolution_is_explicit_and_unresolved_context_carries_no_token() -> None:
    request = ReconstructionAdvisorRequest(
        reconstruction_mode="offline_reconstruction",
        source_workflow="image_sequence",
        source_ready=False,
        source_item_count=0,
        source_fingerprint="",
        workspace_state="idle",
        authoritative_sfm_usable=False,
        runtime_integrity_clean=True,
        evidence_context_resolution=AdvisorContextResolution.UNRESOLVED,
        evidence_context_id="",
        available_provider_ids=(),
        resources=_resource(),
        provider_evidence=(),
    )

    assert request.evidence_context_resolution is AdvisorContextResolution.UNRESOLVED
    assert request.evidence_context_id == ""

    with pytest.raises(ValueError, match="requires evidence_context_id"):
        ReconstructionAdvisorRequest(
            reconstruction_mode="offline_reconstruction",
            source_workflow="image_sequence",
            source_ready=True,
            source_item_count=4,
            source_fingerprint="x",
            workspace_state="ready",
            authoritative_sfm_usable=True,
            runtime_integrity_clean=True,
            evidence_context_resolution=AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
            evidence_context_id="",
            available_provider_ids=("provider-a",),
            resources=_resource(),
            provider_evidence=(),
        )


def test_i6_3a_provider_family_must_match_canonical_implementation_contract() -> None:
    with pytest.raises(ValueError, match="provider family does not match"):
        AdvisorEvidenceObservation(
            provider_id="bad",
            provider_family=DenseProviderFamily.MVS,
            provider_implementation=DenseProviderImplementation.DA3,
            outcome=AdvisorEvidenceOutcome.PASS,
            evidence_context_id="context",
            matched_scenario_ids=("scenario",),
            reason_codes=("pass",),
            limitations=(),
            bindings=(_binding(),),
        )


def test_i6_3a_exclusion_is_reason_coded_and_immutable() -> None:
    exclusion = AdvisorProviderExclusion("provider-a", ("rejected_exact_context",))
    assert exclusion.as_dict() == {
        "provider_id": "provider-a",
        "reason_codes": ["rejected_exact_context"],
    }


def test_i6_3a_serialization_makes_non_effect_and_no_score_contract_explicit() -> None:
    request = _request(evidence=(_provider_evidence(),))
    payload = request.as_dict()

    assert payload["authority"]["camera_authority_owner"] == "pycolmap_sfm"
    assert payload["production_auto"]["qualified"] is False
    assert payload["production_auto"]["authorized"] is False
    assert payload["effects"]["provider_activation_authorized"] is False
    assert payload["effects"]["execution_authorized"] is False
    assert payload["effects"]["settings_mutation_authorized"] is False
    assert payload["claim_boundaries"]["global_provider_ranking"] is False
    assert payload["claim_boundaries"]["global_scalar_score"] is False
    assert payload["claim_boundaries"]["cross_family_rank"] is False
    assert payload["claim_boundaries"]["universal_provider_winner"] is False
