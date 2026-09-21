from __future__ import annotations

from pathlib import Path

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
    AdvisorResourceObservation,
    ReconstructionAdvisorDecision,
    ReconstructionAdvisorRequest,
)


def test_i6_3a_contract_can_carry_bounded_i5_evidence_without_execution_effects() -> None:
    binding = AdvisorEvidenceBinding(
        evidence_id="post-i5-final-closure",
        phase="I5.14B",
        path="/evidence/i5_14b_final_closure.json",
        sha256="c" * 64,
    )
    observation = AdvisorEvidenceObservation(
        provider_id="depth-anything-3-external",
        provider_family=DenseProviderFamily.MULTI_VIEW_AI_DEPTH,
        provider_implementation=DenseProviderImplementation.DA3,
        outcome=AdvisorEvidenceOutcome.PASS,
        evidence_context_id="clean_static_multiview",
        matched_scenario_ids=("i4.3g2-clean-static",),
        reason_codes=("frozen_exact_context_pass",),
        limitations=("no_cross_context_generalization",),
        bindings=(binding,),
    )
    request = ReconstructionAdvisorRequest(
        reconstruction_mode="offline_reconstruction",
        source_workflow="image_sequence",
        source_ready=True,
        source_item_count=80,
        source_fingerprint="fixture",
        workspace_state="partial_result",
        authoritative_sfm_usable=True,
        runtime_integrity_clean=True,
        evidence_context_resolution=AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
        evidence_context_id="clean_static_multiview",
        available_provider_ids=(observation.provider_id,),
        resources=AdvisorResourceObservation(
            gpu_name="NVIDIA GeForce RTX 2060",
            cuda_ready=True,
            total_vram_bytes=6 * 1024**3,
            free_vram_bytes=4 * 1024**3,
            available_ram_bytes=8 * 1024**3,
        ),
        provider_evidence=(observation,),
        camera_authority_owner=ADVISOR_CAMERA_AUTHORITY_OWNER,
        production_auto_policy_scope=ADVISOR_PRODUCTION_AUTO_SCOPE,
    )
    decision = ReconstructionAdvisorDecision(
        disposition=AdvisorDisposition.RECOMMEND,
        evidence_support=AdvisorEvidenceSupport.EXACT_QUALIFIED,
        recommended_provider_id=observation.provider_id,
        recommended_family=observation.provider_family,
        recommended_implementation=observation.provider_implementation,
        reason_codes=("exact_unique_evidence_support",),
        explanations=("Recommendation is advisory and evidence-bound.",),
        limitations=observation.limitations,
        matched_scenario_ids=observation.matched_scenario_ids,
        evidence_bindings=observation.bindings,
        considered_provider_ids=(observation.provider_id,),
        excluded_providers=(),
    )

    assert request.camera_authority_owner == "pycolmap_sfm"
    assert decision.recommended_provider_id == observation.provider_id
    assert decision.production_effect is False
    assert decision.provider_selection_effect is False
    assert decision.provider_execution is False
    assert decision.settings_mutation is False
    assert decision.start_action is False


def test_i6_3a_contract_module_is_execution_free_and_does_not_import_auto_engine() -> None:
    text = Path("core/reconstruction/reconstruction_advisor_contract.py").read_text(encoding="utf-8")

    forbidden = (
        "i5_auto_policy_engine",
        "OfflineReconstructionController",
        "subprocess",
        "torch",
        "import pycolmap",
        "QWidget",
        "QApplication",
    )
    for token in forbidden:
        assert token not in text

    assert "global_provider_ranking_created" in text
    assert "global_scalar_score_created" in text
    assert "production_auto_authorized" in text
    assert "provider_selection_effect" in text
