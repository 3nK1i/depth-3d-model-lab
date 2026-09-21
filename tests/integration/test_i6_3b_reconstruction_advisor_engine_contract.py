from __future__ import annotations

from pathlib import Path

from core.reconstruction.dense_provider_architecture import DenseProviderFamily, DenseProviderImplementation
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorDisposition,
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorResourceObservation,
    ReconstructionAdvisorRequest,
)
from core.reconstruction.reconstruction_advisor_engine import evaluate_reconstruction_advisor


def _request() -> ReconstructionAdvisorRequest:
    binding = AdvisorEvidenceBinding(
        evidence_id="frozen-i5-qualification",
        phase="I5.14B",
        path="/evidence/i5_14b.json",
        sha256="d" * 64,
    )
    observation = AdvisorEvidenceObservation(
        provider_id="colmap-mvs-external",
        provider_family=DenseProviderFamily.MVS,
        provider_implementation=DenseProviderImplementation.COLMAP_MVS,
        outcome=AdvisorEvidenceOutcome.PASS,
        evidence_context_id="i5_6_current_mvs_envelope",
        matched_scenario_ids=("low_overlap_weak_parallax", "i5_6_current_mvs_envelope"),
        reason_codes=("frozen_exact_pass",),
        limitations=("bounded_to_frozen_i5_context",),
        bindings=(binding,),
    )
    return ReconstructionAdvisorRequest(
        reconstruction_mode="offline_reconstruction",
        source_workflow="image_sequence",
        source_ready=True,
        source_item_count=128,
        source_fingerprint="fingerprint",
        workspace_state="partial_result",
        authoritative_sfm_usable=True,
        runtime_integrity_clean=True,
        evidence_context_resolution=AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
        evidence_context_id="i5_6_current_mvs_envelope",
        available_provider_ids=("colmap-mvs-external",),
        resources=AdvisorResourceObservation(
            gpu_name="NVIDIA GeForce RTX 2060",
            cuda_ready=True,
            total_vram_bytes=6 * 1024**3,
            free_vram_bytes=5 * 1024**3,
            available_ram_bytes=8 * 1024**3,
        ),
        provider_evidence=(observation,),
    )


def test_i6_3b_engine_consumes_i6_3a_contract_and_returns_i6_3a_decision() -> None:
    decision = evaluate_reconstruction_advisor(_request())
    assert decision.disposition is AdvisorDisposition.RECOMMEND
    assert decision.recommended_provider_id == "colmap-mvs-external"
    assert decision.evidence_bindings[0].sha256 == "d" * 64
    assert decision.matched_scenario_ids == ("i5_6_current_mvs_envelope", "low_overlap_weak_parallax")


def test_i6_3b_engine_has_no_controller_ui_worker_or_auto_engine_dependency() -> None:
    text = Path("core/reconstruction/reconstruction_advisor_engine.py").read_text(encoding="utf-8")
    forbidden = (
        "from app.",
        "import app.",
        "OfflineReconstructionController",
        "i5_auto_policy_engine",
        "DenseProviderSelectionDecision",
        "subprocess",
        "import torch",
        "import pycolmap",
        "PyQt6",
        "QWidget",
        "QApplication",
    )
    for token in forbidden:
        assert token not in text


def test_i6_3b_engine_contains_no_score_rank_or_hardware_tiebreak_path() -> None:
    text = Path("core/reconstruction/reconstruction_advisor_engine.py").read_text(encoding="utf-8")
    assert "global_scalar_score" not in text
    assert "def rank_provider" not in text
    assert "provider_scores" not in text
    assert "vram_tiebreak" not in text
    assert "gpu_tiebreak" not in text
    assert "while retry" not in text
    assert "retry_candidates" not in text
    assert "fallback_provider_id" not in text


def test_i6_3b_engine_preserves_pycolmap_authority_and_auto_boundary() -> None:
    decision = evaluate_reconstruction_advisor(_request())
    assert decision.camera_authority_owner == "pycolmap_sfm"
    assert decision.production_auto_qualified is False
    assert decision.production_auto_authorized is False
    assert decision.provider_selection_effect is False
    assert decision.provider_execution is False
    assert decision.start_action is False
