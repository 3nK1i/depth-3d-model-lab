from __future__ import annotations

from copy import deepcopy
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_final_closure import (
    build_i5_14b_final_closure,
    validate_i5_14a_readiness,
    validate_i5_14b_receipt,
)
from tests.i5_14b_helpers import make_binding, make_closure_receipt, make_readiness


def build():
    return build_i5_14b_final_closure(
        readiness=make_readiness(),
        readiness_binding=make_binding("i5_14a.json", "a"),
        post_i5_14a_checkpoint_binding=make_binding("post_i5_14a.tar.gz", "b"),
        closure_receipt=make_closure_receipt(),
        closure_receipt_binding=make_binding("i5_14b_receipt.json", "c"),
    )


def test_final_i5_closure_passes_with_bounded_limitations():
    out = build()
    assert out["phase"] == "I5.14B"
    assert out["pass"] is True
    assert out["quality_status"] == "final_i5_closed_with_bounded_limitations"
    assert out["qualification"]["i5_complete"] is True
    assert out["qualification"]["i5_closed"] is True


def test_all_fourteen_i5_phases_are_closed():
    summary = build()["phase_closure_summary"]
    assert len(summary) == 14
    assert summary == {f"I5.{idx}": "closed" for idx in range(1, 15)}


def test_readiness_synthesis_is_preserved_exactly():
    readiness = make_readiness()
    out = build()
    assert out["evidence_synthesis"] == readiness["evidence_synthesis"]
    assert out["bounded_limitations"] == readiness["bounded_limitations"]


def test_negative_results_remain_visible_in_final_closure():
    limitations = build()["bounded_limitations"]
    for item in (
        "forward_motion_sparse_sfm_rejected",
        "video_depth_anything_degraded",
        "matching_policy_candidate_rejected",
        "hybrid_not_justified",
        "production_auto_deferred",
    ):
        assert item in limitations


def test_production_auto_remains_deferred():
    out = build()
    assert out["decision"]["offline_auto_policy_qualified"] is True
    assert out["decision"]["production_auto_authorized"] is False
    assert out["decision"]["production_auto_integration_deferred"] is True


def test_final_closure_does_not_change_production_defaults():
    decision = build()["decision"]
    assert decision["production_default_modified"] is False
    assert decision["provider_promoted"] is False
    assert decision["provider_ranking_created"] is False
    assert decision["global_scalar_score_created"] is False


def test_final_handoff_contract_requires_delta_and_full_checkpoint():
    handoff = build()["immutable_handoff_contract"]
    assert handoff["post_i5_14b_delta_required"] is True
    assert handoff["full_post_i5_source_checkpoint_required"] is True
    assert handoff["sha256_sidecars_required"] is True
    assert handoff["full_checkpoint_becomes_new_recovery_base"] is True
    assert handoff["post_closure_handoff_checkpoint_created"] is False


def test_no_git_requirement_is_introduced():
    handoff = build()["immutable_handoff_contract"]
    assert handoff["source_control_mode"] == "immutable_local_checkpoints"
    assert handoff["git_or_gitea_required"] is False


def test_claim_boundaries_remain_conservative():
    claims = build()["claim_boundaries"]
    assert all(value is False for value in claims.values())


def test_scope_is_evidence_only_and_execution_free():
    scope = build()["scope"]
    assert scope["evidence_only_final_i5_closure"] is True
    for key in (
        "provider_execution", "sfm_execution", "mvs_execution", "scene_classifier_execution",
        "scenario_rematerialization", "hybrid_execution", "matching_policy_modified",
        "thresholds_modified", "production_auto_integration", "production_default_modified",
        "provider_promoted",
    ):
        assert scope[key] is False


def test_final_decision_marks_i5_complete_and_closed():
    decision = build()["decision"]
    assert decision["i5_14a_complete"] is True
    assert decision["i5_14b_complete"] is True
    assert decision["i5_14_complete"] is True
    assert decision["i5_complete"] is True
    assert decision["i5_closed"] is True


def test_readiness_must_be_closure_ready():
    readiness = make_readiness()
    readiness["qualification"]["closure_ready"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_14a_readiness(readiness)


def test_readiness_cannot_claim_i5_already_complete():
    readiness = make_readiness()
    readiness["decision"]["i5_complete"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_14a_readiness(readiness)


def test_production_auto_authorization_drift_is_fatal():
    readiness = make_readiness()
    readiness["decision"]["production_auto_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_14a_readiness(readiness)


def test_closure_receipt_requires_zero_return_codes():
    receipt = make_closure_receipt()
    receipt["return_codes"]["focused"] = 1
    with pytest.raises(ReconstructionValidationError):
        validate_i5_14b_receipt(receipt)


def test_binding_sha_must_be_valid():
    bad = make_binding("i5_14a.json", "a")
    bad["sha256"] = "bad"
    with pytest.raises(ReconstructionValidationError):
        build_i5_14b_final_closure(
            readiness=make_readiness(),
            readiness_binding=bad,
            post_i5_14a_checkpoint_binding=make_binding("post.tar.gz", "b"),
            closure_receipt=make_closure_receipt(),
            closure_receipt_binding=make_binding("receipt.json", "c"),
        )
