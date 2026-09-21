from __future__ import annotations

from copy import deepcopy
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_final_qualification_readiness import (
    build_i5_14a_final_qualification_readiness,
    validate_i5_14a_receipt,
)
from tests.i5_14a_helpers import make_bindings, make_closures, make_receipt, make_binding


def build():
    return build_i5_14a_final_qualification_readiness(
        closures=make_closures(),
        closure_bindings=make_bindings(),
        post_i5_13d_checkpoint_binding=make_binding("post_i5_13d.tar.gz", "f"),
        readiness_receipt=make_receipt(),
        readiness_receipt_binding=make_binding("receipt.json", "e"),
    )


def test_readiness_passes_with_bounded_limitations_preserved():
    out = build()
    assert out["phase"] == "I5.14A"
    assert out["pass"] is True
    assert out["quality_status"] == "final_i5_closure_ready_with_bounded_limitations"
    assert out["qualification"]["closure_ready"] is True
    assert out["decision"]["i5_14b_authorized"] is True


def test_sparse_sfm_envelope_is_preserved():
    out = build()["evidence_synthesis"]["sparse_sfm"]
    assert out == {
        "difficult_handheld": "degraded",
        "low_overlap_weak_parallax": "pass",
        "visual_stress": "degraded",
        "forward_motion": "rejected",
        "dynamic_scene_proxy": "pass",
    }


def test_dense_provider_envelope_is_preserved():
    out = build()["evidence_synthesis"]["dense_provider"]
    assert out["mvs"] == "pass"
    assert out["rolling_depth"] == "pass"
    assert out["video_depth_anything"] == "degraded"
    assert out["remaining_single_image"] == "mixed_terminal_quality"


def test_negative_results_do_not_block_closure():
    out = build()
    assert out["qualification"]["negative_results_preserved"] is True
    assert out["qualification"]["negative_results_are_not_closure_failures"] is True
    assert "forward_motion_sparse_sfm_rejected" in out["bounded_limitations"]
    assert "production_auto_deferred" in out["bounded_limitations"]


def test_auto_conclusion_remains_deferred_in_production():
    auto = build()["evidence_synthesis"]["auto_policy"]
    assert auto["offline_shadow_qualified"] is True
    assert auto["production_auto_qualified"] is False
    assert auto["production_auto_authorized"] is False
    assert auto["integration_decision"] == "defer_production_auto"


def test_matching_policy_candidate_remains_rejected():
    matching = build()["evidence_synthesis"]["matching_policy"]
    assert matching["sensitivity_observed"] is True
    assert matching["production_quality_recovery"] is False
    assert matching["production_default_change_authorized"] is False


def test_hybrid_remains_not_justified():
    hybrid = build()["evidence_synthesis"]["hybrid"]
    assert hybrid["quality_status"] == "hybrid_not_justified"
    assert hybrid["execution_authorized"] is False


def test_claim_boundaries_remain_conservative():
    claims = build()["claim_boundaries"]
    assert all(value is False for value in claims.values())


def test_scope_is_evidence_only():
    scope = build()["scope"]
    for key in (
        "provider_execution", "sfm_execution", "mvs_execution", "scene_classifier_execution",
        "production_auto_integration", "production_default_modified", "provider_promoted",
        "matching_policy_modified", "hybrid_execution", "thresholds_modified",
    ):
        assert scope[key] is False


def test_all_thirteen_closures_are_bound():
    out = build()
    assert out["qualification"]["closure_artifact_count"] == 13
    assert len(out["evidence_bindings"]["phase_closures"]) == 13


def test_wrong_forward_motion_quality_fails():
    closures = make_closures()
    closures["i5_4c"]["quality_status"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        build_i5_14a_final_qualification_readiness(
            closures=closures,
            closure_bindings=make_bindings(),
            post_i5_13d_checkpoint_binding=make_binding("post.tar.gz", "f"),
            readiness_receipt=make_receipt(),
            readiness_receipt_binding=make_binding("receipt.json", "e"),
        )


def test_production_auto_authorization_drift_fails():
    closures = make_closures()
    closures["i5_13d"]["qualification"]["production_auto_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        build_i5_14a_final_qualification_readiness(
            closures=closures,
            closure_bindings=make_bindings(),
            post_i5_13d_checkpoint_binding=make_binding("post.tar.gz", "f"),
            readiness_receipt=make_receipt(),
            readiness_receipt_binding=make_binding("receipt.json", "e"),
        )


def test_receipt_requires_zero_return_codes():
    receipt = make_receipt()
    receipt["return_codes"]["focused"] = 1
    with pytest.raises(ReconstructionValidationError):
        validate_i5_14a_receipt(receipt)


def test_binding_sha_validation():
    bindings = make_bindings()
    bindings["i5_1c"]["sha256"] = "bad"
    with pytest.raises(ReconstructionValidationError):
        build_i5_14a_final_qualification_readiness(
            closures=make_closures(),
            closure_bindings=bindings,
            post_i5_13d_checkpoint_binding=make_binding("post.tar.gz", "f"),
            readiness_receipt=make_receipt(),
            readiness_receipt_binding=make_binding("receipt.json", "e"),
        )
