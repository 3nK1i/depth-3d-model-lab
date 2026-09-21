from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_auto_policy_admission import (
    I513A_DECISION_CLASSES,
    I513A_PHASE,
    I513A_POLICY_PROFILE,
    build_i5_13a_auto_policy_admission,
    validate_i5_10d_for_i5_13a,
    validate_i5_11d_for_i5_13a,
    validate_i5_12a_for_i5_13a,
    validate_i5_13a_receipt,
    validate_i5_1c_for_i5_13a,
    validate_i5_4c_for_i5_13a,
    validate_i5_7c_for_i5_13a,
    validate_i5_8c_for_i5_13a,
    validate_i5_9c_for_i5_13a,
)
from tests.i5_13a_helpers import (
    build_args,
    clone,
    i510d,
    i511d,
    i512a,
    i51c,
    i54c,
    i57c,
    i58c,
    i59c,
    receipt,
)


def build():
    return build_i5_13a_auto_policy_admission(**build_args())


def test_phase_and_profile_constants():
    assert I513A_PHASE == "I5.13A"
    assert I513A_POLICY_PROFILE == "evidence_bound_conservative_v1"


def test_decision_classes_include_abstain():
    assert I513A_DECISION_CLASSES == ("select", "abstain", "engineer_override")


def test_build_admits_only_design():
    out = build()
    assert out["pass"] is True
    assert out["quality_status"] == "auto_policy_design_admitted"
    assert out["decision"]["i5_13b_authorized"] is True
    assert out["decision"]["production_auto_authorized"] is False


def test_auto_is_policy_not_provider():
    arch = build()["architecture_contract"]
    assert arch["auto_policy_id"] == "auto"
    assert arch["auto_is_selection_policy_only"] is True
    assert arch["auto_is_executable_provider"] is False
    assert "auto" not in arch["provider_implementations"]


def test_pycolmap_authority_is_locked():
    assert build()["architecture_contract"]["camera_pose_authority_owner"] == "pycolmap_sfm"


def test_abstention_is_first_class():
    design = build()["auto_policy_v1_design"]
    assert design["primary_principle"] == "select_only_on_exact_unique_evidence_support_else_abstain"
    assert design["ambiguity_rules"]["multiple_eligible_without_direct_scenario_preference_abstain"] is True
    assert design["ambiguity_rules"]["zero_eligible_abstain"] is True


def test_rejected_evidence_is_never_auto_primary():
    rules = build()["auto_policy_v1_design"]["eligibility_rules"]
    assert rules["rejected_evidence_is_ineligible"] is True
    assert rules["sparse_sfm_failure_cannot_be_masked_by_dense_provider_selection"] is True


def test_degraded_evidence_is_not_auto_primary():
    assert build()["auto_policy_v1_design"]["eligibility_rules"]["degraded_evidence_is_not_automatic_primary"] is True


def test_historical_rejection_cannot_be_overwritten():
    rules = build()["auto_policy_v1_design"]["eligibility_rules"]
    assert rules["historical_exact_scenario_rejection_remains_active"] is True
    assert rules["later_distinct_scenario_pass_does_not_overwrite_rejection"] is True


def test_no_global_score_or_provider_win_count():
    rules = build()["auto_policy_v1_design"]["ambiguity_rules"]
    assert rules["no_global_scalar_score"] is True
    assert rules["no_provider_win_count"] is True
    assert rules["no_cross_family_rank"] is True


def test_no_runtime_or_gpu_tiebreak():
    rules = build()["auto_policy_v1_design"]["ambiguity_rules"]
    assert rules["no_runtime_performance_tiebreak"] is True
    assert rules["no_gpu_model_performance_ranking"] is True


def test_fallback_is_explicit_and_no_retry_until_pass():
    rules = build()["auto_policy_v1_design"]["fallback_rules"]
    assert rules["fallback_must_be_explicit_in_decision"] is True
    assert rules["retry_until_pass_forbidden"] is True
    assert rules["implicit_global_fallback_chain_forbidden"] is True


def test_engineer_override_is_not_auto():
    rules = build()["auto_policy_v1_design"]["engineer_override_rules"]
    assert rules["override_is_not_automatic_selection"] is True
    assert rules["override_does_not_rewrite_evidence"] is True


def test_matching_policy_cannot_be_changed_by_auto():
    out = build()
    assert out["evidence_synthesis"]["matching_policy_context"]["auto_matching_policy_change_authorized"] is False
    assert out["auto_policy_v1_design"]["forbidden_policy_actions"]["change_sfm_matching_strategy"] is True


def test_hybrid_is_not_auto_eligible():
    out = build()
    assert out["evidence_synthesis"]["hybrid_context"]["hybrid_family_auto_eligible"] is False
    assert out["auto_policy_v1_design"]["hard_gates"]["hybrid_family_forbidden"] is True


def test_i5_4_forward_motion_forces_abstention_boundary():
    info = validate_i5_4c_for_i5_13a(i54c())
    assert info["quality_status"] == "rejected"
    assert info["hard_auto_abstention_required"] is True


def test_i5_1_difficult_handheld_is_degraded_boundary():
    info = validate_i5_1c_for_i5_13a(i51c())
    assert info["quality_status"] == "degraded"
    assert "abstention" in info["auto_implication"]


def test_rolling_depth_historical_motion_rejection_is_preserved():
    info = validate_i5_7c_for_i5_13a(i57c())
    assert info["current_castle_quality_status"] == "pass"
    assert info["historical_motion"] == "rejected"
    assert info["conflicting_or_mixed_scenario_evidence_preserved"] is True


def test_vda_is_execution_qualified_but_not_clean_auto_primary():
    info = validate_i5_8c_for_i5_13a(i58c())
    assert info["execution_qualified"] is True
    assert info["clean_auto_primary_eligible"] is False


def test_single_image_low_texture_rejections_remain_active():
    info = validate_i5_9c_for_i5_13a(i59c())
    assert info["historical_low_texture_rejections_remain_active"] == ["depth_anything_v2", "marigold"]


def test_mp_sfm_partial_recovery_does_not_authorize_matcher_change():
    info = validate_i5_10d_for_i5_13a(i510d())
    assert info["matching_policy_sensitivity_observed"] is True
    assert info["production_quality_recovery_observed"] is False
    assert info["auto_matching_policy_change_authorized"] is False


def test_failure_recovery_evidence_supports_integrity_gate_only():
    info = validate_i5_11d_for_i5_13a(i511d())
    assert info["bounded_fault_recovery_pass"] is True
    assert info["system_integrity_gate_supported"] is True


def test_i512_closure_authorizes_i513_but_not_hybrid():
    info = validate_i5_12a_for_i5_13a(i512a())
    assert info["i5_13_authorized"] is True
    assert info["hybrid_execution_authorized"] is False


def test_i513b_is_offline_shadow_only():
    boundary = build()["auto_policy_v1_design"]["production_context_boundary"]
    assert boundary["i5_13b_engine_is_offline_shadow_only"] is True
    assert boundary["production_auto_integration_deferred_to_i5_13d"] is True


def test_i513c_is_frozen_evidence_replay_only():
    replay = build()["offline_replay_contract"]
    assert replay["i5_13c_must_replay_frozen_evidence_only"] is True
    assert replay["provider_execution"] is False
    assert replay["same_input_same_decision_required"] is True


def test_scope_is_execution_free():
    scope = build()["scope"]
    for key in (
        "auto_policy_engine_implemented", "auto_provider_selection_performed", "production_auto_integration",
        "provider_execution", "sfm_execution", "mvs_execution", "hybrid_execution", "production_default_modified",
    ):
        assert scope[key] is False


def test_claim_boundaries_remain_conservative():
    claims = build()["claim_boundaries"]
    assert claims["universal_provider_winner"] is False
    assert claims["global_provider_ranking"] is False
    assert claims["production_auto_qualified"] is False


def test_receipt_validation():
    assert validate_i5_13a_receipt(receipt())["pass"] is True


def test_rejects_receipt_that_claims_engine_implementation():
    bad = clone(receipt())
    bad["scope"]["auto_policy_engine_implemented"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_13a_receipt(bad)


def test_rejects_forward_motion_if_not_rejected():
    bad = clone(i54c())
    bad["quality_status"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_4c_for_i5_13a(bad)


def test_rejects_vda_clean_pass_claim_drift():
    bad = clone(i58c())
    bad["qualification"]["current_frozen_foundation_vda_quality_clean_pass_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8c_for_i5_13a(bad)


def test_rejects_hybrid_authorization_drift():
    bad = clone(i512a())
    bad["decision"]["hybrid_execution_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_12a_for_i5_13a(bad)


def test_rejects_auto_already_implemented_upstream():
    args = build_args()
    args["i5_12a_closure"]["decision"]["auto_policy_implemented"] = True
    with pytest.raises(ReconstructionValidationError):
        build_i5_13a_auto_policy_admission(**args)


def test_all_13_evidence_classes_are_sha_bound():
    bindings = build()["evidence_bindings"]
    assert len(bindings) == 15  # 13 evidence artifacts + source checkpoint + admission receipt
    assert all(len(item["sha256"]) == 64 for item in bindings.values())


def test_next_step_is_i513b_only():
    out = build()
    assert out["next"].startswith("I5.13B")
    assert out["decision"]["i5_13c_authorized"] is False
    assert out["decision"]["i5_13d_authorized"] is False
