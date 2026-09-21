from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_hybrid_admission import (
    I512A_HYBRID_RULE,
    I512A_PHASE,
    build_i5_12a_hybrid_admission_review,
    evaluate_g2_hybrid_evidence,
    validate_i4_3a_hybrid_rule,
    validate_i5_11d_for_i5_12a,
    validate_i5_12a_receipt,
    validate_i5_2a_for_i5_12a,
    validate_i5_6c_for_i5_12a,
)
from tests.i5_12a_helpers import build_args, clone, g2, i43a_text, i511d, i52a, i56c, receipt


def build(*, justified: bool = False):
    return build_i5_12a_hybrid_admission_review(**build_args(justified=justified))


def test_phase_constant():
    assert I512A_PHASE == "I5.12A"


def test_historical_rule_is_explicit():
    rule = validate_i4_3a_hybrid_rule(i43a_text())
    assert rule["rule_id"] == I512A_HYBRID_RULE
    assert rule["global_completeness_alone_sufficient"] is False


def test_validates_latest_roadmap_boundary():
    assert validate_i5_11d_for_i5_12a(i511d())["pass"] is True


def test_validates_mvs_robustness_context():
    assert validate_i5_6c_for_i5_12a(i56c())["quality_status"] == "pass"


def test_validates_i52a_g2_binding():
    assert validate_i5_2a_for_i5_12a(i52a())["g2_binding"]["sha256"] == "4" * 64


def test_validates_receipt():
    assert validate_i5_12a_receipt(receipt())["pass"] is True


def test_global_completeness_alone_does_not_justify_hybrid():
    evidence = evaluate_g2_hybrid_evidence(g2(justified=False, completeness_da3=True))
    assert evidence["global_completeness_da3_observed"] is True
    assert evidence["ai_legitimate_hole_recovery_observed"] is False
    assert evidence["historical_hybrid_condition_met"] is False


def test_direct_legitimate_hole_recovery_plus_mvs_cleanliness_justifies():
    evidence = evaluate_g2_hybrid_evidence(g2(justified=True))
    assert evidence["mvs_cleaner_observed"] is True
    assert evidence["ai_legitimate_hole_recovery_observed"] is True
    assert evidence["historical_hybrid_condition_met"] is True


def test_default_review_closes_i512_without_experiment_when_not_justified():
    out = build(justified=False)
    assert out["quality_status"] == "hybrid_not_justified"
    assert out["decision"]["i5_12_complete"] is True
    assert out["decision"]["i5_12b_authorized"] is False
    assert out["decision"]["i5_13_authorized"] is True


def test_justified_review_authorizes_only_b():
    out = build(justified=True)
    assert out["quality_status"] == "hybrid_justified"
    assert out["decision"]["i5_12b_authorized"] is True
    assert out["decision"]["i5_12c_authorized"] is False
    assert out["decision"]["hybrid_execution_authorized"] is False


def test_candidate_is_mvs_primary_da3_hole_supplement():
    candidate = build(justified=True)["candidate_hypothesis"]
    assert candidate["provider_pair"] == ["colmap_mvs", "da3"]
    assert candidate["mvs_remains_primary_geometry"] is True
    assert candidate["provider_rerun_authorized_by_i5_12a"] is False
    assert candidate["single_candidate_attempt_if_authorized"] == 1


def test_review_is_execution_free():
    scope = build()["scope"]
    for key in ("provider_execution", "sfm_execution", "mvs_execution", "da3_execution", "hybrid_execution", "matched_view_rendering", "manual_review_rerun"):
        assert scope[key] is False


def test_authority_and_claims_remain_locked():
    out = build()
    assert out["scope"]["provider_selected"] is False
    assert out["scope"]["provider_promoted"] is False
    assert out["scope"]["auto_policy_implemented"] is False
    assert out["scope"]["metric_scale_claimed"] is False
    assert out["comparison_contract"]["global_scalar_score_forbidden"] is True
    assert out["comparison_contract"]["universal_provider_winner_forbidden"] is True


def test_rejects_g2_binding_mismatch():
    args = build_args()
    args["i4_3g2_binding"] = {"path": "/evidence/other.json", "sha256": "4" * 64}
    with pytest.raises(ReconstructionValidationError):
        build_i5_12a_hybrid_admission_review(**args)


def test_rejects_hybrid_already_authorized_upstream():
    bad = clone(i511d())
    bad["decision"]["hybrid_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11d_for_i5_12a(bad)


def test_rejects_mvs_not_qualified():
    bad = clone(i56c())
    bad["qualification"]["current_production_colmap_mvs_path_qualified_within_admitted_envelope"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6c_for_i5_12a(bad)


def test_rejects_missing_direct_hole_dimension():
    bad = g2()
    del bad["scenario_synthesis"]["g2_weak_overlap_mvs_da3"]["provider_preferences"]["legitimate_hole_recovery"]
    with pytest.raises(ReconstructionValidationError):
        evaluate_g2_hybrid_evidence(bad)


def test_rejects_wrong_review_scope():
    bad = g2()
    bad["scenario_synthesis"]["g2_weak_overlap_mvs_da3"]["review_scope"] = "full_scene"
    with pytest.raises(ReconstructionValidationError):
        evaluate_g2_hybrid_evidence(bad)


def test_rejects_global_winner_claim():
    bad = g2()
    bad["scenario_synthesis"]["g2_weak_overlap_mvs_da3"]["global_provider_winner_declared"] = True
    with pytest.raises(ReconstructionValidationError):
        evaluate_g2_hybrid_evidence(bad)


def test_rejects_rule_drift():
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3a_hybrid_rule("hybrid someday")


def test_rejects_receipt_that_claims_hybrid_execution():
    bad = clone(receipt())
    bad["scope"]["hybrid_execution"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_12a_receipt(bad)


def test_comparison_contract_forbids_metric_and_runtime_ranking():
    c = build()["comparison_contract"]
    assert c["metric_scale_comparison_forbidden"] is True
    assert c["runtime_performance_ranking_forbidden"] is True
    assert c["absolute_cross_run_pose_comparison_forbidden"] is True


def test_not_justified_next_goes_to_i513():
    assert build()["next"].startswith("I5.13")


def test_justified_next_goes_to_i512b():
    assert build(justified=True)["next"].startswith("I5.12B")
