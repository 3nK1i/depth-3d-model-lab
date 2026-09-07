from __future__ import annotations

import json

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mvs_robustness import (
    I56C_PHASE,
    build_i5_6c_mvs_robustness_closure,
    validate_i5_6b_for_i5_6c,
    validate_i5_6c_receipt,
)
from tests.i5_6b_helpers import admission
from tests.i5_6c_helpers import (
    I56A_SHA,
    I56B_SHA,
    I56C_RECEIPT_SHA,
    binding,
    clone,
    degraded_i5_6b,
    passing_i5_6b,
    passing_i5_6c_receipt,
)


def build(*, directional_reduction: bool = True, recovered: bool = True) -> dict:
    return build_i5_6c_mvs_robustness_closure(
        i5_6a_result=admission(),
        i5_6a_binding=binding("/evidence/i5_6a.json", I56A_SHA),
        i5_6b_result=passing_i5_6b(directional_reduction=directional_reduction, recovered=recovered),
        i5_6b_binding=binding("/evidence/i5_6b.json", I56B_SHA),
        qualification_receipt=passing_i5_6c_receipt(),
        qualification_receipt_binding=binding("/evidence/i5_6c_receipt.json", I56C_RECEIPT_SHA),
    )


def test_phase_constant() -> None:
    assert I56C_PHASE == "I5.6C"


def test_passing_recovered_b_is_valid_for_closure() -> None:
    out = validate_i5_6b_for_i5_6c(passing_i5_6b())
    assert out["quality_status"] == "pass"
    assert out["scope"]["interrupted_parent_recovery_finalization"] is True


def test_sorted_json_round_trip_does_not_create_scenario_order_dependency() -> None:
    payload = json.loads(json.dumps(passing_i5_6b(), sort_keys=True))
    assert tuple(payload["scenario_records"]) != tuple(payload["execution_policy"]["scenario_ids"])
    assert validate_i5_6b_for_i5_6c(payload)["quality_status"] == "pass"


def test_receipt_validates() -> None:
    assert validate_i5_6c_receipt(passing_i5_6c_receipt())["pass"] is True


def test_closure_preserves_pass_quality_and_all_six_runtime_passes() -> None:
    q = build()["qualification"]
    assert q["quality_status"] == "pass"
    assert q["runtime_pass_scenario_count"] == 6
    assert q["all_six_admitted_scenarios_runtime_pass"] is True


def test_closure_qualifies_current_colmap_mvs_only_within_admitted_envelope() -> None:
    q = build()["qualification"]
    assert q["current_production_colmap_mvs_path_qualified_within_admitted_envelope"] is True
    assert q["universal_mvs_robustness_claimed"] is False
    assert q["universal_provider_winner_claimed"] is False
    assert q["provider_ranking_claimed"] is False


def test_closure_records_directional_native_support_reductions_without_threshold() -> None:
    result = build()
    d = result["directional_native_support_evidence"]
    assert d["directional_native_support_reduction_observed"] is True
    assert "castle_symmetry" in d["directional_native_support_reduction_scenarios"]
    assert d["classification_threshold_used"] is None
    assert d["global_scalar_score_created"] is False
    assert d["directional_comparison_only"] is True


def test_closure_operating_envelope_names_directional_reductions() -> None:
    q = build()["qualification"]
    assert q["operating_envelope"] == "qualified_on_all_six_admitted_castle_mvs_scenarios_with_directional_native_support_reductions"


def test_closure_without_directional_reduction_does_not_invent_one() -> None:
    result = build(directional_reduction=False)
    assert result["directional_native_support_evidence"]["directional_native_support_reduction_observed"] is False
    assert result["qualification"]["operating_envelope"] == "qualified_on_all_six_admitted_castle_mvs_scenarios"


def test_closure_preserves_recovery_provenance_without_quality_upgrade() -> None:
    result = build()
    assert result["scope"]["interrupted_recovery_evidence_reused"] is True
    assert result["recovery_provenance"]["interrupted_parent_exit_code"] == 130
    assert result["provenance"]["worker_termination_cause_resolved"] is False
    assert result["provenance"]["i5_6b_quality_status_preserved"] is True


def test_closure_preserves_missing_recovered_wall_time_as_limitation() -> None:
    limitations = build()["limitations"]
    assert limitations["recovered_dynamic_proxy_parent_wall_time_unavailable"] is True
    assert limitations["runtime_performance_envelope_complete"] is False
    assert limitations["resource_envelope_complete"] is False


def test_closure_does_not_claim_runtime_or_resource_robustness() -> None:
    q = build()["qualification"]
    assert q["runtime_performance_qualification_claimed"] is False
    assert q["resource_robustness_qualification_claimed"] is False


def test_closure_is_evidence_only() -> None:
    scope = build()["scope"]
    assert scope["evidence_only_qualification_and_closure"] is True
    for key in ("sfm_rerun", "mvs_rerun", "patch_match_rerun", "stereo_fusion_rerun"):
        assert scope[key] is False


def test_closure_preserves_authority_guards() -> None:
    authority = build()["authority"]
    assert authority["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert authority["permanent_global_view_cap"] is None
    assert authority["metric_scale_claimed"] is False
    assert authority["provider_promoted"] is False


def test_closure_completes_i56_and_authorizes_i57_only() -> None:
    decision = build()["decision"]
    assert decision["i5_6_complete"] is True
    assert decision["i5_7_authorized"] is True
    assert decision["i5_7_started"] is False
    assert decision["mp_sfm_authorized"] is False
    assert decision["auto_policy_implemented"] is False
    assert decision["hybrid_authorized"] is False


def test_degraded_b_stays_degraded_in_closure() -> None:
    result = build_i5_6c_mvs_robustness_closure(
        i5_6a_result=admission(),
        i5_6a_binding=binding("/evidence/i5_6a.json", I56A_SHA),
        i5_6b_result=degraded_i5_6b(),
        i5_6b_binding=binding("/evidence/i5_6b.json", I56B_SHA),
        qualification_receipt=passing_i5_6c_receipt(),
        qualification_receipt_binding=binding("/evidence/i5_6c_receipt.json", I56C_RECEIPT_SHA),
    )
    assert result["quality_status"] == "degraded"
    assert result["qualification"]["operating_envelope"] == "degraded_on_admitted_castle_mvs_envelope"
    assert result["qualification"]["current_production_colmap_mvs_path_qualified_within_admitted_envelope"] is False


def test_validator_rejects_quality_upgrade() -> None:
    payload = clone(passing_i5_6b())
    payload["quality_status"] = "degraded"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6b_for_i5_6c(payload)


def test_validator_rejects_recovery_rerun_claim() -> None:
    payload = clone(passing_i5_6b())
    payload["recovery_provenance"]["mvs_rerun"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6b_for_i5_6c(payload)


def test_validator_rejects_invented_recovered_wall_time() -> None:
    payload = clone(passing_i5_6b())
    payload["scenario_records"]["castle_dynamic_proxy"]["native_metrics"]["wall_time_s"] = 123.0
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6b_for_i5_6c(payload)


def test_validator_rejects_cross_run_absolute_geometry_claim() -> None:
    payload = clone(passing_i5_6b())
    payload["control_relative_comparisons"]["castle_symmetry"]["cross_run_absolute_geometry_comparison"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6b_for_i5_6c(payload)


def test_validator_rejects_early_provider_promotion() -> None:
    payload = clone(passing_i5_6b())
    payload["authority"]["provider_promoted"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6b_for_i5_6c(payload)


def test_receipt_rejects_mvs_rerun() -> None:
    payload = clone(passing_i5_6c_receipt())
    payload["scope"]["mvs_rerun"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6c_receipt(payload)
