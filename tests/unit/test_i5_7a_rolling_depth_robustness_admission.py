from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_rolling_depth_robustness import (
    I57A_FROZEN_G3_ROLLING_DEPTH_STATUSES,
    I57A_SCENARIO_IDS,
    I57AAdmissionPolicy,
    build_i5_7a_rolling_depth_robustness_admission,
    validate_i4_3g3c_for_i5_7,
    validate_i5_6c_for_i5_7,
    validate_i5_7a_receipt,
)
from tests.i5_7a_helpers import build_args, clone, g3c_closure, i56c_closure, receipt


def test_policy_freezes_existing_rolling_depth_production_contract() -> None:
    policy = I57AAdmissionPolicy().as_dict()
    assert policy["provider_id"] == "rolling-depth-temporal"
    assert policy["provider_implementation"] == "rolling_depth"
    assert policy["stateful_temporal_sequence"] is True
    assert policy["source_order_required"] is True
    assert policy["sequence_alignment"] == "one_global_affine_to_authoritative_sfm"
    assert policy["per_frame_alignment"] is False
    assert policy["default_device"] == "cuda:0"
    assert policy["default_precision"] == "fp16"
    assert policy["production_config"]["processing_resolution"] == 512
    assert policy["production_config"]["max_vae_batch_size"] == 1
    assert policy["production_config"]["dilations"] == [1, 25]
    assert policy["permanent_global_view_cap"] is None


def test_policy_refuses_result_driven_retuning_or_view_cap() -> None:
    with pytest.raises(ReconstructionValidationError):
        I57AAdmissionPolicy(result_driven_retuning=True)
    with pytest.raises(ReconstructionValidationError):
        I57AAdmissionPolicy(permanent_global_view_cap=12)


def test_i56c_must_explicitly_authorize_i57() -> None:
    payload = i56c_closure()
    payload["decision"]["i5_7_authorized"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6c_for_i5_7(payload)


def test_i56c_authority_drift_fails_closed() -> None:
    payload = i56c_closure()
    payload["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6c_for_i5_7(payload)


def test_g3c_frozen_rolling_depth_statuses_are_immutable() -> None:
    payload = validate_i4_3g3c_for_i5_7(g3c_closure())
    observed = {
        sid: payload["scenario_synthesis"][sid]["production_statuses"]["rolling_depth"]
        for sid in I57A_FROZEN_G3_ROLLING_DEPTH_STATUSES
    }
    assert observed == I57A_FROZEN_G3_ROLLING_DEPTH_STATUSES


def test_g3c_status_rewrite_fails_closed() -> None:
    payload = g3c_closure()
    payload["scenario_synthesis"]["g3_temporal_motion"]["production_statuses"]["rolling_depth"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3g3c_for_i5_7(payload)


def test_receipt_rejects_hidden_execution() -> None:
    payload = receipt()
    payload["scope"]["rolling_depth_inference"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7a_receipt(payload)


def test_build_admits_exact_six_castle_scenarios() -> None:
    result = build_i5_7a_rolling_depth_robustness_admission(**build_args())
    assert tuple(result["scenario_matrix"]) == I57A_SCENARIO_IDS
    assert all(node["admitted_for_i5_7b"] for node in result["scenario_matrix"].values())
    assert all(node["rolling_depth_execution_required"] for node in result["scenario_matrix"].values())


def test_low_overlap_is_reinterpreted_for_temporal_semantics() -> None:
    result = build_i5_7a_rolling_depth_robustness_admission(**build_args())
    row = result["scenario_matrix"]["castle_low_overlap"]
    assert row["scientific_interpretation"] == "temporal_support_spacing_stress_not_mvs_overlap_equivalence"
    assert row["mvs_specific_metrics_are_rolling_depth_criteria"] is False


def test_symmetry_scope_preserves_partial_registered_authority() -> None:
    result = build_i5_7a_rolling_depth_robustness_admission(**build_args())
    row = result["scenario_matrix"]["castle_symmetry"]
    assert row["input_images"] == 19
    assert row["registered_images"] == 12
    assert row["source_registered_scope_only"] is True


def test_upstream_exclusions_are_not_provider_failures() -> None:
    result = build_i5_7a_rolling_depth_robustness_admission(**build_args())
    exclusions = result["excluded_or_deferred_scenarios"]
    assert exclusions["castle_blur"]["status"] == "BLOCKED_UPSTREAM"
    assert exclusions["castle_underexposure"]["status"] == "BLOCKED_UPSTREAM"
    assert exclusions["kitti_forward_motion"]["status"] == "BLOCKED_UPSTREAM"
    assert exclusions["castle_clutter"]["status"] == "NOT_FORMALLY_ADMITTED"
    assert exclusions["difficult_handheld_horse"]["status"] == "DEFERRED_NOT_ADMITTED"
    assert all(node["rolling_depth_failure"] is False for node in exclusions.values())


def test_admission_is_execution_free_and_authorizes_only_i57b() -> None:
    result = build_i5_7a_rolling_depth_robustness_admission(**build_args())
    assert result["scope"]["rolling_depth_inference"] is False
    assert result["scope"]["sfm_rerun"] is False
    assert result["scope"]["mvs_rerun"] is False
    assert result["scope"]["provider_comparison_started"] is False
    assert result["decision"]["i5_7a_complete"] is True
    assert result["decision"]["i5_7b_authorized"] is True
    assert result["decision"]["i5_7b_started"] is False
    assert result["decision"]["i5_8_started"] is False


def test_authority_and_metric_claims_remain_locked() -> None:
    result = build_i5_7a_rolling_depth_robustness_admission(**build_args())
    authority = result["authority"]
    assert authority["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert authority["rolling_depth_pose_authority"] is False
    assert authority["metric_scale_claimed"] is False
    assert authority["measurement_authoritative"] is False
    assert authority["certified_metrology"] is False
    assert authority["permanent_global_view_cap"] is None
