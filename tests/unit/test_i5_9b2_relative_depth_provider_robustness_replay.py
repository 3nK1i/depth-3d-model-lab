from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_relative_depth_provider_robustness_replay import (
    I59B2_EXPECTED_RUN_COUNT,
    I59B2_PROVIDERS,
    I59B2_SCENARIO_IDS,
    I59B2ReplayPolicy,
    build_i5_9b2_replay,
    validate_i5_9a_for_i5_9b2,
    validate_i5_9b1_for_i5_9b2,
    validate_i5_9b2_provider_scenario_record,
    validate_i5_9b2_runtime_preflight,
)
from tests.i5_9b2_helpers import admission, b1_replay, build_replay_args, preflight, record


def test_b2_provider_and_scenario_matrix_is_exactly_frozen():
    assert I59B2_PROVIDERS == ("depth_anything_v2", "marigold")
    assert len(I59B2_SCENARIO_IDS) == 6
    assert I59B2_EXPECTED_RUN_COUNT == 12


def test_policy_preserves_existing_relative_depth_defaults():
    policy = I59B2ReplayPolicy().as_dict()
    assert policy["production_max_voxels"] == 1_500_000
    assert policy["minimum_accepted_views"] == 2
    assert policy["minimum_correspondences"] == 8
    assert policy["maximum_relative_residual"] == 0.35
    assert policy["fusion_capacity_drops_change_single_image_status"] is False
    assert policy["result_driven_retuning"] is False
    assert policy["retry_until_pass"] is False
    assert policy["provider_comparison"] is False


def test_policy_refuses_capacity_tuning():
    with pytest.raises(ReconstructionValidationError):
        I59B2ReplayPolicy(production_max_voxels=2_000_000)


def test_i59a_preserves_b2_alignment_semantics_and_historical_rejections():
    result = validate_i5_9a_for_i5_9b2(admission())
    assert result["provider_inventory"]["depth_anything_v2"]["alignment_semantics"] == "per_view_affine_inverse_depth_to_authoritative_sfm"
    assert result["provider_inventory"]["marigold"]["alignment_semantics"] == "per_view_affine_native_domain_to_authoritative_sfm"
    assert result["historical_g4_anchors"]["low_texture"]["depth_anything_v2"] == "rejected"
    assert result["historical_g4_anchors"]["low_texture"]["marigold"] == "rejected"


def test_i59a_historical_rejection_drift_is_rejected():
    payload = admission()
    payload["historical_g4_anchors"]["low_texture"]["marigold"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9a_for_i5_9b2(payload)


def test_b1_prerequisite_must_be_closed_clean_pass_and_authorize_b2():
    result = validate_i5_9b1_for_i5_9b2(b1_replay())
    assert result["terminal_status_counts"] == {"pass": 24, "degraded": 0, "rejected": 0}
    assert result["decision"]["i5_9b1_complete"] is True
    assert result["decision"]["i5_9b2_authorized"] is True


def test_b1_prerequisite_quality_drift_fails_closed():
    payload = b1_replay()
    payload["terminal_status_counts"] = {"pass": 23, "degraded": 1, "rejected": 0}
    payload["quality_status"] = "mixed_terminal_quality"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b1_for_i5_9b2(payload)


def test_b1_prerequisite_record_matrix_must_match_frozen_24_pass_outcome():
    payload = b1_replay()
    payload["records"]["castle_control"]["depth_pro"]["runtime_status"] = "degraded"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b1_for_i5_9b2(payload)


def test_preflight_requires_all_12_provider_scenario_plans():
    result = validate_i5_9b2_runtime_preflight(preflight())
    assert result["pass"] is True
    assert sum(len(node["providers"]) for node in result["scenario_plans"].values()) == 12


def test_preflight_requires_b1_evidence_binding():
    payload = preflight()
    del payload["evidence_bindings"]["i5_9b1_replay"]
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_runtime_preflight(payload)


def test_preflight_cannot_start_provider_inference():
    payload = preflight()
    payload["scope"]["provider_inference_started"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_runtime_preflight(payload)


def test_terminal_pass_record_preserves_capacity_as_evidence_only():
    payload = record("castle_control", "depth_anything_v2", status="pass")
    result = validate_i5_9b2_provider_scenario_record(payload)
    assert result["fusion_capacity_drops"] == 123
    assert result["fusion_capacity_drops_change_status"] is False
    assert result["runtime_status"] == "pass"


def test_terminal_degraded_record_is_valid_without_becoming_failure():
    payload = record("castle_symmetry", "marigold", status="degraded")
    assert validate_i5_9b2_provider_scenario_record(payload)["runtime_status"] == "degraded"


def test_terminal_recognized_quality_rejection_is_valid():
    payload = record("castle_low_overlap", "depth_anything_v2", status="rejected")
    result = validate_i5_9b2_provider_scenario_record(payload)
    assert result["rejection"]["recognized_existing_quality_gate"] is True


def test_nonterminal_runtime_failure_is_not_valid_scientific_outcome():
    payload = record("castle_control", "marigold")
    payload["runtime_status"] = "failed"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_provider_scenario_record(payload)


def test_build_replay_closes_b2_after_12_terminal_runs():
    result = build_i5_9b2_replay(**build_replay_args())
    assert result["pass"] is True
    assert result["quality_status"] == "pass"
    assert result["terminal_provider_scenario_runs"] == 12
    assert result["historical_g4_low_texture_anchor"]["preserved"] is True
    assert result["decision"]["i5_9b2_complete"] is True
    assert result["decision"]["i5_9c_authorized"] is True
    assert result["decision"]["i5_9c_started"] is False


def test_build_replay_preserves_mixed_current_quality_and_historical_rejections():
    result = build_i5_9b2_replay(
        **build_replay_args(overrides={
            ("castle_low_overlap", "depth_anything_v2"): "rejected",
            ("castle_symmetry", "marigold"): "degraded",
        })
    )
    assert result["pass"] is True
    assert result["quality_status"] == "mixed_terminal_quality"
    assert result["terminal_status_counts"] == {"pass": 10, "degraded": 1, "rejected": 1}
    assert result["historical_g4_low_texture_anchor"]["depth_anything_v2"] == "rejected"
    assert result["historical_g4_low_texture_anchor"]["marigold"] == "rejected"
    assert result["scope"]["retry_until_pass"] is False


def test_missing_provider_scenario_record_fails_closed():
    args = build_replay_args()
    del args["records"]["castle_control"]["marigold"]
    with pytest.raises(ReconstructionValidationError):
        build_i5_9b2_replay(**args)
