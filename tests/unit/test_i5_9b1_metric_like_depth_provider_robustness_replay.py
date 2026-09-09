from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_remaining_depth_provider_robustness_replay import (
    I59B1_EXPECTED_RUN_COUNT,
    I59B1_PROVIDERS,
    I59B1_SCENARIO_IDS,
    I59B1ReplayPolicy,
    build_i5_9b1_replay,
    validate_i5_9a_for_i5_9b1,
    validate_i5_9b1_provider_scenario_record,
    validate_i5_9b1_runtime_preflight,
)
from tests.i5_9b1_helpers import admission, build_replay_args, clone, preflight, record


def test_b1_provider_and_scenario_matrix_is_exactly_frozen():
    assert I59B1_PROVIDERS == ("metric3d_v2", "unidepth_v2", "moge_2", "depth_pro")
    assert len(I59B1_SCENARIO_IDS) == 6
    assert I59B1_EXPECTED_RUN_COUNT == 24


def test_policy_preserves_existing_single_image_defaults():
    policy = I59B1ReplayPolicy().as_dict()
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
        I59B1ReplayPolicy(production_max_voxels=2_000_000)


def test_i59a_authorizes_b1_and_keeps_b2_unstarted():
    result = validate_i5_9a_for_i5_9b1(admission())
    assert result["decision"]["i5_9b1_authorized"] is True
    assert result["decision"]["i5_9b1_started"] is False
    assert result["decision"]["i5_9b2_started"] is False


def test_i59a_provider_alignment_drift_is_rejected():
    payload = admission()
    payload["provider_inventory"]["moge_2"]["alignment_semantics"] = "sequence_global_affine"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9a_for_i5_9b1(payload)


def test_preflight_requires_all_24_provider_scenario_plans():
    result = validate_i5_9b1_runtime_preflight(preflight())
    assert result["pass"] is True
    assert sum(len(node["providers"]) for node in result["scenario_plans"].values()) == 24


def test_preflight_cannot_start_provider_inference():
    payload = preflight()
    payload["scope"]["provider_inference_started"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b1_runtime_preflight(payload)


def test_terminal_pass_record_preserves_capacity_as_evidence_only():
    payload = record("castle_control", "metric3d_v2", status="pass")
    result = validate_i5_9b1_provider_scenario_record(payload)
    assert result["fusion_capacity_drops"] == 123
    assert result["fusion_capacity_drops_change_status"] is False
    assert result["runtime_status"] == "pass"


def test_terminal_degraded_record_is_valid_without_becoming_failure():
    payload = record("castle_symmetry", "depth_pro", status="degraded")
    assert validate_i5_9b1_provider_scenario_record(payload)["runtime_status"] == "degraded"


def test_terminal_recognized_quality_rejection_is_valid():
    payload = record("castle_low_overlap", "unidepth_v2", status="rejected")
    result = validate_i5_9b1_provider_scenario_record(payload)
    assert result["rejection"]["recognized_existing_quality_gate"] is True


def test_nonterminal_runtime_failure_is_not_valid_scientific_outcome():
    payload = record("castle_control", "moge_2")
    payload["runtime_status"] = "failed"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b1_provider_scenario_record(payload)


def test_build_replay_closes_b1_and_authorizes_b2_after_24_terminal_runs():
    result = build_i5_9b1_replay(**build_replay_args())
    assert result["pass"] is True
    assert result["quality_status"] == "pass"
    assert result["terminal_provider_scenario_runs"] == 24
    assert result["decision"]["i5_9b1_complete"] is True
    assert result["decision"]["i5_9b2_authorized"] is True
    assert result["decision"]["i5_9b2_started"] is False
    assert result["next"].startswith("I5.9B2")


def test_build_replay_preserves_mixed_terminal_quality_without_retrying():
    result = build_i5_9b1_replay(
        **build_replay_args(overrides={
            ("castle_low_overlap", "unidepth_v2"): "rejected",
            ("castle_symmetry", "depth_pro"): "degraded",
        })
    )
    assert result["pass"] is True
    assert result["quality_status"] == "mixed_terminal_quality"
    assert result["terminal_status_counts"] == {"pass": 22, "degraded": 1, "rejected": 1}
    assert result["scope"]["retry_until_pass"] is False


def test_missing_provider_scenario_record_fails_closed():
    args = build_replay_args()
    del args["records"]["castle_control"]["depth_pro"]
    with pytest.raises(ReconstructionValidationError):
        build_i5_9b1_replay(**args)
