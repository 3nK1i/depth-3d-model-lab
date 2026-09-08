from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_rolling_depth_robustness_replay import (
    I57BReplayPolicy,
    I57B_PHASE,
    I57B_SCENARIO_IDS,
    analyze_i5_7b_provider_summary,
    build_i5_7b_recovered_control_record,
    build_i5_7b_rolling_depth_robustness_replay,
    compare_i5_7b_scenarios,
    validate_i5_7a_for_i5_7b,
    validate_i5_7b_receipt,
)
from tests.i5_7b_helpers import (
    admission, build_args, provider_summary, receipt, recovery_artifact_bindings,
    recovery_provenance, scenario_record,
)


def test_policy_freezes_existing_production_defaults_and_all_registered_sequence():
    policy = I57BReplayPolicy()
    assert policy.provider_id == "rolling-depth-temporal"
    assert policy.device == "cuda:0"
    assert policy.precision == "fp16"
    assert policy.processing_resolution == 512
    assert policy.dilations == (1, 25)
    assert policy.max_vae_batch_size == 1
    assert policy.all_registered_views is True
    assert policy.adaptive_view_reduction is False
    assert policy.sequential_execution is True
    assert policy.permanent_global_view_cap is None
    assert policy.result_driven_retuning is False


def test_validate_i57a_requires_exact_authorized_six_scenario_contract():
    out = validate_i5_7a_for_i5_7b(admission())
    assert tuple(out["execution_policy"]["scenario_ids"]) == I57B_SCENARIO_IDS
    assert set(out["scenario_matrix"]) == set(I57B_SCENARIO_IDS)
    bad = admission()
    bad["decision"]["i5_7b_authorized"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7a_for_i5_7b(bad)


def test_validate_i57a_rejects_sfm_rerun_or_provider_comparison_authority_drift():
    bad = admission()
    bad["scenario_matrix"]["castle_control"]["sfm_rerun_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7a_for_i5_7b(bad)
    bad = admission()
    bad["authority"]["provider_ranking_created"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7a_for_i5_7b(bad)


def test_receipt_requires_execution_free_validation_only():
    assert validate_i5_7b_receipt(receipt())["pass"] is True
    bad = receipt()
    bad["scope"]["rolling_depth_inference"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7b_receipt(bad)


def test_provider_summary_accepts_real_dense_execution_result_nested_authority():
    summary = provider_summary("pass", views=19)
    execution = summary["execution_result"]
    assert "metric_scale_claimed" not in execution
    assert execution["authority"]["metric_scale_claimed"] is False
    out = analyze_i5_7b_provider_summary(
        summary, wall_time_s=1.0, resource_observation={}
    )
    assert out["runtime_status"] == "pass"


def test_recovered_control_preserves_missing_wrapper_performance_without_rerun():
    record = build_i5_7b_recovered_control_record(
        provider_summary=provider_summary("pass", views=19),
        artifact_bindings=recovery_artifact_bindings(),
        recovery_provenance=recovery_provenance(),
    )
    assert record["scenario_id"] == "castle_control"
    assert record["provider_execution_terminal"] is True
    assert record["recovery"]["provider_execution_rerun"] is False
    assert record["native_metrics"]["wall_time_s"] is None
    assert record["native_metrics"]["peak_vram_mib"] is None
    assert record["native_metrics"]["peak_system_ram_used_mib"] is None
    assert record["resource_observation"]["missing_metrics_preserved_as_null"] is True


def test_recovered_control_rejects_invented_performance_or_wrong_failure_identity():
    bad = recovery_provenance()
    bad["original_exception_message"] = "different failure"
    with pytest.raises(ReconstructionValidationError):
        build_i5_7b_recovered_control_record(
            provider_summary=provider_summary("pass", views=19),
            artifact_bindings=recovery_artifact_bindings(),
            recovery_provenance=bad,
        )

    with pytest.raises(ReconstructionValidationError):
        analyze_i5_7b_provider_summary(
            provider_summary("pass", views=19),
            wall_time_s=1.0,
            resource_observation={},
            allow_missing_performance_observation=True,
        )


def test_provider_summary_extracts_native_alignment_fusion_and_resources():
    out = analyze_i5_7b_provider_summary(
        provider_summary("pass", views=19, residual=0.12),
        wall_time_s=9.25,
        resource_observation={"peak_vram_mib": 1536.0, "peak_system_ram_used_mib": 7000.0},
    )
    assert out["runtime_status"] == "pass"
    metrics = out["native_metrics"]
    assert metrics["processed_views"] == 19
    assert metrics["alignment_accepted"] is True
    assert metrics["retained_correspondences"] == 80
    assert metrics["alignment_median_relative_residual"] == pytest.approx(0.12)
    assert metrics["accepted_fusion_views"] == 19
    assert metrics["canonical_point_count"] == 10019
    assert metrics["wall_time_s"] == pytest.approx(9.25)
    assert metrics["peak_vram_mib"] == pytest.approx(1536.0)
    assert metrics["peak_system_ram_used_mib"] == pytest.approx(7000.0)


def test_provider_summary_preserves_degraded_and_quality_terminal_rejected():
    degraded = analyze_i5_7b_provider_summary(provider_summary("degraded", views=12, accepted_views=10), wall_time_s=2.0, resource_observation={})
    assert degraded["runtime_status"] == "degraded"
    rejected = analyze_i5_7b_provider_summary(provider_summary("rejected", views=7), wall_time_s=2.0, resource_observation={})
    assert rejected["runtime_status"] == "rejected"
    assert rejected["failure"]["quality_terminal"] is True
    assert rejected["native_metrics"]["canonical_point_count"] is None


def test_provider_summary_rejects_config_or_authority_drift():
    bad = provider_summary("pass")
    bad["config"]["processing_resolution"] = 768
    with pytest.raises(ReconstructionValidationError):
        analyze_i5_7b_provider_summary(bad, wall_time_s=1.0, resource_observation={})
    bad = provider_summary("pass")
    bad["authority"]["per_frame_alignment_performed"] = True
    with pytest.raises(ReconstructionValidationError):
        analyze_i5_7b_provider_summary(bad, wall_time_s=1.0, resource_observation={})
    bad = provider_summary("pass")
    bad["execution_result"]["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        analyze_i5_7b_provider_summary(bad, wall_time_s=1.0, resource_observation={})


def test_control_relative_comparison_is_directional_only_and_not_a_score():
    records = {sid: scenario_record(sid) for sid in I57B_SCENARIO_IDS}
    records["castle_weak_parallax"] = scenario_record("castle_weak_parallax", residual=0.20)
    out = compare_i5_7b_scenarios(records)
    weak = out["castle_weak_parallax"]
    assert weak["alignment_median_residual_ratio_to_control"] == pytest.approx(2.0)
    assert weak["classification_threshold_used"] is None
    assert weak["global_scalar_score_created"] is False
    assert weak["cross_run_absolute_geometry_comparison"] is False
    assert weak["cross_run_scale_shift_comparison"] is False


def test_build_replay_passes_when_all_six_are_terminal_pass():
    result = build_i5_7b_rolling_depth_robustness_replay(**build_args())
    assert result["phase"] == I57B_PHASE
    assert result["pass"] is True
    assert result["quality_status"] == "pass"
    assert result["decision"]["i5_7b_complete"] is True
    assert result["decision"]["i5_7c_authorized"] is True
    assert result["scope"]["sfm_rerun"] is False
    assert result["scope"]["retry_until_pass"] is False


def test_build_replay_degraded_when_stress_is_degraded_or_rejected():
    args = build_args()
    args["scenario_records"]["castle_dynamic_proxy"] = scenario_record("castle_dynamic_proxy", "rejected")
    result = build_i5_7b_rolling_depth_robustness_replay(**args)
    assert result["pass"] is True
    assert result["quality_status"] == "degraded"
    assert result["decision"]["i5_7c_authorized"] is True


def test_build_replay_rejected_when_control_is_rejected_but_evidence_is_complete():
    args = build_args()
    args["scenario_records"]["castle_control"] = scenario_record("castle_control", "rejected")
    result = build_i5_7b_rolling_depth_robustness_replay(**args)
    assert result["pass"] is True
    assert result["quality_status"] == "rejected"
    assert result["decision"]["i5_7c_authorized"] is True


def test_build_replay_rejects_missing_scenario_or_authority_mutation():
    args = build_args()
    args["scenario_records"].pop("castle_symmetry")
    with pytest.raises(ReconstructionValidationError):
        build_i5_7b_rolling_depth_robustness_replay(**args)
    args = build_args()
    args["scenario_records"]["castle_control"]["authority"]["authoritative_sfm_modified"] = True
    with pytest.raises(ReconstructionValidationError):
        build_i5_7b_rolling_depth_robustness_replay(**args)


def test_replay_output_does_not_promote_provider_metric_scale_or_auto():
    out = build_i5_7b_rolling_depth_robustness_replay(**build_args())
    assert out["authority"]["provider_promoted"] is False
    assert out["authority"]["provider_ranking_created"] is False
    assert out["authority"]["metric_scale_claimed"] is False
    assert out["authority"]["certified_metrology"] is False
    assert out["decision"]["auto_policy_implemented"] is False
    assert out["decision"]["hybrid_authorized"] is False
    assert out["decision"]["mp_sfm_authorized"] is False
