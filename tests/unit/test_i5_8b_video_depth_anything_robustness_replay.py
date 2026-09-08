from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_video_depth_anything_robustness_replay import (
    I58B_I58A_FROZEN_SHA256,
    I58B_SCENARIO_IDS,
    I58BReplayPolicy,
    analyze_i5_8b_provider_summary,
    build_i5_8b_video_depth_anything_robustness_replay,
    compare_i5_8b_scenarios,
    validate_i5_8a_for_i5_8b,
    validate_i5_8b_receipt,
)
from tests.i5_8b_helpers import build_args, clone, provider_summary, receipt, runtime_preflight, scenario_record


def resources() -> dict:
    return {"peak_vram_mib": 2600.0, "peak_system_ram_used_mib": 9100.0}


def test_i58b_policy_preserves_vda_streaming_semantics() -> None:
    p = I58BReplayPolicy().as_dict()
    assert p["temporal_mode"] == "official_streaming_cache"
    assert p["stateful_temporal_sequence"] is True
    assert p["warmup_reset_before_user_sequence"] is True
    assert p["per_frame_alignment_performed"] is True
    assert p["sequence_global_alignment_performed"] is False
    assert p["runtime_profile_resolution"] == "hardware_resolved_compute_profile_auto"
    assert p["retry_until_pass"] is False


def test_i58b_validates_closed_i58a_admission() -> None:
    admitted = validate_i5_8a_for_i5_8b(build_args()["i5_8a_admission"])
    assert admitted["decision"]["i5_8b_authorized"] is True
    assert admitted["decision"]["i5_8b_started"] is False


def test_i58b_refuses_historical_vda_motion_overwrite() -> None:
    payload = clone(build_args()["i5_8a_admission"])
    payload["historical_temporal_anchors"]["g3_temporal_motion"]["video_depth_anything_production_status"] = "pass"
    with pytest.raises(ReconstructionValidationError, match="g3_temporal_motion"):
        validate_i5_8a_for_i5_8b(payload)


def test_i58b_receipt_accepts_execution_free_gate() -> None:
    assert validate_i5_8b_receipt(receipt())["pass"] is True


def test_i58b_receipt_rejects_nonzero_predecessor_gate() -> None:
    payload = receipt()
    payload["return_codes"]["predecessor_unit"] = 1
    with pytest.raises(ReconstructionValidationError, match="predecessor_unit"):
        validate_i5_8b_receipt(payload)


def test_i58b_analyzes_real_vda_pass_summary() -> None:
    a = analyze_i5_8b_provider_summary(
        provider_summary(), wall_time_s=10.5, resource_observation=resources(), expected_runtime_config=runtime_preflight()["runtime_config"]
    )
    assert a["runtime_status"] == "pass"
    assert a["native_metrics"]["processed_views"] == 19
    assert a["native_metrics"]["alignment_accepted_views"] == 19
    assert a["native_metrics"]["alignment_rejected_views"] == 0
    assert a["native_metrics"]["frame_latency_count"] == 19
    assert a["native_metrics"]["frame_latency_mean_ms"] > 0


def test_i58b_analyzes_provider_native_degradation_without_retuning() -> None:
    a = analyze_i5_8b_provider_summary(
        provider_summary("degraded", rejected_views=2), wall_time_s=11.0, resource_observation=resources(), expected_runtime_config=runtime_preflight()["runtime_config"]
    )
    assert a["runtime_status"] == "degraded"
    assert a["native_metrics"]["alignment_accepted_views"] == 17
    assert a["native_metrics"]["alignment_rejected_views"] == 2
    assert a["failure"] is None
    assert a["degradation_reasons"] == ["alignment_rejected_views:2"]


def test_i58b_refuses_runtime_config_profile_drift() -> None:
    summary = provider_summary()
    summary["runtime_config"]["encoder"] = "vitl"
    with pytest.raises(ReconstructionValidationError, match="runtime config drifted"):
        analyze_i5_8b_provider_summary(summary, wall_time_s=1.0, resource_observation=resources(), expected_runtime_config=runtime_preflight()["runtime_config"])


def test_i58b_refuses_metric_authority_claim_in_nested_execution_result() -> None:
    summary = provider_summary()
    summary["execution_result"]["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError, match="metric_scale_claimed"):
        analyze_i5_8b_provider_summary(summary, wall_time_s=1.0, resource_observation=resources(), expected_runtime_config=runtime_preflight()["runtime_config"])


def test_i58b_refuses_model_intrinsics_promotion() -> None:
    summary = provider_summary()
    summary["authority"]["estimated_model_intrinsics_consumed_for_canonical_projection"] = True
    with pytest.raises(ReconstructionValidationError, match="estimated_model_intrinsics"):
        analyze_i5_8b_provider_summary(summary, wall_time_s=1.0, resource_observation=resources(), expected_runtime_config=runtime_preflight()["runtime_config"])


def test_i58b_refuses_worker_native_metric_promotion() -> None:
    summary = provider_summary()
    summary["worker"]["authority"]["native_metric_depth_promoted_to_canonical_metric_scale"] = True
    with pytest.raises(ReconstructionValidationError, match="native_metric_depth_promoted"):
        analyze_i5_8b_provider_summary(summary, wall_time_s=1.0, resource_observation=resources(), expected_runtime_config=runtime_preflight()["runtime_config"])


def test_i58b_refuses_worker_order_drift() -> None:
    summary = provider_summary()
    summary["worker"]["frame_records"][1]["source_sequence_index"] = 9
    with pytest.raises(ReconstructionValidationError, match="worker frame/source order"):
        analyze_i5_8b_provider_summary(summary, wall_time_s=1.0, resource_observation=resources(), expected_runtime_config=runtime_preflight()["runtime_config"])


def test_i58b_refuses_temporal_timestamp_regression() -> None:
    summary = provider_summary()
    summary["temporal_sequence"]["ordered_timestamps_ns"][2] = summary["temporal_sequence"]["ordered_timestamps_ns"][1]
    with pytest.raises(ReconstructionValidationError, match="timestamps"):
        analyze_i5_8b_provider_summary(summary, wall_time_s=1.0, resource_observation=resources(), expected_runtime_config=runtime_preflight()["runtime_config"])


def test_i58b_requires_native_frame_latency_records() -> None:
    summary = provider_summary()
    summary["worker"]["frame_records"][0].pop("latency_ms")
    with pytest.raises(ReconstructionValidationError, match="latency"):
        analyze_i5_8b_provider_summary(summary, wall_time_s=1.0, resource_observation=resources(), expected_runtime_config=runtime_preflight()["runtime_config"])


def test_i58b_compare_is_descriptive_not_a_quality_score() -> None:
    records = {sid: scenario_record(sid) for sid in I58B_SCENARIO_IDS}
    c = compare_i5_8b_scenarios(records)
    assert set(c) == set(I58B_SCENARIO_IDS)
    assert c["castle_control"]["classification_threshold_used"] is None
    assert c["castle_control"]["global_scalar_score_created"] is False
    assert c["castle_control"]["provider_ranking_created"] is False


def test_i58b_builds_pass_when_all_six_are_pass() -> None:
    out = build_i5_8b_video_depth_anything_robustness_replay(**build_args())
    assert out["phase"] == "I5.8B"
    assert out["pass"] is True
    assert out["quality_status"] == "pass"
    assert out["decision"]["i5_8b_complete"] is True
    assert out["decision"]["i5_8c_authorized"] is True
    assert out["decision"]["i5_8c_started"] is False


def test_i58b_builds_degraded_if_any_stress_case_is_degraded() -> None:
    args = build_args()
    args["scenario_records"]["castle_dynamic_proxy"] = scenario_record("castle_dynamic_proxy", "degraded", rejected_views=2)
    out = build_i5_8b_video_depth_anything_robustness_replay(**args)
    assert out["quality_status"] == "degraded"
    assert out["decision"]["i5_8c_authorized"] is True


def test_i58b_preserves_historical_motion_degraded_anchor() -> None:
    out = build_i5_8b_video_depth_anything_robustness_replay(**build_args())
    assert out["historical_temporal_anchors_preserved"]["g3_temporal_motion"]["video_depth_anything_production_status"] == "degraded"
    assert out["observations"]["historical_motion_degraded_can_be_overwritten_by_current_pass"] is False


def test_i58b_scenario_record_requires_complete_registered_sequence() -> None:
    args = build_args()
    args["scenario_records"]["castle_low_overlap"]["processed_views"] = 5
    with pytest.raises(ReconstructionValidationError, match="complete registered sequence"):
        build_i5_8b_video_depth_anything_robustness_replay(**args)


def test_i58b_scenario_authority_requires_per_frame_not_global_alignment() -> None:
    args = build_args()
    node = args["scenario_records"]["castle_control"]["authority"]
    node["per_frame_alignment_performed"] = False
    node["sequence_global_alignment_performed"] = True
    with pytest.raises(ReconstructionValidationError, match="per_frame_alignment_performed"):
        build_i5_8b_video_depth_anything_robustness_replay(**args)


def test_i58b_rejects_wrong_frozen_i58a_binding_sha() -> None:
    args = build_args()
    args["i5_8a_binding"] = args["i5_8a_binding"].__class__("/evidence/i5_8a.json", "0" * 64)
    with pytest.raises(ReconstructionValidationError, match="SHA256 drifted"):
        build_i5_8b_video_depth_anything_robustness_replay(**args)
    assert I58B_I58A_FROZEN_SHA256 != "0" * 64


def test_i58b_final_never_claims_provider_or_metric_authority() -> None:
    out = build_i5_8b_video_depth_anything_robustness_replay(**build_args())
    assert out["authority"]["provider_selected"] is False
    assert out["authority"]["provider_promoted"] is False
    assert out["authority"]["metric_scale_claimed"] is False
    assert out["authority"]["native_estimated_intrinsics_are_canonical_authority"] is False
    assert out["decision"]["i5_9_started"] is False
