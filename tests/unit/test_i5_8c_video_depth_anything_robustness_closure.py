from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_video_depth_anything_robustness_qualification import (
    I58C_PHASE,
    build_i5_8c_video_depth_anything_robustness_closure,
    validate_i5_8b_for_i5_8c,
)
from tests.i5_8c_helpers import (
    CAPACITY_DROPS,
    I58A_SHA,
    I58B_SHA,
    I58C_RECEIPT_SHA,
    binding,
    passing_i5_8a,
    passing_i5_8b_final,
    passing_i5_8c_receipt,
)


def _build() -> dict:
    return build_i5_8c_video_depth_anything_robustness_closure(
        i5_8a_result=passing_i5_8a(),
        i5_8a_binding=binding("/evidence/i5_8a.json", I58A_SHA),
        i5_8b_result=passing_i5_8b_final(),
        i5_8b_binding=binding("/evidence/i5_8b.json", I58B_SHA),
        qualification_receipt=passing_i5_8c_receipt(),
        qualification_receipt_binding=binding("/evidence/i5_8c_receipt.json", I58C_RECEIPT_SHA),
    )


def test_i5_8c_closes_vda_with_degraded_capacity_bounded_envelope() -> None:
    result = _build()
    assert result["phase"] == I58C_PHASE
    assert result["pass"] is True
    assert result["quality_status"] == "degraded"
    q = result["qualification"]
    assert q["current_replay_terminal_scenario_count"] == 6
    assert q["current_replay_degraded_scenario_count"] == 6
    assert q["all_current_registered_frames_alignment_accepted"] is True
    assert q["all_six_current_scenarios_fusion_capacity_degraded"] is True
    assert q["universal_video_depth_anything_robustness_claimed"] is False
    assert result["decision"]["i5_8_complete"] is True
    assert result["decision"]["i5_9_authorized"] is True
    assert result["decision"]["i5_9_started"] is False


def test_i5_8c_preserves_exact_capacity_drop_observations() -> None:
    result = _build()
    assert result["fusion_capacity_observations"] == CAPACITY_DROPS
    assert result["limitations"]["fusion_capacity_degradation_remains_active"] is True
    assert result["limitations"]["production_fusion_capacity_increased"] is False


def test_i5_8c_preserves_historical_motion_degraded_anchor() -> None:
    result = _build()
    q = result["qualification"]
    assert q["historical_g3_static_sweep_pass_preserved"] is True
    assert q["historical_g3_motion_degraded_preserved"] is True
    assert q["historical_g3_low_texture_pass_preserved"] is True


def test_i5_8c_rejects_current_pass_upgrade() -> None:
    payload = passing_i5_8b_final()
    payload["quality_status"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8b_for_i5_8c(payload)


def test_i5_8c_rejects_missing_capacity_reason() -> None:
    payload = passing_i5_8b_final()
    payload["scenario_records"]["castle_control"]["execution_result"]["degradation_reasons"] = []
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8b_for_i5_8c(payload)


def test_i5_8c_rejects_alignment_rejection_hidden_inside_degraded_result() -> None:
    payload = passing_i5_8b_final()
    record = payload["scenario_records"]["castle_weak_parallax"]
    record["native_metrics"]["alignment_rejected_views"] = 1
    record["native_metrics"]["alignment_accepted_fraction"] = 6 / 7
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8b_for_i5_8c(payload)


def test_i5_8c_rejects_capacity_cap_change() -> None:
    payload = passing_i5_8b_final()
    payload["runtime_preflight"]["production_config"]["max_voxels"] = 2000000
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8b_for_i5_8c(payload)


def test_i5_8c_rejects_metric_authority_promotion() -> None:
    payload = passing_i5_8b_final()
    payload["scenario_records"]["castle_dynamic_proxy"]["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8b_for_i5_8c(payload)


def test_i5_8c_rejects_historical_motion_upgrade() -> None:
    payload = passing_i5_8b_final()
    payload["historical_temporal_anchors_preserved"]["g3_temporal_motion"]["video_depth_anything_production_status"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8b_for_i5_8c(payload)


def test_i5_8c_never_creates_provider_ranking_or_metric_claim() -> None:
    result = _build()
    q = result["qualification"]
    assert q["provider_ranking_claimed"] is False
    assert q["universal_provider_winner_claimed"] is False
    assert q["metric_scale_claimed"] is False
    assert result["authority"]["provider_selected"] is False
    assert result["authority"]["provider_promoted"] is False


def test_i5_8c_current_performance_observations_are_not_cross_gpu_qualification() -> None:
    result = _build()
    limits = result["limitations"]
    assert limits["current_six_scenario_performance_observations_complete"] is True
    assert limits["runtime_performance_envelope_complete"] is False
    assert limits["resource_envelope_complete"] is False
    assert limits["rtx_4090_not_qualified"] is True
    assert limits["rtx_5090_not_qualified"] is True
