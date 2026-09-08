from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_rolling_depth_robustness_qualification import (
    I57C_PHASE,
    build_i5_7c_rolling_depth_robustness_closure,
    validate_i5_7b_for_i5_7c,
)
from tests.i5_7c_helpers import (
    I57A_SHA,
    I57B_SHA,
    I57C_RECEIPT_SHA,
    binding,
    clone,
    passing_i5_7b_final,
    passing_i5_7c_receipt,
)


def _build() -> dict:
    return build_i5_7c_rolling_depth_robustness_closure(
        i5_7a_result=passing_i5_7b_final()["evidence_bindings"] and __import__("tests.i5_7b_helpers", fromlist=["admission"]).admission(),
        i5_7a_binding=binding("/evidence/i5_7a.json", I57A_SHA),
        i5_7b_result=passing_i5_7b_final(),
        i5_7b_binding=binding("/evidence/i5_7b.json", I57B_SHA),
        qualification_receipt=passing_i5_7c_receipt(),
        qualification_receipt_binding=binding("/evidence/i5_7c_receipt.json", I57C_RECEIPT_SHA),
    )


def test_i5_7c_closes_rolling_depth_with_bounded_envelope() -> None:
    result = _build()
    assert result["phase"] == I57C_PHASE
    assert result["pass"] is True
    assert result["quality_status"] == "pass"
    q = result["qualification"]
    assert q["all_six_admitted_current_replay_scenarios_pass"] is True
    assert q["current_frozen_foundation_rolling_depth_path_qualified_within_admitted_envelope"] is True
    assert q["universal_rolling_depth_robustness_claimed"] is False
    assert q["historical_g3_static_sweep_degraded_preserved"] is True
    assert q["historical_g3_motion_rejected_preserved"] is True
    assert result["decision"]["i5_7_complete"] is True
    assert result["decision"]["i5_8_authorized"] is True
    assert result["decision"]["i5_8_started"] is False


def test_i5_7c_preserves_recovered_control_without_inventing_performance() -> None:
    result = _build()
    recovery = result["recovery_provenance"]
    assert recovery["provider_execution_rerun"] is False
    assert recovery["manifest_verified"] is True
    assert recovery["console_log_verified"] is True
    assert result["limitations"]["recovered_control_wall_time_unavailable"] is True
    assert result["limitations"]["recovered_control_resource_metrics_unavailable"] is True
    assert result["limitations"]["runtime_performance_envelope_complete"] is False
    assert result["limitations"]["resource_envelope_complete"] is False


def test_i5_7c_rejects_missing_recovery_provenance() -> None:
    payload = passing_i5_7b_final()
    payload.pop("recovery_provenance")
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7b_for_i5_7c(payload)


def test_i5_7c_rejects_control_rerun_claim() -> None:
    payload = passing_i5_7b_final()
    payload["scope"]["castle_control_provider_rerun"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7b_for_i5_7c(payload)


def test_i5_7c_rejects_fabricated_recovered_control_wall_time() -> None:
    payload = passing_i5_7b_final()
    payload["scenario_records"]["castle_control"]["native_metrics"]["wall_time_s"] = 123.0
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7b_for_i5_7c(payload)


def test_i5_7c_rejects_historical_g3_motion_upgrade() -> None:
    payload = passing_i5_7b_final()
    payload["historical_temporal_anchors_preserved"]["g3_temporal_motion"]["rolling_depth_production_status"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7b_for_i5_7c(payload)


def test_i5_7c_rejects_authority_promotion() -> None:
    payload = passing_i5_7b_final()
    payload["scenario_records"]["castle_dynamic_proxy"]["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7b_for_i5_7c(payload)


def test_i5_7c_rejects_non_pass_current_scenario() -> None:
    payload = passing_i5_7b_final()
    payload["scenario_records"]["castle_symmetry"]["runtime_status"] = "degraded"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7b_for_i5_7c(payload)


def test_i5_7c_never_creates_provider_ranking_or_metric_claim() -> None:
    result = _build()
    assert result["qualification"]["provider_ranking_claimed"] is False
    assert result["qualification"]["universal_provider_winner_claimed"] is False
    assert result["qualification"]["metric_scale_claimed"] is False
    assert result["authority"]["provider_selected"] is False
    assert result["authority"]["provider_promoted"] is False
