from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_failure_robustness_campaign import (
    I511C_FAULT_SCENARIOS,
    build_i5_11c_campaign_header,
    classify_fault_attempt,
    classify_recovery_attempt,
    finalize_i5_11c_campaign,
    validate_i5_11b_for_i5_11c,
    validate_i5_11c_receipt,
)
from tests.i5_11c_helpers import (
    I511A_SHA,
    I511B_SHA,
    POST_I510D_DELTA_SHA,
    binding,
    fault_outcome,
    mutate,
    recovery_outcome,
    synthetic_anchors,
    synthetic_disk,
    synthetic_i511b_preflight,
    synthetic_i511c_receipt,
)


def test_validate_frozen_b_contract() -> None:
    out = validate_i5_11b_for_i5_11c(synthetic_i511b_preflight())
    assert out["decision"]["i5_11c_authorized"] is True
    assert tuple(out["campaign_contract"]["fault_scenarios"]) == I511C_FAULT_SCENARIOS


def test_b_scenario_order_drift_fails_closed() -> None:
    b = mutate(synthetic_i511b_preflight(), "campaign_contract", "fault_scenarios", value=["disk_enospc", "cuda_oom", "crash_orphan_descendant"])
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11b_for_i5_11c(b)


def test_b_real_resource_exhaustion_fails_closed() -> None:
    b = mutate(synthetic_i511b_preflight(), "campaign_contract", "real_resource_exhaustion", value=True)
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11b_for_i5_11c(b)


def test_b_decision_must_authorize_c() -> None:
    b = mutate(synthetic_i511b_preflight(), "decision", "i5_11c_authorized", value=False)
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11b_for_i5_11c(b)


def test_receipt_is_execution_free() -> None:
    out = validate_i5_11c_receipt(synthetic_i511c_receipt())
    assert out["pass"] is True


@pytest.mark.parametrize("field", ["real_campaign_execution", "fault_injection", "supervised_fault_worker_execution"])
def test_receipt_cannot_claim_real_campaign(field: str) -> None:
    receipt = mutate(synthetic_i511c_receipt(), "scope", field, value=True)
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11c_receipt(receipt)


@pytest.mark.parametrize("scenario", I511C_FAULT_SCENARIOS)
def test_expected_fault_attempt_is_accepted(scenario: str) -> None:
    plan = synthetic_i511b_preflight()["fault_plan"][scenario]
    result = classify_fault_attempt(scenario=scenario, plan=plan, outcome=fault_outcome(scenario=scenario))
    assert result["expected_fault_observed"] is True
    assert result["terminal_kind"] == "expected_fault_contained"


def test_unexpected_fault_signature_is_runtime_integrity_failure() -> None:
    scenario = "cuda_oom"
    plan = synthetic_i511b_preflight()["fault_plan"][scenario]
    result = classify_fault_attempt(scenario=scenario, plan=plan, outcome=fault_outcome(scenario=scenario, signature="different failure"))
    assert result["terminal_kind"] == "runtime_integrity_failure"


def test_wrong_crash_exit_code_is_runtime_integrity_failure() -> None:
    scenario = "crash_orphan_descendant"
    plan = synthetic_i511b_preflight()["fault_plan"][scenario]
    result = classify_fault_attempt(scenario=scenario, plan=plan, outcome=fault_outcome(scenario=scenario, exit_code=1))
    assert result["terminal_kind"] == "runtime_integrity_failure"


def test_recovery_pass_requires_completed_clean_preserved_workspace() -> None:
    result = classify_recovery_attempt(outcome=recovery_outcome(completed=True), result_payload={"phase": "I5.11", "mode": "complete", "pass": True, "child_reaped_by_worker": True, "real_gpu_exhaustion": False, "real_filesystem_fill": False})
    assert result["terminal_kind"] == "recovery_pass"


def test_failed_recovery_is_robustness_failure() -> None:
    result = classify_recovery_attempt(outcome=recovery_outcome(completed=False), result_payload=None)
    assert result["terminal_kind"] == "robustness_failure"


def _header_args() -> dict:
    return {
        "i5_11a_binding": binding("/evidence/a.json", I511A_SHA),
        "i5_11b_preflight": synthetic_i511b_preflight(),
        "i5_11b_binding": binding("/evidence/b.json", I511B_SHA),
        "post_i5_10d_delta_binding": binding("/checkpoint/d.tar.gz", POST_I510D_DELTA_SHA),
        "post_i5_11b_delta_binding": binding("/checkpoint/b.tar.gz", "c" * 64),
        "campaign_receipt": synthetic_i511c_receipt(),
        "campaign_receipt_binding": binding("/evidence/c_receipt.json", "d" * 64),
        "repository_anchors_before": synthetic_anchors(),
        "disk_before": synthetic_disk(),
    }


def test_campaign_header_freezes_no_retry_and_no_resource_exhaustion() -> None:
    h = build_i5_11c_campaign_header(**_header_args())
    assert h["campaign_contract"]["retry_until_pass"] is False
    assert h["scope"]["real_gpu_oom"] is False
    assert h["scope"]["real_disk_fill"] is False
    assert h["attempt_accounting"]["authorized_fault_attempts_total"] == 3


def test_campaign_header_rejects_bad_b_sha() -> None:
    args = _header_args()
    args["i5_11b_binding"] = binding("/evidence/b.json", "0" * 64)
    with pytest.raises(ReconstructionValidationError, match="SHA256 drift"):
        build_i5_11c_campaign_header(**args)


def test_campaign_header_rejects_low_disk() -> None:
    args = _header_args()
    args["disk_before"] = synthetic_disk(free_bytes=2 * 1024**3)
    with pytest.raises(ReconstructionValidationError, match="disk safety floor"):
        build_i5_11c_campaign_header(**args)


def test_finalize_pass_authorizes_d() -> None:
    header = build_i5_11c_campaign_header(**_header_args())
    scenarios = [{"scenario": s, "terminal_status": "pass"} for s in I511C_FAULT_SCENARIOS]
    final = finalize_i5_11c_campaign(header, scenario_records=scenarios, terminal_status="pass", terminal_reason="all_scenarios_passed")
    assert final["pass"] is True
    assert final["decision"]["i5_11c_complete"] is True
    assert final["decision"]["i5_11d_authorized"] is True


def test_finalize_robustness_failure_is_valid_evidence_and_authorizes_d() -> None:
    header = build_i5_11c_campaign_header(**_header_args())
    final = finalize_i5_11c_campaign(header, scenario_records=[{"scenario": "cuda_oom", "terminal_status": "robustness_failure"}], terminal_status="robustness_failure", terminal_reason="recovery_failed")
    assert final["pass"] is True
    assert final["decision"]["i5_11d_authorized"] is True
    assert final["campaign_quality_status"] == "robustness_failure_observed"


def test_finalize_runtime_integrity_failure_does_not_authorize_d() -> None:
    header = build_i5_11c_campaign_header(**_header_args())
    final = finalize_i5_11c_campaign(header, scenario_records=[], terminal_status="runtime_integrity_failure", terminal_reason="anchor_drift")
    assert final["pass"] is False
    assert final["decision"]["i5_11c_complete"] is False
    assert final["decision"]["i5_11d_authorized"] is False
