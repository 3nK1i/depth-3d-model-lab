from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_failure_robustness import (
    I511A_PHASE,
    I511A_POST_I510D_DELTA_FROZEN_SHA256,
    I511AFailureAdmissionPolicy,
    build_i5_11a_failure_mode_admission,
    repository_failure_robustness_inventory,
    validate_i5_10d_for_i5_11a,
)
from tests.i5_11a_helpers import (
    I510D_SHA,
    POST_I510D_DELTA_SHA,
    binding,
    mutate,
    synthetic_i510d_closure,
    synthetic_i511a_receipt,
)


def _build() -> dict:
    return build_i5_11a_failure_mode_admission(
        i5_10d_closure=synthetic_i510d_closure(),
        i5_10d_binding=binding("/evidence/i5_10d.json", I510D_SHA),
        post_i5_10d_delta_binding=binding("/checkpoints/post_i5_10d_delta.tar.gz", POST_I510D_DELTA_SHA),
        admission_receipt=synthetic_i511a_receipt(),
        admission_receipt_binding=binding("/evidence/i5_11a_receipt.json", "a" * 64),
        repository_anchors={"x": {"verified": True, "sha256": "b" * 64, "path": "/repo/x"}},
    )


def test_phase_and_checkpoint_constants_are_frozen() -> None:
    assert I511A_PHASE == "I5.11A"
    assert I511A_POST_I510D_DELTA_FROZEN_SHA256 == POST_I510D_DELTA_SHA


def test_repository_inventory_records_existing_recovery_architecture() -> None:
    inv = repository_failure_robustness_inventory()
    assert inv["whole_engine_supervisor"]["process_group_boundary"] is True
    assert inv["whole_engine_supervisor"]["failed_workspace_cleanup"] is True
    assert inv["whole_engine_supervisor"]["safe_for_recovery_contract"] is True
    assert inv["runtime_failure_classification"]["cuda_oom_first_class"] is True
    assert inv["runtime_failure_classification"]["worker_crash_first_class"] is True
    assert inv["runtime_failure_classification"]["disk_enospc_first_class"] is False


def test_policy_admits_only_three_bounded_faults_and_clean_recovery() -> None:
    policy = I511AFailureAdmissionPolicy()
    assert policy.fault_scenarios == ("cuda_oom", "disk_enospc", "crash_orphan_descendant")
    assert policy.fault_attempts_each == 1
    assert policy.clean_recovery_after_each is True
    assert policy.actual_gpu_exhaustion is False
    assert policy.actual_root_filesystem_fill is False


def test_policy_rejects_real_gpu_exhaustion() -> None:
    with pytest.raises(ReconstructionValidationError, match="real GPU exhaustion"):
        I511AFailureAdmissionPolicy(actual_gpu_exhaustion=True)


def test_policy_rejects_real_root_disk_fill() -> None:
    with pytest.raises(ReconstructionValidationError, match="root filesystem"):
        I511AFailureAdmissionPolicy(actual_root_filesystem_fill=True)


def test_i510d_validation_requires_i511_authorization() -> None:
    payload = mutate(synthetic_i510d_closure(), "decision", "i5_11_authorized", value=False)
    with pytest.raises(ReconstructionValidationError, match="i5_11_authorized"):
        validate_i5_10d_for_i5_11a(payload)


def test_build_admission_is_execution_free_and_authorizes_only_b() -> None:
    out = _build()
    assert out["pass"] is True
    assert out["scope"]["fault_injection"] is False
    assert out["scope"]["real_gpu_oom"] is False
    assert out["scope"]["real_disk_fill"] is False
    assert out["decision"]["i5_11a_complete"] is True
    assert out["decision"]["i5_11b_authorized"] is True
    assert out["decision"]["i5_11b_started"] is False
    assert out["decision"]["i5_11c_authorized"] is False
    assert out["decision"]["i5_12_started"] is False


def test_fault_matrix_uses_safe_synthetic_mechanisms() -> None:
    out = _build()
    matrix = out["fault_matrix"]
    assert matrix["cuda_oom"]["injection"] == "synthetic_cuda_oom_signature_in_isolated_fault_worker"
    assert matrix["cuda_oom"]["consumes_real_gpu_memory"] is False
    assert matrix["disk_enospc"]["injection"] == "synthetic_errno_enospc_in_isolated_owned_workspace"
    assert matrix["disk_enospc"]["fills_real_filesystem"] is False
    assert matrix["crash_orphan_descendant"]["existing_g1_control"] is True


def test_disk_classifier_gap_is_explicit_not_silently_fixed() -> None:
    out = _build()
    disk = out["fault_matrix"]["disk_enospc"]
    assert disk["generic_failure_classifier_has_first_class_enospc"] is False
    assert disk["raw_enospc_signature_must_be_preserved"] is True
    assert out["decision"]["production_failure_classifier_change_authorized"] is False


def test_historical_cancel_timeout_are_context_not_campaign_repeats() -> None:
    out = _build()
    exclusions = out["excluded_or_context_only"]
    assert exclusions["cancel"] == "historically_qualified_context_only"
    assert exclusions["timeout"] == "historically_qualified_context_only"


def test_b_requirements_keep_fault_injection_off() -> None:
    out = _build()
    req = out["i5_11b_preflight_requirements"]
    assert req["execute_fault_injection"] is False
    assert req["freeze_three_fault_mechanisms"] is True
    assert req["freeze_clean_recovery_after_each"] is True
    assert req["verify_disk_free_space_without_pressure"] is True


def test_claim_boundaries_do_not_turn_fault_test_into_quality_claim() -> None:
    out = _build()
    bounds = out["claim_boundaries"]
    assert bounds["reconstruction_quality_comparison"] is False
    assert bounds["provider_ranking"] is False
    assert bounds["metric_scale_claimed"] is False
    assert bounds["certified_metrology"] is False
