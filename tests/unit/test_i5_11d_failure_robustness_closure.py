from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_failure_robustness_qualification import (
    I511D_PHASE,
    build_i5_11d_failure_robustness_closure,
    validate_i5_11a_for_i5_11d,
    validate_i5_11b_for_i5_11d,
    validate_i5_11c_for_i5_11d,
    validate_i5_11d_receipt,
)
from tests.i5_11d_helpers import (
    I511A_SHA,
    I511B_SHA,
    I511C_SHA,
    binding,
    clone,
    synthetic_i511a_admission,
    synthetic_i511b_preflight,
    synthetic_i511c_pass,
    synthetic_i511d_receipt,
)


def build():
    return build_i5_11d_failure_robustness_closure(
        i5_11a_admission=synthetic_i511a_admission(),
        i5_11a_binding=binding('/evidence/i511a.json', I511A_SHA),
        i5_11b_preflight=synthetic_i511b_preflight(),
        i5_11b_binding=binding('/evidence/i511b.json', I511B_SHA),
        i5_11c_campaign=synthetic_i511c_pass(),
        i5_11c_binding=binding('/evidence/i511c.json', I511C_SHA),
        closure_receipt=synthetic_i511d_receipt(),
        closure_receipt_binding=binding('/evidence/i511d_receipt.json', 'd' * 64),
    )


def test_phase_constant():
    assert I511D_PHASE == 'I5.11D'


def test_validates_a_b_c_and_receipt():
    assert validate_i5_11a_for_i5_11d(synthetic_i511a_admission())['pass'] is True
    assert validate_i5_11b_for_i5_11d(synthetic_i511b_preflight())['pass'] is True
    assert validate_i5_11c_for_i5_11d(synthetic_i511c_pass())['terminal_status'] == 'pass'
    assert validate_i5_11d_receipt(synthetic_i511d_receipt())['pass'] is True


def test_closure_qualifies_all_three_bounded_faults():
    q = build()['qualification']
    assert q['bounded_fault_recovery_pass'] is True
    assert q['fault_scenarios_qualified'] == ['cuda_oom', 'disk_enospc', 'crash_orphan_descendant']
    assert q['fault_attempts_executed'] == 3
    assert q['recovery_attempts_executed'] == 3
    assert q['retry_attempts_executed'] == 0
    assert q['all_expected_fault_signatures_observed'] is True
    assert q['all_process_groups_clean'] is True
    assert q['all_failed_workspaces_removed'] is True
    assert q['all_recoveries_completed'] is True


def test_closure_preserves_enospc_classifier_gap_without_fix():
    out = build()
    gap = out['observed_gaps']['disk_enospc_classifier']
    assert gap['first_class_before_campaign'] is False
    assert gap['observed_kind'] == 'unknown_model_error'
    assert gap['gap_preserved'] is True
    assert gap['production_classifier_change_authorized'] is False


def test_closure_is_bounded_not_universal():
    limits = build()['observed_limitations']
    assert limits['real_gpu_oom_was_not_executed'] is True
    assert limits['real_disk_fill_was_not_executed'] is True
    assert limits['arbitrary_failure_recovery_not_proven'] is True
    assert limits['only_three_admitted_fault_contracts_qualified'] is True


def test_closure_is_evidence_only():
    scope = build()['scope']
    assert scope['evidence_only_qualification_and_closure'] is True
    assert scope['fault_injection'] is False
    assert scope['supervised_fault_worker_execution'] is False
    assert scope['real_resource_exhaustion'] is False
    assert scope['production_failure_classifier_modified'] is False
    assert scope['production_recovery_engine_modified'] is False


def test_closure_does_not_claim_reconstruction_or_runtime_winner():
    claims = build()['claim_boundaries']
    assert claims['reconstruction_quality_comparison'] is False
    assert claims['runtime_performance_ranking'] is False
    assert claims['universal_failure_recovery_claim'] is False
    assert claims['real_resource_exhaustion_qualified'] is False


def test_closure_closes_i511_and_authorizes_only_i512_admission_review():
    d = build()['decision']
    assert d['i5_11_complete'] is True
    assert d['i5_11d_complete'] is True
    assert d['i5_12_admission_review_authorized'] is True
    assert d['i5_12_started'] is False
    assert d['hybrid_authorized'] is False
    assert d['i5_13_started'] is False
    assert d['i5_14_started'] is False


def test_rejects_runtime_integrity_failure_c():
    bad = clone(synthetic_i511c_pass())
    bad['pass'] = False
    bad['status'] = 'bounded_failure_recovery_campaign_incomplete'
    bad['terminal_status'] = 'runtime_integrity_failure'
    bad['campaign_quality_status'] = 'runtime_integrity_failure'
    bad['decision']['i5_11c_complete'] = False
    bad['decision']['i5_11d_authorized'] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11c_for_i5_11d(bad)


def test_rejects_missing_recovery():
    bad = clone(synthetic_i511c_pass())
    bad['attempt_accounting']['recovery_attempts_executed'] = 2
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11c_for_i5_11d(bad)


def test_rejects_retry_attempt():
    bad = clone(synthetic_i511c_pass())
    bad['attempt_accounting']['retry_attempts_executed'] = 1
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11c_for_i5_11d(bad)


def test_rejects_scenario_order_drift():
    bad = clone(synthetic_i511c_pass())
    bad['scenario_records'] = list(reversed(bad['scenario_records']))
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11c_for_i5_11d(bad)


def test_rejects_leaked_descendant():
    bad = clone(synthetic_i511c_pass())
    bad['scenario_records'][2]['no_leaked_descendants'] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11c_for_i5_11d(bad)


def test_rejects_production_classifier_change():
    bad = clone(synthetic_i511c_pass())
    bad['scope']['production_failure_classifier_modified'] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11c_for_i5_11d(bad)


def test_rejects_binding_sha_drift():
    with pytest.raises(ReconstructionValidationError):
        build_i5_11d_failure_robustness_closure(
            i5_11a_admission=synthetic_i511a_admission(),
            i5_11a_binding=binding('/evidence/i511a.json', '0' * 64),
            i5_11b_preflight=synthetic_i511b_preflight(),
            i5_11b_binding=binding('/evidence/i511b.json', I511B_SHA),
            i5_11c_campaign=synthetic_i511c_pass(),
            i5_11c_binding=binding('/evidence/i511c.json', I511C_SHA),
            closure_receipt=synthetic_i511d_receipt(),
            closure_receipt_binding=binding('/evidence/i511d_receipt.json', 'd' * 64),
        )
