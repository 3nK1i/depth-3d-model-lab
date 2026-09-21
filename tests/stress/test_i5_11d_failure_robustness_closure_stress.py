from __future__ import annotations

import json

from core.reconstruction.i5_failure_robustness_qualification import build_i5_11d_failure_robustness_closure
from tests.i5_11d_helpers import (
    I511A_SHA, I511B_SHA, I511C_SHA, binding,
    synthetic_i511a_admission, synthetic_i511b_preflight, synthetic_i511c_pass, synthetic_i511d_receipt,
)


def _build():
    return build_i5_11d_failure_robustness_closure(
        i5_11a_admission=synthetic_i511a_admission(),
        i5_11a_binding=binding('/evidence/a.json', I511A_SHA),
        i5_11b_preflight=synthetic_i511b_preflight(),
        i5_11b_binding=binding('/evidence/b.json', I511B_SHA),
        i5_11c_campaign=synthetic_i511c_pass(),
        i5_11c_binding=binding('/evidence/c.json', I511C_SHA),
        closure_receipt=synthetic_i511d_receipt(),
        closure_receipt_binding=binding('/evidence/d_receipt.json', 'd' * 64),
    )


def test_repeated_closure_build_is_deterministic() -> None:
    baseline = _build()
    for _ in range(100):
        assert _build() == baseline


def test_sorted_json_roundtrip_preserves_semantics() -> None:
    out = _build()
    restored = json.loads(json.dumps(out, sort_keys=True))
    assert restored['qualification']['all_recoveries_completed'] is True
    assert restored['decision']['i5_11_complete'] is True
    assert restored['decision']['hybrid_authorized'] is False


def test_closure_has_no_execution_or_resource_pressure_surface() -> None:
    scope = _build()['scope']
    forbidden = [
        'fault_injection', 'supervised_fault_worker_execution', 'real_gpu_oom', 'real_disk_fill',
        'sfm_execution', 'mvs_execution', 'provider_inference', 'production_default_modified',
        'production_failure_classifier_modified', 'production_recovery_engine_modified',
    ]
    assert all(scope[key] is False for key in forbidden)
