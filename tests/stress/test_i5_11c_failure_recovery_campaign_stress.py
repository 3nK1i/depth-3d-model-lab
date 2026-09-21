from __future__ import annotations

import json

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_failure_robustness_campaign import (
    I511C_FAULT_SCENARIOS,
    build_i5_11c_campaign_header,
    validate_i5_11b_for_i5_11c,
)
from tests.i5_11c_helpers import (
    I511A_SHA,
    I511B_SHA,
    POST_I510D_DELTA_SHA,
    binding,
    synthetic_anchors,
    synthetic_disk,
    synthetic_i511b_preflight,
    synthetic_i511c_receipt,
)


def _args() -> dict:
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


def test_campaign_header_is_deterministic() -> None:
    assert build_i5_11c_campaign_header(**_args()) == build_i5_11c_campaign_header(**_args())


def test_sorted_json_roundtrip_does_not_change_frozen_scenario_order() -> None:
    b = synthetic_i511b_preflight()
    roundtrip = json.loads(json.dumps(b, sort_keys=True))
    out = validate_i5_11b_for_i5_11c(roundtrip)
    assert tuple(out["campaign_contract"]["fault_scenarios"]) == I511C_FAULT_SCENARIOS


@pytest.mark.parametrize("field", ["retry_until_pass", "result_driven_retuning", "real_gpu_oom", "real_disk_fill"])
def test_b_campaign_scope_drift_fails_closed(field: str) -> None:
    b = synthetic_i511b_preflight()
    b["campaign_contract"][field] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_11b_for_i5_11c(b)


def test_existing_output_or_attempt_reuse_is_forbidden_by_contract() -> None:
    h = build_i5_11c_campaign_header(**_args())
    assert h["campaign_contract"]["attempt_directories_new_and_immutable"] is True
    assert h["campaign_contract"]["no_retry_after_terminal_scenario"] is True
