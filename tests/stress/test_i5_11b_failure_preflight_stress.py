from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_failure_robustness_preflight import build_i5_11b_preflight
from tests.i5_11b_helpers import (
    I511A_SHA,
    POST_I510D_DELTA_SHA,
    binding,
    synthetic_anchors,
    synthetic_classifier_probe,
    synthetic_disk_observation,
    synthetic_i511a_admission,
    synthetic_i511b_receipt,
)


def _args() -> dict:
    return {
        "i5_11a_admission": synthetic_i511a_admission(),
        "i5_11a_binding": binding("/evidence/a.json", I511A_SHA),
        "post_i5_10d_delta_binding": binding("/checkpoint/d.tar.gz", POST_I510D_DELTA_SHA),
        "preflight_receipt": synthetic_i511b_receipt(),
        "preflight_receipt_binding": binding("/evidence/b_receipt.json", "c" * 64),
        "repository_anchors": synthetic_anchors(),
        "classifier_probe": synthetic_classifier_probe(),
        "disk_observation": synthetic_disk_observation(),
        "live_fault_workers": [],
    }


def test_repeated_build_is_deterministic() -> None:
    a = build_i5_11b_preflight(**_args())
    b = build_i5_11b_preflight(**_args())
    assert a == b


@pytest.mark.parametrize("field", ["fault_injection", "supervised_fault_worker_execution", "real_gpu_oom", "real_disk_fill"])
def test_receipt_scope_cannot_claim_execution(field: str) -> None:
    args = _args()
    args["preflight_receipt"]["scope"][field] = True
    with pytest.raises(ReconstructionValidationError):
        build_i5_11b_preflight(**args)


def test_bad_i511a_sha_fails_closed() -> None:
    args = _args()
    args["i5_11a_binding"] = binding("/evidence/a.json", "0" * 64)
    with pytest.raises(ReconstructionValidationError, match="SHA256 drift"):
        build_i5_11b_preflight(**args)


def test_bad_post_i510d_delta_sha_fails_closed() -> None:
    args = _args()
    args["post_i5_10d_delta_binding"] = binding("/checkpoint/d.tar.gz", "0" * 64)
    with pytest.raises(ReconstructionValidationError, match="SHA256 drift"):
        build_i5_11b_preflight(**args)


def test_anchor_drift_fails_closed() -> None:
    args = _args()
    args["repository_anchors"]["x"]["verified"] = False
    with pytest.raises(ReconstructionValidationError):
        build_i5_11b_preflight(**args)
