from __future__ import annotations

import json

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_rolling_depth_robustness import (
    I57AAdmissionPolicy,
    build_i5_7a_rolling_depth_robustness_admission,
)
from tests.i5_7a_helpers import build_args


def test_repeated_admission_is_deterministic() -> None:
    a = build_i5_7a_rolling_depth_robustness_admission(**build_args())
    b = build_i5_7a_rolling_depth_robustness_admission(**build_args())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.parametrize("field,value", [
    ("default_precision", "fp32"),
    ("default_device", "cuda:1"),
    ("sequential_execution", False),
    ("provider_comparison", True),
])
def test_policy_drift_is_rejected(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises(ReconstructionValidationError):
        I57AAdmissionPolicy(**kwargs)


def test_frozen_predecessor_hash_drift_is_rejected() -> None:
    args = build_args()
    args["i5_6c_binding"] = type(args["i5_6c_binding"])(args["i5_6c_binding"].path, "f" * 64)
    with pytest.raises(ReconstructionValidationError):
        build_i5_7a_rolling_depth_robustness_admission(**args)
