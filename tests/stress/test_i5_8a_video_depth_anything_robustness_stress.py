from __future__ import annotations

import json

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_video_depth_anything_robustness import (
    I58AAdmissionPolicy,
    build_i5_8a_video_depth_anything_robustness_admission,
)
from tests.i5_8a_helpers import build_args


def test_repeated_i58a_admission_is_deterministic() -> None:
    a = build_i5_8a_video_depth_anything_robustness_admission(**build_args())
    b = build_i5_8a_video_depth_anything_robustness_admission(**build_args())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.parametrize("field,value", [
    ("stateful_temporal_sequence", False),
    ("source_order_required", False),
    ("stable_pixel_grid_required", False),
    ("worker_model_loads_once_per_sequence", False),
    ("warmup_reset_before_user_sequence", False),
    ("monotonic_worker_frame_ids", False),
    ("sequential_scenario_execution", False),
])
def test_vda_streaming_policy_drift_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ReconstructionValidationError):
        I58AAdmissionPolicy(**{field: value})


def test_frozen_i57c_binding_drift_is_rejected() -> None:
    args = build_args()
    current = args["i5_7c_binding"]
    args["i5_7c_binding"] = type(current)(current.path, "f" * 64)
    with pytest.raises(ReconstructionValidationError):
        build_i5_8a_video_depth_anything_robustness_admission(**args)


def test_frozen_i56a_binding_drift_is_rejected() -> None:
    args = build_args()
    current = args["i5_6a_binding"]
    args["i5_6a_binding"] = type(current)(current.path, "e" * 64)
    with pytest.raises(ReconstructionValidationError):
        build_i5_8a_video_depth_anything_robustness_admission(**args)
