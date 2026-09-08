from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_remaining_depth_provider_robustness import (
    I59A_PROVIDER_IMPLEMENTATIONS,
    build_i5_9a_remaining_depth_provider_robustness_admission,
    validate_i4_3g4c_for_i5_9,
)
from tests.i5_9a_helpers import build_args, g4c_closure


@pytest.mark.parametrize("provider", I59A_PROVIDER_IMPLEMENTATIONS)
def test_g4_complete_room_provider_status_drift_is_rejected(provider):
    payload = g4c_closure()
    payload["scenario_synthesis"]["g4_single_image_complete_room"]["production_statuses"][provider] = "degraded"
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3g4c_for_i5_9(payload)


@pytest.mark.parametrize("provider", ("metric3d_v2", "unidepth_v2", "moge_2", "depth_pro"))
def test_g4_low_texture_b1_pass_drift_is_rejected(provider):
    payload = g4c_closure()
    payload["scenario_synthesis"]["g4_single_image_low_texture"]["production_statuses"][provider] = "rejected"
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3g4c_for_i5_9(payload)


def test_receipt_cannot_claim_inference_started():
    args = build_args()
    args["admission_receipt"]["scope"]["single_image_provider_inference"] = True
    with pytest.raises(ReconstructionValidationError):
        build_i5_9a_remaining_depth_provider_robustness_admission(**args)


def test_source_scenario_set_cannot_drift():
    args = build_args()
    args["i5_6a_admission"]["scenario_matrix"].pop("castle_control")
    with pytest.raises(ReconstructionValidationError):
        build_i5_9a_remaining_depth_provider_robustness_admission(**args)
