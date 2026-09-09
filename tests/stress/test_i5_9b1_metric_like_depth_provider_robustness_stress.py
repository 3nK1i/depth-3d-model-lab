from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_remaining_depth_provider_robustness_replay import (
    I59B1_PROVIDERS,
    I59B1_SCENARIO_IDS,
    build_i5_9b1_replay,
    validate_i5_9b1_provider_scenario_record,
    validate_i5_9b1_runtime_preflight,
)
from tests.i5_9b1_helpers import build_replay_args, preflight, record


@pytest.mark.parametrize("provider", I59B1_PROVIDERS)
def test_every_b1_provider_rejects_metric_authority_promotion(provider):
    payload = record("castle_control", provider)
    payload["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b1_provider_scenario_record(payload)


@pytest.mark.parametrize("sid", I59B1_SCENARIO_IDS)
def test_every_scenario_preflight_requires_source_integrity(sid):
    payload = preflight()
    payload["scenario_plans"][sid]["source_integrity_verified"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b1_runtime_preflight(payload)


def test_replay_refuses_silent_provider_substitution():
    args = build_replay_args()
    bad = args["records"]["castle_control"].pop("metric3d_v2")
    bad["provider_implementation"] = "depth_anything_v2"
    args["records"]["castle_control"]["depth_anything_v2"] = bad
    with pytest.raises(ReconstructionValidationError):
        build_i5_9b1_replay(**args)


def test_replay_refuses_result_driven_selected_view_expansion_beyond_source():
    payload = record("castle_low_overlap", "metric3d_v2")
    payload["selected_views"] = payload["source_registered_views"] + 1
    payload["processed_views"] = payload["selected_views"]
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b1_provider_scenario_record(payload)
