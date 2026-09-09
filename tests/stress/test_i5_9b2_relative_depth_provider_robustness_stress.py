from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_relative_depth_provider_robustness_replay import (
    I59B2_PROVIDERS,
    I59B2_SCENARIO_IDS,
    build_i5_9b2_replay,
    validate_i5_9b2_provider_scenario_record,
    validate_i5_9b2_runtime_preflight,
)
from tests.i5_9b2_helpers import build_replay_args, preflight, record


@pytest.mark.parametrize("provider", I59B2_PROVIDERS)
def test_every_b2_provider_rejects_metric_authority_promotion(provider):
    payload = record("castle_control", provider)
    payload["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_provider_scenario_record(payload)


@pytest.mark.parametrize("sid", I59B2_SCENARIO_IDS)
def test_every_scenario_preflight_requires_source_integrity(sid):
    payload = preflight()
    payload["scenario_plans"][sid]["source_integrity_verified"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_runtime_preflight(payload)


def test_replay_refuses_silent_provider_substitution():
    args = build_replay_args()
    bad = args["records"]["castle_control"].pop("depth_anything_v2")
    bad["provider_implementation"] = "depth_pro"
    args["records"]["castle_control"]["depth_pro"] = bad
    with pytest.raises(ReconstructionValidationError):
        build_i5_9b2_replay(**args)


def test_replay_refuses_selected_view_expansion_beyond_source():
    payload = record("castle_low_overlap", "depth_anything_v2")
    payload["selected_views"] = payload["source_registered_views"] + 1
    payload["processed_views"] = payload["selected_views"]
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_provider_scenario_record(payload)


def test_replay_refuses_alignment_semantics_cross_contamination():
    payload = record("castle_control", "marigold")
    payload["alignment_semantics"] = "per_view_affine_inverse_depth_to_authoritative_sfm"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_provider_scenario_record(payload)
