from __future__ import annotations

import copy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from tests.i5_13d_helpers import make_i5_13c_qualification, make_integration_inventory
from core.reconstruction.i5_auto_policy_production_qualification import qualify_i5_13d_production_integration


@pytest.mark.parametrize("_", range(8))
def test_decision_is_deterministic_across_repeated_evidence_only_reviews(_):
    a = qualify_i5_13d_production_integration(
        i5_13c_qualification=make_i5_13c_qualification(),
        integration_inventory=make_integration_inventory(),
    )
    b = qualify_i5_13d_production_integration(
        i5_13c_qualification=make_i5_13c_qualification(),
        integration_inventory=make_integration_inventory(),
    )
    assert a == b
    assert a["production_auto_authorized"] is False


@pytest.mark.parametrize("field", [
    "live_context_to_evidence_token_mapping_qualified",
    "production_scene_classifier_implemented",
])
def test_partial_context_claims_cannot_silently_authorize_production_auto(field):
    inv = make_integration_inventory()
    inv[field] = True
    out = qualify_i5_13d_production_integration(
        i5_13c_qualification=make_i5_13c_qualification(),
        integration_inventory=inv,
    )
    assert out["production_auto_authorized"] is False


def test_offline_replay_failure_blocks_d():
    c = make_i5_13c_qualification()
    c["replay_summary"]["all_expected_decisions_matched"] = False
    with pytest.raises(ReconstructionValidationError):
        qualify_i5_13d_production_integration(
            i5_13c_qualification=c,
            integration_inventory=make_integration_inventory(),
        )
