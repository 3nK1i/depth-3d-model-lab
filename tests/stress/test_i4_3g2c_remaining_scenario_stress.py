from __future__ import annotations

import copy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_matched_view import matched_view_scope_from_frozen_verdict


BASE = {
    "g2_complete_room_mvs_da3": {
        "scenario_id": "g2_complete_room_mvs_da3",
        "scenario_classification": "operating_envelope_failure_authoritative_sfm_coverage",
        "full_room_authoritative_sfm_qualified": False,
        "full_room_provider_comparison_qualified": False,
        "localized_provider_comparison_valid": True,
        "execution_defect_established": False,
        "infrastructure_rerun_authorized": False,
    },
    "g2_low_texture_mvs_da3": {
        "scenario_id": "g2_low_texture_mvs_da3",
        "scenario_classification": "operating_envelope_failure_authoritative_sfm_coverage",
        "authoritative_sfm_qualified_for_full_scenario": False,
        "full_scenario_provider_comparison_qualified": False,
        "localized_provider_comparison_valid": True,
        "execution_defect_established": False,
        "infrastructure_rerun_authorized": False,
    },
    "g2_weak_overlap_mvs_da3": {
        "scenario_id": "g2_weak_overlap_mvs_da3",
        "authoritative_sfm_connectivity_qualified": True,
        "production_native_comparison_valid": True,
        "mvs_status": "pass",
        "da3_status": "pass",
        "same_authoritative_sfm_shared": True,
        "execution_defect_established": False,
        "infrastructure_rerun_authorized": False,
        "scenario_verdict_pending_g2c_manual_review": True,
    },
}


def test_repeated_remaining_scenario_validation_is_deterministic():
    expected = {
        "g2_complete_room_mvs_da3": "localized_authoritative_registered_views",
        "g2_low_texture_mvs_da3": "localized_authoritative_registered_views",
        "g2_weak_overlap_mvs_da3": "authoritative_registered_views",
    }
    for _ in range(500):
        for scenario_id, verdict in BASE.items():
            assert matched_view_scope_from_frozen_verdict(
                copy.deepcopy(verdict), scenario_id=scenario_id
            ) == expected[scenario_id]


def test_single_contract_bit_drift_is_rejected_across_remaining_scenarios():
    critical = {
        "g2_complete_room_mvs_da3": "localized_provider_comparison_valid",
        "g2_low_texture_mvs_da3": "localized_provider_comparison_valid",
        "g2_weak_overlap_mvs_da3": "authoritative_sfm_connectivity_qualified",
    }
    for _ in range(100):
        for scenario_id, key in critical.items():
            verdict = copy.deepcopy(BASE[scenario_id])
            verdict[key] = False
            with pytest.raises(ReconstructionValidationError):
                matched_view_scope_from_frozen_verdict(verdict, scenario_id=scenario_id)
