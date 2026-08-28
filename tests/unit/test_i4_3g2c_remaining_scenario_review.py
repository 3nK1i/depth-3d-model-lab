from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_g2c_review import (
    verify_g1_manual_review_requirement,
    verify_g2b_scenario_closure_for_g2c,
)
from core.reconstruction.mvs_da3_matched_view import (
    matched_view_scope_from_frozen_verdict,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority() -> dict:
    return {
        "camera_pose_authority_owner": "pycolmap_sfm",
        "provider_selected": False,
        "provider_promoted": False,
        "universal_provider_winner_assumed": False,
        "global_scalar_score_forbidden": True,
        "auto_policy_implemented": False,
        "hybrid_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "permanent_global_view_cap": None,
    }


def _g2b_closure(scenario_id: str, verdict_sha: str) -> dict:
    if scenario_id in {"g2_complete_room_mvs_da3", "g2_low_texture_mvs_da3"}:
        scenario = {
            "status": "closed_operating_envelope_result",
            "full_scenario_sfm_qualified": False,
            "localized_provider_evidence_valid": True,
            "rerun_authorized": False,
            "verdict_sha256": verdict_sha,
        }
    elif scenario_id == "g2_temple_ring_mvs_da3":
        scenario = {
            "status": "closed_production_native_result",
            "authoritative_sfm_qualified": True,
            "production_native_comparison_valid": True,
            "rerun_authorized": False,
            "verdict_sha256": verdict_sha,
        }
    elif scenario_id == "g2_weak_overlap_mvs_da3":
        scenario = {
            "status": "closed_production_native_result",
            "authoritative_sfm_connectivity_qualified": True,
            "production_native_comparison_valid": True,
            "rerun_authorized": False,
            "verdict_sha256": verdict_sha,
        }
    else:
        raise AssertionError(scenario_id)
    return {
        "phase": "I4.3G2B",
        "status": "closed",
        "i4_3g2b_closed": True,
        "i4_3g2c_ready": True,
        "all_five_g2_scenarios_accounted_for": True,
        "real_provider_execution_phase_complete": True,
        "scenarios": {scenario_id: scenario},
        "authority": _authority(),
    }


def test_g1_requirement_explicitly_requires_manual_matched_view(tmp_path: Path):
    scenario_id = "g2_complete_room_mvs_da3"
    g1 = _write(
        tmp_path / "g1.json",
        {
            "phase": "I4.3G1",
            "pass": True,
            "scenario_matrix": [
                {
                    "scenario_id": scenario_id,
                    "required": True,
                    "manual_review_required": True,
                    "comparison_modes": ["production_native", "matched_view_diagnostic"],
                }
            ],
        },
    )
    entry = verify_g1_manual_review_requirement(
        g1,
        expected_g1_sha256=_sha(g1),
        scenario_id=scenario_id,
    )
    assert entry["manual_review_required"] is True


def test_g1_requirement_rejects_missing_matched_view_mode(tmp_path: Path):
    scenario_id = "g2_complete_room_mvs_da3"
    g1 = _write(
        tmp_path / "g1_bad.json",
        {
            "phase": "I4.3G1",
            "pass": True,
            "scenario_matrix": [
                {
                    "scenario_id": scenario_id,
                    "required": True,
                    "manual_review_required": True,
                    "comparison_modes": ["production_native"],
                }
            ],
        },
    )
    with pytest.raises(ReconstructionValidationError):
        verify_g1_manual_review_requirement(
            g1,
            expected_g1_sha256=_sha(g1),
            scenario_id=scenario_id,
        )


@pytest.mark.parametrize(
    ("scenario_id", "verdict", "scope"),
    [
        (
            "g2_complete_room_mvs_da3",
            {
                "scenario_id": "g2_complete_room_mvs_da3",
                "scenario_classification": "operating_envelope_failure_authoritative_sfm_coverage",
                "full_room_authoritative_sfm_qualified": False,
                "full_room_provider_comparison_qualified": False,
                "localized_provider_comparison_valid": True,
                "execution_defect_established": False,
                "infrastructure_rerun_authorized": False,
            },
            "localized_authoritative_registered_views",
        ),
        (
            "g2_low_texture_mvs_da3",
            {
                "scenario_id": "g2_low_texture_mvs_da3",
                "scenario_classification": "operating_envelope_failure_authoritative_sfm_coverage",
                "authoritative_sfm_qualified_for_full_scenario": False,
                "full_scenario_provider_comparison_qualified": False,
                "localized_provider_comparison_valid": True,
                "execution_defect_established": False,
                "infrastructure_rerun_authorized": False,
            },
            "localized_authoritative_registered_views",
        ),
        (
            "g2_weak_overlap_mvs_da3",
            {
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
            "authoritative_registered_views",
        ),
    ],
)
def test_remaining_scenario_verdicts_preserve_their_frozen_scope(
    scenario_id: str,
    verdict: dict,
    scope: str,
):
    assert matched_view_scope_from_frozen_verdict(verdict, scenario_id=scenario_id) == scope


def test_operating_envelope_result_cannot_be_promoted_to_full_scenario_review():
    verdict = {
        "scenario_id": "g2_low_texture_mvs_da3",
        "scenario_classification": "operating_envelope_failure_authoritative_sfm_coverage",
        "authoritative_sfm_qualified_for_full_scenario": True,
        "full_scenario_provider_comparison_qualified": False,
        "localized_provider_comparison_valid": True,
        "execution_defect_established": False,
        "infrastructure_rerun_authorized": False,
    }
    with pytest.raises(ReconstructionValidationError):
        matched_view_scope_from_frozen_verdict(
            verdict,
            scenario_id="g2_low_texture_mvs_da3",
        )


@pytest.mark.parametrize(
    "scenario_id",
    [
        "g2_complete_room_mvs_da3",
        "g2_temple_ring_mvs_da3",
        "g2_low_texture_mvs_da3",
        "g2_weak_overlap_mvs_da3",
    ],
)
def test_formal_g2b_closure_binds_each_fresh_scenario(tmp_path: Path, scenario_id: str):
    verdict_sha = hashlib.sha256(scenario_id.encode()).hexdigest()
    closure = _write(tmp_path / f"{scenario_id}.json", _g2b_closure(scenario_id, verdict_sha))
    verified = verify_g2b_scenario_closure_for_g2c(
        closure,
        expected_closure_sha256=_sha(closure),
        scenario_id=scenario_id,
        expected_verdict_sha256=verdict_sha,
    )
    assert verified["i4_3g2c_ready"] is True
