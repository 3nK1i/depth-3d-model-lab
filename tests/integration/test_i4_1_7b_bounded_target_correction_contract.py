from __future__ import annotations

from pathlib import Path

import numpy as np

from core.reconstruction.offline_dense_i4_clean_static_orbit_diagnosis import (
    build_i4_1_7a_orbit_geometry_diagnosis,
)
from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import build_clean_static_orbit_assessment
from core.reconstruction.offline_dense_i4_clean_static_orbit_target_correction import (
    build_i4_1_7b_bounded_target_correction,
)
from tests.unit.test_i4_1_7_clean_static_orbit_qualification import _arc, _plan


def _fixture():
    matrices = _arc(60, 220.0)
    provisional = np.asarray([6.0, 0.0, 0.0])
    plan = _plan(selected=60, aligned=60, pair_fraction=1.0)
    plan.update(planned_candidates=60, alignment_accepted=60, alignment_fraction=1.0, dense_gate_pass=True)
    assessment = build_clean_static_orbit_assessment(
        input_global_views=60,
        registered_global_views=60,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        selected_source_indices=list(range(60)),
        selected_pose_provenance=["global_sfm_anchor"] * 60,
        provisional_target_reconstruction=provisional,
        target_diagnostics={},
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 20, "pair_pass_fraction": 1.0},
        prior_guardrails={"guardrails_pass": True},
        capture_confirmed_static=True,
    )
    source = {"phase": "I4.1.7", "status": "degraded", "qualification_plan": plan["plan_name"], "assessment": assessment}
    diag = build_i4_1_7a_orbit_geometry_diagnosis(
        source_i4_1_7_summary=source,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        provisional_target_reconstruction=provisional,
    )
    return matrices, source, diag, plan


def test_i4_1_7b_contract_is_lightweight_provisional_only_and_keeps_final_core_requalification() -> None:
    matrices, source, diag, plan = _fixture()
    result = build_i4_1_7b_bounded_target_correction(
        source_i4_1_7_summary=source,
        source_i4_1_7a_summary=diag,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 20, "pair_pass_fraction": 1.0},
        selected_source_indices=list(range(60)),
        selected_pose_provenance=["global_sfm_anchor"] * 60,
    )
    assert result["status"] == "qualified_for_i4_1_8"
    assert result["authority"]["sfm_rerun"] is False
    assert result["authority"]["da3_rerun"] is False
    assert result["authority"]["fusion_rerun"] is False
    assert result["authority"]["production_binding_executed"] is False
    assert result["target_correction"]["production_target_authority"] is False
    assert result["decision"]["i4_1_8_core_target_requalification_required"] is True
    assert result["authority"]["metric_scale_claimed"] is False


def test_i4_1_7b_runner_is_additive() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / "scripts/run_i4_1_7b_bounded_target_correction.py").is_file()
    assert (root / "core/reconstruction/offline_dense_i4_clean_static_orbit_target_correction.py").is_file()
