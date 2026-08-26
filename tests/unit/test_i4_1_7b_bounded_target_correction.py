from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_clean_static_orbit_diagnosis import (
    build_i4_1_7a_orbit_geometry_diagnosis,
)
from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import (
    build_clean_static_orbit_assessment,
)
from core.reconstruction.offline_dense_i4_clean_static_orbit_target_correction import (
    build_i4_1_7b_bounded_target_correction,
)
from tests.unit.test_i4_1_7_clean_static_orbit_qualification import _arc, _plan


def _source(matrices: np.ndarray, provisional: np.ndarray, *, capture: bool = True) -> dict:
    plan = _plan(selected=len(matrices), aligned=100, pair_fraction=1.0)
    plan["planned_candidates"] = len(matrices)
    plan["alignment_accepted"] = len(matrices)
    plan["alignment_fraction"] = 1.0
    plan["dense_gate_pass"] = True
    assessment = build_clean_static_orbit_assessment(
        input_global_views=40,
        registered_global_views=40,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        selected_source_indices=list(range(len(matrices))),
        selected_pose_provenance=["global_sfm_anchor"] * len(matrices),
        provisional_target_reconstruction=provisional,
        target_diagnostics={},
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 40, "pair_pass_fraction": 1.0},
        prior_guardrails={"guardrails_pass": True},
        capture_confirmed_static=capture,
    )
    return {
        "phase": "I4.1.7",
        "status": "degraded",
        "qualification_plan": plan["plan_name"],
        "assessment": assessment,
    }


def _inputs(*, capture: bool = True):
    matrices = _arc(118, 220.0)
    provisional = np.asarray([6.0, 0.0, 0.0], dtype=np.float64)
    source = _source(matrices, provisional, capture=capture)
    diag = build_i4_1_7a_orbit_geometry_diagnosis(
        source_i4_1_7_summary=source,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        provisional_target_reconstruction=provisional,
    )
    plan = _plan(selected=len(matrices), aligned=len(matrices), pair_fraction=1.0)
    plan["planned_candidates"] = len(matrices)
    plan["alignment_fraction"] = 1.0
    plan["dense_gate_pass"] = True
    return matrices, source, diag, plan


def test_i4_1_7b_uses_trajectory_circle_only_for_diagnosed_target_collapse() -> None:
    matrices, source, diag, plan = _inputs()
    result = build_i4_1_7b_bounded_target_correction(
        source_i4_1_7_summary=source,
        source_i4_1_7a_summary=diag,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 40, "pair_pass_fraction": 1.0},
        selected_source_indices=list(range(len(matrices))),
        selected_pose_provenance=["global_sfm_anchor"] * len(matrices),
    )
    assert result["correction_preconditions_pass"] is True
    assert result["status"] == "qualified_for_i4_1_8"
    assert result["assessment"]["target"]["mode"] == "provisional_registered_trajectory_circle_center"
    assert result["assessment"]["pass"] is True
    assert result["decision"]["i4_1_8_production_binding_authorized"] is True
    assert result["target_correction"]["production_target_authority"] is False
    assert result["authority"]["orbit_thresholds_changed"] is False


def test_i4_1_7b_fails_closed_when_source_has_unrelated_gate_failure() -> None:
    matrices, source, diag, plan = _inputs(capture=False)
    result = build_i4_1_7b_bounded_target_correction(
        source_i4_1_7_summary=source,
        source_i4_1_7a_summary=diag,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 40, "pair_pass_fraction": 1.0},
        selected_source_indices=list(range(len(matrices))),
        selected_pose_provenance=["global_sfm_anchor"] * len(matrices),
    )
    assert "capture_confirmed_static" in result["unrelated_source_gate_failures"]
    assert result["correction_preconditions_pass"] is False
    assert result["decision"]["i4_1_8_production_binding_authorized"] is False


def test_i4_1_7b_does_not_use_trajectory_fallback_when_optical_target_is_already_valid() -> None:
    matrices = _arc(40, 220.0)
    provisional = np.zeros(3, dtype=np.float64)
    source = _source(matrices, provisional)
    diag = build_i4_1_7a_orbit_geometry_diagnosis(
        source_i4_1_7_summary=source,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        provisional_target_reconstruction=provisional,
    )
    plan = _plan(selected=40, aligned=40, pair_fraction=1.0)
    plan.update(planned_candidates=40, alignment_accepted=40, alignment_fraction=1.0, dense_gate_pass=True)
    result = build_i4_1_7b_bounded_target_correction(
        source_i4_1_7_summary=source,
        source_i4_1_7a_summary=diag,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 20, "pair_pass_fraction": 1.0},
        selected_source_indices=list(range(40)),
        selected_pose_provenance=["global_sfm_anchor"] * 40,
    )
    assert result["preconditions"]["source_a_classifies_provisional_target_collapse"] is False
    assert result["correction_preconditions_pass"] is False
    assert result["decision"]["i4_1_8_production_binding_authorized"] is False


def test_i4_1_7b_rejects_degenerate_registered_camera_trajectory() -> None:
    pose = np.eye(4, dtype=np.float64)
    matrices = np.repeat(pose[None, :, :], 12, axis=0)
    source = {
        "phase": "I4.1.7",
        "status": "degraded",
        "assessment": {
            "target": {"mode": "provisional_optical_axis_consensus", "target_reconstruction": [1.0, 0.0, 0.0]},
            "gates": {"capture_confirmed_static": True, "prior_decoupled_guardrails": True},
            "counts": {"input_global_views": 12, "registered_global_views": 12},
            "selected_orbit": {},
        },
    }
    diag = {"phase": "I4.1.7A", "status": "diagnosis_ready", "diagnosis": {}}
    with pytest.raises(ReconstructionValidationError):
        build_i4_1_7b_bounded_target_correction(
            source_i4_1_7_summary=source,
            source_i4_1_7a_summary=diag,
            registered_camera_matrices=matrices,
            selected_camera_matrices=matrices,
            plan_payload={},
            baseline_payload={},
            selected_source_indices=list(range(12)),
            selected_pose_provenance=["global_sfm_anchor"] * 12,
        )
