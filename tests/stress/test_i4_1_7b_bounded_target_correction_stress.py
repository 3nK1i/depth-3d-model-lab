from __future__ import annotations

import numpy as np

from core.reconstruction.offline_dense_i4_clean_static_orbit_diagnosis import build_i4_1_7a_orbit_geometry_diagnosis
from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import build_clean_static_orbit_assessment
from core.reconstruction.offline_dense_i4_clean_static_orbit_target_correction import build_i4_1_7b_bounded_target_correction
from tests.unit.test_i4_1_7_clean_static_orbit_qualification import _arc, _plan


def _transform(matrices: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(37.0)
    R = np.asarray([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    t = np.asarray([3.2, -1.7, 4.5])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    transformed = np.asarray([T @ m for m in matrices])
    return transformed, R @ target + t


def _run(matrices: np.ndarray, provisional: np.ndarray):
    count = len(matrices)
    plan = _plan(selected=count, aligned=count, pair_fraction=1.0)
    plan.update(planned_candidates=count, alignment_accepted=count, alignment_fraction=1.0, dense_gate_pass=True)
    assessment = build_clean_static_orbit_assessment(
        input_global_views=count,
        registered_global_views=count,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        selected_source_indices=list(range(count)),
        selected_pose_provenance=["global_sfm_anchor"] * count,
        provisional_target_reconstruction=provisional,
        target_diagnostics={},
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 40, "pair_pass_fraction": 1.0},
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
    return build_i4_1_7b_bounded_target_correction(
        source_i4_1_7_summary=source,
        source_i4_1_7a_summary=diag,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 40, "pair_pass_fraction": 1.0},
        selected_source_indices=list(range(count)),
        selected_pose_provenance=["global_sfm_anchor"] * count,
    )


def test_i4_1_7b_large_pose_set_is_coordinate_invariant_and_threshold_preserving() -> None:
    matrices = _arc(120, 235.0)
    provisional = np.asarray([6.0, 0.0, 0.0])
    base = _run(matrices, provisional)
    transformed_matrices, transformed_target = _transform(matrices, provisional)
    moved = _run(transformed_matrices, transformed_target)
    assert base["status"] == moved["status"] == "qualified_for_i4_1_8"
    a = base["assessment"]["selected_orbit"]
    b = moved["assessment"]["selected_orbit"]
    assert np.isclose(a["azimuth_coverage_degrees"], b["azimuth_coverage_degrees"], atol=1e-7)
    assert a["occupied_azimuth_bins"] == b["occupied_azimuth_bins"]
    assert np.isclose(a["maximum_opposed_view_angle_degrees"], b["maximum_opposed_view_angle_degrees"], atol=1e-7)
    assert base["authority"]["orbit_thresholds_changed"] is False
    assert base["authority"]["dense_consistency_threshold_relaxation_authorized"] is False
