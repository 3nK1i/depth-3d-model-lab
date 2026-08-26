from __future__ import annotations

from pathlib import Path

import numpy as np

from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import (
    assemble_selected_pose_sequence,
    build_clean_static_orbit_assessment,
    validate_prior_decoupled_evidence,
)
from tests.unit.test_i4_1_7_clean_static_orbit_qualification import _arc, _prior, _plan


def test_i4_1_7_contract_preserves_one_sfm_frame_and_no_production_binding():
    selected = _arc(12, 220.0)
    assessment = build_clean_static_orbit_assessment(
        input_global_views=40,
        registered_global_views=40,
        registered_camera_matrices=_arc(40, 230.0),
        selected_camera_matrices=selected,
        selected_source_indices=list(range(12)),
        selected_pose_provenance=["localized_extra_observation"] * 12,
        provisional_target_reconstruction=np.zeros(3),
        target_diagnostics={"inlier_fraction": 0.95},
        plan_payload=_plan(selected=12),
        baseline_payload={"selected_qualified_views": 2, "pair_pass_fraction": 0.02},
        prior_guardrails=validate_prior_decoupled_evidence(_prior()),
        capture_confirmed_static=True,
    )
    assert assessment["pass"]
    assert assessment["authority"]["single_authoritative_sfm_frame"]
    assert not assessment["authority"]["production_default_changed"]
    assert not assessment["authority"]["dense_consistency_threshold_relaxation_authorized"]
    assert assessment["decision"]["i4_2_detail_preservation_authorized"] is False


def test_i4_1_7_contract_requires_real_dense_improvement_over_global_baseline():
    plan = _plan(selected=12, pair_fraction=0.02)
    assessment = build_clean_static_orbit_assessment(
        input_global_views=40,
        registered_global_views=40,
        registered_camera_matrices=_arc(40, 230.0),
        selected_camera_matrices=_arc(12, 220.0),
        selected_source_indices=list(range(12)),
        selected_pose_provenance=["global_sfm_anchor"] * 12,
        provisional_target_reconstruction=np.zeros(3),
        target_diagnostics={},
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 2, "pair_pass_fraction": 0.02},
        prior_guardrails=validate_prior_decoupled_evidence(_prior()),
        capture_confirmed_static=True,
    )
    assert not assessment["pass"]
    assert not assessment["gates"]["real_dense_plan_improves_over_global_only"]


def test_i4_1_7_mixed_pose_sources_keep_exact_selected_count():
    sources = list(range(12))
    poses = _arc(12, 220.0)
    global_map = {sources[i]: poses[i] for i in range(0, 12, 2)}
    local_map = {sources[i]: poses[i] for i in range(1, 12, 2)}
    selected, provenance = assemble_selected_pose_sequence(sources, global_pose_by_source=global_map, localized_pose_by_source=local_map)
    assert selected.shape == (12, 4, 4)
    assert provenance.count("global_sfm_anchor") == 6
    assert provenance.count("localized_extra_observation") == 6


def test_i4_1_7_runner_and_preflight_exist_and_are_additive():
    root = Path(__file__).resolve().parents[2]
    assert (root / "scripts/run_i4_1_7_capture_preflight.py").is_file()
    assert (root / "scripts/run_i4_1_7_clean_static_orbit_campaign.py").is_file()
    assert (root / "core/reconstruction/offline_dense_i4_clean_static_orbit_qualification.py").is_file()


def test_i4_1_7_contract_handles_saturated_global_pair_quality_without_relaxation():
    selected = _arc(20, 220.0)
    assessment = build_clean_static_orbit_assessment(
        input_global_views=40,
        registered_global_views=40,
        registered_camera_matrices=_arc(40, 230.0),
        selected_camera_matrices=selected,
        selected_source_indices=list(range(20)),
        selected_pose_provenance=["localized_extra_observation"] * 20,
        provisional_target_reconstruction=np.zeros(3),
        target_diagnostics={},
        plan_payload=_plan(selected=20, pair_fraction=1.0),
        baseline_payload={"selected_qualified_views": 12, "pair_pass_fraction": 1.0},
        prior_guardrails=validate_prior_decoupled_evidence(_prior()),
        capture_confirmed_static=True,
    )
    assert assessment["gates"]["real_dense_plan_improves_over_global_only"] is True
    assert assessment["counts"]["baseline_pair_pass_fraction_saturated"] is True
    assert assessment["authority"]["dense_consistency_threshold_relaxation_authorized"] is False
