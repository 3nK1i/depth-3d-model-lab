from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import (
    assemble_selected_pose_sequence,
    build_clean_static_orbit_assessment,
    validate_prior_decoupled_evidence,
)
from tests.unit.test_i4_1_7_clean_static_orbit_qualification import _arc, _prior, _plan


def test_i4_1_7_stress_40_selected_views_remain_bounded_and_scale_invariant():
    matrices = _arc(40, 240.0) * 1.0
    assessment = build_clean_static_orbit_assessment(
        input_global_views=40,
        registered_global_views=40,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        selected_source_indices=list(range(40)),
        selected_pose_provenance=["global_sfm_anchor"] * 40,
        provisional_target_reconstruction=np.zeros(3),
        target_diagnostics={},
        plan_payload={**_plan(selected=40, aligned=100, pair_fraction=0.5), "planned_candidates": 110, "alignment_fraction": 100/110, "alignment_accepted": 100},
        baseline_payload={"selected_qualified_views": 2, "pair_pass_fraction": 0.02},
        prior_guardrails=validate_prior_decoupled_evidence(_prior()),
        capture_confirmed_static=True,
    )
    assert assessment["pass"]
    assert assessment["counts"]["selected_qualified_views"] == 40
    assert assessment["authority"]["metric_scale_claimed"] is False


def test_i4_1_7_stress_duplicate_selected_sources_are_rejected():
    pose = np.eye(4)
    with pytest.raises(ReconstructionValidationError):
        assemble_selected_pose_sequence([1, 1], global_pose_by_source={1: pose}, localized_pose_by_source={})


def test_i4_1_7_stress_nonrigid_localized_pose_is_rejected():
    bad = np.eye(4); bad[0, 0] = 2.0
    with pytest.raises(ReconstructionValidationError):
        assemble_selected_pose_sequence([5], global_pose_by_source={}, localized_pose_by_source={5: bad})
