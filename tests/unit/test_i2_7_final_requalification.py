from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i2_closure import (
    I2ClosurePolicy,
    build_i2_closure_record,
    closure_quality_gates,
    voxel_component_profile,
)


def _good_gates(**overrides):
    values = dict(
        policy=I2ClosurePolicy(), i2_6_pass=True, exact_counterfactual_equivalence=True,
        weak_manifold_attachment_rejections=412, expected_manifold_rejections=412,
        extended_loss_fraction=0.0031, core_loss_fraction=0.0015,
        full_to_robust_improvement_fraction=0.056, full_diagonal_shrink_fraction=0.079,
        robust_diagonal_shrink_fraction=0.025, far_peripheral_reduction_fraction=0.215,
        largest_component_fraction_of_extended=0.99, driver_rejection_fraction=0.522,
        stubborn_rejection_fraction=0.571, upstream_quality_gates_pass=True,
        persistent_integrity_pass=True, authority_boundaries_preserved=True,
    ); values.update(overrides); return closure_quality_gates(**values)


def test_i2_7_policy_defaults_are_conservative():
    p = I2ClosurePolicy(); assert p.maximum_core_loss_fraction < p.maximum_extended_loss_fraction
    assert p.minimum_largest_component_fraction_of_extended >= 0.98


def test_i2_7_policy_rejects_invalid_fraction():
    with pytest.raises(ReconstructionValidationError): I2ClosurePolicy(maximum_core_loss_fraction=1.2)


def test_i2_7_component_profile_single_connected_block():
    points = np.asarray([[x+.2, y+.2, z+.2] for x in range(3) for y in range(3) for z in range(2)], float)
    result = voxel_component_profile(points, 1.0); assert result["component_count"] == 1; assert result["largest_component_fraction_of_extended"] == pytest.approx(1.0)


def test_i2_7_component_profile_detects_detached_voxel():
    points = np.asarray([[.2,.2,.2],[1.2,.2,.2],[20.2,.2,.2]], float)
    result = voxel_component_profile(points, 1.0); assert result["component_count"] == 2; assert result["largest_component_voxels"] == 2


def test_i2_7_component_profile_rejects_duplicate_voxel():
    with pytest.raises(ReconstructionValidationError): voxel_component_profile(np.asarray([[.1,.1,.1],[.2,.2,.2]]), 1.0)


def test_i2_7_current_d12_profile_passes_all_automated_gates():
    assert all(_good_gates().values())


def test_i2_7_rejects_excessive_core_loss():
    assert not _good_gates(core_loss_fraction=0.02)["core_loss_within_gate"]


def test_i2_7_requires_meaningful_ratio_improvement():
    assert not _good_gates(full_to_robust_improvement_fraction=0.01)["full_to_robust_improvement"]


def test_i2_7_requires_robust_denominator_preservation():
    assert not _good_gates(robust_diagonal_shrink_fraction=0.08)["robust_diagonal_preservation"]


def test_i2_7_requires_far_artifact_reduction():
    assert not _good_gates(far_peripheral_reduction_fraction=0.02)["far_peripheral_reduction"]


def test_i2_7_requires_exact_manifold_rejection_count():
    assert not _good_gates(weak_manifold_attachment_rejections=411)["manifold_rejection_count_exact"]


def _automated():
    return {"phase":"I2.7","automated_pass":True,"scene":{"scene_id":"scene","fingerprint":"abc"}}


def _ui():
    return {"phase":"I2.7-UI","pass":True,"scene_fingerprint_sha256":"abc"}


def test_i2_7_closure_requires_fixed_view_confirmation():
    with pytest.raises(ReconstructionValidationError): build_i2_closure_record(_automated(), _ui(), fixed_view_review_pass=False)


def test_i2_7_closure_requires_matching_ui_scene():
    ui = _ui(); ui["scene_fingerprint_sha256"] = "other"
    with pytest.raises(ReconstructionValidationError): build_i2_closure_record(_automated(), ui, fixed_view_review_pass=True)


def test_i2_7_closure_authorizes_i3_but_not_mesh():
    result = build_i2_closure_record(_automated(), _ui(), fixed_view_review_pass=True)
    assert result["i2_complete"] is True
    assert result["decision"]["authorizes_i3_second_scene_campaign"] is True
    assert result["decision"]["authorizes_surface_mesh_reconstruction"] is False
    assert result["authority"]["metric_scale_claimed"] is False
