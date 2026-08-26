from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.dense_confidence_filtering import DenseArtifactReason, artifact_reason_histogram
from core.reconstruction.dense_manifold_refinement import (
    DenseManifoldRefinementPolicy,
    manifold_attachment_rejection_mask,
    promotion_loss_fractions,
    validate_exact_counterfactual_equivalence,
    validate_promotion_evidence,
)
from core.reconstruction.errors import ReconstructionValidationError


def _i252() -> dict:
    return {
        "phase": "I2.5.2",
        "status": "diagnosis_ready",
        "scene_id": "scene-a",
        "diagnostic_signal": {
            "strength": "strong",
            "supports_local_geometry_manifold_hypothesis": True,
        },
    }


def _record() -> dict:
    return {
        "name": "d12_sparse",
        "resolved_rule": {
            "scenario_name": "d12_sparse",
            "anchor_distance_scale": 12.0,
            "knn_spacing_scale": 1.0,
            "uses_peripheral_or_far_labels": False,
            "uses_extremum_driver_labels": False,
            "uses_source_view_blacklist": False,
            "uses_metric_distance": False,
        },
        "comparison": {
            "balanced_for_manual_review": True,
            "full_to_robust_ratio_improvement_fraction": .05,
            "central_core_loss_fraction": 0.0,
        },
        "effects_against_baseline": {"rejected_points": 2},
    }


def _i253() -> dict:
    return {
        "phase": "I2.5.3",
        "status": "counterfactuals_ready",
        "scene_id": "scene-a",
        "authority": {
            "source_geometry_changed": False,
            "production_f_policy_changed": False,
            "sfm_rerun": False,
            "da3_rerun": False,
            "fusion_rerun": False,
            "persistence_changed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "ranking": {"balanced_candidates_for_manual_review": ["d12_sparse"]},
        "scenarios": [_record()],
    }


def test_policy_is_locked_to_d12() -> None:
    with pytest.raises(ReconstructionValidationError):
        DenseManifoldRefinementPolicy(scenario_name="d8_sparse")


def test_policy_is_locked_to_anchor_scale_12() -> None:
    with pytest.raises(ReconstructionValidationError):
        DenseManifoldRefinementPolicy(anchor_distance_scale=8.0)


def test_policy_contract_has_no_scene_specific_inputs() -> None:
    payload = DenseManifoldRefinementPolicy().as_dict()
    assert payload["uses_peripheral_or_far_labels"] is False
    assert payload["uses_extremum_driver_labels"] is False
    assert payload["uses_source_view_blacklist"] is False
    assert payload["uses_metric_distance"] is False


def test_promotion_requires_manual_confirmation() -> None:
    with pytest.raises(ReconstructionValidationError):
        validate_promotion_evidence(_i252(), _i253(), scenario_name="d12_sparse", manual_review_pass=False)


def test_promotion_accepts_balanced_manual_candidate() -> None:
    policy, record, signal = validate_promotion_evidence(
        _i252(), _i253(), scenario_name="d12_sparse", manual_review_pass=True
    )
    assert policy.scenario_name == "d12_sparse"
    assert record["comparison"]["balanced_for_manual_review"]
    assert signal["strength"] == "strong"


def test_promotion_rejects_non_balanced_candidate() -> None:
    i253 = _i253()
    i253["ranking"]["balanced_candidates_for_manual_review"] = []
    with pytest.raises(ReconstructionValidationError):
        validate_promotion_evidence(_i252(), i253, scenario_name="d12_sparse", manual_review_pass=True)


def test_promotion_rejects_forbidden_rule_input() -> None:
    i253 = _i253()
    i253["scenarios"][0]["resolved_rule"]["uses_metric_distance"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_promotion_evidence(_i252(), i253, scenario_name="d12_sparse", manual_review_pass=True)


def test_production_mask_uses_strict_thresholds() -> None:
    policy = DenseManifoldRefinementPolicy()
    reject, resolved = manifold_attachment_rejection_mask(
        [12.0, 12.0001, 24.0],
        [2.0, 2.0, 1.0],
        anchor_distance_reference_q90_voxels=1.0,
        knn_spacing_reference_q90_voxels=1.5,
        policy=policy,
    )
    assert reject.tolist() == [False, True, False]
    assert resolved["resolved_anchor_distance_limit_voxels"] == 12.0


def test_production_mask_requires_aligned_vectors() -> None:
    with pytest.raises(ReconstructionValidationError):
        manifold_attachment_rejection_mask(
            [1, 2], [1], anchor_distance_reference_q90_voxels=1,
            knn_spacing_reference_q90_voxels=1, policy=DenseManifoldRefinementPolicy()
        )


def test_counterfactual_equivalence_passes_exact_mask() -> None:
    result = validate_exact_counterfactual_equivalence([0, 1, 1], [0, 1, 1], expected_rejected_points=2)
    assert result == {"exact_mask_match": True, "mismatched_voxels": 0, "rejected_points": 2}


def test_counterfactual_equivalence_fails_on_one_bit() -> None:
    with pytest.raises(ReconstructionValidationError):
        validate_exact_counterfactual_equivalence([0, 1, 1], [0, 1, 0], expected_rejected_points=2)


def test_promotion_loss_fractions_are_exact() -> None:
    result = promotion_loss_fractions(
        source_extended_points=1000, promoted_extended_points=990,
        source_core_points=800, promoted_core_points=798,
    )
    assert result["extended_loss_fraction"] == pytest.approx(.01)
    assert result["core_loss_fraction"] == pytest.approx(.0025)


def test_promotion_loss_rejects_invalid_hierarchy() -> None:
    with pytest.raises(ReconstructionValidationError):
        promotion_loss_fractions(
            source_extended_points=100, promoted_extended_points=101,
            source_core_points=80, promoted_core_points=80,
        )


def test_artifact_reason_histogram_reports_manifold_attachment() -> None:
    code = int(DenseArtifactReason.WEAK_MANIFOLD_ATTACHMENT)
    hist = artifact_reason_histogram(np.asarray([0, code, code], dtype=np.uint16))
    assert hist["weak_manifold_attachment"] == 2
