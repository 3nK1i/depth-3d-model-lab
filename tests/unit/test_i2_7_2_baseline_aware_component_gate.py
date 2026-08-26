from __future__ import annotations

import copy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i2_closure import (
    I2ClosurePolicy,
    baseline_aware_component_topology_gates,
    closure_quality_gates,
)


def _qualification() -> dict:
    return {
        "phase": "I2.7.1",
        "status": "diagnosis_ready",
        "source_scene": {"scene_id": "scene", "fingerprint": "fp"},
        "failure_classification": "preexisting_baseline_fragmentation",
        "surviving_evidence_equivalence": {
            "points": True,
            "colors": True,
            "support": True,
            "confidence": True,
            "dispersion": True,
            "core_membership": True,
        },
        "component_comparison": {
            "baseline": {
                "largest_component_voxels": 128490,
                "largest_component_fraction": 0.970505,
                "detached_voxels": 3905,
            },
            "promoted": {
                "largest_component_voxels": 128490,
                "largest_component_fraction": 0.973534,
                "detached_voxels": 3493,
            },
            "promoted_dominant_from_single_baseline_component_fraction": 1.0,
            "newly_detached_voxels": 0,
            "removed_from_baseline_dominant": 0,
        },
        "promotion_cut_relationship": {"newly_detached_component_count": 0},
    }


def _closure_kwargs() -> dict:
    return dict(
        policy=I2ClosurePolicy(),
        i2_6_pass=True,
        exact_counterfactual_equivalence=True,
        weak_manifold_attachment_rejections=412,
        expected_manifold_rejections=412,
        extended_loss_fraction=0.0031,
        core_loss_fraction=0.0015,
        full_to_robust_improvement_fraction=0.056,
        full_diagonal_shrink_fraction=0.079,
        robust_diagonal_shrink_fraction=0.025,
        far_peripheral_reduction_fraction=0.154,
        largest_component_fraction_of_extended=0.973534,
        driver_rejection_fraction=0.522,
        stubborn_rejection_fraction=0.571,
        upstream_quality_gates_pass=True,
        persistent_integrity_pass=True,
        authority_boundaries_preserved=True,
    )


def test_i2_7_2_real_lineage_shape_passes_baseline_aware_gate():
    gates = baseline_aware_component_topology_gates(
        _qualification(), expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=True
    )
    assert all(gates.values())


def test_i2_7_2_requires_diagnosis_ready_summary():
    q = _qualification(); q["status"] = "degraded"
    with pytest.raises(ReconstructionValidationError):
        baseline_aware_component_topology_gates(q, expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=True)


def test_i2_7_2_requires_matching_scene_identity():
    gates = baseline_aware_component_topology_gates(
        _qualification(), expected_scene_id="other", expected_scene_fingerprint="fp", manual_lineage_review_pass=True
    )
    assert gates["scene_identity_matches"] is False


def test_i2_7_2_requires_preexisting_baseline_fragmentation_classification():
    q = _qualification(); q["failure_classification"] = "promotion_induced_threshold_crossing"
    gates = baseline_aware_component_topology_gates(q, expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=True)
    assert gates["preexisting_baseline_fragmentation_confirmed"] is False


def test_i2_7_2_requires_surviving_evidence_equivalence():
    q = _qualification(); q["surviving_evidence_equivalence"]["support"] = False
    gates = baseline_aware_component_topology_gates(q, expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=True)
    assert gates["surviving_evidence_equivalence"] is False


def test_i2_7_2_rejects_dominant_component_size_regression():
    q = _qualification(); q["component_comparison"]["promoted"]["largest_component_voxels"] = 128489
    gates = baseline_aware_component_topology_gates(q, expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=True)
    assert gates["dominant_component_size_not_degraded"] is False


def test_i2_7_2_requires_exact_dominant_lineage():
    q = _qualification(); q["component_comparison"]["promoted_dominant_from_single_baseline_component_fraction"] = 0.999
    gates = baseline_aware_component_topology_gates(q, expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=True)
    assert gates["dominant_component_lineage_exact"] is False


def test_i2_7_2_rejects_new_detached_voxels():
    q = _qualification(); q["component_comparison"]["newly_detached_voxels"] = 1
    gates = baseline_aware_component_topology_gates(q, expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=True)
    assert gates["no_newly_detached_voxels"] is False


def test_i2_7_2_rejects_new_detached_components():
    q = _qualification(); q["promotion_cut_relationship"]["newly_detached_component_count"] = 1
    gates = baseline_aware_component_topology_gates(q, expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=True)
    assert gates["no_newly_detached_components"] is False


def test_i2_7_2_rejects_removal_from_baseline_dominant():
    q = _qualification(); q["component_comparison"]["removed_from_baseline_dominant"] = 1
    gates = baseline_aware_component_topology_gates(q, expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=True)
    assert gates["no_removed_baseline_dominant_voxels"] is False


def test_i2_7_2_requires_component_fraction_non_regression():
    q = _qualification(); q["component_comparison"]["promoted"]["largest_component_fraction"] = 0.969
    gates = baseline_aware_component_topology_gates(q, expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=True)
    assert gates["largest_component_fraction_not_degraded"] is False


def test_i2_7_2_requires_detached_count_non_increase():
    q = _qualification(); q["component_comparison"]["promoted"]["detached_voxels"] = 3906
    gates = baseline_aware_component_topology_gates(q, expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=True)
    assert gates["detached_voxel_count_not_increased"] is False


def test_i2_7_2_requires_explicit_manual_lineage_review():
    gates = baseline_aware_component_topology_gates(
        _qualification(), expected_scene_id="scene", expected_scene_fingerprint="fp", manual_lineage_review_pass=False
    )
    assert gates["manual_lineage_review_pass"] is False


def test_i2_7_2_baseline_override_replaces_invalid_absolute_fraction_gate_only_when_supplied():
    legacy = closure_quality_gates(**_closure_kwargs())
    assert legacy["largest_component_preserved"] is False
    updated = _closure_kwargs(); updated["baseline_component_topology_preserved"] = True
    corrected = closure_quality_gates(**updated)
    assert corrected["largest_component_preserved"] is True
    assert all(corrected.values())
