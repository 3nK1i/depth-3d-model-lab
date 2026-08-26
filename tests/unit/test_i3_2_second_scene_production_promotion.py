from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i3_promotion import (
    I3PromotionGatePolicy,
    build_i3_2_requalification,
    build_i3_closure_record,
    validate_exact_i3_1_candidate,
    validate_i3_1_promotion_evidence,
)


def _summary() -> dict:
    return {
        "phase": "I3.1",
        "status": "ready_for_manual_transfer_review",
        "i2_transfer_preflight": {"pass": True},
        "baseline_quality": {"automated_pass": True},
        "scene": {"scene_id": "scene-a", "fingerprint": "abc"},
        "authority": {
            "geometry_changed": False,
            "production_f_policy_changed": False,
            "persistence_changed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "candidate": {
            "rejected_points": 2,
            "extended_points": {"baseline": 6, "candidate": 4, "loss_fraction": 2 / 6},
            "core_points": {"baseline": 4, "candidate": 3, "loss_fraction": 0.25},
            "envelope": {
                "baseline": {"full_to_robust_diagonal_ratio": 2.0},
                "candidate": {"full_to_robust_diagonal_ratio": 2.01},
                "full_diagonal_shrink_fraction": 0.0,
                "robust_diagonal_shrink_fraction": 0.01,
                "full_to_robust_improvement_fraction": -0.005,
            },
            "gates": {
                "extended_loss_within_gate": True,
                "core_loss_within_gate": True,
                "robust_diagonal_preserved": True,
                "full_to_robust_not_materially_worse": True,
                "dominant_lineage_preserved": True,
                "no_newly_detached_voxels": True,
                "no_baseline_dominant_removal": True,
                "largest_component_fraction_not_degraded": True,
                "detached_voxels_not_increased": True,
            },
            "safe_for_manual_transfer_review": True,
        },
    }


def _payloads():
    points = np.asarray([[i, 0.0, 0.0] for i in range(6)], dtype=np.float64)
    source = {
        "points_reconstruction": points,
        "colors_rgb": np.arange(18, dtype=np.uint8).reshape(6, 3),
        "class_code": np.asarray([3, 3, 3, 3, 2, 2], dtype=np.uint8),
        "view_observation_counts": np.asarray([3, 3, 3, 3, 2, 2], dtype=np.uint32),
        "filter_confidence": np.linspace(0.5, 0.9, 6, dtype=np.float32),
        "p90_dispersion_voxels": np.linspace(0.1, 0.3, 6, dtype=np.float32),
    }
    reject = np.asarray([False, False, False, True, False, True])
    is_core = source["class_code"] == 3
    diag = {"points_reconstruction": points.copy(), "is_rejected_by_d12_transfer": reject, "is_core": is_core}
    def sliced(mask):
        return {k: np.asarray(v)[mask].copy() for k, v in source.items() if np.asarray(v).ndim >= 1 and np.asarray(v).shape[0] == 6}
    return source, diag, sliced(~reject), sliced(is_core & ~reject)


def _lineage():
    return {
        "baseline": {"largest_component_fraction": 0.96, "detached_voxels": 4},
        "promoted": {"largest_component_fraction": 0.97, "detached_voxels": 2},
        "removed_voxels": 2,
        "removed_from_baseline_dominant": 0,
        "newly_detached_voxels": 0,
        "promoted_dominant_from_single_baseline_component_fraction": 1.0,
    }


def test_i3_2_gate_policy_defaults_are_locked():
    p = I3PromotionGatePolicy()
    assert p.maximum_extended_loss_fraction == pytest.approx(0.01)
    assert p.maximum_full_to_robust_regression_fraction == pytest.approx(0.01)


def test_i3_2_gate_policy_rejects_invalid_fraction():
    with pytest.raises(ReconstructionValidationError):
        I3PromotionGatePolicy(maximum_core_loss_fraction=1.5)


def test_i3_2_requires_explicit_manual_review_confirmation():
    with pytest.raises(ReconstructionValidationError):
        validate_i3_1_promotion_evidence(_summary(), manual_transfer_review_pass=False)


def test_i3_2_accepts_complete_i3_1_evidence():
    result = validate_i3_1_promotion_evidence(_summary(), manual_transfer_review_pass=True)
    assert result["transfer_safety_gates"] == "pass"


def test_i3_2_rejects_failed_i3_1_transfer_gate():
    s = _summary(); s["candidate"]["gates"]["no_newly_detached_voxels"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i3_1_promotion_evidence(s, manual_transfer_review_pass=True)


def test_i3_2_exact_candidate_equivalence_passes():
    source, diag, ext, core = _payloads()
    result = validate_exact_i3_1_candidate(source, diag, ext, core, _summary())
    assert result["exact_candidate_equivalence"] is True
    assert result["rejected_points"] == 2


def test_i3_2_exact_candidate_equivalence_rejects_mutated_candidate():
    source, diag, ext, core = _payloads(); ext["points_reconstruction"][0, 0] += 1
    with pytest.raises(ReconstructionValidationError):
        validate_exact_i3_1_candidate(source, diag, ext, core, _summary())


def test_i3_2_exact_candidate_equivalence_rejects_mask_count_mismatch():
    source, diag, ext, core = _payloads(); s = _summary(); s["candidate"]["rejected_points"] = 3
    with pytest.raises(ReconstructionValidationError):
        validate_exact_i3_1_candidate(source, diag, ext, core, s)


def test_i3_2_requalification_passes_safe_transfer_with_realistic_policy():
    s = _summary()
    s["candidate"]["extended_points"] = {"baseline": 1000, "candidate": 996, "loss_fraction": 0.004}
    s["candidate"]["core_points"] = {"baseline": 800, "candidate": 799, "loss_fraction": 0.00125}
    s["candidate"]["rejected_points"] = 4
    result = build_i3_2_requalification(
        s, exact_candidate_equivalence=True, promoted_extended_points=996, promoted_core_points=799,
        weak_manifold_attachment_rejections=4, exact_supported_accounting=True, component_lineage=_lineage(),
        persistence_reference_count=10, persistence_relative=True, persistence_self_contained=True, relocation_restore_pass=True,
    )
    assert result["pass"] is True


def test_i3_2_requalification_allows_tiny_ratio_regression_within_locked_gate():
    s = _summary()
    s["candidate"]["extended_points"] = {"baseline": 1000, "candidate": 996, "loss_fraction": 0.004}
    s["candidate"]["core_points"] = {"baseline": 800, "candidate": 799, "loss_fraction": 0.00125}
    s["candidate"]["rejected_points"] = 4
    s["candidate"]["envelope"]["full_to_robust_improvement_fraction"] = -0.002
    result = build_i3_2_requalification(
        s, exact_candidate_equivalence=True, promoted_extended_points=996, promoted_core_points=799,
        weak_manifold_attachment_rejections=4, exact_supported_accounting=True, component_lineage=_lineage(),
        persistence_reference_count=10, persistence_relative=True, persistence_self_contained=True, relocation_restore_pass=True,
    )
    assert result["gates"]["full_to_robust_not_materially_worse"] is True


def test_i3_2_requalification_fails_new_fragmentation():
    s = _summary()
    s["candidate"]["extended_points"] = {"baseline": 1000, "candidate": 996, "loss_fraction": 0.004}
    s["candidate"]["core_points"] = {"baseline": 800, "candidate": 799, "loss_fraction": 0.00125}
    s["candidate"]["rejected_points"] = 4
    lineage = _lineage(); lineage["newly_detached_voxels"] = 1
    result = build_i3_2_requalification(
        s, exact_candidate_equivalence=True, promoted_extended_points=996, promoted_core_points=799,
        weak_manifold_attachment_rejections=4, exact_supported_accounting=True, component_lineage=lineage,
        persistence_reference_count=10, persistence_relative=True, persistence_self_contained=True, relocation_restore_pass=True,
    )
    assert result["pass"] is False


def test_i3_2_requalification_requires_portable_persistence():
    s = _summary()
    s["candidate"]["extended_points"] = {"baseline": 1000, "candidate": 996, "loss_fraction": 0.004}
    s["candidate"]["core_points"] = {"baseline": 800, "candidate": 799, "loss_fraction": 0.00125}
    s["candidate"]["rejected_points"] = 4
    result = build_i3_2_requalification(
        s, exact_candidate_equivalence=True, promoted_extended_points=996, promoted_core_points=799,
        weak_manifold_attachment_rejections=4, exact_supported_accounting=True, component_lineage=_lineage(),
        persistence_reference_count=10, persistence_relative=False, persistence_self_contained=True, relocation_restore_pass=True,
    )
    assert result["pass"] is False


def test_i3_2_closure_requires_fixed_view_review():
    automated = {"phase": "I3.2", "automated_pass": True, "scene_id": "s", "scene_fingerprint_sha256": "f", "authority": {"metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False}}
    ui = {"phase": "I3.2-UI", "pass": True, "scene_id": "s"}
    with pytest.raises(ReconstructionValidationError):
        build_i3_closure_record(automated, ui, fixed_view_review_pass=False)


def test_i3_2_closure_authorizes_i4_but_not_mesh():
    automated = {"phase": "I3.2", "automated_pass": True, "scene_id": "s", "scene_fingerprint_sha256": "f", "authority": {"metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False}}
    ui = {"phase": "I3.2-UI", "pass": True, "scene_id": "s"}
    result = build_i3_closure_record(automated, ui, fixed_view_review_pass=True)
    assert result["decision"]["i4_connector_orbit_authorized"] is True
    assert result["decision"]["mesh_authorization"] is False
