from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i3_transfer import (
    I3TransferSafetyPolicy,
    build_scene_relative_manifold_evidence,
    exact_reference_transfer_preflight,
    transfer_candidate_assessment,
)


def _synthetic_payload(*, add_tail: bool = True) -> dict[str, np.ndarray]:
    # 6x6x6 compact Core manifold (216 points), plus optional detached Extended
    # tail.  Unit voxel size keeps component lineage exact and easy to reason.
    core = np.asarray([(x, y, z) for x in range(6) for y in range(6) for z in range(6)], dtype=np.float64)
    if add_tail:
        tail = np.asarray([(30 + 3 * i, 2, 2) for i in range(10)], dtype=np.float64)
        points = np.vstack([core, tail])
    else:
        points = core
    n = len(points)
    classes = np.full(n, 3, dtype=np.uint8)
    if add_tail:
        classes[len(core):] = 2
    support = np.full(n, 3, dtype=np.uint32)
    high = np.full(n, 4, dtype=np.int16)
    if add_tail:
        support[len(core):] = 2
        high[len(core):] = 0
    return {
        "points_reconstruction": points,
        "colors_rgb": np.full((n, 3), 128, dtype=np.uint8),
        "class_code": classes,
        "view_observation_counts": support,
        "high_support_neighbor_count": high,
        "voxel_size_reconstruction_units": np.asarray(1.0, dtype=np.float64),
        "filter_confidence": np.full(n, 0.8, dtype=np.float32),
        "p90_dispersion_voxels": np.full(n, 0.2, dtype=np.float32),
    }


def test_i3_1_safety_policy_defaults_are_conservative():
    p = I3TransferSafetyPolicy()
    assert p.maximum_extended_loss_fraction == pytest.approx(0.01)
    assert p.maximum_core_loss_fraction == pytest.approx(0.005)
    assert p.require_zero_newly_detached is True


def test_i3_1_safety_policy_rejects_invalid_fraction():
    with pytest.raises(ReconstructionValidationError):
        I3TransferSafetyPolicy(maximum_core_loss_fraction=1.1)


def test_i3_1_build_evidence_returns_aligned_masks():
    payload = _synthetic_payload()
    e = build_scene_relative_manifold_evidence(payload)
    n = len(payload["points_reconstruction"])
    for name in ("core", "peripheral", "far", "extremum_drivers", "strong_anchor", "reference", "reject"):
        assert np.asarray(e[name]).shape == (n,)


def test_i3_1_build_evidence_reference_is_large_enough():
    e = build_scene_relative_manifold_evidence(_synthetic_payload())
    assert e["reference_thresholds"]["reference_count"] >= 128
    assert e["reference_thresholds"]["anchor_distance_q90_voxels"] > 0
    assert e["reference_thresholds"]["knn_spacing_q90_voxels"] > 0


def test_i3_1_filter_predicate_rejects_some_far_sparse_tail_points():
    e = build_scene_relative_manifold_evidence(_synthetic_payload())
    reject = np.asarray(e["reject"], dtype=bool)
    assert np.count_nonzero(reject) > 0
    assert np.count_nonzero(reject[:216]) == 0


def test_i3_1_filter_predicate_does_not_use_metric_distance():
    e = build_scene_relative_manifold_evidence(_synthetic_payload())
    rule = e["resolved_rule"]
    assert rule["uses_metric_distance"] is False
    assert rule["uses_source_view_blacklist"] is False


def test_i3_1_preflight_accepts_exact_mask_and_thresholds():
    e = build_scene_relative_manifold_evidence(_synthetic_payload())
    t = e["reference_thresholds"]
    result = exact_reference_transfer_preflight(
        e,
        expected_reject_mask=e["reject"],
        expected_anchor_reference_q90=t["anchor_distance_q90_voxels"],
        expected_knn_reference_q90=t["knn_spacing_q90_voxels"],
    )
    assert result["pass"] is True
    assert result["mismatched_voxels"] == 0


def test_i3_1_preflight_rejects_mask_mismatch():
    e = build_scene_relative_manifold_evidence(_synthetic_payload())
    expected = np.asarray(e["reject"], dtype=bool).copy(); expected[0] = ~expected[0]
    t = e["reference_thresholds"]
    result = exact_reference_transfer_preflight(
        e,
        expected_reject_mask=expected,
        expected_anchor_reference_q90=t["anchor_distance_q90_voxels"],
        expected_knn_reference_q90=t["knn_spacing_q90_voxels"],
    )
    assert result["pass"] is False
    assert result["mismatched_voxels"] == 1


def test_i3_1_preflight_rejects_threshold_mismatch():
    e = build_scene_relative_manifold_evidence(_synthetic_payload())
    t = e["reference_thresholds"]
    result = exact_reference_transfer_preflight(
        e,
        expected_reject_mask=e["reject"],
        expected_anchor_reference_q90=t["anchor_distance_q90_voxels"] + 1.0,
        expected_knn_reference_q90=t["knn_spacing_q90_voxels"],
    )
    assert result["pass"] is False
    assert result["anchor_reference_match"] is False


def test_i3_1_candidate_assessment_preserves_compact_core():
    e = build_scene_relative_manifold_evidence(_synthetic_payload())
    a = transfer_candidate_assessment(e, policy=I3TransferSafetyPolicy(maximum_extended_loss_fraction=0.10))
    assert a["core_points"]["loss_fraction"] == pytest.approx(0.0)
    assert a["component_lineage"]["removed_from_baseline_dominant"] == 0


def test_i3_1_candidate_assessment_creates_no_new_detached_voxels_for_tail_removal():
    e = build_scene_relative_manifold_evidence(_synthetic_payload())
    a = transfer_candidate_assessment(e, policy=I3TransferSafetyPolicy(maximum_extended_loss_fraction=0.10))
    assert a["component_lineage"]["newly_detached_voxels"] == 0


def test_i3_1_candidate_assessment_improves_or_preserves_component_fraction():
    e = build_scene_relative_manifold_evidence(_synthetic_payload())
    a = transfer_candidate_assessment(e, policy=I3TransferSafetyPolicy(maximum_extended_loss_fraction=0.10))
    assert a["component_lineage"]["promoted"]["largest_component_fraction"] >= a["component_lineage"]["baseline"]["largest_component_fraction"]


def test_i3_1_candidate_assessment_is_counterfactual_only_by_contract():
    e = build_scene_relative_manifold_evidence(_synthetic_payload())
    a = transfer_candidate_assessment(e, policy=I3TransferSafetyPolicy(maximum_extended_loss_fraction=0.10))
    assert "gates" in a
    assert "safe_for_manual_transfer_review" in a
    assert "production_f_policy_changed" not in a


def test_i3_1_build_evidence_fails_with_too_small_core_reference():
    payload = _synthetic_payload(add_tail=False)
    # Keep fewer than 128 Core points while preserving non-empty geometry.
    for name, value in list(payload.items()):
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == 216:
            payload[name] = arr[:100]
    with pytest.raises(ReconstructionValidationError):
        build_scene_relative_manifold_evidence(payload)
