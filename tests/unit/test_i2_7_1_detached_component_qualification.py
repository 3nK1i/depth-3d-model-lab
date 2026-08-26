from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_detached_component_qualification import (
    classify_component_gate_failure,
    compare_component_lineage,
    component_lineage,
    detached_distance_statistics,
    mask_statistics,
)


def _p(*xs: float) -> np.ndarray:
    return np.asarray([[x + 0.2, 0.2, 0.2] for x in xs], dtype=float)


def test_i2_7_1_component_lineage_single_component():
    result = component_lineage(_p(0, 1, 2), 1.0)
    assert result.profile["component_count"] == 1
    assert result.profile["largest_component_fraction"] == pytest.approx(1.0)


def test_i2_7_1_component_lineage_detects_detached_fragment():
    result = component_lineage(_p(0, 1, 10, 11), 1.0)
    assert result.profile["component_count"] == 2
    assert result.profile["largest_component_voxels"] == 2


def test_i2_7_1_component_lineage_rejects_duplicate_voxel():
    with pytest.raises(ReconstructionValidationError):
        component_lineage(np.asarray([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]), 1.0)


def test_i2_7_1_compare_requires_promoted_subset():
    with pytest.raises(ReconstructionValidationError):
        compare_component_lineage(baseline_points=_p(0, 1), promoted_points=_p(0, 9), voxel_size=1.0)


def test_i2_7_1_compare_detects_new_detachment_from_cut():
    result = compare_component_lineage(baseline_points=_p(0, 1, 2, 3), promoted_points=_p(0, 2, 3), voxel_size=1.0)
    assert result["removed_voxels"] == 1
    assert result["newly_detached_voxels"] == 1
    assert result["surviving_preexisting_detached_voxels"] == 0


def test_i2_7_1_compare_preserves_preexisting_detachment():
    result = compare_component_lineage(baseline_points=_p(0, 1, 2, 10), promoted_points=_p(0, 1, 10), voxel_size=1.0)
    assert result["newly_detached_voxels"] == 0
    assert result["surviving_preexisting_detached_voxels"] == 1


def test_i2_7_1_compare_accounts_removed_lineage():
    result = compare_component_lineage(baseline_points=_p(0, 1, 2, 10), promoted_points=_p(0, 1), voxel_size=1.0)
    assert result["removed_voxels"] == 2
    assert result["removed_from_baseline_dominant"] == 1
    assert result["removed_from_preexisting_detached"] == 1


def test_i2_7_1_cut_component_reports_touching_removed_voxel():
    result = compare_component_lineage(baseline_points=_p(0, 1, 2, 3), promoted_points=_p(0, 2, 3), voxel_size=1.0)
    new = [r for r in result["promoted_component_records"] if r["origin"] == "newly_detached_from_baseline_dominant"]
    assert len(new) == 1
    assert new[0]["touching_removed_voxels"] == 1


def test_i2_7_1_classifies_promotion_threshold_crossing():
    assert classify_component_gate_failure(baseline_fraction=.985, promoted_fraction=.974, threshold=.98, newly_detached_voxels=20) == "promotion_induced_threshold_crossing"


def test_i2_7_1_classifies_preexisting_baseline_fragmentation():
    assert classify_component_gate_failure(baseline_fraction=.975, promoted_fraction=.974, threshold=.98, newly_detached_voxels=0) == "preexisting_baseline_fragmentation"


def test_i2_7_1_classifies_preexisting_plus_promotion_fragmentation():
    assert classify_component_gate_failure(baseline_fraction=.975, promoted_fraction=.974, threshold=.98, newly_detached_voxels=4) == "preexisting_plus_promotion_fragmentation"


def test_i2_7_1_mask_statistics_reports_core_and_far_fraction():
    mask = np.asarray([True, True, False])
    result = mask_statistics(
        mask,
        is_core=np.asarray([True, False, True]), peripheral=np.asarray([True, True, False]),
        far=np.asarray([True, False, False]), weak=np.asarray([False, True, False]),
        support=np.asarray([2, 3, 5]), confidence=np.asarray([.4, .6, .8]), dispersion=np.asarray([.2, .3, .1]),
    )
    assert result["count"] == 2
    assert result["core_fraction"] == pytest.approx(.5)
    assert result["far_fraction"] == pytest.approx(.5)


def test_i2_7_1_mask_statistics_handles_empty_mask():
    result = mask_statistics(
        np.asarray([False, False]), is_core=np.asarray([True, False]), peripheral=np.asarray([False, False]),
        far=np.asarray([False, False]), weak=np.asarray([False, False]), support=np.asarray([2, 3]),
        confidence=np.asarray([.5, .6]), dispersion=np.asarray([.2, .3]),
    )
    assert result["count"] == 0
    assert result["support_quantiles"]["p50"] is None


def test_i2_7_1_detached_distance_is_reported_in_voxels():
    points = _p(0, 1, 4)
    result = detached_distance_statistics(points, np.asarray([True, True, False]), np.asarray([False, False, True]), 1.0)
    assert result["count"] == 1
    assert result["p50_vox"] == pytest.approx(3.0)
