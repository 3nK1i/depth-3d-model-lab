from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_extremum_local_geometry import (
    I252LocalGeometryPolicy,
    derive_local_geometry_reference,
    diagnostic_signal,
    local_geometry_region_statistics,
    local_geometry_signatures,
    local_shape_metrics,
    strong_manifold_metrics,
    supporting_ray_alignment,
)


def _cloud() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    plane = np.column_stack([
        rng.uniform(-2, 2, 400),
        rng.uniform(-2, 2, 400),
        rng.normal(0, .03, 400),
    ])
    tail = np.column_stack([
        np.linspace(2.3, 7.0, 80),
        rng.normal(0, .03, 80),
        rng.normal(0, .03, 80),
    ])
    points = np.vstack([plane, tail])
    anchor = np.zeros(len(points), dtype=bool); anchor[:400] = True
    return points, anchor


def test_policy_rejects_small_knn() -> None:
    with pytest.raises(ReconstructionValidationError):
        I252LocalGeometryPolicy(knn_size=4)


def test_policy_rejects_bad_reference_quantile() -> None:
    with pytest.raises(ReconstructionValidationError):
        I252LocalGeometryPolicy(reference_quantile=.5)


def test_local_shape_metrics_identify_linear_tail() -> None:
    points, _ = _cloud()
    metrics = local_shape_metrics(points, voxel_size=.1, knn_size=12)
    assert metrics["local_linearity"][-40:].mean() > metrics["local_linearity"][:200].mean()
    assert metrics["principal_axis"].shape == points.shape


def test_local_shape_metrics_reject_invalid_voxel_size() -> None:
    points, _ = _cloud()
    with pytest.raises(ReconstructionValidationError):
        local_shape_metrics(points, voxel_size=0.0, knn_size=12)


def test_strong_manifold_distance_grows_along_tail() -> None:
    points, anchor = _cloud()
    metrics = strong_manifold_metrics(points, strong_anchor_mask=anchor, voxel_size=.1, radius_voxels=3)
    assert metrics["nearest_strong_anchor_distance_voxels"][-1] > metrics["nearest_strong_anchor_distance_voxels"][405]


def test_strong_manifold_requires_anchors() -> None:
    points, _ = _cloud()
    with pytest.raises(ReconstructionValidationError):
        strong_manifold_metrics(points, strong_anchor_mask=np.zeros(len(points)), voxel_size=.1, radius_voxels=3)


def test_supporting_ray_alignment_is_high_for_ray_aligned_axis() -> None:
    points = np.asarray([[2.,0,0],[3.,0,0],[4.,0,0]])
    axis = np.tile([1.,0,0], (3,1))
    matrix = np.asarray([[1,1,1],[1,1,1]], dtype=bool)
    centers = np.asarray([[0.,0,0],[0.,.1,0]])
    alignment = supporting_ray_alignment(points, principal_axis=axis, observation_matrix=matrix, camera_centers=centers)
    assert np.all(alignment > .99)


def test_supporting_ray_alignment_rejects_bad_centers() -> None:
    with pytest.raises(ReconstructionValidationError):
        supporting_ray_alignment(np.ones((4,3)), principal_axis=np.ones((4,3)), observation_matrix=np.ones((2,4)), camera_centers=np.ones((3,3)))


def _metrics_for_reference(n: int = 300) -> dict[str, np.ndarray]:
    x = np.linspace(0, 1, n)
    return {
        "nearest_strong_anchor_distance_voxels": x,
        "strong_anchor_neighbors_within_radius": np.rint(20 - 10*x),
        "knn_median_distance_voxels": .5 + x,
        "knn_p90_distance_voxels": .8 + x,
        "local_linearity": .1 + .4*x,
        "local_planarity": .6 - .2*x,
        "local_scattering": .1 + .05*x,
        "supporting_ray_alignment": .2 + .5*x,
    }


def test_reference_thresholds_are_finite() -> None:
    metrics = _metrics_for_reference()
    thresholds = derive_local_geometry_reference(metrics, reference_mask=np.ones(300, dtype=bool), policy=I252LocalGeometryPolicy(minimum_reference_points=64))
    assert thresholds["anchor_distance_high"] > 0
    assert thresholds["anchor_count_low"] >= 0


def test_reference_requires_minimum_points() -> None:
    metrics = _metrics_for_reference(20)
    with pytest.raises(ReconstructionValidationError):
        derive_local_geometry_reference(metrics, reference_mask=np.ones(20, dtype=bool), policy=I252LocalGeometryPolicy(minimum_reference_points=64))


def test_signatures_combine_manifold_and_shape_evidence() -> None:
    metrics = _metrics_for_reference(10)
    metrics["nearest_strong_anchor_distance_voxels"][-1] = 10
    metrics["knn_median_distance_voxels"][-1] = 5
    thresholds = {"anchor_distance_high":2.,"knn_spacing_high":2.,"linearity_high":.8,"ray_alignment_high":.8,"anchor_count_low":2.}
    sig = local_geometry_signatures(metrics, thresholds)
    assert sig["combined_extremum_signature"][-1]


def test_region_statistics_reports_signature_fraction() -> None:
    metrics = _metrics_for_reference(10)
    signatures = {name: np.asarray([1,0,1,0,0,0,0,0,0,0], dtype=bool) for name in ("manifold_separated","locally_sparse","ray_like","combined_extremum_signature")}
    stats = local_geometry_region_statistics(np.asarray([1,1,0,0,0,0,0,0,0,0], dtype=bool), metrics=metrics, signatures=signatures)
    assert stats["count"] == 2
    assert stats["combined_extremum_signature_fraction"] == .5


def test_region_statistics_rejects_misalignment() -> None:
    metrics = _metrics_for_reference(10)
    signatures = {"combined_extremum_signature": np.zeros(9, dtype=bool)}
    with pytest.raises(ReconstructionValidationError):
        local_geometry_region_statistics(np.ones(10, dtype=bool), metrics=metrics, signatures=signatures)


def test_diagnostic_signal_can_be_strong() -> None:
    policy = I252LocalGeometryPolicy()
    ref = {"combined_extremum_signature_fraction": .10}
    drivers = {"combined_extremum_signature_fraction": .35}
    stubborn = {"combined_extremum_signature_fraction": .55}
    signal = diagnostic_signal(reference_stats=ref, stubborn_stats=stubborn, driver_stats=drivers, policy=policy)
    assert signal["strength"] == "strong"
    assert signal["supports_local_geometry_manifold_hypothesis"]
