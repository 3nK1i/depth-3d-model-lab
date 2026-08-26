"""I4.1.8B retained-Core object-support isolation.

This module is intentionally *not* a final I4.1 target-binding step.  It takes
retained production Core voxels plus exact per-view attribution reconstructed
from the already-saved bound source clouds and asks a narrower question:

    can a compact, broadly observed, camera-attended subset of retained Core be
    isolated as the object-support region without looking at the I4.1 orbit
    pass/fail gates while selecting that subset?

The selection evidence is limited to retained-Core geometry, exact supporting
views, camera optical attention, support/span provenance and filter confidence.
Locked orbit metrics are evaluated only *after* isolation for diagnosis.  They
are never part of component scoring or acceptance.  No production target,
sampling policy, metric scale, measurement authority or mesh state is changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import (
    coverage_by_plane_mode,
)
from core.reconstruction.offline_dense_i4_orbit_qualification import (
    I4OrbitQualificationPolicy,
    orbit_coverage_metrics,
)

I4_1_8B_SCHEMA_VERSION = 1
_EPS = 1e-12
_TARGET_DEPENDENT_FINAL_GATES = frozenset(
    {
        "selected_azimuth_coverage",
        "selected_azimuth_bins",
        "selected_opposed_view_diversity",
        "selected_target_facing",
        "selected_standoff_consistency",
    }
)


@dataclass(frozen=True, slots=True)
class I418BObjectSupportIsolationPolicy:
    """Isolation heuristics only; these are not I4.1 qualification gates."""

    per_view_attention_quantile: float = 0.30
    minimum_supporting_views: int = 2
    minimum_central_observation_count: int = 2
    minimum_central_observation_fraction: float = 0.45
    minimum_support_rank_span_fraction: float = 0.025
    seed_score_quantile: float = 0.82
    maximum_seed_points: int = 30_000
    minimum_seed_points: int = 128
    cluster_radius_voxels: float = 4.0
    minimum_cluster_points: int = 48
    minimum_isolated_support_points: int = 96
    minimum_isolated_view_fraction: float = 0.45
    minimum_isolated_source_span_fraction: float = 0.60
    minimum_isolated_view_quartiles: int = 3
    minimum_best_to_second_score_ratio: float = 1.10
    maximum_isolated_fraction_of_core: float = 0.30
    maximum_bootstrap_target_drift_voxels: float = 4.0
    bootstrap_trials: int = 8
    bootstrap_fraction: float = 0.80

    def __post_init__(self) -> None:
        if not 0.05 <= float(self.per_view_attention_quantile) <= 0.50:
            raise ReconstructionValidationError("I4.1.8B per-view attention quantile must lie in [0.05,0.50].")
        if int(self.minimum_supporting_views) < 2 or int(self.minimum_central_observation_count) < 1:
            raise ReconstructionValidationError("I4.1.8B support minima are invalid.")
        if not 0.0 < float(self.minimum_central_observation_fraction) <= 1.0:
            raise ReconstructionValidationError("I4.1.8B central-observation fraction must lie in (0,1].")
        if not 0.0 <= float(self.minimum_support_rank_span_fraction) <= 1.0:
            raise ReconstructionValidationError("I4.1.8B support-rank span fraction must lie in [0,1].")
        if not 0.50 <= float(self.seed_score_quantile) < 1.0:
            raise ReconstructionValidationError("I4.1.8B seed score quantile must lie in [0.50,1).")
        if int(self.maximum_seed_points) < int(self.minimum_seed_points) or int(self.minimum_seed_points) < 32:
            raise ReconstructionValidationError("I4.1.8B seed-count limits are invalid.")
        if float(self.cluster_radius_voxels) <= 0.0 or int(self.minimum_cluster_points) < 16:
            raise ReconstructionValidationError("I4.1.8B clustering policy is invalid.")
        if int(self.minimum_isolated_support_points) < 32:
            raise ReconstructionValidationError("I4.1.8B minimum isolated support must be >=32.")
        if not 0.0 < float(self.minimum_isolated_view_fraction) <= 1.0:
            raise ReconstructionValidationError("I4.1.8B minimum view fraction must lie in (0,1].")
        if not 0.0 < float(self.minimum_isolated_source_span_fraction) <= 1.0:
            raise ReconstructionValidationError("I4.1.8B minimum source-span fraction must lie in (0,1].")
        if not 1 <= int(self.minimum_isolated_view_quartiles) <= 4:
            raise ReconstructionValidationError("I4.1.8B view-quartile minimum must lie in [1,4].")
        if float(self.minimum_best_to_second_score_ratio) <= 1.0:
            raise ReconstructionValidationError("I4.1.8B uniqueness ratio must be >1.")
        if not 0.0 < float(self.maximum_isolated_fraction_of_core) < 1.0:
            raise ReconstructionValidationError("I4.1.8B maximum isolated fraction must lie in (0,1).")
        if float(self.maximum_bootstrap_target_drift_voxels) <= 0.0:
            raise ReconstructionValidationError("I4.1.8B bootstrap drift limit must be positive.")
        if int(self.bootstrap_trials) < 3 or not 0.5 <= float(self.bootstrap_fraction) < 1.0:
            raise ReconstructionValidationError("I4.1.8B bootstrap policy is invalid.")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _points(value: object, *, name: str, minimum: int = 64) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum or not np.isfinite(points).all():
        raise ReconstructionValidationError(f"I4.1.8B {name} must be finite Nx3 with at least {minimum} points.")
    return np.ascontiguousarray(points, dtype=np.float64)


def _poses(value: object) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 4 or not np.isfinite(poses).all():
        raise ReconstructionValidationError("I4.1.8B production poses must be finite Nx4x4 with N>=4.")
    if not np.allclose(poses[:, 3, :], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ReconstructionValidationError("I4.1.8B production poses have invalid homogeneous rows.")
    return np.ascontiguousarray(poses, dtype=np.float64)


def _observation_matrix(value: object, *, views: int, points: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=bool)
    if matrix.shape != (views, points):
        raise ReconstructionValidationError(
            f"I4.1.8B observation matrix must have shape {(views, points)}, got {matrix.shape}."
        )
    return np.ascontiguousarray(matrix, dtype=bool)


def _vector(value: object | None, *, name: str, length: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(length, float(default), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) != length or not np.isfinite(array).all():
        raise ReconstructionValidationError(f"I4.1.8B {name} must be finite and aligned to retained Core.")
    return array


def _geometric_median(points: np.ndarray, *, maximum_iterations: int = 100) -> np.ndarray:
    values = _points(points, name="geometric-median support", minimum=3)
    x = np.mean(values, axis=0)
    scale = max(float(np.median(np.linalg.norm(values - x, axis=1))), _EPS)
    tolerance = max(scale * 1e-9, 1e-12)
    for _ in range(int(maximum_iterations)):
        distance = np.linalg.norm(values - x.reshape(1, 3), axis=1)
        near = distance <= tolerance
        if np.any(near):
            return np.asarray(values[np.flatnonzero(near)[0]], dtype=np.float64)
        weights = 1.0 / np.maximum(distance, tolerance)
        nxt = np.sum(values * weights[:, None], axis=0) / float(np.sum(weights))
        if float(np.linalg.norm(nxt - x)) <= tolerance:
            x = nxt
            break
        x = nxt
    if not np.isfinite(x).all():
        raise ReconstructionValidationError("I4.1.8B geometric median became non-finite.")
    return np.asarray(x, dtype=np.float64)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "p10": float(np.quantile(data, 0.10)),
        "p25": float(np.quantile(data, 0.25)),
        "p50": float(np.quantile(data, 0.50)),
        "p75": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.90)),
    }


def _point_evidence(
    points: np.ndarray,
    poses: np.ndarray,
    matrix: np.ndarray,
    support_counts: np.ndarray,
    confidence: np.ndarray,
    *,
    policy: I418BObjectSupportIsolationPolicy,
) -> dict[str, np.ndarray | list[float]]:
    """Compute target-independent observation/attention evidence per Core point."""

    views, count = matrix.shape
    support = matrix.sum(axis=0, dtype=np.int32)
    if not np.array_equal(support.astype(np.int64), np.asarray(support_counts, dtype=np.int64)):
        mismatch = int(np.count_nonzero(support.astype(np.int64) != np.asarray(support_counts, dtype=np.int64)))
        raise ReconstructionValidationError(
            f"I4.1.8B exact per-view attribution disagrees with retained Core support counts for {mismatch} voxels."
        )

    central_votes = np.zeros(count, dtype=np.int32)
    angle_sum = np.zeros(count, dtype=np.float64)
    attention_cutoffs: list[float] = []
    centers = poses[:, :3, 3]
    forward = np.asarray(poses[:, :3, 2], dtype=np.float64)
    forward /= np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), _EPS)

    for rank in range(views):
        indices = np.flatnonzero(matrix[rank])
        if len(indices) == 0:
            raise ReconstructionValidationError(f"I4.1.8B production view rank {rank} contributes no retained Core voxels.")
        vectors = points[indices] - centers[rank].reshape(1, 3)
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > _EPS
        if not np.all(valid):
            raise ReconstructionValidationError("I4.1.8B retained Core contains a point at a production camera center.")
        cosine = np.sum(vectors * forward[rank].reshape(1, 3), axis=1) / lengths
        angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        cutoff = float(np.quantile(angles, float(policy.per_view_attention_quantile)))
        attention_cutoffs.append(cutoff)
        central = angles <= cutoff + 1e-12
        central_votes[indices[central]] += 1
        angle_sum[indices] += angles

    mean_angle = angle_sum / np.maximum(support, 1)
    central_fraction = central_votes / np.maximum(support, 1)
    ranks = np.arange(views, dtype=np.int32)[:, None]
    min_rank = np.min(np.where(matrix, ranks, views), axis=0)
    max_rank = np.max(np.where(matrix, ranks, -1), axis=0)
    rank_span_fraction = (max_rank - min_rank) / float(max(views - 1, 1))

    support_scale = max(float(np.quantile(support.astype(np.float64), 0.90)), float(policy.minimum_supporting_views) + 1.0)
    support_score = np.clip((support - float(policy.minimum_supporting_views)) / max(support_scale - float(policy.minimum_supporting_views), 1.0), 0.0, 1.0)
    angle_scale = max(float(np.quantile(mean_angle, 0.75)), 1e-6)
    angle_score = np.clip(1.0 - mean_angle / angle_scale, 0.0, 1.0)
    confidence_min = float(np.min(confidence)); confidence_span = float(np.ptp(confidence))
    confidence_score = np.ones(count, dtype=np.float64) if confidence_span <= _EPS else (confidence - confidence_min) / confidence_span

    score = (
        0.36 * central_fraction
        + 0.24 * np.clip(rank_span_fraction, 0.0, 1.0)
        + 0.20 * support_score
        + 0.10 * angle_score
        + 0.10 * confidence_score
    )
    return {
        "support": support,
        "central_votes": central_votes,
        "central_fraction": central_fraction,
        "mean_attention_angle_degrees": mean_angle,
        "support_rank_span_fraction": rank_span_fraction,
        "score": score,
        "attention_cutoffs_degrees": attention_cutoffs,
    }


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int32)

    def find(self, value: int) -> int:
        root = int(value)
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[value]) != value:
            nxt = int(self.parent[value]); self.parent[value] = root; value = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra = self.find(int(a)); rb = self.find(int(b))
        if ra == rb:
            return
        if int(self.size[ra]) < int(self.size[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _seed_components(
    points: np.ndarray,
    seed_indices: np.ndarray,
    *,
    voxel_size: float,
    policy: I418BObjectSupportIsolationPolicy,
) -> list[np.ndarray]:
    if len(seed_indices) == 0:
        return []
    seed_points = points[seed_indices]
    tree = cKDTree(seed_points)
    pairs = tree.query_pairs(r=float(policy.cluster_radius_voxels) * float(voxel_size), output_type="ndarray")
    union = _UnionFind(len(seed_indices))
    if len(pairs):
        for a, b in np.asarray(pairs, dtype=np.int64):
            union.union(int(a), int(b))
    groups: dict[int, list[int]] = {}
    for local in range(len(seed_indices)):
        root = union.find(local)
        groups.setdefault(root, []).append(int(seed_indices[local]))
    components = [np.asarray(values, dtype=np.int64) for values in groups.values() if len(values) >= int(policy.minimum_cluster_points)]
    components.sort(key=lambda arr: len(arr), reverse=True)
    return components


def _component_record(
    indices: np.ndarray,
    *,
    points: np.ndarray,
    matrix: np.ndarray,
    source_indices: np.ndarray,
    evidence: Mapping[str, np.ndarray | list[float]],
    voxel_size: float,
) -> dict[str, Any]:
    subset = points[indices]
    target = _geometric_median(subset)
    radial = np.linalg.norm(subset - target.reshape(1, 3), axis=1)
    covariance = np.cov(subset.T, bias=True) if len(subset) >= 3 else np.zeros((3, 3), dtype=np.float64)
    eigen = np.maximum(np.sort(np.linalg.eigvalsh(covariance)), 0.0)
    eig_sum = max(float(np.sum(eigen)), _EPS)
    volumetric = float(eigen[0] / eig_sum)
    active_views = np.any(matrix[:, indices], axis=1)
    active_ranks = np.flatnonzero(active_views)
    view_fraction = float(len(active_ranks) / matrix.shape[0])
    if len(active_ranks) >= 2:
        source_values = source_indices[active_ranks]
        source_span = float((np.max(source_values) - np.min(source_values)) / max(float(np.max(source_indices) - np.min(source_indices)), 1.0))
    else:
        source_span = 0.0
    quartile_ids = np.minimum((active_ranks * 4) // max(matrix.shape[0], 1), 3) if len(active_ranks) else np.empty((0,), dtype=np.int64)
    quartiles = int(len(np.unique(quartile_ids)))
    score_arr = np.asarray(evidence["score"], dtype=np.float64)[indices]
    central_arr = np.asarray(evidence["central_fraction"], dtype=np.float64)[indices]
    rank_span_arr = np.asarray(evidence["support_rank_span_fraction"], dtype=np.float64)[indices]
    support_arr = np.asarray(evidence["support"], dtype=np.float64)[indices]
    radius_vox = float(np.quantile(radial, 0.90) / float(voxel_size))
    # Selection score deliberately excludes every I4.1 orbit metric.  It rewards
    # retained-Core attention, provenance breadth, 3-D structure, size and
    # compactness only.
    shape_factor = 0.25 + 0.75 * min(1.0, volumetric / 0.025)
    compactness_factor = 1.0 / np.sqrt(max(radius_vox, 1.0))
    selection_score = float(
        np.mean(score_arr)
        * np.sqrt(float(len(indices)))
        * (0.45 + 0.55 * view_fraction)
        * (0.45 + 0.55 * source_span)
        * shape_factor
        * compactness_factor
    )
    return {
        "indices": indices,
        "target": target,
        "seed_count": int(len(indices)),
        "selection_score": selection_score,
        "mean_point_evidence_score": float(np.mean(score_arr)),
        "central_fraction_quantiles": _quantiles(central_arr),
        "support_rank_span_fraction_quantiles": _quantiles(rank_span_arr),
        "support_count_quantiles": _quantiles(support_arr),
        "active_view_count": int(len(active_ranks)),
        "active_view_fraction": view_fraction,
        "active_view_ranks": [int(v) for v in active_ranks],
        "active_source_span_fraction": source_span,
        "active_view_quartiles": quartiles,
        "robust_radius_voxels": radius_vox,
        "covariance_eigenvalues": [float(v) for v in eigen],
        "volumetric_fraction": volumetric,
    }


def _bootstrap_target_stability(
    points: np.ndarray,
    *,
    voxel_size: float,
    policy: I418BObjectSupportIsolationPolicy,
) -> dict[str, Any]:
    target = _geometric_median(points)
    rng = np.random.default_rng(41802)
    sample_count = max(3, int(np.floor(float(policy.bootstrap_fraction) * len(points))))
    drifts: list[float] = []
    for _ in range(int(policy.bootstrap_trials)):
        selected = rng.choice(len(points), size=sample_count, replace=False)
        trial = _geometric_median(points[selected])
        drifts.append(float(np.linalg.norm(trial - target)))
    maximum = float(max(drifts)) if drifts else 0.0
    return {
        "target_reconstruction": [float(v) for v in target],
        "trials": int(policy.bootstrap_trials),
        "sample_fraction": float(policy.bootstrap_fraction),
        "drift_reconstruction_units": _quantiles(np.asarray(drifts, dtype=np.float64)),
        "maximum_drift_reconstruction_units": maximum,
        "maximum_drift_voxels": float(maximum / float(voxel_size)),
        "stable": bool(maximum / float(voxel_size) <= float(policy.maximum_bootstrap_target_drift_voxels)),
    }


def _posthoc_orbit_evaluation(
    poses: np.ndarray,
    target: np.ndarray,
    *,
    policy: I4OrbitQualificationPolicy,
) -> dict[str, Any]:
    metric = orbit_coverage_metrics(
        poses,
        target,
        azimuth_bin_count=int(policy.azimuth_bin_count),
        facing_angle_degrees=float(policy.maximum_facing_angle_degrees),
    )
    modes = coverage_by_plane_mode(
        poses,
        target,
        azimuth_bin_count=int(policy.azimuth_bin_count),
        facing_angle_degrees=float(policy.maximum_facing_angle_degrees),
    )
    geometry_pass = all(
        float(item["azimuth_coverage_degrees"]) >= float(policy.minimum_selected_azimuth_coverage_degrees)
        and int(item["occupied_azimuth_bins"]) >= int(policy.minimum_selected_occupied_azimuth_bins)
        and float(item["maximum_opposed_view_angle_degrees"]) >= float(policy.minimum_opposed_view_angle_degrees)
        for item in modes.values()
    )
    current_all = bool(
        float(metric["azimuth_coverage_degrees"]) >= float(policy.minimum_selected_azimuth_coverage_degrees)
        and int(metric["occupied_azimuth_bins"]) >= int(policy.minimum_selected_occupied_azimuth_bins)
        and float(metric["maximum_opposed_view_angle_degrees"]) >= float(policy.minimum_opposed_view_angle_degrees)
        and float(metric["target_facing_fraction"]) >= float(policy.minimum_target_facing_fraction)
        and float(metric["standoff_p90_to_p10_ratio"]) <= float(policy.maximum_standoff_p90_to_p10_ratio)
    )
    return {
        "selection_used_orbit_metrics": False,
        "current_metric": metric,
        "plane_modes": modes,
        "robust_locked_orbit_pass": bool(geometry_pass and current_all),
    }


def build_i4_1_8b_retained_core_object_support_isolation(
    *,
    source_i4_1_8_summary: Mapping[str, Any],
    source_i4_1_8a_summary: Mapping[str, Any],
    production_selected_camera_matrices: object,
    production_selected_source_sequence_indices: object,
    retained_core_points: object,
    retained_core_voxel_size: float,
    exact_view_observation_matrix: object,
    core_support_counts: object,
    core_filter_confidence: object | None = None,
    qualification_policy: I4OrbitQualificationPolicy | None = None,
    isolation_policy: I418BObjectSupportIsolationPolicy | None = None,
) -> dict[str, Any]:
    """Isolate object-support Core without consulting orbit pass/fail for selection."""

    source = source_i4_1_8_summary
    diag = source_i4_1_8a_summary
    if not isinstance(source, Mapping) or str(source.get("phase")) != "I4.1.8" or str(source.get("status")) != "degraded":
        raise ReconstructionValidationError("I4.1.8B requires a completed DEGRADED I4.1.8 summary.")
    if not isinstance(diag, Mapping) or str(diag.get("phase")) != "I4.1.8A" or str(diag.get("status")) != "diagnosis_ready":
        raise ReconstructionValidationError("I4.1.8B requires completed I4.1.8A diagnosis evidence.")
    diagnosis = diag.get("diagnosis")
    if not isinstance(diagnosis, Mapping):
        raise ReconstructionValidationError("I4.1.8B I4.1.8A diagnosis payload is missing.")
    if str(diagnosis.get("classification")) not in {
        "trajectory_circle_not_yet_retained_core_supported",
        "insufficient_retained_core_evidence_for_target_correction",
    }:
        raise ReconstructionValidationError(
            "I4.1.8B is intended only for unresolved retained-Core object-support diagnosis cases."
        )

    final_assessment = source.get("final_assessment")
    stage_gates = source.get("stage_gates")
    authority = source.get("authority")
    if not isinstance(final_assessment, Mapping) or not isinstance(stage_gates, Mapping) or not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.1.8B source summary lacks final/stage/authority evidence.")
    failed = tuple(sorted(str(v) for v in final_assessment.get("failed_gates", [])))
    unrelated = tuple(name for name in failed if name not in _TARGET_DEPENDENT_FINAL_GATES)
    if unrelated:
        raise ReconstructionValidationError("I4.1.8B refuses an I4.1.8 source with unrelated final failures.")
    if not all(bool(stage_gates.get(name)) for name in (
        "production_fusion", "formal_filtering", "canonical_persistence", "ordinary_i2_bound_scene_quality"
    )):
        raise ReconstructionValidationError("I4.1.8B requires the complete production F/G/I2 stage chain to have passed.")
    if any(bool(authority.get(name)) for name in (
        "metric_scale_claimed", "measurement_authoritative", "surface_mesh_generated", "dense_consistency_threshold_relaxation_authorized"
    )):
        raise ReconstructionValidationError("I4.1.8B refuses promoted metric/measurement/mesh/threshold authority.")
    diag_authority = diag.get("authority")
    if not isinstance(diag_authority, Mapping) or any(bool(diag_authority.get(name)) for name in (
        "production_target_changed", "orbit_thresholds_changed", "metric_scale_claimed", "measurement_authoritative", "surface_mesh_generated"
    )):
        raise ReconstructionValidationError("I4.1.8B requires diagnosis-only I4.1.8A evidence with no authority promotion.")

    points = _points(retained_core_points, name="retained Core points")
    poses = _poses(production_selected_camera_matrices)
    sources = np.asarray(production_selected_source_sequence_indices, dtype=np.int64).reshape(-1)
    if len(sources) != len(poses) or len(np.unique(sources)) != len(sources):
        raise ReconstructionValidationError("I4.1.8B selected source indices must be unique and aligned to production poses.")
    matrix = _observation_matrix(exact_view_observation_matrix, views=len(poses), points=len(points))
    support = _vector(core_support_counts, name="Core support counts", length=len(points), default=0.0)
    confidence = _vector(core_filter_confidence, name="Core filter confidence", length=len(points), default=1.0)
    voxel_size = float(retained_core_voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ReconstructionValidationError("I4.1.8B retained Core voxel size must be finite and positive.")
    ip = isolation_policy or I418BObjectSupportIsolationPolicy()
    qp = qualification_policy or I4OrbitQualificationPolicy()

    evidence = _point_evidence(points, poses, matrix, support, confidence, policy=ip)
    support_i = np.asarray(evidence["support"], dtype=np.int32)
    central_votes = np.asarray(evidence["central_votes"], dtype=np.int32)
    central_fraction = np.asarray(evidence["central_fraction"], dtype=np.float64)
    rank_span = np.asarray(evidence["support_rank_span_fraction"], dtype=np.float64)
    score = np.asarray(evidence["score"], dtype=np.float64)

    eligible = (
        (support_i >= int(ip.minimum_supporting_views))
        & (central_votes >= int(ip.minimum_central_observation_count))
        & (central_fraction >= float(ip.minimum_central_observation_fraction))
        & (rank_span >= float(ip.minimum_support_rank_span_fraction))
    )
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) >= int(ip.minimum_seed_points):
        cutoff = float(np.quantile(score[eligible_indices], float(ip.seed_score_quantile)))
        seeds = eligible_indices[score[eligible_indices] >= cutoff - 1e-12]
        if len(seeds) > int(ip.maximum_seed_points):
            order = np.argsort(score[seeds], kind="stable")
            seeds = seeds[order[-int(ip.maximum_seed_points):]]
    else:
        cutoff = float("nan")
        seeds = np.empty((0,), dtype=np.int64)

    components = _seed_components(points, seeds, voxel_size=voxel_size, policy=ip)
    records = [
        _component_record(
            component,
            points=points,
            matrix=matrix,
            source_indices=sources,
            evidence=evidence,
            voxel_size=voxel_size,
        )
        for component in components
    ]
    records.sort(key=lambda item: (float(item["selection_score"]), int(item["seed_count"])), reverse=True)

    best = records[0] if records else None
    second = records[1] if len(records) > 1 else None
    if best is not None:
        best_score = float(best["selection_score"])
        second_score = float(second["selection_score"]) if second is not None else 0.0
        uniqueness = float(best_score / max(second_score, _EPS)) if second is not None else float("inf")
        best_indices = np.asarray(best["indices"], dtype=np.int64)
        isolated_fraction = float(len(best_indices) / len(points))
        stability = _bootstrap_target_stability(points[best_indices], voxel_size=voxel_size, policy=ip)
        count_ok = len(best_indices) >= int(ip.minimum_isolated_support_points)
        view_ok = float(best["active_view_fraction"]) >= float(ip.minimum_isolated_view_fraction)
        span_ok = float(best["active_source_span_fraction"]) >= float(ip.minimum_isolated_source_span_fraction)
        quartiles_ok = int(best["active_view_quartiles"]) >= int(ip.minimum_isolated_view_quartiles)
        uniqueness_ok = second is None or uniqueness >= float(ip.minimum_best_to_second_score_ratio)
        bounded_fraction_ok = isolated_fraction <= float(ip.maximum_isolated_fraction_of_core)
        stability_ok = bool(stability["stable"])
        isolation_ready = bool(count_ok and view_ok and span_ok and quartiles_ok and uniqueness_ok and bounded_fraction_ok and stability_ok)
        candidate_target = np.asarray(best["target"], dtype=np.float64)
        posthoc = _posthoc_orbit_evaluation(poses, candidate_target, policy=qp)
    else:
        best_score = 0.0; second_score = 0.0; uniqueness = 0.0
        best_indices = np.empty((0,), dtype=np.int64)
        isolated_fraction = 0.0
        stability = None
        count_ok = view_ok = span_ok = quartiles_ok = uniqueness_ok = bounded_fraction_ok = stability_ok = False
        isolation_ready = False
        candidate_target = None
        posthoc = None

    if len(eligible_indices) < int(ip.minimum_seed_points):
        classification = "insufficient_multiview_attention_seed_support"
    elif not records:
        classification = "no_compact_attention_supported_core_component"
    elif not uniqueness_ok:
        classification = "ambiguous_retained_core_object_support"
    elif not (count_ok and view_ok and span_ok and quartiles_ok):
        classification = "best_component_lacks_broad_retained_view_support"
    elif not bounded_fraction_ok:
        classification = "best_component_is_not_object_compact_relative_to_core"
    elif not stability_ok:
        classification = "isolated_support_target_is_not_stable"
    else:
        classification = "retained_core_object_support_isolated"

    public_components: list[dict[str, Any]] = []
    for rank, record in enumerate(records[:10]):
        public_components.append({
            key: value
            for key, value in record.items()
            if key not in {"indices", "target"}
        } | {
            "rank": int(rank),
            "target_reconstruction": [float(v) for v in np.asarray(record["target"], dtype=np.float64)],
        })

    return {
        "phase": "I4.1.8B",
        "schema_version": I4_1_8B_SCHEMA_VERSION,
        "status": "object_support_isolated" if isolation_ready else "degraded",
        "source": {
            "i4_1_8_status": str(source.get("status")),
            "i4_1_8a_status": str(diag.get("status")),
            "i4_1_8a_classification": str(diagnosis.get("classification")),
            "i4_1_8_failed_final_gates": list(failed),
        },
        "counts": {
            "production_selected_cameras": int(len(poses)),
            "retained_core_points": int(len(points)),
            "attention_eligible_core_points": int(len(eligible_indices)),
            "seed_core_points": int(len(seeds)),
            "seed_components": int(len(records)),
            "isolated_support_points": int(len(best_indices)),
        },
        "isolation_policy": ip.as_dict(),
        "evidence": {
            "exact_view_attribution_support_counts_match": True,
            "per_view_attention_cutoffs_degrees": [float(v) for v in evidence["attention_cutoffs_degrees"]],
            "support_count_quantiles": _quantiles(support_i.astype(np.float64)),
            "central_observation_fraction_quantiles": _quantiles(central_fraction),
            "support_rank_span_fraction_quantiles": _quantiles(rank_span),
            "point_evidence_score_quantiles": _quantiles(score),
            "seed_score_cutoff": None if not np.isfinite(cutoff) else float(cutoff),
            "component_selection_uses_orbit_metrics": False,
            "component_selection_uses_locked_i4_1_gates": False,
        },
        "candidate_components": public_components,
        "selected_support": None if best is None else {
            "selection_basis": "retained-Core exact view attribution + camera optical attention + support span + compact 3-D geometry",
            "selection_score": best_score,
            "second_best_selection_score": second_score,
            "best_to_second_score_ratio": None if second is None else uniqueness,
            "core_fraction": isolated_fraction,
            "active_view_count": int(best["active_view_count"]),
            "active_view_fraction": float(best["active_view_fraction"]),
            "active_view_ranks": list(best["active_view_ranks"]),
            "active_source_span_fraction": float(best["active_source_span_fraction"]),
            "active_view_quartiles": int(best["active_view_quartiles"]),
            "robust_radius_voxels": float(best["robust_radius_voxels"]),
            "volumetric_fraction": float(best["volumetric_fraction"]),
            "target_reconstruction": [float(v) for v in candidate_target],
            "target_bootstrap_stability": stability,
            "readiness_gates": {
                "minimum_point_count": bool(count_ok),
                "broad_view_support": bool(view_ok),
                "broad_source_span": bool(span_ok),
                "view_quartile_coverage": bool(quartiles_ok),
                "unique_best_component": bool(uniqueness_ok),
                "object_compact_relative_to_core": bool(bounded_fraction_ok),
                "target_bootstrap_stable": bool(stability_ok),
            },
            "posthoc_orbit_diagnosis": posthoc,
            "posthoc_orbit_used_for_selection_or_readiness": False,
        },
        "diagnosis": {
            "classification": classification,
            "object_support_isolation_ready": bool(isolation_ready),
            "recommends_i4_1_8c_final_target_binding_design": bool(isolation_ready),
        },
        "authority": {
            "single_authoritative_sfm_frame": True,
            "sfm_rerun": False,
            "da3_rerun": False,
            "dense_pair_rerun": False,
            "fusion_rerun": False,
            "filtering_rerun": False,
            "persistence_rerun": False,
            "production_binding_rerun": False,
            "production_target_changed": False,
            "production_sampling_integration_authorized": False,
            "orbit_thresholds_changed": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_target_change": False,
            "authorizes_i4_1_final_promotion": False,
            "authorizes_i4_2_detail_preservation": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": (
                "Design I4.1.8C final retained-Core target binding and locked I4.1 closure using this independently isolated Core support."
                if isolation_ready
                else "Do not bind a final object target; review object-support isolation evidence and remain fail-closed."
            ),
        },
        "_selected_core_indices": best_indices,
    }


__all__ = [
    "I4_1_8B_SCHEMA_VERSION",
    "I418BObjectSupportIsolationPolicy",
    "build_i4_1_8b_retained_core_object_support_isolation",
]
