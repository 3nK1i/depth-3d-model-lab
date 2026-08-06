"""Dependency-light depth and 3D point filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.object_geometry.diagnostics import (
    SupportRetentionDiagnostics,
    support_retention_diagnostics,
)
from core.object_geometry.policy import (
    BackgroundPlanePolicy,
    DepthDiscontinuityPolicy,
    StatisticalOutlierPolicy,
    SupportRetentionPolicy,
)


@dataclass(frozen=True, slots=True)
class FilterDecision:
    keep: np.ndarray
    method: str
    threshold: float | None = None
    warning: str = ""
    support_diagnostics: SupportRetentionDiagnostics | None = None
    support_guard_applied: bool = False


def depth_discontinuity_keep_mask(
    depth: np.ndarray,
    candidate_mask: np.ndarray,
    policy: DepthDiscontinuityPolicy,
) -> FilterDecision:
    """Reject isolated depth spikes while preserving locally coherent thin structures.

    The previous maximum-neighbour rule could remove both sides of one valid edge.
    This implementation requires neighbour consensus instead: a pixel survives when
    it has at least one locally consistent neighbour, or when it belongs to sparse
    image support where demanding dense neighbourhood evidence would erase thin
    legs, rails or handles.
    """

    candidate = np.asarray(candidate_mask, dtype=np.bool_)
    if not policy.enabled or np.count_nonzero(candidate) < 2:
        return FilterDecision(np.array(candidate, copy=True), "disabled")

    depth_array = np.asarray(depth, dtype=np.float64)
    pair_offsets = ((0, 1), (1, 0), (1, 1), (1, -1))
    pair_records: list[tuple[tuple[slice, slice], tuple[slice, slice], np.ndarray, np.ndarray]] = []
    samples: list[np.ndarray] = []

    height, width = candidate.shape
    for dy, dx in pair_offsets:
        if height <= abs(dy) or width <= abs(dx):
            continue
        if dx >= 0:
            source_x = slice(0, width - dx)
            target_x = slice(dx, width)
        else:
            source_x = slice(-dx, width)
            target_x = slice(0, width + dx)
        source_y = slice(0, height - dy)
        target_y = slice(dy, height)
        a = candidate[source_y, source_x]
        b = candidate[target_y, target_x]
        both = a & b
        if not np.any(both):
            continue
        da = depth_array[source_y, source_x]
        db = depth_array[target_y, target_x]
        diff = np.abs(da - db)
        samples.append(diff[both])
        pair_records.append(((source_y, source_x), (target_y, target_x), both, diff))

    if not samples:
        return FilterDecision(
            np.array(candidate, copy=True),
            "insufficient_neighbors",
            warning="depth_discontinuity_insufficient_neighbors",
        )

    values = np.concatenate(samples)
    if values.size < policy.minimum_neighbor_samples:
        return FilterDecision(
            np.array(candidate, copy=True),
            "insufficient_neighbors",
            warning="depth_discontinuity_insufficient_neighbors",
        )

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    candidate_depth = depth_array[candidate]
    representative_depth = float(np.median(np.abs(candidate_depth))) if candidate_depth.size else 0.0
    threshold = max(
        policy.minimum_absolute_threshold,
        policy.minimum_relative_threshold * representative_depth,
        median + policy.mad_multiplier * 1.4826 * mad,
    )

    comparisons = np.zeros(candidate.shape, dtype=np.uint8)
    consistent = np.zeros(candidate.shape, dtype=np.uint8)
    for source, target, both, diff in pair_records:
        source_count = comparisons[source]
        target_count = comparisons[target]
        source_count[both] += 1
        target_count[both] += 1
        agrees = both & (diff <= threshold)
        source_consistent = consistent[source]
        target_consistent = consistent[target]
        source_consistent[agrees] += 1
        target_consistent[agrees] += 1

    fractions = np.zeros(candidate.shape, dtype=np.float64)
    np.divide(consistent, comparisons, out=fractions, where=comparisons > 0)
    coherent = (
        (consistent >= policy.minimum_consistent_neighbors)
        & (fractions >= policy.minimum_consistent_fraction)
    )
    sparse = comparisons <= policy.sparse_neighbor_threshold
    keep = candidate & (coherent | (policy.preserve_sparse_support & sparse))
    sparse_preserved = bool(np.any(candidate & sparse & ~coherent & keep))
    return FilterDecision(
        keep,
        "robust_eight_neighbour_consensus_mad",
        threshold=threshold,
        warning="thin_sparse_depth_support_preserved" if sparse_preserved else "",
    )


def robust_xyz_outlier_keep_mask(
    points_xyz: np.ndarray,
    policy: StatisticalOutlierPolicy,
    *,
    source_pixel_indices_xy: np.ndarray | None = None,
    support_policy: SupportRetentionPolicy | None = None,
) -> FilterDecision:
    """Remove gross 3D outliers without collapsing articulated object support."""

    points = np.asarray(points_xyz, dtype=np.float64)
    count = len(points)
    if not policy.enabled or count < policy.minimum_points:
        warning = "outlier_filter_insufficient_points" if policy.enabled and count else ""
        return FilterDecision(
            np.ones(count, dtype=bool),
            "disabled_or_insufficient",
            warning=warning,
        )

    median = np.median(points, axis=0)
    mad = np.median(np.abs(points - median), axis=0)
    scale = 1.4826 * mad
    active = scale > np.finfo(np.float64).eps
    if not np.any(active):
        return FilterDecision(
            np.ones(count, dtype=bool),
            "degenerate_axes",
            warning="outlier_filter_degenerate_axes",
        )

    residual = np.zeros_like(points)
    residual[:, active] = np.abs(points[:, active] - median[active]) / scale[active]
    strict_keep = np.all(residual[:, active] <= policy.mad_multiplier, axis=1)

    diagnostics = None
    if (
        source_pixel_indices_xy is not None
        and support_policy is not None
        and support_policy.enabled
    ):
        diagnostics = support_retention_diagnostics(
            source_pixel_indices_xy,
            strict_keep,
            grid_size=support_policy.grid_size,
            minimum_point_retention=support_policy.minimum_point_retention,
            minimum_axis_span_retention=support_policy.minimum_axis_span_retention,
            minimum_grid_coverage_retention=support_policy.minimum_grid_coverage_retention,
        )
        if diagnostics.thin_support_loss:
            quantile = support_policy.fallback_quantile
            if quantile <= 0.0:
                fallback_keep = np.ones(count, dtype=bool)
            else:
                lower = np.quantile(points, quantile, axis=0)
                upper = np.quantile(points, 1.0 - quantile, axis=0)
                fallback_keep = np.all((points >= lower) & (points <= upper), axis=1)
            fallback_diagnostics = support_retention_diagnostics(
                source_pixel_indices_xy,
                fallback_keep,
                grid_size=support_policy.grid_size,
                minimum_point_retention=support_policy.minimum_point_retention,
                minimum_axis_span_retention=support_policy.minimum_axis_span_retention,
                minimum_grid_coverage_retention=support_policy.minimum_grid_coverage_retention,
            )
            if (
                not fallback_diagnostics.thin_support_loss
                or fallback_diagnostics.retained_points > diagnostics.retained_points
            ):
                return FilterDecision(
                    fallback_keep,
                    "support_guard_quantile_fallback",
                    threshold=quantile,
                    warning="thin_structure_support_preserved_after_outlier_guard",
                    support_diagnostics=fallback_diagnostics,
                    support_guard_applied=True,
                )

    return FilterDecision(
        strict_keep,
        "axiswise_robust_mad",
        threshold=policy.mad_multiplier,
        support_diagnostics=diagnostics,
    )


def fit_plane_ransac(points_xyz: np.ndarray, policy: BackgroundPlanePolicy) -> tuple[np.ndarray | None, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64)
    count = len(points)
    if count < max(3, policy.minimum_ring_points):
        return None, np.zeros(count, dtype=bool)
    rng = np.random.default_rng(policy.random_seed)
    best_plane: np.ndarray | None = None
    best_inliers = np.zeros(count, dtype=bool)
    for _ in range(policy.ransac_iterations):
        sample = rng.choice(count, size=3, replace=False)
        p0, p1, p2 = points[sample]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal /= norm
        offset = -float(np.dot(normal, p0))
        distances = np.abs(points @ normal + offset)
        inliers = distances <= policy.distance_threshold
        if np.count_nonzero(inliers) > np.count_nonzero(best_inliers):
            best_plane = np.array([normal[0], normal[1], normal[2], offset], dtype=np.float64)
            best_inliers = inliers
    if best_plane is None:
        return None, best_inliers
    support = np.count_nonzero(best_inliers) / count
    if support < policy.minimum_support_ratio:
        return None, best_inliers
    inlier_points = points[best_inliers]
    centroid = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, centroid))
    refined = np.array([normal[0], normal[1], normal[2], offset], dtype=np.float64)
    distances = np.abs(points @ normal + offset)
    return refined, distances <= policy.distance_threshold


def background_plane_keep_mask(
    object_points: np.ndarray,
    ring_points: np.ndarray,
    policy: BackgroundPlanePolicy,
) -> FilterDecision:
    count = len(object_points)
    if not policy.enabled:
        return FilterDecision(np.ones(count, dtype=bool), "disabled")
    plane, ring_inliers = fit_plane_ransac(ring_points, policy)
    if plane is None:
        return FilterDecision(np.ones(count, dtype=bool), "plane_unavailable", warning="background_plane_unavailable")
    ring_support = ring_points[ring_inliers]
    if ring_support.size == 0:
        return FilterDecision(np.ones(count, dtype=bool), "plane_unavailable", warning="background_plane_unavailable")
    if float(np.median(ring_support[:, 2])) < float(np.median(object_points[:, 2])) + policy.minimum_behind_separation:
        return FilterDecision(np.ones(count, dtype=bool), "plane_not_behind_object", warning="background_plane_not_behind_object")
    distances = np.abs(object_points @ plane[:3] + plane[3])
    remove = distances <= policy.distance_threshold
    fraction = float(np.count_nonzero(remove)) / max(1, count)
    if fraction > policy.maximum_removal_fraction:
        return FilterDecision(np.ones(count, dtype=bool), "unsafe_removal_fraction", warning="background_plane_removal_rejected")
    return FilterDecision(~remove, "exterior_ring_ransac", threshold=policy.distance_threshold)
