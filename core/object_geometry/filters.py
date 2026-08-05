"""Dependency-light depth and 3D point filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.object_geometry.policy import BackgroundPlanePolicy, DepthDiscontinuityPolicy, StatisticalOutlierPolicy


@dataclass(frozen=True, slots=True)
class FilterDecision:
    keep: np.ndarray
    method: str
    threshold: float | None = None
    warning: str = ""


def depth_discontinuity_keep_mask(
    depth: np.ndarray,
    candidate_mask: np.ndarray,
    policy: DepthDiscontinuityPolicy,
) -> FilterDecision:
    candidate = np.asarray(candidate_mask, dtype=np.bool_)
    if not policy.enabled or np.count_nonzero(candidate) < 2:
        return FilterDecision(np.array(candidate, copy=True), "disabled")
    depth_array = np.asarray(depth, dtype=np.float64)
    maximum_difference = np.zeros(depth_array.shape, dtype=np.float64)
    samples: list[np.ndarray] = []
    for dy, dx in ((0, 1), (1, 0)):
        if candidate.shape[0] <= dy or candidate.shape[1] <= dx:
            continue
        source_y = slice(0, candidate.shape[0] - dy)
        source_x = slice(0, candidate.shape[1] - dx)
        target_y = slice(dy, candidate.shape[0])
        target_x = slice(dx, candidate.shape[1])
        a = candidate[source_y, source_x]
        b = candidate[target_y, target_x]
        both = a & b
        if not np.any(both):
            continue
        da = depth_array[source_y, source_x]
        db = depth_array[target_y, target_x]
        diff = np.abs(da - db)
        values = diff[both]
        samples.append(values)
        left = maximum_difference[source_y, source_x]
        right = maximum_difference[target_y, target_x]
        left[both] = np.maximum(left[both], values)
        right[both] = np.maximum(right[both], values)
    if not samples:
        return FilterDecision(np.array(candidate, copy=True), "insufficient_neighbors", warning="depth_discontinuity_insufficient_neighbors")
    values = np.concatenate(samples)
    if values.size < policy.minimum_neighbor_samples:
        return FilterDecision(np.array(candidate, copy=True), "insufficient_neighbors", warning="depth_discontinuity_insufficient_neighbors")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(policy.minimum_absolute_threshold, median + policy.mad_multiplier * 1.4826 * mad)
    keep = candidate & (maximum_difference <= threshold)
    return FilterDecision(keep, "robust_four_neighbour_mad", threshold=threshold)


def robust_xyz_outlier_keep_mask(
    points_xyz: np.ndarray,
    policy: StatisticalOutlierPolicy,
) -> FilterDecision:
    points = np.asarray(points_xyz, dtype=np.float64)
    count = len(points)
    if not policy.enabled or count < policy.minimum_points:
        warning = "outlier_filter_insufficient_points" if policy.enabled and count else ""
        return FilterDecision(np.ones(count, dtype=bool), "disabled_or_insufficient", warning=warning)
    median = np.median(points, axis=0)
    mad = np.median(np.abs(points - median), axis=0)
    scale = 1.4826 * mad
    active = scale > np.finfo(np.float64).eps
    if not np.any(active):
        return FilterDecision(np.ones(count, dtype=bool), "degenerate_axes", warning="outlier_filter_degenerate_axes")
    residual = np.zeros_like(points)
    residual[:, active] = np.abs(points[:, active] - median[active]) / scale[active]
    keep = np.all(residual[:, active] <= policy.mad_multiplier, axis=1)
    return FilterDecision(keep, "axiswise_robust_mad", threshold=policy.mad_multiplier)


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
