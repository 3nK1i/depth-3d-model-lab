"""Deterministic dependency-light voxel clustering."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np

from core.object_geometry.diagnostics import (
    SupportRetentionDiagnostics,
    support_retention_diagnostics,
)
from core.object_geometry.policy import ClusteringPolicy, SupportRetentionPolicy


@dataclass(frozen=True, slots=True)
class ClusterSelection:
    keep: np.ndarray
    voxel_size: float
    cluster_count: int
    selected_cluster_points: int
    method: str
    ambiguous: bool
    support_diagnostics: SupportRetentionDiagnostics | None = None
    support_guard_applied: bool = False
    merged_cluster_count: int = 1


def _adaptive_voxel_size(points: np.ndarray, policy: ClusteringPolicy) -> float:
    if policy.voxel_size is not None:
        return policy.voxel_size
    extent = np.ptp(points, axis=0)
    positive = extent[extent > np.finfo(np.float64).eps]
    base = float(np.median(positive)) if positive.size else 1.0
    return max(np.finfo(np.float64).eps * 1024, base * policy.adaptive_extent_fraction)


def _interval_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    if a1 < b0:
        return b0 - a1 - 1.0
    if b1 < a0:
        return a0 - b1 - 1.0
    return 0.0


def _pixel_bbox(pixels: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.min(pixels[:, 0])),
        float(np.max(pixels[:, 0])),
        float(np.min(pixels[:, 1])),
        float(np.max(pixels[:, 1])),
    )


def _clusters_are_image_adjacent(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    maximum_gap: int,
) -> bool:
    ax0, ax1, ay0, ay1 = _pixel_bbox(first_pixels)
    bx0, bx1, by0, by1 = _pixel_bbox(second_pixels)
    return max(
        _interval_gap(ax0, ax1, bx0, bx1),
        _interval_gap(ay0, ay1, by0, by1),
    ) <= float(maximum_gap)


def _clusters_are_depth_consistent(
    first_points: np.ndarray,
    second_points: np.ndarray,
    policy: ClusteringPolicy,
) -> bool:
    first_z = float(np.median(first_points[:, 2]))
    second_z = float(np.median(second_points[:, 2]))
    allowed = max(
        policy.maximum_absolute_depth_gap,
        policy.maximum_relative_depth_gap * max(abs(first_z), abs(second_z), 1e-9),
    )
    return abs(first_z - second_z) <= allowed


def select_voxel_cluster(
    points_xyz: np.ndarray,
    source_pixel_indices_xy: np.ndarray,
    *,
    mask_centroid_xy: tuple[float, float] | None,
    policy: ClusteringPolicy,
    support_policy: SupportRetentionPolicy | None = None,
) -> ClusterSelection:
    points = np.asarray(points_xyz, dtype=np.float64)
    pixels = np.asarray(source_pixel_indices_xy, dtype=np.float64)
    count = len(points)
    if not policy.enabled or count < policy.minimum_cluster_points:
        return ClusterSelection(
            np.ones(count, dtype=bool),
            0.0,
            1 if count else 0,
            count,
            "disabled_or_insufficient",
            False,
        )

    voxel_size = _adaptive_voxel_size(points, policy)
    origin = np.min(points, axis=0)
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    voxel_to_points: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(keys):
        voxel_to_points.setdefault(tuple(int(v) for v in key), []).append(index)

    unvisited = set(voxel_to_points)
    clusters: list[list[int]] = []
    offsets = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    )
    while unvisited:
        start = min(unvisited)
        unvisited.remove(start)
        queue = deque([start])
        component_voxels = [start]
        while queue:
            current = queue.popleft()
            for offset in offsets:
                neighbour = (
                    current[0] + offset[0],
                    current[1] + offset[1],
                    current[2] + offset[2],
                )
                if neighbour in unvisited:
                    unvisited.remove(neighbour)
                    queue.append(neighbour)
                    component_voxels.append(neighbour)
        point_indices: list[int] = []
        for voxel in component_voxels:
            point_indices.extend(voxel_to_points[voxel])
        clusters.append(sorted(point_indices))

    clusters.sort(key=lambda values: (-len(values), values[0]))
    selected_index = 0
    method = "largest_cluster"
    if mask_centroid_xy is not None:
        centre = np.asarray(mask_centroid_xy, dtype=np.float64)
        nearest_point = int(np.argmin(np.sum((pixels - centre) ** 2, axis=1)))
        for index, cluster in enumerate(clusters):
            if nearest_point in cluster and len(cluster) >= policy.minimum_cluster_points:
                selected_index = index
                method = "mask_centroid_cluster"
                break

    selected_clusters = {selected_index}
    selected = clusters[selected_index]
    strict_keep = np.zeros(count, dtype=bool)
    strict_keep[selected] = True
    diagnostics = None
    support_guard_applied = False

    if support_policy is not None and support_policy.enabled:
        diagnostics = support_retention_diagnostics(
            pixels,
            strict_keep,
            grid_size=support_policy.grid_size,
            minimum_point_retention=support_policy.minimum_point_retention,
            minimum_axis_span_retention=support_policy.minimum_axis_span_retention,
            minimum_grid_coverage_retention=support_policy.minimum_grid_coverage_retention,
        )

    if (
        diagnostics is not None
        and diagnostics.thin_support_loss
        and policy.preserve_mask_connected_support
    ):
        changed = True
        while changed:
            changed = False
            for candidate_index, candidate in enumerate(clusters):
                if candidate_index in selected_clusters:
                    continue
                if len(candidate) < policy.minimum_preserved_cluster_points:
                    continue
                candidate_points = points[candidate]
                candidate_pixels = pixels[candidate]
                for retained_index in tuple(selected_clusters):
                    retained = clusters[retained_index]
                    if not _clusters_are_image_adjacent(
                        pixels[retained], candidate_pixels, policy.maximum_pixel_gap
                    ):
                        continue
                    if not _clusters_are_depth_consistent(
                        points[retained], candidate_points, policy
                    ):
                        continue
                    selected_clusters.add(candidate_index)
                    changed = True
                    break

        if len(selected_clusters) > 1:
            merged_keep = np.zeros(count, dtype=bool)
            for index in selected_clusters:
                merged_keep[clusters[index]] = True
            merged_diagnostics = support_retention_diagnostics(
                pixels,
                merged_keep,
                grid_size=support_policy.grid_size,
                minimum_point_retention=support_policy.minimum_point_retention,
                minimum_axis_span_retention=support_policy.minimum_axis_span_retention,
                minimum_grid_coverage_retention=support_policy.minimum_grid_coverage_retention,
            )
            if merged_diagnostics.retained_points > diagnostics.retained_points:
                strict_keep = merged_keep
                diagnostics = merged_diagnostics
                method = f"{method}_with_mask_support_guard"
                support_guard_applied = True

    sizes = sorted((len(cluster) for cluster in clusters), reverse=True)
    ambiguous = (
        len(sizes) > 1
        and sizes[1] > 0
        and sizes[0] / sizes[1] < policy.weak_dominance_ratio
    )
    return ClusterSelection(
        strict_keep,
        voxel_size,
        len(clusters),
        int(np.count_nonzero(strict_keep)),
        method,
        ambiguous,
        support_diagnostics=diagnostics,
        support_guard_applied=support_guard_applied,
        merged_cluster_count=len(selected_clusters),
    )
