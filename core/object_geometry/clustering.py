"""Deterministic dependency-light voxel clustering."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np

from core.object_geometry.policy import ClusteringPolicy


@dataclass(frozen=True, slots=True)
class ClusterSelection:
    keep: np.ndarray
    voxel_size: float
    cluster_count: int
    selected_cluster_points: int
    method: str
    ambiguous: bool


def _adaptive_voxel_size(points: np.ndarray, policy: ClusteringPolicy) -> float:
    if policy.voxel_size is not None:
        return policy.voxel_size
    extent = np.ptp(points, axis=0)
    positive = extent[extent > np.finfo(np.float64).eps]
    base = float(np.median(positive)) if positive.size else 1.0
    return max(np.finfo(np.float64).eps * 1024, base * policy.adaptive_extent_fraction)


def select_voxel_cluster(
    points_xyz: np.ndarray,
    source_pixel_indices_xy: np.ndarray,
    *,
    mask_centroid_xy: tuple[float, float] | None,
    policy: ClusteringPolicy,
) -> ClusterSelection:
    points = np.asarray(points_xyz, dtype=np.float64)
    pixels = np.asarray(source_pixel_indices_xy, dtype=np.float64)
    count = len(points)
    if not policy.enabled or count < policy.minimum_cluster_points:
        return ClusterSelection(np.ones(count, dtype=bool), 0.0, 1 if count else 0, count, "disabled_or_insufficient", False)
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
                neighbour = (current[0] + offset[0], current[1] + offset[1], current[2] + offset[2])
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
    selected = clusters[selected_index]
    keep = np.zeros(count, dtype=bool)
    keep[selected] = True
    sizes = sorted((len(cluster) for cluster in clusters), reverse=True)
    ambiguous = len(sizes) > 1 and sizes[1] > 0 and sizes[0] / sizes[1] < policy.weak_dominance_ratio
    return ClusterSelection(keep, voxel_size, len(clusters), len(selected), method, ambiguous)
