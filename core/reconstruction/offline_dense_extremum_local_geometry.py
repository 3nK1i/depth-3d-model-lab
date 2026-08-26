"""I2.5.2 extremum local-geometry / manifold diagnostics.

This module characterises the tiny population of I2.5.1 extremum-driving and
stubborn survivor voxels using local 3D shape, neighbourhood spacing,
proximity to strong multi-view manifold anchors, and alignment of local
principal structure with the supporting camera rays.

The helpers are diagnostic only.  They never mutate F, source geometry,
persistence, metric/measurement authority, or mesh authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_quality import quantiles


I252_LOCAL_GEOMETRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class I252LocalGeometryPolicy:
    knn_size: int = 16
    manifold_radius_voxels: float = 3.0
    strong_anchor_min_support: int = 3
    strong_anchor_min_high_neighbors: int = 2
    reference_quantile: float = 0.90
    minimum_reference_points: int = 128
    strong_enrichment_ratio: float = 1.75
    strong_enrichment_absolute_margin: float = 0.15

    def __post_init__(self) -> None:
        if int(self.knn_size) < 6:
            raise ReconstructionValidationError("I2.5.2 knn_size must be >= 6.")
        if not isfinite(float(self.manifold_radius_voxels)) or float(self.manifold_radius_voxels) <= 0.0:
            raise ReconstructionValidationError("I2.5.2 manifold radius must be positive and finite.")
        if int(self.strong_anchor_min_support) < 2 or int(self.strong_anchor_min_high_neighbors) < 0:
            raise ReconstructionValidationError("I2.5.2 strong-anchor thresholds are invalid.")
        q = float(self.reference_quantile)
        if not isfinite(q) or not 0.5 < q < 1.0:
            raise ReconstructionValidationError("I2.5.2 reference_quantile must be in (0.5,1).")
        if int(self.minimum_reference_points) <= 0:
            raise ReconstructionValidationError("I2.5.2 minimum_reference_points must be positive.")
        if float(self.strong_enrichment_ratio) <= 1.0 or float(self.strong_enrichment_absolute_margin) <= 0.0:
            raise ReconstructionValidationError("I2.5.2 enrichment gates must be positive and meaningful.")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _xyz(points: object) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ReconstructionValidationError("I2.5.2 points must be finite non-empty Nx3.")
    return xyz


def _aligned(mask: object, length: int, *, name: str) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).reshape(-1)
    if result.shape != (length,):
        raise ReconstructionValidationError(f"I2.5.2 {name} mask must align with points.")
    return result


def local_shape_metrics(points: object, *, voxel_size: float, knn_size: int = 16) -> dict[str, np.ndarray]:
    """Compute KNN spacing and covariance-eigenvalue shape descriptors.

    Eigenvalues are ordered lambda1 >= lambda2 >= lambda3.  The conventional
    linearity / planarity / scattering descriptors are scale-free and useful
    for distinguishing thin ray-like tails from legitimate surface patches.
    """
    xyz = _xyz(points)
    voxel = float(voxel_size)
    k = int(knn_size)
    if not isfinite(voxel) or voxel <= 0.0:
        raise ReconstructionValidationError("I2.5.2 voxel_size must be finite and positive.")
    if k < 6 or len(xyz) <= k:
        raise ReconstructionValidationError("I2.5.2 requires more points than knn_size and knn_size >= 6.")

    tree = cKDTree(xyz)
    distances, indices = tree.query(xyz, k=k + 1, workers=-1)
    distances = np.asarray(distances[:, 1:], dtype=np.float64)
    indices = np.asarray(indices[:, 1:], dtype=np.int64)
    neighbours = xyz[indices]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / float(k)
    eigenvalues = np.linalg.eigvalsh(covariance)[:, ::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    l1 = np.maximum(eigenvalues[:, 0], 1e-15)
    l2 = eigenvalues[:, 1]
    l3 = eigenvalues[:, 2]
    linearity = np.clip((l1 - l2) / l1, 0.0, 1.0)
    planarity = np.clip((l2 - l3) / l1, 0.0, 1.0)
    scattering = np.clip(l3 / l1, 0.0, 1.0)

    # Principal axis associated with lambda1.  eigvalsh does not return vectors;
    # use eigh once on the already-small 3x3 covariance batches.
    _, vectors = np.linalg.eigh(covariance)
    principal = vectors[:, :, -1]
    principal /= np.maximum(np.linalg.norm(principal, axis=1, keepdims=True), 1e-15)

    return {
        "knn_median_distance_voxels": np.median(distances, axis=1) / voxel,
        "knn_p90_distance_voxels": np.quantile(distances, 0.90, axis=1) / voxel,
        "local_linearity": linearity,
        "local_planarity": planarity,
        "local_scattering": scattering,
        "principal_axis": principal,
        "knn_indices": indices,
    }


def strong_manifold_metrics(
    points: object,
    *,
    strong_anchor_mask: object,
    voxel_size: float,
    radius_voxels: float,
) -> dict[str, np.ndarray]:
    """Measure distance and local connection to strong multi-view anchors."""
    xyz = _xyz(points)
    anchor = _aligned(strong_anchor_mask, len(xyz), name="strong_anchor")
    voxel = float(voxel_size)
    radius_v = float(radius_voxels)
    if not isfinite(voxel) or voxel <= 0.0 or not isfinite(radius_v) or radius_v <= 0.0:
        raise ReconstructionValidationError("I2.5.2 manifold scale must be finite and positive.")
    anchor_indices = np.flatnonzero(anchor)
    if len(anchor_indices) < 8:
        raise ReconstructionValidationError("I2.5.2 requires at least eight strong manifold anchors.")
    anchor_points = xyz[anchor_indices]
    tree = cKDTree(anchor_points)

    distance, nearest = tree.query(xyz, k=2, workers=-1)
    distance = np.asarray(distance, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.int64)
    # Non-anchor points use nearest anchor.  Anchors themselves use the nearest
    # *other* anchor so the reference distribution is not artificially zero.
    first = distance[:, 0]
    nearest_distance = first.copy()
    nearest_global = anchor_indices[np.clip(nearest[:, 0], 0, len(anchor_indices) - 1)]
    self_hit = anchor & (nearest_global == np.arange(len(xyz))) & (distance[:, 0] <= 1e-12)
    nearest_distance[self_hit] = distance[self_hit, 1]

    radius = radius_v * voxel
    counts = np.asarray(tree.query_ball_point(xyz, r=radius, return_length=True, workers=-1), dtype=np.int32)
    counts[anchor] = np.maximum(counts[anchor] - 1, 0)
    return {
        "nearest_strong_anchor_distance_voxels": nearest_distance / voxel,
        "strong_anchor_neighbors_within_radius": counts,
    }


def supporting_ray_alignment(
    points: object,
    *,
    principal_axis: object,
    observation_matrix: object,
    camera_centers: object,
) -> np.ndarray:
    """Absolute alignment between local principal axis and mean supporting ray."""
    xyz = _xyz(points)
    axis = np.asarray(principal_axis, dtype=np.float64)
    matrix = np.asarray(observation_matrix, dtype=bool)
    centers = np.asarray(camera_centers, dtype=np.float64)
    if axis.shape != xyz.shape or matrix.ndim != 2 or matrix.shape[1] != len(xyz):
        raise ReconstructionValidationError("I2.5.2 ray-alignment inputs are misaligned.")
    if centers.shape != (matrix.shape[0], 3) or not np.isfinite(centers).all():
        raise ReconstructionValidationError("I2.5.2 camera centers must be finite Vx3.")
    if not np.isfinite(axis).all():
        raise ReconstructionValidationError("I2.5.2 principal axes must be finite Nx3.")

    ray_sum = np.zeros_like(xyz, dtype=np.float64)
    counts = matrix.sum(axis=0, dtype=np.int32)
    if np.any(counts <= 0):
        raise ReconstructionValidationError("I2.5.2 retained voxels must have at least one supporting camera.")
    for row, center in enumerate(centers):
        selected = matrix[row]
        if not np.any(selected):
            continue
        rays = xyz[selected] - center[None, :]
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-15)
        ray_sum[selected] += rays
    mean_ray = ray_sum / np.maximum(np.linalg.norm(ray_sum, axis=1, keepdims=True), 1e-15)
    return np.clip(np.abs(np.einsum("ij,ij->i", axis, mean_ray)), 0.0, 1.0)


def derive_local_geometry_reference(
    metrics: Mapping[str, object],
    *,
    reference_mask: object,
    policy: I252LocalGeometryPolicy,
) -> dict[str, float]:
    reference = np.asarray(reference_mask, dtype=bool).reshape(-1)
    arrays = {
        "anchor_distance": np.asarray(metrics["nearest_strong_anchor_distance_voxels"], dtype=np.float64).reshape(-1),
        "knn_spacing": np.asarray(metrics["knn_median_distance_voxels"], dtype=np.float64).reshape(-1),
        "linearity": np.asarray(metrics["local_linearity"], dtype=np.float64).reshape(-1),
        "ray_alignment": np.asarray(metrics["supporting_ray_alignment"], dtype=np.float64).reshape(-1),
        "anchor_count": np.asarray(metrics["strong_anchor_neighbors_within_radius"], dtype=np.float64).reshape(-1),
    }
    if any(array.shape != reference.shape for array in arrays.values()):
        raise ReconstructionValidationError("I2.5.2 reference metrics must align.")
    count = int(np.count_nonzero(reference))
    if count < int(policy.minimum_reference_points):
        raise ReconstructionValidationError("I2.5.2 central Core reference is too small.")
    q = float(policy.reference_quantile)
    # Distance / spacing / linearity / ray alignment are suspicious when high;
    # manifold-neighbour count is suspicious when low, therefore use 1-q.
    return {
        "reference_count": float(count),
        "anchor_distance_high": float(np.quantile(arrays["anchor_distance"][reference], q)),
        "knn_spacing_high": float(np.quantile(arrays["knn_spacing"][reference], q)),
        "linearity_high": float(np.quantile(arrays["linearity"][reference], q)),
        "ray_alignment_high": float(np.quantile(arrays["ray_alignment"][reference], q)),
        "anchor_count_low": float(np.quantile(arrays["anchor_count"][reference], 1.0 - q)),
    }


def local_geometry_signatures(
    metrics: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> dict[str, np.ndarray]:
    anchor_distance = np.asarray(metrics["nearest_strong_anchor_distance_voxels"], dtype=np.float64)
    knn = np.asarray(metrics["knn_median_distance_voxels"], dtype=np.float64)
    linearity = np.asarray(metrics["local_linearity"], dtype=np.float64)
    ray = np.asarray(metrics["supporting_ray_alignment"], dtype=np.float64)
    anchor_count = np.asarray(metrics["strong_anchor_neighbors_within_radius"], dtype=np.float64)
    shapes = {arr.shape for arr in (anchor_distance, knn, linearity, ray, anchor_count)}
    if len(shapes) != 1:
        raise ReconstructionValidationError("I2.5.2 signature metrics must align.")
    manifold_separated = (
        (anchor_distance > float(thresholds["anchor_distance_high"]))
        | (anchor_count <= float(thresholds["anchor_count_low"]))
    )
    locally_sparse = knn > float(thresholds["knn_spacing_high"])
    ray_like = (
        (linearity > float(thresholds["linearity_high"]))
        & (ray > float(thresholds["ray_alignment_high"]))
    )
    combined = manifold_separated & (locally_sparse | ray_like)
    return {
        "manifold_separated": manifold_separated,
        "locally_sparse": locally_sparse,
        "ray_like": ray_like,
        "combined_extremum_signature": combined,
    }


def local_geometry_region_statistics(
    mask: object,
    *,
    metrics: Mapping[str, object],
    signatures: Mapping[str, object],
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    arrays = {
        "nearest_strong_anchor_distance_voxels": np.asarray(metrics["nearest_strong_anchor_distance_voxels"], dtype=np.float64).reshape(-1),
        "strong_anchor_neighbors_within_radius": np.asarray(metrics["strong_anchor_neighbors_within_radius"], dtype=np.float64).reshape(-1),
        "knn_median_distance_voxels": np.asarray(metrics["knn_median_distance_voxels"], dtype=np.float64).reshape(-1),
        "knn_p90_distance_voxels": np.asarray(metrics["knn_p90_distance_voxels"], dtype=np.float64).reshape(-1),
        "local_linearity": np.asarray(metrics["local_linearity"], dtype=np.float64).reshape(-1),
        "local_planarity": np.asarray(metrics["local_planarity"], dtype=np.float64).reshape(-1),
        "local_scattering": np.asarray(metrics["local_scattering"], dtype=np.float64).reshape(-1),
        "supporting_ray_alignment": np.asarray(metrics["supporting_ray_alignment"], dtype=np.float64).reshape(-1),
    }
    sigs = {name: np.asarray(value, dtype=bool).reshape(-1) for name, value in signatures.items()}
    if any(array.shape != selected.shape for array in list(arrays.values()) + list(sigs.values())):
        raise ReconstructionValidationError("I2.5.2 region evidence arrays must align.")
    count = int(np.count_nonzero(selected))
    result: dict[str, Any] = {"count": count, "fraction_of_extended": float(np.mean(selected))}
    for name, array in arrays.items():
        result[name + "_quantiles"] = quantiles(array[selected])
    for name, array in sigs.items():
        result[name + "_fraction"] = float(np.mean(array[selected])) if count else 0.0
    return result


def diagnostic_signal(
    *,
    reference_stats: Mapping[str, Any],
    stubborn_stats: Mapping[str, Any],
    driver_stats: Mapping[str, Any],
    policy: I252LocalGeometryPolicy,
) -> dict[str, Any]:
    ref = float(reference_stats["combined_extremum_signature_fraction"])
    stubborn = float(stubborn_stats["combined_extremum_signature_fraction"])
    drivers = float(driver_stats["combined_extremum_signature_fraction"])

    def enrichment(value: float) -> float | None:
        return None if ref <= 0.0 else float(value / ref)

    stubborn_enrichment = enrichment(stubborn)
    driver_enrichment = enrichment(drivers)
    stubborn_strong = (
        stubborn >= ref + float(policy.strong_enrichment_absolute_margin)
        and (stubborn_enrichment is None or stubborn_enrichment >= float(policy.strong_enrichment_ratio))
    )
    driver_support = (
        drivers >= ref + float(policy.strong_enrichment_absolute_margin) * 0.5
        and (driver_enrichment is None or driver_enrichment >= 1.25)
    )
    strength = "strong" if stubborn_strong and driver_support else ("mixed" if stubborn_strong or driver_support else "weak")
    return {
        "strength": strength,
        "supports_local_geometry_manifold_hypothesis": strength == "strong",
        "reference_signature_fraction": ref,
        "driver_signature_fraction": drivers,
        "stubborn_signature_fraction": stubborn,
        "driver_enrichment_vs_reference": driver_enrichment,
        "stubborn_enrichment_vs_reference": stubborn_enrichment,
        "interpretation": (
            "diagnostic only; a strong signal means extremum survivors are enriched for local manifold separation, "
            "sparse neighbourhoods, or ray-aligned linear structure relative to legitimate central Core geometry"
        ),
    }
