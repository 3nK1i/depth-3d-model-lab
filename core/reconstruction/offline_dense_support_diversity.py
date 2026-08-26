"""Phase I2.4 support-diversity / parallax diagnostics.

I2.4 is diagnostic only.  It consumes the exact retained Extended voxel
membership from I2.1 together with the already-saved SfM camera poses and asks
whether suspicious peripheral geometry is supported by genuinely diverse
viewpoints or merely by a narrow cluster of nearby views.

No geometry is edited, no F thresholds are changed, and no SfM/DA3/fusion/G/UI
stage is rerun or promoted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_quality import quantiles


I24_SUPPORT_DIVERSITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class I24SupportDiversityPolicy:
    reference_quantile: float = 0.25
    maximum_support_for_low_diversity: int = 3
    minimum_reference_points: int = 128
    strong_enrichment_ratio: float = 1.50
    strong_enrichment_absolute_margin: float = 0.10

    def __post_init__(self) -> None:
        q = float(self.reference_quantile)
        if not isfinite(q) or not 0.0 < q < 0.5:
            raise ReconstructionValidationError("I2.4 reference_quantile must be finite in (0, 0.5).")
        if int(self.maximum_support_for_low_diversity) < 2:
            raise ReconstructionValidationError("I2.4 maximum_support_for_low_diversity must be >= 2.")
        if int(self.minimum_reference_points) <= 0:
            raise ReconstructionValidationError("I2.4 minimum_reference_points must be positive.")
        for name in ("strong_enrichment_ratio", "strong_enrichment_absolute_margin"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ReconstructionValidationError(f"I2.4 {name} must be finite and positive.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_quantile": float(self.reference_quantile),
            "maximum_support_for_low_diversity": int(self.maximum_support_for_low_diversity),
            "minimum_reference_points": int(self.minimum_reference_points),
            "strong_enrichment_ratio": float(self.strong_enrichment_ratio),
            "strong_enrichment_absolute_margin": float(self.strong_enrichment_absolute_margin),
        }


def _validate_inputs(
    points: object,
    observation_matrix: object,
    camera_centers: object,
    view_orders: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(observation_matrix, dtype=bool)
    centers = np.asarray(camera_centers, dtype=np.float64)
    orders = np.asarray(tuple(int(v) for v in view_orders), dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ReconstructionValidationError("I2.4 points must be finite non-empty Nx3.")
    if matrix.ndim != 2 or matrix.shape[1] != len(xyz) or matrix.shape[0] < 2:
        raise ReconstructionValidationError("I2.4 observation matrix must be VxN and align with points.")
    if centers.shape != (matrix.shape[0], 3) or not np.isfinite(centers).all():
        raise ReconstructionValidationError("I2.4 camera centers must be finite Vx3.")
    if orders.shape != (matrix.shape[0],):
        raise ReconstructionValidationError("I2.4 view order count must match observation rows.")
    support = matrix.sum(axis=0, dtype=np.int64)
    if np.any(support < 2):
        raise ReconstructionValidationError("I2.4 retained geometry must have >=2 exact supporting views.")
    return xyz, matrix, centers, orders


def support_diversity_metrics(
    points: object,
    observation_matrix: object,
    camera_centers: object,
    view_orders: Sequence[int],
) -> dict[str, np.ndarray]:
    """Measure the strongest available geometric diversity for every voxel.

    ``maximum_pair_parallax_degrees`` is the maximum angle between any two
    supporting camera-to-voxel rays.  ``maximum_baseline_to_range_ratio`` is
    the maximum supporting-camera baseline divided by the mean range from the
    two cameras to the voxel.  Both are scale invariant except the separately
    reported absolute camera baseline.

    Using the *maximum* pair evidence is conservative: if any pair of supporting
    views provides a useful baseline/parallax, the voxel is not treated as
    narrow-view simply because some other pair is nearly co-located.
    """

    xyz, matrix, centers, orders = _validate_inputs(points, observation_matrix, camera_centers, view_orders)
    support = matrix.sum(axis=0, dtype=np.int64)
    n_points = len(xyz)
    max_angle = np.zeros(n_points, dtype=np.float64)
    max_baseline_ratio = np.zeros(n_points, dtype=np.float64)
    max_baseline = np.zeros(n_points, dtype=np.float64)

    eps = 1e-12
    for i in range(matrix.shape[0] - 1):
        for j in range(i + 1, matrix.shape[0]):
            mask = matrix[i] & matrix[j]
            indices = np.flatnonzero(mask)
            if len(indices) == 0:
                continue
            vi = xyz[indices] - centers[i][None, :]
            vj = xyz[indices] - centers[j][None, :]
            ri = np.linalg.norm(vi, axis=1)
            rj = np.linalg.norm(vj, axis=1)
            valid = (ri > eps) & (rj > eps)
            if not np.any(valid):
                continue
            idx = indices[valid]
            vi = vi[valid]
            vj = vj[valid]
            ri = ri[valid]
            rj = rj[valid]
            cosine = np.einsum("ij,ij->i", vi, vj) / (ri * rj)
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            baseline = float(np.linalg.norm(centers[i] - centers[j]))
            ratio = baseline / np.maximum((ri + rj) * 0.5, eps)
            max_angle[idx] = np.maximum(max_angle[idx], angle)
            max_baseline_ratio[idx] = np.maximum(max_baseline_ratio[idx], ratio)
            max_baseline[idx] = np.maximum(max_baseline[idx], baseline)

    if not (np.isfinite(max_angle).all() and np.isfinite(max_baseline_ratio).all() and np.isfinite(max_baseline).all()):
        raise ReconstructionValidationError("I2.4 support-diversity metrics produced non-finite values.")

    # At least one pair exists for every support>=2 voxel. Zero diversity is
    # allowed only when supporting camera centers/rays truly coincide.
    row_ids = np.arange(matrix.shape[0], dtype=np.int64)[:, None]
    min_rank = np.min(np.where(matrix, row_ids, matrix.shape[0]), axis=0)
    max_rank = np.max(np.where(matrix, row_ids, -1), axis=0)
    order_grid = orders[:, None]
    order_sentinel_hi = int(np.max(orders)) + max(1, int(np.ptp(orders))) + 1
    order_sentinel_lo = int(np.min(orders)) - max(1, int(np.ptp(orders))) - 1
    min_order = np.min(np.where(matrix, order_grid, order_sentinel_hi), axis=0)
    max_order = np.max(np.where(matrix, order_grid, order_sentinel_lo), axis=0)

    return {
        "support": support,
        "maximum_pair_parallax_degrees": max_angle,
        "maximum_baseline_to_range_ratio": max_baseline_ratio,
        "maximum_camera_baseline_reconstruction_units": max_baseline,
        "supporting_rank_span": (max_rank - min_rank).astype(np.int64),
        "supporting_order_span": (max_order - min_order).astype(np.int64),
    }


def derive_reference_thresholds(
    metrics: Mapping[str, object],
    *,
    core_mask: object,
    peripheral_mask: object,
    policy: I24SupportDiversityPolicy,
) -> tuple[dict[str, Any], np.ndarray]:
    support = np.asarray(metrics["support"], dtype=np.int64).reshape(-1)
    angle = np.asarray(metrics["maximum_pair_parallax_degrees"], dtype=np.float64).reshape(-1)
    ratio = np.asarray(metrics["maximum_baseline_to_range_ratio"], dtype=np.float64).reshape(-1)
    core = np.asarray(core_mask, dtype=bool).reshape(-1)
    peripheral = np.asarray(peripheral_mask, dtype=bool).reshape(-1)
    if not (len(support) == len(angle) == len(ratio) == len(core) == len(peripheral)):
        raise ReconstructionValidationError("I2.4 reference evidence arrays must align.")

    low_support = support <= int(policy.maximum_support_for_low_diversity)
    reference = core & ~peripheral & low_support
    source = "central_core_low_support"
    if int(np.count_nonzero(reference)) < int(policy.minimum_reference_points):
        reference = core & ~peripheral
        source = "central_core_all_support_fallback"
    if int(np.count_nonzero(reference)) < max(2, min(int(policy.minimum_reference_points), 8)):
        raise ReconstructionValidationError("I2.4 has insufficient central-Core reference geometry.")

    q = float(policy.reference_quantile)
    angle_threshold = float(np.quantile(angle[reference], q))
    ratio_threshold = float(np.quantile(ratio[reference], q))
    thresholds = {
        "source": source,
        "reference_count": int(np.count_nonzero(reference)),
        "reference_quantile": q,
        "maximum_pair_parallax_degrees_threshold": angle_threshold,
        "maximum_baseline_to_range_ratio_threshold": ratio_threshold,
        "reference_angle_quantiles": quantiles(angle[reference]),
        "reference_baseline_to_range_quantiles": quantiles(ratio[reference]),
        "reference_support_quantiles": quantiles(support[reference].astype(np.float64)),
    }
    return thresholds, reference


def classify_low_diversity(
    metrics: Mapping[str, object],
    thresholds: Mapping[str, Any],
    *,
    policy: I24SupportDiversityPolicy,
) -> np.ndarray:
    support = np.asarray(metrics["support"], dtype=np.int64).reshape(-1)
    angle = np.asarray(metrics["maximum_pair_parallax_degrees"], dtype=np.float64).reshape(-1)
    ratio = np.asarray(metrics["maximum_baseline_to_range_ratio"], dtype=np.float64).reshape(-1)
    return (
        (support <= int(policy.maximum_support_for_low_diversity))
        & (angle <= float(thresholds["maximum_pair_parallax_degrees_threshold"]))
        & (ratio <= float(thresholds["maximum_baseline_to_range_ratio_threshold"]))
    )


def diversity_region_statistics(
    mask: object,
    *,
    metrics: Mapping[str, object],
    low_diversity_mask: object,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    low = np.asarray(low_diversity_mask, dtype=bool).reshape(-1)
    support = np.asarray(metrics["support"], dtype=np.int64).reshape(-1)
    angle = np.asarray(metrics["maximum_pair_parallax_degrees"], dtype=np.float64).reshape(-1)
    ratio = np.asarray(metrics["maximum_baseline_to_range_ratio"], dtype=np.float64).reshape(-1)
    baseline = np.asarray(metrics["maximum_camera_baseline_reconstruction_units"], dtype=np.float64).reshape(-1)
    rank_span = np.asarray(metrics["supporting_rank_span"], dtype=np.int64).reshape(-1)
    order_span = np.asarray(metrics["supporting_order_span"], dtype=np.int64).reshape(-1)
    n = len(selected)
    if any(len(arr) != n for arr in (low, support, angle, ratio, baseline, rank_span, order_span)):
        raise ReconstructionValidationError("I2.4 region evidence arrays must align.")
    count = int(np.count_nonzero(selected))
    low_count = int(np.count_nonzero(selected & low))
    support2 = int(np.count_nonzero(selected & (support == 2)))
    return {
        "count": count,
        "low_diversity_count": low_count,
        "low_diversity_fraction": low_count / float(count) if count else 0.0,
        "support2_count": support2,
        "support2_fraction": support2 / float(count) if count else 0.0,
        "support_quantiles": quantiles(support[selected].astype(np.float64)),
        "maximum_pair_parallax_degrees_quantiles": quantiles(angle[selected]),
        "maximum_baseline_to_range_ratio_quantiles": quantiles(ratio[selected]),
        "maximum_camera_baseline_quantiles": quantiles(baseline[selected]),
        "supporting_rank_span_quantiles": quantiles(rank_span[selected].astype(np.float64)),
        "supporting_order_span_quantiles": quantiles(order_span[selected].astype(np.float64)),
    }


def _safe_enrichment(region_fraction: float, reference_fraction: float) -> float | None:
    if reference_fraction <= 0.0:
        return None
    return float(region_fraction / reference_fraction)


def diagnostic_signal(
    *,
    reference_stats: Mapping[str, Any],
    peripheral_stats: Mapping[str, Any],
    far_stats: Mapping[str, Any],
    policy: I24SupportDiversityPolicy,
) -> dict[str, Any]:
    ref = float(reference_stats["low_diversity_fraction"])
    peripheral = float(peripheral_stats["low_diversity_fraction"])
    far = float(far_stats["low_diversity_fraction"])
    ratio_gate = float(policy.strong_enrichment_ratio)
    margin = float(policy.strong_enrichment_absolute_margin)

    def strong(value: float) -> bool:
        return value >= ref + margin and (ref <= 0.0 or value / ref >= ratio_gate)

    peripheral_strong = strong(peripheral)
    far_strong = strong(far)
    if peripheral_strong and far_strong:
        strength = "strong"
    elif peripheral_strong or far_strong:
        strength = "mixed"
    else:
        strength = "weak"
    return {
        "strength": strength,
        "supports_support_diversity_hypothesis": strength == "strong",
        "reference_low_diversity_fraction": ref,
        "peripheral_low_diversity_fraction": peripheral,
        "far_low_diversity_fraction": far,
        "peripheral_enrichment_vs_reference": _safe_enrichment(peripheral, ref),
        "far_enrichment_vs_reference": _safe_enrichment(far, ref),
        "interpretation": (
            "diagnostic only; a strong signal means low-baseline/low-parallax evidence is enriched in suspicious regions, "
            "not that any production filtering rule is authorized"
        ),
    }


def support2_pair_diversity_records(
    *,
    points: object,
    observation_matrix: object,
    camera_centers: object,
    view_orders: Sequence[int],
    peripheral_mask: object,
    far_mask: object,
    weak_mask: object,
    core_mask: object,
    low_diversity_mask: object,
) -> list[dict[str, Any]]:
    xyz, matrix, centers, orders = _validate_inputs(points, observation_matrix, camera_centers, view_orders)
    support = matrix.sum(axis=0, dtype=np.int64)
    masks = [np.asarray(v, dtype=bool).reshape(-1) for v in (peripheral_mask, far_mask, weak_mask, core_mask, low_diversity_mask)]
    if any(len(v) != len(xyz) for v in masks):
        raise ReconstructionValidationError("I2.4 support-2 pair masks must align with points.")
    peripheral, far, weak, core, low = masks
    records: list[dict[str, Any]] = []
    eps = 1e-12
    for i in range(matrix.shape[0] - 1):
        for j in range(i + 1, matrix.shape[0]):
            exact = matrix[i] & matrix[j] & (support == 2)
            count = int(np.count_nonzero(exact))
            if count == 0:
                continue
            idx = np.flatnonzero(exact)
            vi = xyz[idx] - centers[i][None, :]
            vj = xyz[idx] - centers[j][None, :]
            ri = np.linalg.norm(vi, axis=1)
            rj = np.linalg.norm(vj, axis=1)
            cosine = np.einsum("ij,ij->i", vi, vj) / np.maximum(ri * rj, eps)
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            baseline = float(np.linalg.norm(centers[i] - centers[j]))
            ratio = baseline / np.maximum((ri + rj) * 0.5, eps)
            records.append({
                "order_a": int(orders[i]),
                "order_b": int(orders[j]),
                "selected_rank_gap": int(j - i),
                "workspace_order_gap": int(abs(int(orders[j]) - int(orders[i]))),
                "camera_baseline_reconstruction_units": baseline,
                "support2_extended_count": count,
                "support2_core_count": int(np.count_nonzero(exact & core)),
                "support2_peripheral_count": int(np.count_nonzero(exact & peripheral)),
                "support2_far_count": int(np.count_nonzero(exact & far)),
                "support2_weak_peripheral_count": int(np.count_nonzero(exact & weak)),
                "support2_low_diversity_count": int(np.count_nonzero(exact & low)),
                "support2_low_diversity_peripheral_count": int(np.count_nonzero(exact & low & peripheral)),
                "support2_low_diversity_far_count": int(np.count_nonzero(exact & low & far)),
                "parallax_degrees_quantiles": quantiles(angle),
                "baseline_to_range_ratio_quantiles": quantiles(ratio),
            })
    records.sort(
        key=lambda item: (
            item["support2_low_diversity_far_count"],
            item["support2_low_diversity_peripheral_count"],
            item["support2_peripheral_count"],
            -item["selected_rank_gap"],
            -item["order_a"],
            -item["order_b"],
        ),
        reverse=True,
    )
    return records
