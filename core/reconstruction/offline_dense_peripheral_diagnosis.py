"""Phase I2.1 targeted peripheral-artifact diagnosis contracts.

I2.1 is diagnostic only. It consumes the retained I2 Extended geometry plus
exact per-view voxel membership and measures how strongly peripheral/far
geometry depends on individual views, temporal view groups, and low-support
co-observation pairs. It never edits geometry, changes F filtering policy,
reruns SfM/DA3, promotes metric authority, or authorizes mesh reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_quality import quantiles, spatial_envelope


I21_DIAGNOSIS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class I21DiagnosisPolicy:
    minimum_support_views: int = 2
    top_views_per_temporal_half: int = 6
    robust_lower_quantile: float = 0.02
    robust_upper_quantile: float = 0.98

    def __post_init__(self) -> None:
        if int(self.minimum_support_views) < 2:
            raise ReconstructionValidationError("I2.1 minimum_support_views must be >= 2.")
        if int(self.top_views_per_temporal_half) <= 0:
            raise ReconstructionValidationError("I2.1 top_views_per_temporal_half must be positive.")
        lo = float(self.robust_lower_quantile)
        hi = float(self.robust_upper_quantile)
        if not (isfinite(lo) and isfinite(hi) and 0.0 <= lo < 0.5 < hi <= 1.0):
            raise ReconstructionValidationError("I2.1 robust quantiles must be finite and straddle 0.5.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_support_views": int(self.minimum_support_views),
            "top_views_per_temporal_half": int(self.top_views_per_temporal_half),
            "robust_lower_quantile": float(self.robust_lower_quantile),
            "robust_upper_quantile": float(self.robust_upper_quantile),
        }


def validate_observation_matrix(
    observation_matrix: object,
    baseline_support: object,
    *,
    minimum_support_views: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(observation_matrix, dtype=bool)
    support = np.asarray(baseline_support, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != len(support) or matrix.shape[0] == 0 or len(support) == 0:
        raise ReconstructionValidationError("I2.1 observation matrix must be VxN and align with baseline support.")
    if np.any(support < int(minimum_support_views)):
        raise ReconstructionValidationError("I2.1 baseline retained geometry must meet the support floor.")
    recomputed = matrix.sum(axis=0, dtype=np.int64)
    mismatch = recomputed != support
    if np.any(mismatch):
        raise ReconstructionValidationError(
            f"I2.1 exact retained-support accounting failed for {int(np.count_nonzero(mismatch))} voxels."
        )
    return matrix, support


def automatic_temporal_suspect_segments(
    selected_orders: Sequence[int],
    per_view_records: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
) -> dict[str, tuple[int, ...]]:
    orders = tuple(int(value) for value in selected_orders)
    if not orders:
        raise ReconstructionValidationError("I2.1 requires selected workspace orders.")
    selected = set(orders)
    records = [item for item in per_view_records if int(item.get("workspace_order", -1)) in selected]
    if not records:
        raise ReconstructionValidationError("I2.1 requires I2 per-view attribution records.")
    midpoint = float(np.median(np.asarray(orders, dtype=np.float64)))

    def _pick(*, early: bool) -> tuple[int, ...]:
        candidates = [
            item for item in records
            if (int(item["workspace_order"]) <= midpoint if early else int(item["workspace_order"]) > midpoint)
        ]
        candidates.sort(
            key=lambda item: (
                -int(item.get("peripheral_retained_observations", 0)),
                -int(item.get("far_peripheral_retained_observations", 0)),
                int(item["workspace_order"]),
            )
        )
        chosen = [int(item["workspace_order"]) for item in candidates[: int(top_n)]]
        return tuple(sorted(chosen))

    early = _pick(early=True)
    late = _pick(early=False)
    if not early or not late:
        raise ReconstructionValidationError("I2.1 could not form both temporal suspect segments.")
    return {"early_suspect": early, "late_suspect": late}


def _region_collapse(mask: np.ndarray, collapse: np.ndarray) -> dict[str, Any]:
    source_count = int(np.count_nonzero(mask))
    collapsed = int(np.count_nonzero(mask & collapse))
    return {
        "source_count": source_count,
        "collapsed_count": collapsed,
        "collapsed_fraction": collapsed / float(source_count) if source_count else 0.0,
    }


def removal_sensitivity(
    *,
    points: object,
    observation_matrix: object,
    baseline_support: object,
    removed_view_indices: Sequence[int],
    peripheral_mask: object,
    far_mask: object,
    weak_mask: object,
    core_mask: object,
    sector_codes: object,
    policy: I21DiagnosisPolicy,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    xyz = np.asarray(points, dtype=np.float64)
    matrix, support = validate_observation_matrix(
        observation_matrix, baseline_support, minimum_support_views=policy.minimum_support_views
    )
    if xyz.ndim != 2 or xyz.shape != (len(support), 3) or not np.isfinite(xyz).all():
        raise ReconstructionValidationError("I2.1 sensitivity points must be finite Nx3 and align with support.")
    masks = [np.asarray(value, dtype=bool).reshape(-1) for value in (peripheral_mask, far_mask, weak_mask, core_mask)]
    if any(len(mask) != len(support) for mask in masks):
        raise ReconstructionValidationError("I2.1 region masks must align with retained geometry.")
    peripheral, far, weak, core = masks
    sectors = np.asarray(sector_codes, dtype=np.uint8).reshape(-1)
    if len(sectors) != len(support):
        raise ReconstructionValidationError("I2.1 sector codes must align with retained geometry.")
    removed = tuple(sorted({int(value) for value in removed_view_indices}))
    if not removed or removed[0] < 0 or removed[-1] >= matrix.shape[0]:
        raise ReconstructionValidationError("I2.1 removed view indices are invalid.")

    removed_observations = matrix[np.asarray(removed, dtype=np.int64)].sum(axis=0, dtype=np.int64)
    remaining_support = support - removed_observations
    collapse = remaining_support < int(policy.minimum_support_views)
    survive = ~collapse

    baseline_env = spatial_envelope(
        xyz,
        lower_quantile=policy.robust_lower_quantile,
        upper_quantile=policy.robust_upper_quantile,
    )
    survivor_profile: dict[str, float | None]
    if int(np.count_nonzero(survive)) >= 8:
        survivor_env = spatial_envelope(
            xyz[survive],
            lower_quantile=policy.robust_lower_quantile,
            upper_quantile=policy.robust_upper_quantile,
        )
        survivor_profile = {
            "robust_diagonal": float(survivor_env.robust_diagonal),
            "full_diagonal": float(survivor_env.full_diagonal),
            "robust_diagonal_ratio_to_baseline": (
                float(survivor_env.robust_diagonal / baseline_env.robust_diagonal)
                if baseline_env.robust_diagonal > 0.0 else None
            ),
            "full_diagonal_ratio_to_baseline": (
                float(survivor_env.full_diagonal / baseline_env.full_diagonal)
                if baseline_env.full_diagonal > 0.0 else None
            ),
        }
    else:
        survivor_profile = {
            "robust_diagonal": None,
            "full_diagonal": None,
            "robust_diagonal_ratio_to_baseline": None,
            "full_diagonal_ratio_to_baseline": None,
        }

    labels = {1: "+x", 2: "-x", 3: "+y", 4: "-y", 5: "+z", 6: "-z"}
    directional = {
        label: int(np.count_nonzero(collapse & (sectors == code)))
        for code, label in labels.items()
    }
    result = {
        "removed_view_count": len(removed),
        "collapsed_extended": _region_collapse(np.ones(len(support), dtype=bool), collapse),
        "collapsed_peripheral": _region_collapse(peripheral, collapse),
        "collapsed_far": _region_collapse(far, collapse),
        "collapsed_weak_peripheral": _region_collapse(weak, collapse),
        "collapsed_core": _region_collapse(core, collapse),
        "remaining_support_quantiles": quantiles(remaining_support.astype(np.float64)),
        "surviving_point_count": int(np.count_nonzero(survive)),
        "surviving_geometry": survivor_profile,
        "collapsed_directional_counts": directional,
        "diagnostic_semantics": (
            "support-only counterfactual on baseline retained voxel keys; point positions and F topology/dispersion "
            "are not recomputed"
        ),
    }
    return result, collapse, remaining_support


def pair_cosupport_records(
    observation_matrix: object,
    baseline_support: object,
    view_orders: Sequence[int],
    *,
    peripheral_mask: object,
    far_mask: object,
    weak_mask: object,
) -> list[dict[str, Any]]:
    matrix = np.asarray(observation_matrix, dtype=bool)
    support = np.asarray(baseline_support, dtype=np.int64).reshape(-1)
    orders = tuple(int(value) for value in view_orders)
    if matrix.ndim != 2 or matrix.shape != (len(orders), len(support)):
        raise ReconstructionValidationError("I2.1 co-support matrix/order shape mismatch.")
    peripheral = np.asarray(peripheral_mask, dtype=bool).reshape(-1)
    far = np.asarray(far_mask, dtype=bool).reshape(-1)
    weak = np.asarray(weak_mask, dtype=bool).reshape(-1)
    if not (len(peripheral) == len(far) == len(weak) == len(support)):
        raise ReconstructionValidationError("I2.1 co-support masks are not aligned.")
    support2 = support == 2
    records: list[dict[str, Any]] = []
    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            co = matrix[i] & matrix[j]
            records.append({
                "order_a": orders[i],
                "order_b": orders[j],
                "coobserved_extended": int(np.count_nonzero(co)),
                "coobserved_peripheral": int(np.count_nonzero(co & peripheral)),
                "coobserved_far": int(np.count_nonzero(co & far)),
                "coobserved_weak_peripheral": int(np.count_nonzero(co & weak)),
                "support2_peripheral_pair": int(np.count_nonzero(co & peripheral & support2)),
                "support2_far_pair": int(np.count_nonzero(co & far & support2)),
            })
    records.sort(
        key=lambda item: (
            item["support2_peripheral_pair"],
            item["coobserved_far"],
            item["coobserved_peripheral"],
            -item["order_a"],
            -item["order_b"],
        ),
        reverse=True,
    )
    return records


def scale_profile(
    scale_by_order: Mapping[int, float],
    suspect_orders: Sequence[int],
) -> dict[str, Any]:
    values = {int(order): float(value) for order, value in scale_by_order.items() if float(value) > 0.0 and isfinite(float(value))}
    if not values:
        return {"available": False, "reason": "no_finite_positive_scales"}
    suspect = [values[int(order)] for order in suspect_orders if int(order) in values]
    if not suspect:
        return {"available": False, "reason": "suspect_segment_has_no_scale_values"}
    all_log = np.asarray([log(value) for value in values.values()], dtype=np.float64)
    median_log = float(np.median(all_log))
    mad = float(np.median(np.abs(all_log - median_log)))
    denom = max(1.4826 * mad, 1e-9)
    z_by_order = {order: float((log(value) - median_log) / denom) for order, value in values.items()}
    suspect_z = [abs(z_by_order[int(order)]) for order in suspect_orders if int(order) in z_by_order]
    return {
        "available": True,
        "segment_count": len(suspect),
        "segment_scale_quantiles": quantiles(np.asarray(suspect, dtype=np.float64)),
        "global_scale_quantiles": quantiles(np.asarray(list(values.values()), dtype=np.float64)),
        "maximum_absolute_robust_log_z": max(suspect_z) if suspect_z else 0.0,
        "robust_log_z_by_order": {str(order): value for order, value in sorted(z_by_order.items()) if order in set(map(int, suspect_orders))},
        "diagnostic_only": True,
    }
