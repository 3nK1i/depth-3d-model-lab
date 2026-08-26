"""I2.5.1 spatial-extremum survivor diagnostics.

This module identifies the small retained population that actually drives the
pathological full-scene extent after I2.5 showed that conservative diversity
filters remove many suspicious voxels without improving the full/robust scene
envelope.

The helpers are diagnostic only.  They never modify F, source geometry,
persistence, metric/measurement authority, or mesh authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_quality import quantiles, spatial_envelope


I251_EXTREMUM_SURVIVOR_SCHEMA_VERSION = 1
_DIRECTION_LABELS = ("+x", "-x", "+y", "-y", "+z", "-z")


@dataclass(frozen=True)
class I251ExtremumPolicy:
    robust_lower_quantile: float = 0.02
    robust_upper_quantile: float = 0.98
    directional_tail_quantile: float = 0.90
    minimum_fraction_of_directional_max: float = 0.50
    minimum_direction_points: int = 8

    def __post_init__(self) -> None:
        lo = float(self.robust_lower_quantile)
        hi = float(self.robust_upper_quantile)
        q = float(self.directional_tail_quantile)
        frac = float(self.minimum_fraction_of_directional_max)
        if not (isfinite(lo) and isfinite(hi) and 0.0 <= lo < 0.5 < hi <= 1.0):
            raise ReconstructionValidationError("I2.5.1 robust quantiles must be finite and straddle 0.5.")
        if not isfinite(q) or not 0.5 <= q < 1.0:
            raise ReconstructionValidationError("I2.5.1 directional_tail_quantile must be in [0.5,1).")
        if not isfinite(frac) or not 0.0 < frac <= 1.0:
            raise ReconstructionValidationError("I2.5.1 minimum_fraction_of_directional_max must be in (0,1].")
        if int(self.minimum_direction_points) <= 0:
            raise ReconstructionValidationError("I2.5.1 minimum_direction_points must be positive.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "robust_lower_quantile": float(self.robust_lower_quantile),
            "robust_upper_quantile": float(self.robust_upper_quantile),
            "directional_tail_quantile": float(self.directional_tail_quantile),
            "minimum_fraction_of_directional_max": float(self.minimum_fraction_of_directional_max),
            "minimum_direction_points": int(self.minimum_direction_points),
        }


def _xyz(points: object) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ReconstructionValidationError("I2.5.1 points must be finite non-empty Nx3.")
    return xyz


def robust_directional_excess(
    points: object,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, np.ndarray | float]:
    """Measure normalized distance beyond the six sides of a robust box.

    Excess is normalized by each robust axis span, making the diagnostic stable
    in the arbitrary SfM gauge and directly comparable across axes.
    """
    xyz = _xyz(points)
    lo = np.quantile(xyz, float(lower_quantile), axis=0)
    hi = np.quantile(xyz, float(upper_quantile), axis=0)
    span = hi - lo
    if np.any(~np.isfinite(span)) or np.any(span <= 0.0):
        raise ReconstructionValidationError("I2.5.1 robust scene span must be finite and positive on all axes.")

    positive = np.maximum((xyz - hi[None, :]) / span[None, :], 0.0)
    negative = np.maximum((lo[None, :] - xyz) / span[None, :], 0.0)
    # +x,-x,+y,-y,+z,-z
    directional = np.column_stack(
        [positive[:, 0], negative[:, 0], positive[:, 1], negative[:, 1], positive[:, 2], negative[:, 2]]
    )
    max_excess = directional.max(axis=1)
    direction = np.argmax(directional, axis=1).astype(np.uint8) + 1
    direction[max_excess <= 0.0] = 0
    return {
        "lower": lo,
        "upper": hi,
        "span": span,
        "directional_excess": directional,
        "maximum_normalized_excess": max_excess,
        "dominant_direction_code": direction,
        "outside_robust_box": max_excess > 0.0,
    }


def extremum_driver_mask(
    directional_excess: object,
    *,
    policy: I251ExtremumPolicy,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(directional_excess, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0 or not np.isfinite(values).all():
        raise ReconstructionValidationError("I2.5.1 directional excess must be finite Nx6.")
    if np.any(values < 0.0):
        raise ReconstructionValidationError("I2.5.1 directional excess must be non-negative.")

    mask = np.zeros(len(values), dtype=bool)
    records: list[dict[str, Any]] = []
    for column, label in enumerate(_DIRECTION_LABELS):
        data = values[:, column]
        positive = data[data > 0.0]
        if len(positive) == 0:
            records.append({"direction": label, "outside_count": 0, "driver_count": 0, "threshold": None, "maximum": 0.0})
            continue
        maximum = float(np.max(positive))
        if len(positive) >= int(policy.minimum_direction_points):
            quantile_threshold = float(np.quantile(positive, float(policy.directional_tail_quantile)))
        else:
            quantile_threshold = 0.0
        threshold = max(quantile_threshold, maximum * float(policy.minimum_fraction_of_directional_max))
        selected = data >= threshold
        mask |= selected
        records.append(
            {
                "direction": label,
                "outside_count": int(len(positive)),
                "driver_count": int(np.count_nonzero(selected)),
                "threshold": float(threshold),
                "maximum": maximum,
            }
        )
    if not np.any(mask):
        raise ReconstructionValidationError("I2.5.1 could not identify any extremum-driving geometry.")
    return mask, {"directions": records, "driver_count": int(np.count_nonzero(mask)), "driver_fraction": float(np.mean(mask))}


def aligned_region_statistics(
    mask: object,
    *,
    support: object,
    confidence: object,
    dispersion: object,
    high_support_neighbors: object,
    radius2_neighbors: object,
    parallax_degrees: object,
    baseline_to_range: object,
    low_diversity: object,
    is_core: object,
    is_peripheral: object,
    is_far: object,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    arrays = {
        "support": np.asarray(support, dtype=np.float64).reshape(-1),
        "confidence": np.asarray(confidence, dtype=np.float64).reshape(-1),
        "dispersion": np.asarray(dispersion, dtype=np.float64).reshape(-1),
        "high_support_neighbors": np.asarray(high_support_neighbors, dtype=np.float64).reshape(-1),
        "radius2_neighbors": np.asarray(radius2_neighbors, dtype=np.float64).reshape(-1),
        "parallax_degrees": np.asarray(parallax_degrees, dtype=np.float64).reshape(-1),
        "baseline_to_range": np.asarray(baseline_to_range, dtype=np.float64).reshape(-1),
    }
    bools = {
        "low_diversity": np.asarray(low_diversity, dtype=bool).reshape(-1),
        "core": np.asarray(is_core, dtype=bool).reshape(-1),
        "peripheral": np.asarray(is_peripheral, dtype=bool).reshape(-1),
        "far": np.asarray(is_far, dtype=bool).reshape(-1),
    }
    if any(array.shape != selected.shape for array in list(arrays.values()) + list(bools.values())):
        raise ReconstructionValidationError("I2.5.1 region evidence arrays must align.")
    count = int(np.count_nonzero(selected))
    result: dict[str, Any] = {"count": count, "fraction_of_extended": float(np.mean(selected))}
    if count == 0:
        for name in arrays:
            result[name + "_quantiles"] = quantiles([])
        for name in bools:
            result[name + "_fraction"] = 0.0
        return result
    for name, array in arrays.items():
        result[name + "_quantiles"] = quantiles(array[selected])
    for name, array in bools.items():
        result[name + "_fraction"] = float(np.mean(array[selected]))
    return result


def policy_envelope_decomposition(
    points: object,
    reject_mask: object,
    *,
    lower_quantile: float,
    upper_quantile: float,
    extremum_driver_mask_fixed: object,
) -> dict[str, Any]:
    xyz = _xyz(points)
    reject = np.asarray(reject_mask, dtype=bool).reshape(-1)
    drivers = np.asarray(extremum_driver_mask_fixed, dtype=bool).reshape(-1)
    if reject.shape != (len(xyz),) or drivers.shape != reject.shape:
        raise ReconstructionValidationError("I2.5.1 policy masks must align with points.")
    kept = ~reject
    if int(np.count_nonzero(kept)) < 10:
        raise ReconstructionValidationError("I2.5.1 policy candidate retains too few points.")
    baseline = spatial_envelope(xyz, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
    candidate = spatial_envelope(xyz[kept], lower_quantile=lower_quantile, upper_quantile=upper_quantile)
    base_ratio = baseline.full_diagonal / baseline.robust_diagonal
    cand_ratio = candidate.full_diagonal / candidate.robust_diagonal
    full_shrink = 1.0 - candidate.full_diagonal / baseline.full_diagonal
    robust_shrink = 1.0 - candidate.robust_diagonal / baseline.robust_diagonal
    driver_count = int(np.count_nonzero(drivers))
    rejected_drivers = int(np.count_nonzero(reject & drivers))
    return {
        "kept_points": int(np.count_nonzero(kept)),
        "rejected_points": int(np.count_nonzero(reject)),
        "baseline_full_diagonal": float(baseline.full_diagonal),
        "candidate_full_diagonal": float(candidate.full_diagonal),
        "full_diagonal_shrink_fraction": float(full_shrink),
        "baseline_robust_diagonal": float(baseline.robust_diagonal),
        "candidate_robust_diagonal": float(candidate.robust_diagonal),
        "robust_diagonal_shrink_fraction": float(robust_shrink),
        "baseline_full_to_robust_ratio": float(base_ratio),
        "candidate_full_to_robust_ratio": float(cand_ratio),
        "ratio_improvement_fraction": float(1.0 - cand_ratio / base_ratio),
        "extremum_driver_count": driver_count,
        "rejected_extremum_drivers": rejected_drivers,
        "extremum_driver_rejection_fraction": rejected_drivers / float(driver_count) if driver_count else 0.0,
        "ratio_worsening_explained_by_robust_shrink": bool(cand_ratio > base_ratio and robust_shrink > full_shrink),
    }


def rejection_consensus(rejection_masks: Sequence[object], *, length: int) -> dict[str, np.ndarray]:
    if not rejection_masks:
        raise ReconstructionValidationError("I2.5.1 requires at least one I2.5 rejection mask.")
    stack = np.asarray([np.asarray(mask, dtype=bool).reshape(-1) for mask in rejection_masks], dtype=bool)
    if stack.ndim != 2 or stack.shape[1] != int(length):
        raise ReconstructionValidationError("I2.5.1 rejection masks must all align with baseline geometry.")
    count = stack.sum(axis=0, dtype=np.int16)
    return {
        "rejection_count": count,
        "rejected_by_any_policy": count > 0,
        "rejected_by_all_policies": count == stack.shape[0],
        "survives_all_policies": count == 0,
    }


def source_view_attribution(
    observation_matrix: object,
    view_orders: Sequence[int],
    mask: object,
    *,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    matrix = np.asarray(observation_matrix, dtype=bool)
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    orders = tuple(int(v) for v in view_orders)
    if matrix.ndim != 2 or matrix.shape[1] != len(selected) or matrix.shape[0] != len(orders):
        raise ReconstructionValidationError("I2.5.1 observation matrix/view orders do not align.")
    if int(top_k) <= 0:
        raise ReconstructionValidationError("I2.5.1 top_k must be positive.")
    denominator = int(np.count_nonzero(selected))
    records: list[dict[str, Any]] = []
    for row, order in enumerate(orders):
        count = int(np.count_nonzero(matrix[row] & selected))
        records.append(
            {
                "workspace_order": order,
                "observed_extremum_voxels": count,
                "fraction_of_selected_extrema": count / float(denominator) if denominator else 0.0,
            }
        )
    records.sort(key=lambda item: (-int(item["observed_extremum_voxels"]), int(item["workspace_order"])))
    return records[: int(top_k)]
