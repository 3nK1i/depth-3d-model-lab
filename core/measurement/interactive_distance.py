"""Shared two-point interactive distance estimation.

This module is deliberately dependency-light.  It owns the numeric contract used
by both Measure & Inspect image picking and Reconstruction canonical-surface
picking.  Results are estimates with explicit heuristic uncertainty unless the
available scale/calibration evidence qualifies them further; none of these
contracts claim certified metrology.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np

from core.contracts.spatial import CalibrationStatus
from core.models.metadata import DepthScale, IntrinsicsSource


class InteractiveDistanceState(str, Enum):
    ESTIMATED = "estimated"
    QUALIFIED = "qualified"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class InteractiveDistanceEstimate:
    """One immutable two-point measurement result.

    ``distance_m`` / ``expanded_uncertainty_m`` are exposed only when a metric
    estimate is justified.  ``distance_source_units`` always retains the native
    distance used by the source geometry when available.
    """

    state: InteractiveDistanceState
    source_kind: str
    point_a_source: tuple[float, float, float] | None
    point_b_source: tuple[float, float, float] | None
    distance_source_units: float | None
    distance_m: float | None
    expanded_uncertainty_m: float | None
    uncertainty_basis: str
    calibration_status: str
    scale_status: str
    unit_to_metre: float | None = None
    reasons: tuple[str, ...] = ()
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, InteractiveDistanceState):
            raise TypeError("state must be InteractiveDistanceState.")
        source = str(self.source_kind).strip()
        if not source:
            raise ValueError("source_kind is required.")
        object.__setattr__(self, "source_kind", source)
        object.__setattr__(self, "reasons", tuple(str(value).strip() for value in self.reasons if str(value).strip()))
        if self.certified_metrology:
            raise ValueError("Interactive measurement cannot claim certified metrology.")
        for field_name in ("distance_source_units", "distance_m", "expanded_uncertainty_m", "unit_to_metre"):
            value = getattr(self, field_name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{field_name} must be finite and non-negative when present.")
                object.__setattr__(self, field_name, value)
        if self.state in {InteractiveDistanceState.ESTIMATED, InteractiveDistanceState.QUALIFIED}:
            if self.distance_m is None or self.expanded_uncertainty_m is None:
                raise ValueError("Metric interactive distance states require distance and uncertainty.")
        if self.state in {InteractiveDistanceState.UNAVAILABLE, InteractiveDistanceState.REJECTED}:
            if self.distance_m is not None or self.expanded_uncertainty_m is not None:
                raise ValueError("Unavailable/rejected interactive distance cannot expose metric values.")
        object.__setattr__(self, "certified_metrology", False)

    @property
    def has_metric_estimate(self) -> bool:
        return self.distance_m is not None and self.expanded_uncertainty_m is not None

    @property
    def is_provisional(self) -> bool:
        return self.state is InteractiveDistanceState.ESTIMATED


def _as_pixel(value: Sequence[int | float], *, width: int, height: int) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("pixel must contain x and y.")
    x, y = int(round(float(value[0]))), int(round(float(value[1])))
    if x < 0 or y < 0 or x >= width or y >= height:
        raise ValueError("pixel is outside the depth image.")
    return x, y


def _local_depth(depth: np.ndarray, valid: np.ndarray, x: int, y: int, *, radius: int = 2) -> tuple[float, float]:
    h, w = depth.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64)
    patch_valid = np.asarray(valid[y0:y1, x0:x1], dtype=bool)
    mask = patch_valid & np.isfinite(patch) & (patch > 0.0)
    samples = patch[mask]
    if samples.size < 1:
        raise ValueError("No valid positive depth exists around the selected point.")
    z = float(np.median(samples))
    mad = float(np.median(np.abs(samples - z))) if samples.size > 1 else 0.0
    robust_sigma = 1.4826 * mad
    # Depth models remain estimates even with calibrated intrinsics.  Keep an
    # explicit conservative floor instead of presenting pixel-perfect certainty.
    sigma = max(robust_sigma, 0.0125 * z, 0.0015)
    return z, sigma


def _backproject(x: int, y: int, z: float, K: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.asarray(((x - cx) * z / fx, (y - cy) * z / fy, z), dtype=np.float64)


def estimate_depth_pixel_distance(
    *,
    depth: np.ndarray,
    depth_valid_mask: np.ndarray,
    intrinsics_matrix: np.ndarray,
    pixel_a_xy: Sequence[int | float],
    pixel_b_xy: Sequence[int | float],
    depth_scale: DepthScale | str,
    intrinsics_source: IntrinsicsSource | str,
    calibration_status: CalibrationStatus | str,
) -> InteractiveDistanceEstimate:
    """Estimate real-world distance between two pixels on a metric depth map."""

    depth_values = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(depth_valid_mask, dtype=bool)
    K = np.asarray(intrinsics_matrix, dtype=np.float64)
    if depth_values.ndim != 2 or valid.shape != depth_values.shape:
        raise ValueError("depth and depth_valid_mask must be matching HxW arrays.")
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        return InteractiveDistanceEstimate(
            state=InteractiveDistanceState.REJECTED,
            source_kind="image_depth",
            point_a_source=None,
            point_b_source=None,
            distance_source_units=None,
            distance_m=None,
            expanded_uncertainty_m=None,
            uncertainty_basis="heuristic_depth_intrinsics",
            calibration_status=str(getattr(calibration_status, "value", calibration_status)),
            scale_status=str(getattr(depth_scale, "value", depth_scale)),
            reasons=("invalid_intrinsics",),
        )
    scale = depth_scale if isinstance(depth_scale, DepthScale) else DepthScale(str(depth_scale))
    intr_source = intrinsics_source if isinstance(intrinsics_source, IntrinsicsSource) else IntrinsicsSource(str(intrinsics_source))
    cal = calibration_status if isinstance(calibration_status, CalibrationStatus) else CalibrationStatus(str(calibration_status))
    h, w = depth_values.shape
    try:
        ax, ay = _as_pixel(pixel_a_xy, width=w, height=h)
        bx, by = _as_pixel(pixel_b_xy, width=w, height=h)
        za, sigma_a = _local_depth(depth_values, valid, ax, ay)
        zb, sigma_b = _local_depth(depth_values, valid, bx, by)
    except ValueError as exc:
        return InteractiveDistanceEstimate(
            state=InteractiveDistanceState.REJECTED,
            source_kind="image_depth",
            point_a_source=None,
            point_b_source=None,
            distance_source_units=None,
            distance_m=None,
            expanded_uncertainty_m=None,
            uncertainty_basis="heuristic_depth_intrinsics",
            calibration_status=cal.value,
            scale_status=scale.value,
            reasons=(str(exc),),
        )

    point_a = _backproject(ax, ay, za, K)
    point_b = _backproject(bx, by, zb, K)
    native_distance = float(np.linalg.norm(point_b - point_a))
    if native_distance <= 1e-9:
        return InteractiveDistanceEstimate(
            state=InteractiveDistanceState.REJECTED,
            source_kind="image_depth",
            point_a_source=tuple(float(v) for v in point_a),
            point_b_source=tuple(float(v) for v in point_b),
            distance_source_units=native_distance,
            distance_m=None,
            expanded_uncertainty_m=None,
            uncertainty_basis="heuristic_depth_intrinsics",
            calibration_status=cal.value,
            scale_status=scale.value,
            reasons=("selected_points_are_coincident",),
        )

    if scale is not DepthScale.METRIC:
        return InteractiveDistanceEstimate(
            state=InteractiveDistanceState.UNAVAILABLE,
            source_kind="image_depth",
            point_a_source=tuple(float(v) for v in point_a),
            point_b_source=tuple(float(v) for v in point_b),
            distance_source_units=native_distance,
            distance_m=None,
            expanded_uncertainty_m=None,
            uncertainty_basis="heuristic_depth_intrinsics",
            calibration_status=cal.value,
            scale_status=scale.value,
            reasons=("real_world_scale_not_available_for_selected_depth_model",),
        )

    # One-pixel localization sensitivity plus explicit intrinsics/model-scale
    # terms.  These are heuristic components, not an empirical confidence interval.
    mean_z = 0.5 * (za + zb)
    pixel_sigma = 0.5 * mean_z * math.sqrt((1.0 / float(K[0, 0])) ** 2 + (1.0 / float(K[1, 1])) ** 2)
    intrinsics_rel = 0.008 if cal is CalibrationStatus.CALIBRATED else (0.020 if intr_source is IntrinsicsSource.MODEL_PREDICTED else 0.030)
    model_scale_rel = 0.030
    uncertainty = math.sqrt(
        sigma_a * sigma_a
        + sigma_b * sigma_b
        + pixel_sigma * pixel_sigma
        + (intrinsics_rel * native_distance) ** 2
        + (model_scale_rel * native_distance) ** 2
    )
    uncertainty = max(uncertainty, 0.002)
    state = (
        InteractiveDistanceState.QUALIFIED
        if cal is CalibrationStatus.CALIBRATED and intr_source is IntrinsicsSource.CALIBRATED
        else InteractiveDistanceState.ESTIMATED
    )
    return InteractiveDistanceEstimate(
        state=state,
        source_kind="image_depth",
        point_a_source=tuple(float(v) for v in point_a),
        point_b_source=tuple(float(v) for v in point_b),
        distance_source_units=native_distance,
        distance_m=native_distance,
        expanded_uncertainty_m=uncertainty,
        uncertainty_basis="heuristic_depth_intrinsics",
        calibration_status=cal.value,
        scale_status=scale.value,
        unit_to_metre=1.0,
    )


def estimate_reconstruction_distance(
    *,
    point_a_reconstruction: Sequence[float],
    point_b_reconstruction: Sequence[float],
    unit_to_metre: float | None,
    relative_scale_uncertainty: float | None,
    scale_state: str,
    calibration_available: bool,
) -> InteractiveDistanceEstimate:
    """Estimate distance between two canonical points using existing scale evidence."""

    a = np.asarray(point_a_reconstruction, dtype=np.float64)
    b = np.asarray(point_b_reconstruction, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Reconstruction measurement endpoints must be finite 3-vectors.")
    native = float(np.linalg.norm(b - a))
    if native <= 1e-12:
        return InteractiveDistanceEstimate(
            state=InteractiveDistanceState.REJECTED,
            source_kind="canonical_surface",
            point_a_source=tuple(float(v) for v in a),
            point_b_source=tuple(float(v) for v in b),
            distance_source_units=native,
            distance_m=None,
            expanded_uncertainty_m=None,
            uncertainty_basis="canonical_scale_plus_surface_pick",
            calibration_status="available" if calibration_available else "not_available",
            scale_status=str(scale_state),
            reasons=("selected_points_are_coincident",),
        )
    if unit_to_metre is None:
        return InteractiveDistanceEstimate(
            state=InteractiveDistanceState.UNAVAILABLE,
            source_kind="canonical_surface",
            point_a_source=tuple(float(v) for v in a),
            point_b_source=tuple(float(v) for v in b),
            distance_source_units=native,
            distance_m=None,
            expanded_uncertainty_m=None,
            uncertainty_basis="canonical_scale_plus_surface_pick",
            calibration_status="available" if calibration_available else "not_available",
            scale_status=str(scale_state),
            reasons=("canonical_metric_scale_not_available",),
        )
    factor = float(unit_to_metre)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("unit_to_metre must be positive when present.")
    distance_m = native * factor
    scale_rel = 0.10 if relative_scale_uncertainty is None else max(0.0, float(relative_scale_uncertainty))
    # Surface picking has finite mesh discretization even when the scale itself
    # is qualified; keep a small explicit floor until richer per-vertex QC exists.
    surface_pick_rel = 0.010
    uncertainty = max(
        math.sqrt((distance_m * scale_rel) ** 2 + (distance_m * surface_pick_rel) ** 2),
        0.001,
    )
    normalized_state = str(scale_state).strip().lower()
    qualified = normalized_state == "qualified" and bool(calibration_available)
    return InteractiveDistanceEstimate(
        state=InteractiveDistanceState.QUALIFIED if qualified else InteractiveDistanceState.ESTIMATED,
        source_kind="canonical_surface",
        point_a_source=tuple(float(v) for v in a),
        point_b_source=tuple(float(v) for v in b),
        distance_source_units=native,
        distance_m=distance_m,
        expanded_uncertainty_m=uncertainty,
        uncertainty_basis="canonical_scale_plus_surface_pick",
        calibration_status="available" if calibration_available else "not_available",
        scale_status=normalized_state or "not_qualified",
        unit_to_metre=factor,
    )


def format_interactive_distance(estimate: InteractiveDistanceEstimate) -> tuple[str, str]:
    """Return compact primary/secondary UI text for one estimate."""

    if estimate.has_metric_estimate:
        assert estimate.distance_m is not None and estimate.expanded_uncertainty_m is not None
        if estimate.distance_m < 1.0:
            primary = f"{estimate.distance_m * 1000.0:.1f} ± {estimate.expanded_uncertainty_m * 1000.0:.1f} mm"
        else:
            primary = f"{estimate.distance_m:.3f} ± {estimate.expanded_uncertainty_m:.3f} m"
        status = "QUALIFIED" if estimate.state is InteractiveDistanceState.QUALIFIED else "ESTIMATED"
        secondary = f"{status} • uncertainty is heuristic • certified metrology: no"
        return primary, secondary
    if estimate.distance_source_units is not None and estimate.source_kind == "canonical_surface":
        return (
            "Metric estimate unavailable",
            f"{estimate.distance_source_units:.4f} reconstruction units • scale not qualified",
        )
    reason = estimate.reasons[0].replace("_", " ") if estimate.reasons else "measurement unavailable"
    return "Measurement unavailable", reason
