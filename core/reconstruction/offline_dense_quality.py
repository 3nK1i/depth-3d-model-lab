"""Phase I2 offline Dense RGB real-scene quality diagnostics.

I2 is deliberately diagnostic.  It does not alter geometry, relax dense
qualification thresholds, rerun SfM/DA3, or promote metric/measurement/mesh
authority.  The helpers in this module summarize spatial coverage and retained
peripheral geometry so the full-video reconstruction can be reviewed before
surface reconstruction is authorized.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError


I2_QUALITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class I2QualityPolicy:
    minimum_registered_fraction: float = 0.80
    minimum_dense_alignment_fraction: float = 0.80
    minimum_selected_fraction_of_registered: float = 0.60
    minimum_largest_component_fraction: float = 0.85
    minimum_core_fraction_of_supported: float = 0.50
    minimum_extended_fraction_of_supported: float = 0.65
    maximum_rejected_supported_fraction: float = 0.35
    robust_lower_quantile: float = 0.02
    robust_upper_quantile: float = 0.98
    peripheral_margin_fraction: float = 0.05
    far_radial_multiplier: float = 1.25

    def __post_init__(self) -> None:
        fraction_fields = (
            "minimum_registered_fraction",
            "minimum_dense_alignment_fraction",
            "minimum_selected_fraction_of_registered",
            "minimum_largest_component_fraction",
            "minimum_core_fraction_of_supported",
            "minimum_extended_fraction_of_supported",
            "maximum_rejected_supported_fraction",
            "robust_lower_quantile",
            "robust_upper_quantile",
            "peripheral_margin_fraction",
        )
        for name in fraction_fields:
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ReconstructionValidationError(f"I2 policy {name} must be finite in [0,1].")
        if not self.robust_lower_quantile < self.robust_upper_quantile:
            raise ReconstructionValidationError("I2 robust quantiles must be strictly ordered.")
        if self.robust_lower_quantile >= 0.5 or self.robust_upper_quantile <= 0.5:
            raise ReconstructionValidationError("I2 robust envelope must straddle the median.")
        if float(self.far_radial_multiplier) <= 1.0 or not isfinite(float(self.far_radial_multiplier)):
            raise ReconstructionValidationError("I2 far_radial_multiplier must be finite and > 1.")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class SpatialEnvelope:
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    robust_diagonal: float
    full_diagonal: float
    radial_p95: float
    radial_p99: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "robust_diagonal": float(self.robust_diagonal),
            "full_diagonal": float(self.full_diagonal),
            "full_to_robust_diagonal_ratio": (
                float(self.full_diagonal / self.robust_diagonal) if self.robust_diagonal > 0.0 else None
            ),
            "radial_p95": float(self.radial_p95),
            "radial_p99": float(self.radial_p99),
        }


def _xyz(points: object, *, name: str) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ReconstructionValidationError(f"I2 {name} points must be non-empty Nx3.")
    if not np.isfinite(values).all():
        raise ReconstructionValidationError(f"I2 {name} points must be finite.")
    return values


def quantiles(values: object) -> dict[str, float | None]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return {"p05": None, "p25": None, "p50": None, "p75": None, "p90": None, "p95": None}
    return {
        "p05": float(np.quantile(finite, 0.05)),
        "p25": float(np.quantile(finite, 0.25)),
        "p50": float(np.quantile(finite, 0.50)),
        "p75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def spatial_envelope(
    points: object,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> SpatialEnvelope:
    xyz = _xyz(points, name="spatial-envelope")
    lo = np.quantile(xyz, float(lower_quantile), axis=0)
    hi = np.quantile(xyz, float(upper_quantile), axis=0)
    center = np.median(xyz, axis=0)
    full_lo = np.min(xyz, axis=0)
    full_hi = np.max(xyz, axis=0)
    robust_diagonal = float(np.linalg.norm(hi - lo))
    full_diagonal = float(np.linalg.norm(full_hi - full_lo))
    radial = np.linalg.norm(xyz - center[None, :], axis=1)
    return SpatialEnvelope(
        center=np.asarray(center, dtype=np.float64),
        lower=np.asarray(lo, dtype=np.float64),
        upper=np.asarray(hi, dtype=np.float64),
        robust_diagonal=robust_diagonal,
        full_diagonal=full_diagonal,
        radial_p95=float(np.quantile(radial, 0.95)),
        radial_p99=float(np.quantile(radial, 0.99)),
    )


def classify_peripheral_geometry(
    extended_points: object,
    core_points: object,
    *,
    voxel_size: float,
    policy: I2QualityPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SpatialEnvelope]:
    """Return peripheral/far-peripheral masks and axis-sector codes.

    The Core cloud is used only as a robust spatial anchor.  The returned masks
    are diagnostics, never filtering decisions.  Extended points remain intact.
    """

    extended = _xyz(extended_points, name="Extended")
    core = _xyz(core_points, name="Core")
    voxel = float(voxel_size)
    if not isfinite(voxel) or voxel <= 0.0:
        raise ReconstructionValidationError("I2 voxel size must be finite and positive.")
    envelope = spatial_envelope(
        core,
        lower_quantile=policy.robust_lower_quantile,
        upper_quantile=policy.robust_upper_quantile,
    )
    span = np.maximum(envelope.upper - envelope.lower, voxel)
    margin = np.maximum(span * float(policy.peripheral_margin_fraction), 2.0 * voxel)
    expanded_lo = envelope.lower - margin
    expanded_hi = envelope.upper + margin
    outside_box = np.any((extended < expanded_lo[None, :]) | (extended > expanded_hi[None, :]), axis=1)

    radial = np.linalg.norm(extended - envelope.center[None, :], axis=1)
    far_threshold = max(float(envelope.radial_p99) * float(policy.far_radial_multiplier), 4.0 * voxel)
    far = radial > far_threshold
    peripheral = outside_box | far

    half_span = np.maximum((envelope.upper - envelope.lower) * 0.5, voxel)
    normalized = (extended - envelope.center[None, :]) / half_span[None, :]
    dominant_axis = np.argmax(np.abs(normalized), axis=1)
    signs = np.sign(normalized[np.arange(len(extended)), dominant_axis])
    # 0 central; 1 +X; 2 -X; 3 +Y; 4 -Y; 5 +Z; 6 -Z
    sector = np.zeros(len(extended), dtype=np.uint8)
    codes_positive = np.asarray([1, 3, 5], dtype=np.uint8)
    codes_negative = np.asarray([2, 4, 6], dtype=np.uint8)
    sector[peripheral & (signs >= 0.0)] = codes_positive[dominant_axis[peripheral & (signs >= 0.0)]]
    sector[peripheral & (signs < 0.0)] = codes_negative[dominant_axis[peripheral & (signs < 0.0)]]
    return peripheral, far, sector, envelope


def region_statistics(
    mask: object,
    *,
    support: object,
    confidence: object,
    dispersion: object,
    component_size: object,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    arrays = [
        np.asarray(support),
        np.asarray(confidence),
        np.asarray(dispersion),
        np.asarray(component_size),
    ]
    if any(array.shape != selected.shape for array in arrays):
        raise ReconstructionValidationError("I2 region evidence arrays must be aligned.")
    count = int(np.count_nonzero(selected))
    if count == 0:
        return {
            "count": 0,
            "fraction": 0.0,
            "support_quantiles": quantiles([]),
            "confidence_quantiles": quantiles([]),
            "p90_dispersion_quantiles": quantiles([]),
            "component_size_quantiles": quantiles([]),
        }
    return {
        "count": count,
        "support_quantiles": quantiles(arrays[0][selected]),
        "confidence_quantiles": quantiles(arrays[1][selected]),
        "p90_dispersion_quantiles": quantiles(arrays[2][selected]),
        "component_size_quantiles": quantiles(arrays[3][selected]),
    }


def directional_sector_summary(
    sector_codes: object,
    *,
    support: object,
    confidence: object,
    dispersion: object,
) -> dict[str, Any]:
    sectors = np.asarray(sector_codes, dtype=np.uint8)
    support_arr = np.asarray(support)
    confidence_arr = np.asarray(confidence)
    dispersion_arr = np.asarray(dispersion)
    if not (support_arr.shape == confidence_arr.shape == dispersion_arr.shape == sectors.shape):
        raise ReconstructionValidationError("I2 sector evidence arrays must be aligned.")
    labels = {0: "central", 1: "+x", 2: "-x", 3: "+y", 4: "-y", 5: "+z", 6: "-z"}
    result: dict[str, Any] = {}
    for code, label in labels.items():
        mask = sectors == int(code)
        count = int(np.count_nonzero(mask))
        result[label] = {
            "count": count,
            "support_quantiles": quantiles(support_arr[mask]),
            "confidence_quantiles": quantiles(confidence_arr[mask]),
            "p90_dispersion_quantiles": quantiles(dispersion_arr[mask]),
        }
    return result


def automated_quality_gates(
    *,
    input_sfm_keyframes: int,
    registered_views: int,
    candidate_dense_views: int,
    alignment_accepted_views: int,
    selected_views: int,
    source_supported: int,
    core_points: int,
    extended_points: int,
    rejected_supported: int,
    largest_component_fraction: float,
    exact_observation_accounting: bool,
    persistence_self_contained: bool,
    authority_boundaries_preserved: bool,
    policy: I2QualityPolicy,
) -> dict[str, bool]:
    if input_sfm_keyframes <= 0 or registered_views <= 0 or candidate_dense_views <= 0 or source_supported <= 0:
        raise ReconstructionValidationError("I2 automated quality gates require positive source counts.")
    registered_fraction = registered_views / float(input_sfm_keyframes)
    alignment_fraction = alignment_accepted_views / float(candidate_dense_views)
    selected_fraction = selected_views / float(registered_views)
    core_fraction = core_points / float(source_supported)
    extended_fraction = extended_points / float(source_supported)
    rejected_fraction = rejected_supported / float(source_supported)
    return {
        "minimum_registered_fraction": registered_fraction >= policy.minimum_registered_fraction,
        "minimum_dense_alignment_fraction": alignment_fraction >= policy.minimum_dense_alignment_fraction,
        "minimum_selected_fraction_of_registered": selected_fraction >= policy.minimum_selected_fraction_of_registered,
        "minimum_largest_component_fraction": float(largest_component_fraction) >= policy.minimum_largest_component_fraction,
        "minimum_core_fraction_of_supported": core_fraction >= policy.minimum_core_fraction_of_supported,
        "minimum_extended_fraction_of_supported": extended_fraction >= policy.minimum_extended_fraction_of_supported,
        "maximum_rejected_supported_fraction": rejected_fraction <= policy.maximum_rejected_supported_fraction,
        "exact_observation_accounting": bool(exact_observation_accounting),
        "persistence_self_contained": bool(persistence_self_contained),
        "authority_boundaries_preserved": bool(authority_boundaries_preserved),
    }
