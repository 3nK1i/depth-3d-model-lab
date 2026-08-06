"""Immutable policy values for object-specific point extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from core.contracts.validation import ContractValidationError


def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ContractValidationError(f"{name} must be an integer.")
    result = int(value)
    if result < 0 or (result == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ContractValidationError(f"{name} must be {qualifier}.")
    return result


def _positive_float(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ContractValidationError(f"{name} must be finite and {qualifier}.")
    return result


@dataclass(frozen=True, slots=True)
class MaskPreprocessingPolicy:
    connectivity: int = 8
    keep_largest_component: bool = True
    erosion_pixels: int = 1
    minimum_mask_pixels: int = 32

    def __post_init__(self) -> None:
        if self.connectivity not in (4, 8):
            raise ContractValidationError("connectivity must be 4 or 8.")
        object.__setattr__(self, "erosion_pixels", _positive_int(self.erosion_pixels, "erosion_pixels", allow_zero=True))
        object.__setattr__(self, "minimum_mask_pixels", _positive_int(self.minimum_mask_pixels, "minimum_mask_pixels"))


@dataclass(frozen=True, slots=True)
class ConfidenceFilterPolicy:
    minimum_confidence: float | None = None
    maximum_uncertainty: float | None = None
    inverse_uncertainty_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.minimum_confidence is not None:
            value = float(self.minimum_confidence)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ContractValidationError("minimum_confidence must be within [0, 1].")
            object.__setattr__(self, "minimum_confidence", value)
        if self.maximum_uncertainty is not None:
            object.__setattr__(self, "maximum_uncertainty", _positive_float(self.maximum_uncertainty, "maximum_uncertainty", allow_zero=True))
        object.__setattr__(self, "inverse_uncertainty_epsilon", _positive_float(self.inverse_uncertainty_epsilon, "inverse_uncertainty_epsilon"))


@dataclass(frozen=True, slots=True)
class DepthRangePolicy:
    minimum_depth_value: float | None = None
    maximum_depth_value: float | None = None

    def __post_init__(self) -> None:
        minimum = self.minimum_depth_value
        maximum = self.maximum_depth_value
        if minimum is not None:
            minimum = _positive_float(minimum, "minimum_depth_value", allow_zero=True)
            object.__setattr__(self, "minimum_depth_value", minimum)
        if maximum is not None:
            maximum = _positive_float(maximum, "maximum_depth_value")
            object.__setattr__(self, "maximum_depth_value", maximum)
        if minimum is not None and maximum is not None and maximum <= minimum:
            raise ContractValidationError("maximum_depth_value must exceed minimum_depth_value.")


@dataclass(frozen=True, slots=True)
class DepthDiscontinuityPolicy:
    enabled: bool = True
    mad_multiplier: float = 6.0
    minimum_absolute_threshold: float = 1e-6
    minimum_relative_threshold: float = 0.01
    minimum_neighbor_samples: int = 16
    minimum_consistent_neighbors: int = 1
    minimum_consistent_fraction: float = 0.25
    sparse_neighbor_threshold: int = 2
    preserve_sparse_support: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mad_multiplier", _positive_float(self.mad_multiplier, "mad_multiplier"))
        object.__setattr__(self, "minimum_absolute_threshold", _positive_float(self.minimum_absolute_threshold, "minimum_absolute_threshold", allow_zero=True))
        object.__setattr__(self, "minimum_relative_threshold", _positive_float(self.minimum_relative_threshold, "minimum_relative_threshold", allow_zero=True))
        object.__setattr__(self, "minimum_neighbor_samples", _positive_int(self.minimum_neighbor_samples, "minimum_neighbor_samples"))
        object.__setattr__(self, "minimum_consistent_neighbors", _positive_int(self.minimum_consistent_neighbors, "minimum_consistent_neighbors"))
        fraction = float(self.minimum_consistent_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ContractValidationError("minimum_consistent_fraction must be within (0, 1].")
        object.__setattr__(self, "minimum_consistent_fraction", fraction)
        object.__setattr__(self, "sparse_neighbor_threshold", _positive_int(self.sparse_neighbor_threshold, "sparse_neighbor_threshold", allow_zero=True))


@dataclass(frozen=True, slots=True)
class StatisticalOutlierPolicy:
    enabled: bool = True
    mad_multiplier: float = 6.0
    minimum_points: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "mad_multiplier", _positive_float(self.mad_multiplier, "mad_multiplier"))
        object.__setattr__(self, "minimum_points", _positive_int(self.minimum_points, "minimum_points"))


@dataclass(frozen=True, slots=True)
class BackgroundPlanePolicy:
    enabled: bool = False
    exterior_ring_pixels: int = 5
    ransac_iterations: int = 64
    distance_threshold: float = 0.02
    minimum_ring_points: int = 64
    minimum_support_ratio: float = 0.60
    minimum_behind_separation: float = 0.01
    maximum_removal_fraction: float = 0.50
    random_seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "exterior_ring_pixels", _positive_int(self.exterior_ring_pixels, "exterior_ring_pixels"))
        object.__setattr__(self, "ransac_iterations", _positive_int(self.ransac_iterations, "ransac_iterations"))
        object.__setattr__(self, "distance_threshold", _positive_float(self.distance_threshold, "distance_threshold"))
        object.__setattr__(self, "minimum_ring_points", _positive_int(self.minimum_ring_points, "minimum_ring_points"))
        for name in ("minimum_support_ratio", "maximum_removal_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ContractValidationError(f"{name} must be within (0, 1].")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "minimum_behind_separation", _positive_float(self.minimum_behind_separation, "minimum_behind_separation", allow_zero=True))
        object.__setattr__(self, "random_seed", int(self.random_seed))


@dataclass(frozen=True, slots=True)
class ClusteringPolicy:
    enabled: bool = True
    voxel_size: float | None = None
    adaptive_extent_fraction: float = 0.03
    minimum_cluster_points: int = 8
    weak_dominance_ratio: float = 1.50
    preserve_mask_connected_support: bool = True
    minimum_preserved_cluster_points: int = 8
    maximum_pixel_gap: int = 4
    maximum_relative_depth_gap: float = 0.25
    maximum_absolute_depth_gap: float = 0.20

    def __post_init__(self) -> None:
        if self.voxel_size is not None:
            object.__setattr__(self, "voxel_size", _positive_float(self.voxel_size, "voxel_size"))
        object.__setattr__(self, "adaptive_extent_fraction", _positive_float(self.adaptive_extent_fraction, "adaptive_extent_fraction"))
        object.__setattr__(self, "minimum_cluster_points", _positive_int(self.minimum_cluster_points, "minimum_cluster_points"))
        object.__setattr__(self, "weak_dominance_ratio", _positive_float(self.weak_dominance_ratio, "weak_dominance_ratio"))
        object.__setattr__(self, "minimum_preserved_cluster_points", _positive_int(self.minimum_preserved_cluster_points, "minimum_preserved_cluster_points"))
        object.__setattr__(self, "maximum_pixel_gap", _positive_int(self.maximum_pixel_gap, "maximum_pixel_gap", allow_zero=True))
        object.__setattr__(self, "maximum_relative_depth_gap", _positive_float(self.maximum_relative_depth_gap, "maximum_relative_depth_gap", allow_zero=True))
        object.__setattr__(self, "maximum_absolute_depth_gap", _positive_float(self.maximum_absolute_depth_gap, "maximum_absolute_depth_gap", allow_zero=True))




@dataclass(frozen=True, slots=True)
class SupportRetentionPolicy:
    """Guard against filters collapsing valid thin or articulated mask support."""

    enabled: bool = True
    grid_size: int = 8
    minimum_point_retention: float = 0.45
    minimum_axis_span_retention: float = 0.75
    minimum_grid_coverage_retention: float = 0.60
    fallback_quantile: float = 0.005

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid_size", _positive_int(self.grid_size, "grid_size"))
        for name in (
            "minimum_point_retention",
            "minimum_axis_span_retention",
            "minimum_grid_coverage_retention",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ContractValidationError(f"{name} must be within (0, 1].")
            object.__setattr__(self, name, value)
        quantile = float(self.fallback_quantile)
        if not math.isfinite(quantile) or not 0.0 <= quantile < 0.5:
            raise ContractValidationError("fallback_quantile must be within [0, 0.5).")
        object.__setattr__(self, "fallback_quantile", quantile)


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    maximum_points: int = 100_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "maximum_points", _positive_int(self.maximum_points, "maximum_points"))


@dataclass(frozen=True, slots=True)
class ObjectPointExtractionPolicy:
    mask: MaskPreprocessingPolicy = field(default_factory=MaskPreprocessingPolicy)
    confidence: ConfidenceFilterPolicy = field(default_factory=ConfidenceFilterPolicy)
    depth_range: DepthRangePolicy = field(default_factory=DepthRangePolicy)
    discontinuity: DepthDiscontinuityPolicy = field(default_factory=DepthDiscontinuityPolicy)
    outliers: StatisticalOutlierPolicy = field(default_factory=StatisticalOutlierPolicy)
    background_plane: BackgroundPlanePolicy = field(default_factory=BackgroundPlanePolicy)
    clustering: ClusteringPolicy = field(default_factory=ClusteringPolicy)
    support_retention: SupportRetentionPolicy = field(default_factory=SupportRetentionPolicy)
    sampling: SamplingPolicy = field(default_factory=SamplingPolicy)
    minimum_retained_points: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_retained_points", _positive_int(self.minimum_retained_points, "minimum_retained_points"))
