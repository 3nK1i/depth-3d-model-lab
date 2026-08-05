"""Policies for robust object-frame and 3D-box fitting."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class RobustPcaPolicy:
    minimum_points: int = 32
    use_point_weights: bool = True
    trim_extreme_tails: bool = True
    trim_mad_multiplier: float = 4.5
    linear_ratio_threshold: float = 0.02
    planar_ratio_threshold: float = 0.02
    stable_axis_gap_threshold: float = 0.08
    def __post_init__(self):
        if self.minimum_points < 3: raise ValueError("minimum_points must be >= 3")
        if self.trim_mad_multiplier <= 0: raise ValueError("trim_mad_multiplier must be positive")
        for n in ("linear_ratio_threshold","planar_ratio_threshold","stable_axis_gap_threshold"):
            v=float(getattr(self,n))
            if not 0 <= v < 1: raise ValueError(f"{n} must be in [0,1)")

@dataclass(frozen=True, slots=True)
class BoxFitPolicy:
    fit_axis_aligned: bool = True
    fit_camera_aligned: bool = True
    fit_oriented: bool = True
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    allow_ambiguous_oriented_box: bool = True
    def __post_init__(self):
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
        if not (self.fit_axis_aligned or self.fit_camera_aligned or self.fit_oriented):
            raise ValueError("at least one box method must be enabled")

@dataclass(frozen=True, slots=True)
class GenericShapeFitPolicy:
    enabled: bool = True
    preferred_box_order: tuple[str,...] = ("oriented","camera_aligned","axis_aligned")

@dataclass(frozen=True, slots=True)
class ObjectGeometryFittingPolicy:
    pca: RobustPcaPolicy = field(default_factory=RobustPcaPolicy)
    boxes: BoxFitPolicy = field(default_factory=BoxFitPolicy)
    shape: GenericShapeFitPolicy = field(default_factory=GenericShapeFitPolicy)
