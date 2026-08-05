"""Generic, model-independent Step 6 measurement policies."""
from __future__ import annotations
from dataclasses import dataclass, field
from core.measurement.contracts import LowerBoundPolicy, MeasurementKind

@dataclass(frozen=True, slots=True)
class MeasurementSelectionPolicy:
    requested_kinds: tuple[MeasurementKind, ...] = (
        MeasurementKind.WIDTH, MeasurementKind.HEIGHT, MeasurementKind.DEPTH,
        MeasurementKind.CAMERA_DISTANCE, MeasurementKind.CENTRE_X,
        MeasurementKind.CENTRE_Y, MeasurementKind.CENTRE_Z,
        MeasurementKind.ORIENTATION_ROLL, MeasurementKind.ORIENTATION_PITCH,
        MeasurementKind.ORIENTATION_YAW, MeasurementKind.VOLUME,
    )
    prefer_stable_oriented_box: bool = True
    reject_width_on_horizontal_border: bool = True
    reject_height_on_vertical_border: bool = True
    reject_volume_on_any_border: bool = True
    allow_weak_orientation: bool = True

    def __post_init__(self) -> None:
        kinds = tuple(self.requested_kinds)
        if not kinds or len(kinds) != len(set(kinds)):
            raise ValueError("requested_kinds must be non-empty and unique.")
        object.__setattr__(self, "requested_kinds", kinds)

@dataclass(frozen=True, slots=True)
class PerturbationPolicy:
    percentile_pairs: tuple[tuple[float, float], ...] = ((0.5, 99.5), (1.0, 99.0), (2.0, 98.0))
    point_subset_strides: tuple[int, ...] = (2, 3)
    depth_relative_perturbation: float = 0.01
    focal_length_relative_perturbation: float = 0.01
    principal_point_perturbation_pixels: float = 1.0
    mask_boundary_pixels: float = 1.0
    confidence_relative_scale: float = 0.02
    fallback_depth_relative_uncertainty: float = 0.01
    border_contact_relative_uncertainty: float = 0.03
    occlusion_relative_uncertainty: float = 0.05
    minimum_uncertainty_si: float = 1e-6

    def __post_init__(self) -> None:
        for low, high in self.percentile_pairs:
            if not 0 <= low < high <= 100:
                raise ValueError("percentile pairs must satisfy 0 <= low < high <= 100.")
        if any(int(v) < 2 for v in self.point_subset_strides):
            raise ValueError("point subset strides must be >= 2.")
        for name in (
            "depth_relative_perturbation", "focal_length_relative_perturbation",
            "principal_point_perturbation_pixels", "mask_boundary_pixels",
            "confidence_relative_scale", "fallback_depth_relative_uncertainty",
            "border_contact_relative_uncertainty", "occlusion_relative_uncertainty",
            "minimum_uncertainty_si",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")

@dataclass(frozen=True, slots=True)
class ReliabilityPolicy:
    high_relative_uncertainty: float = 0.05
    medium_relative_uncertainty: float = 0.15
    maximum_relative_uncertainty: float = 0.50
    minimum_high_support_points: int = 500
    minimum_medium_support_points: int = 100
    policy_id: str = "phase75-step6-heuristic-v1"

    def __post_init__(self) -> None:
        if not 0 <= self.high_relative_uncertainty < self.medium_relative_uncertainty < self.maximum_relative_uncertainty:
            raise ValueError("reliability uncertainty thresholds must be strictly increasing.")
        if self.minimum_medium_support_points < 1 or self.minimum_high_support_points < self.minimum_medium_support_points:
            raise ValueError("support-point thresholds are invalid.")

@dataclass(frozen=True, slots=True)
class MeasurementComputationPolicy:
    selection: MeasurementSelectionPolicy = field(default_factory=MeasurementSelectionPolicy)
    perturbation: PerturbationPolicy = field(default_factory=PerturbationPolicy)
    reliability: ReliabilityPolicy = field(default_factory=ReliabilityPolicy)
    lower_bound_policy: LowerBoundPolicy = LowerBoundPolicy.CLIP_TO_ZERO
