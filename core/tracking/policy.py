"""Generic Step 7 association and temporal-stabilization policies."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssociationPolicy:
    minimum_score: float = 0.25
    mask_iou_weight: float = 0.55
    box_iou_weight: float = 0.35
    centre_similarity_weight: float = 0.10

    def __post_init__(self) -> None:
        values = (self.minimum_score, self.mask_iou_weight, self.box_iou_weight, self.centre_similarity_weight)
        if any(not 0.0 <= float(v) <= 1.0 for v in values):
            raise ValueError("Association policy values must be within [0, 1].")
        if sum(values[1:]) <= 0.0:
            raise ValueError("At least one association weight must be positive.")


@dataclass(frozen=True, slots=True)
class TemporalStabilizationPolicy:
    history_size: int = 15
    minimum_stable_samples: int = 3
    outlier_mad_multiplier: float = 3.5
    minimum_absolute_outlier_tolerance: float = 1e-5
    minimum_relative_outlier_tolerance: float = 0.01
    maximum_temporal_relative_uncertainty: float = 0.35

    def __post_init__(self) -> None:
        if int(self.history_size) < 2:
            raise ValueError("history_size must be at least two.")
        if not 2 <= int(self.minimum_stable_samples) <= int(self.history_size):
            raise ValueError("minimum_stable_samples must be between two and history_size.")
        if float(self.outlier_mad_multiplier) <= 0.0:
            raise ValueError("outlier_mad_multiplier must be positive.")
        if min(float(self.minimum_absolute_outlier_tolerance), float(self.minimum_relative_outlier_tolerance)) < 0.0:
            raise ValueError("Outlier tolerances must not be negative.")
        if float(self.maximum_temporal_relative_uncertainty) <= 0.0:
            raise ValueError("maximum_temporal_relative_uncertainty must be positive.")


@dataclass(frozen=True, slots=True)
class TemporalTrackingPolicy:
    confirmation_hits: int = 2
    maximum_missed_frames: int = 3
    association: AssociationPolicy = AssociationPolicy()
    stabilization: TemporalStabilizationPolicy = TemporalStabilizationPolicy()

    def __post_init__(self) -> None:
        if int(self.confirmation_hits) <= 0:
            raise ValueError("confirmation_hits must be positive.")
        if int(self.maximum_missed_frames) < 0:
            raise ValueError("maximum_missed_frames must be non-negative.")
