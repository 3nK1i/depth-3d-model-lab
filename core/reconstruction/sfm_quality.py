"""Quality assessment for offline sparse SfM results.

These thresholds are operational acceptance heuristics for reconstruction
health, not metrology guarantees. They intentionally evaluate registration,
track support, and reprojection consistency without converting arbitrary SfM
coordinates to metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

from core.contracts.validation import require_finite
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_contracts import SfmStageStatistics
from core.reconstruction.validation import require_nonnegative_int, require_probability


class SfmQualityStatus(str, Enum):
    ACCEPTED = "accepted"
    DEGRADED = "degraded"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class SfmQualityPolicy:
    """Conservative, configurable sparse-SfM acceptance thresholds."""

    minimum_registered_keyframes: int = 3
    reject_below_registration_ratio: float = 0.30
    degrade_below_registration_ratio: float = 0.60
    minimum_landmarks: int = 20
    degrade_below_landmarks_per_registered_keyframe: float = 20.0
    reject_above_mean_reprojection_error_px: float = 8.0
    degrade_above_mean_reprojection_error_px: float = 3.0
    degrade_below_mean_track_length: float = 2.20

    def __post_init__(self) -> None:
        minimum_registered = require_nonnegative_int(
            self.minimum_registered_keyframes,
            field_name="minimum_registered_keyframes",
        )
        if minimum_registered < 2:
            raise ReconstructionValidationError(
                "minimum_registered_keyframes must be at least two."
            )
        object.__setattr__(self, "minimum_registered_keyframes", minimum_registered)

        reject_ratio = require_probability(
            self.reject_below_registration_ratio,
            field_name="reject_below_registration_ratio",
        )
        degrade_ratio = require_probability(
            self.degrade_below_registration_ratio,
            field_name="degrade_below_registration_ratio",
        )
        if degrade_ratio < reject_ratio:
            raise ReconstructionValidationError(
                "degrade_below_registration_ratio cannot be below the rejection threshold."
            )
        object.__setattr__(self, "reject_below_registration_ratio", reject_ratio)
        object.__setattr__(self, "degrade_below_registration_ratio", degrade_ratio)

        minimum_landmarks = require_nonnegative_int(
            self.minimum_landmarks, field_name="minimum_landmarks"
        )
        object.__setattr__(self, "minimum_landmarks", minimum_landmarks)

        for field_name in (
            "degrade_below_landmarks_per_registered_keyframe",
            "reject_above_mean_reprojection_error_px",
            "degrade_above_mean_reprojection_error_px",
            "degrade_below_mean_track_length",
        ):
            value = require_finite(getattr(self, field_name), field_name=field_name)
            if value < 0.0:
                raise ReconstructionValidationError(f"{field_name} must not be negative.")
            object.__setattr__(self, field_name, value)

        if (
            self.degrade_above_mean_reprojection_error_px
            > self.reject_above_mean_reprojection_error_px
        ):
            raise ReconstructionValidationError(
                "The degraded reprojection threshold cannot exceed the rejection threshold."
            )
        if self.degrade_below_mean_track_length < 2.0:
            raise ReconstructionValidationError(
                "degrade_below_mean_track_length must be at least 2.0."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_registered_keyframes": self.minimum_registered_keyframes,
            "reject_below_registration_ratio": self.reject_below_registration_ratio,
            "degrade_below_registration_ratio": self.degrade_below_registration_ratio,
            "minimum_landmarks": self.minimum_landmarks,
            "degrade_below_landmarks_per_registered_keyframe": (
                self.degrade_below_landmarks_per_registered_keyframe
            ),
            "reject_above_mean_reprojection_error_px": (
                self.reject_above_mean_reprojection_error_px
            ),
            "degrade_above_mean_reprojection_error_px": (
                self.degrade_above_mean_reprojection_error_px
            ),
            "degrade_below_mean_track_length": self.degrade_below_mean_track_length,
        }


@dataclass(frozen=True, slots=True)
class SfmQualityAssessment:
    status: SfmQualityStatus
    registration_ratio: float
    landmarks_per_registered_keyframe: float
    mean_track_length: float
    mean_reprojection_error_px: float | None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, SfmQualityStatus):
            raise TypeError("status must be an SfmQualityStatus.")
        object.__setattr__(
            self,
            "registration_ratio",
            require_probability(self.registration_ratio, field_name="registration_ratio"),
        )
        for field_name in (
            "landmarks_per_registered_keyframe",
            "mean_track_length",
        ):
            value = require_finite(getattr(self, field_name), field_name=field_name)
            if value < 0.0:
                raise ReconstructionValidationError(f"{field_name} must not be negative.")
            object.__setattr__(self, field_name, value)
        if self.mean_reprojection_error_px is not None:
            value = require_finite(
                self.mean_reprojection_error_px,
                field_name="mean_reprojection_error_px",
            )
            if value < 0.0:
                raise ReconstructionValidationError(
                    "mean_reprojection_error_px must not be negative."
                )
            object.__setattr__(self, "mean_reprojection_error_px", value)
        reasons = tuple(str(item).strip() for item in self.reasons if str(item).strip())
        object.__setattr__(self, "reasons", reasons)

    @property
    def usable(self) -> bool:
        return self.status is not SfmQualityStatus.REJECTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "usable": self.usable,
            "registration_ratio": self.registration_ratio,
            "landmarks_per_registered_keyframe": self.landmarks_per_registered_keyframe,
            "mean_track_length": self.mean_track_length,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "reasons": list(self.reasons),
        }


def assess_sfm_quality(
    statistics: SfmStageStatistics,
    policy: SfmQualityPolicy | None = None,
) -> SfmQualityAssessment:
    """Classify sparse-SfM health without making any metric-scale claim."""

    if not isinstance(statistics, SfmStageStatistics):
        raise TypeError("statistics must be SfmStageStatistics.")
    active = policy or SfmQualityPolicy()
    if not isinstance(active, SfmQualityPolicy):
        raise TypeError("policy must be SfmQualityPolicy or None.")

    registration_ratio = (
        statistics.registered_keyframes / statistics.input_keyframes
        if statistics.input_keyframes
        else 0.0
    )
    landmarks_per_camera = (
        statistics.landmarks / statistics.registered_keyframes
        if statistics.registered_keyframes
        else 0.0
    )
    mean_track_length = (
        statistics.observations / statistics.landmarks if statistics.landmarks else 0.0
    )
    error = statistics.mean_reprojection_error_px

    rejected: list[str] = []
    degraded: list[str] = []

    if statistics.registered_keyframes < active.minimum_registered_keyframes:
        rejected.append(
            f"Only {statistics.registered_keyframes} keyframes registered; "
            f"at least {active.minimum_registered_keyframes} are required."
        )
    if registration_ratio < active.reject_below_registration_ratio:
        rejected.append(
            f"Registration ratio {registration_ratio:.1%} is below the "
            f"{active.reject_below_registration_ratio:.1%} rejection threshold."
        )
    elif registration_ratio < active.degrade_below_registration_ratio:
        degraded.append(
            f"Registration ratio {registration_ratio:.1%} is below the "
            f"{active.degrade_below_registration_ratio:.1%} preferred threshold."
        )
    if statistics.landmarks < active.minimum_landmarks:
        rejected.append(
            f"Only {statistics.landmarks} sparse landmarks were reconstructed; "
            f"at least {active.minimum_landmarks} are required."
        )
    elif landmarks_per_camera < active.degrade_below_landmarks_per_registered_keyframe:
        degraded.append(
            f"Landmark density {landmarks_per_camera:.1f} per registered keyframe is low."
        )
    if mean_track_length < 2.0 and statistics.landmarks:
        rejected.append("Mean sparse track length is below the two-view geometric minimum.")
    elif mean_track_length < active.degrade_below_mean_track_length:
        degraded.append(f"Mean sparse track length {mean_track_length:.2f} is weak.")
    if error is None:
        degraded.append("Mean reprojection error was not reported by the SfM backend.")
    elif error > active.reject_above_mean_reprojection_error_px:
        rejected.append(
            f"Mean reprojection error {error:.3f}px exceeds the "
            f"{active.reject_above_mean_reprojection_error_px:.3f}px rejection threshold."
        )
    elif error > active.degrade_above_mean_reprojection_error_px:
        degraded.append(
            f"Mean reprojection error {error:.3f}px exceeds the "
            f"{active.degrade_above_mean_reprojection_error_px:.3f}px preferred threshold."
        )

    if rejected:
        status = SfmQualityStatus.REJECTED
        reasons = tuple(rejected + degraded)
    elif degraded:
        status = SfmQualityStatus.DEGRADED
        reasons = tuple(degraded)
    else:
        status = SfmQualityStatus.ACCEPTED
        reasons = ()

    return SfmQualityAssessment(
        status=status,
        registration_ratio=registration_ratio,
        landmarks_per_registered_keyframe=landmarks_per_camera,
        mean_track_length=mean_track_length,
        mean_reprojection_error_px=error,
        reasons=reasons,
    )
