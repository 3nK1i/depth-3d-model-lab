"""Deterministic image-quality metrics for reconstruction ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import cv2
import numpy as np

from core.reconstruction.errors import ReconstructionValidationError


@dataclass(frozen=True, slots=True)
class FrameQualityPolicy:
    """Conservative acceptance thresholds for decoded reconstruction frames.

    The defaults reject only clearly unusable inputs. They are intentionally
    backend-neutral; later SfM/SLAM backends may apply stricter criteria.
    """

    minimum_blur_variance: float = 12.0
    minimum_contrast_stddev: float = 3.0
    minimum_mean_luma: float = 5.0
    maximum_mean_luma: float = 250.0
    maximum_clipped_fraction: float = 0.985
    minimum_quality_score: float = 0.12

    def __post_init__(self) -> None:
        numeric = {
            "minimum_blur_variance": self.minimum_blur_variance,
            "minimum_contrast_stddev": self.minimum_contrast_stddev,
            "minimum_mean_luma": self.minimum_mean_luma,
            "maximum_mean_luma": self.maximum_mean_luma,
            "maximum_clipped_fraction": self.maximum_clipped_fraction,
            "minimum_quality_score": self.minimum_quality_score,
        }
        for field_name, raw in numeric.items():
            value = float(raw)
            if not isfinite(value):
                raise ReconstructionValidationError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, value)
        if self.minimum_blur_variance < 0.0 or self.minimum_contrast_stddev < 0.0:
            raise ReconstructionValidationError("Quality sharpness thresholds must be non-negative.")
        if not 0.0 <= self.minimum_mean_luma < self.maximum_mean_luma <= 255.0:
            raise ReconstructionValidationError(
                "Luma thresholds must satisfy 0 <= minimum < maximum <= 255."
            )
        if not 0.0 <= self.maximum_clipped_fraction <= 1.0:
            raise ReconstructionValidationError(
                "maximum_clipped_fraction must be between 0 and 1."
            )
        if not 0.0 <= self.minimum_quality_score <= 1.0:
            raise ReconstructionValidationError(
                "minimum_quality_score must be between 0 and 1."
            )


@dataclass(frozen=True, slots=True)
class FrameQualityMetrics:
    blur_variance: float
    contrast_stddev: float
    mean_luma: float
    underexposed_fraction: float
    overexposed_fraction: float
    quality_score: float
    accepted: bool
    rejection_details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "blur_variance",
            "contrast_stddev",
            "mean_luma",
            "underexposed_fraction",
            "overexposed_fraction",
            "quality_score",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value):
                raise ReconstructionValidationError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "underexposed_fraction",
            "overexposed_fraction",
            "quality_score",
        ):
            if not 0.0 <= getattr(self, field_name) <= 1.0:
                raise ReconstructionValidationError(f"{field_name} must be between 0 and 1.")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool.")
        details = tuple(str(item).strip() for item in self.rejection_details if str(item).strip())
        if self.accepted and details:
            raise ReconstructionValidationError(
                "Accepted quality metrics cannot contain rejection details."
            )
        if not self.accepted and not details:
            raise ReconstructionValidationError(
                "Rejected quality metrics require at least one detail."
            )
        object.__setattr__(self, "rejection_details", details)

    def as_dict(self) -> dict[str, Any]:
        return {
            "blur_variance": self.blur_variance,
            "contrast_stddev": self.contrast_stddev,
            "mean_luma": self.mean_luma,
            "underexposed_fraction": self.underexposed_fraction,
            "overexposed_fraction": self.overexposed_fraction,
            "quality_score": self.quality_score,
            "accepted": self.accepted,
            "rejection_details": list(self.rejection_details),
        }


def _normalized_score(
    *,
    blur_variance: float,
    contrast_stddev: float,
    mean_luma: float,
    underexposed_fraction: float,
    overexposed_fraction: float,
) -> float:
    sharpness = min(1.0, blur_variance / 250.0)
    contrast = min(1.0, contrast_stddev / 55.0)
    exposure = max(0.0, 1.0 - abs(mean_luma - 127.5) / 127.5)
    clipping = max(0.0, 1.0 - underexposed_fraction - overexposed_fraction)
    return float(np.clip(0.35 * sharpness + 0.25 * contrast + 0.20 * exposure + 0.20 * clipping, 0.0, 1.0))


def analyze_frame_quality(
    image: np.ndarray,
    policy: FrameQualityPolicy | None = None,
) -> FrameQualityMetrics:
    """Measure one BGR/gray uint8 frame without mutating it."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")
    if image.dtype != np.uint8:
        raise ReconstructionValidationError("image must use uint8 pixels.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ReconstructionValidationError(
            "image must have shape HxW or HxWx3 for quality analysis."
        )
    if gray.shape[0] <= 0 or gray.shape[1] <= 0:
        raise ReconstructionValidationError("image dimensions must be positive.")

    active_policy = policy or FrameQualityPolicy()
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_stddev = float(gray.std())
    mean_luma = float(gray.mean())
    underexposed_fraction = float(np.mean(gray <= 5))
    overexposed_fraction = float(np.mean(gray >= 250))
    quality_score = _normalized_score(
        blur_variance=blur_variance,
        contrast_stddev=contrast_stddev,
        mean_luma=mean_luma,
        underexposed_fraction=underexposed_fraction,
        overexposed_fraction=overexposed_fraction,
    )

    details: list[str] = []
    if blur_variance < active_policy.minimum_blur_variance:
        details.append("blur")
    if contrast_stddev < active_policy.minimum_contrast_stddev:
        details.append("low_contrast")
    if mean_luma < active_policy.minimum_mean_luma:
        details.append("underexposed")
    if mean_luma > active_policy.maximum_mean_luma:
        details.append("overexposed")
    if underexposed_fraction > active_policy.maximum_clipped_fraction:
        details.append("underexposed_clipping")
    if overexposed_fraction > active_policy.maximum_clipped_fraction:
        details.append("overexposed_clipping")
    if quality_score < active_policy.minimum_quality_score:
        details.append("low_quality_score")

    unique_details = tuple(dict.fromkeys(details))
    return FrameQualityMetrics(
        blur_variance=blur_variance,
        contrast_stddev=contrast_stddev,
        mean_luma=mean_luma,
        underexposed_fraction=underexposed_fraction,
        overexposed_fraction=overexposed_fraction,
        quality_score=quality_score,
        accepted=not unique_details,
        rejection_details=unique_details,
    )
