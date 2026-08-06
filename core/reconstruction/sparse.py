"""Sparse landmark, observation, and track contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.contracts.validation import require_finite, require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.validation import (
    readonly_vector3,
    require_nonnegative_int,
    require_probability,
)


@dataclass(frozen=True, slots=True)
class SparseObservation:
    observation_id: str
    landmark_id: str
    keyframe_id: str
    feature_xy: tuple[float, float]
    feature_index: int | None = None
    reprojection_error_px: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            require_nonempty(self.observation_id, field_name="observation_id"),
        )
        object.__setattr__(
            self, "landmark_id", require_nonempty(self.landmark_id, field_name="landmark_id")
        )
        object.__setattr__(
            self, "keyframe_id", require_nonempty(self.keyframe_id, field_name="keyframe_id")
        )
        if len(self.feature_xy) != 2:
            raise ReconstructionValidationError("feature_xy must contain two coordinates.")
        xy = tuple(require_finite(value, field_name="feature_xy") for value in self.feature_xy)
        object.__setattr__(self, "feature_xy", xy)
        if self.feature_index is not None:
            object.__setattr__(
                self,
                "feature_index",
                require_nonnegative_int(self.feature_index, field_name="feature_index"),
            )
        if self.reprojection_error_px is not None:
            error = require_finite(
                self.reprojection_error_px, field_name="reprojection_error_px"
            )
            if error < 0.0:
                raise ReconstructionValidationError(
                    "reprojection_error_px must not be negative."
                )
            object.__setattr__(self, "reprojection_error_px", error)
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                require_probability(self.confidence, field_name="confidence"),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "landmark_id": self.landmark_id,
            "keyframe_id": self.keyframe_id,
            "feature_xy": list(self.feature_xy),
            "feature_index": self.feature_index,
            "reprojection_error_px": self.reprojection_error_px,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class SparseLandmark:
    landmark_id: str
    track_id: str
    position_world_m: np.ndarray
    observations: tuple[SparseObservation, ...]
    color_rgb: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "landmark_id", require_nonempty(self.landmark_id, field_name="landmark_id")
        )
        object.__setattr__(self, "track_id", require_nonempty(self.track_id, field_name="track_id"))
        object.__setattr__(
            self,
            "position_world_m",
            readonly_vector3(self.position_world_m, field_name="position_world_m"),
        )
        observations = tuple(self.observations)
        if not observations:
            raise ReconstructionValidationError("Sparse landmarks require observations.")
        if any(not isinstance(item, SparseObservation) for item in observations):
            raise TypeError("observations must contain SparseObservation values.")
        if any(item.landmark_id != self.landmark_id for item in observations):
            raise ReconstructionValidationError(
                "Every sparse observation must reference its parent landmark."
            )
        observation_ids = [item.observation_id for item in observations]
        keyframe_ids = [item.keyframe_id for item in observations]
        if len(set(observation_ids)) != len(observation_ids):
            raise ReconstructionValidationError("Sparse observation IDs must be unique.")
        if len(set(keyframe_ids)) != len(keyframe_ids):
            raise ReconstructionValidationError(
                "A landmark cannot contain duplicate observations from one keyframe."
            )
        color = self.color_rgb
        if color is not None:
            if len(color) != 3 or any(
                isinstance(value, bool) or int(value) != value or not 0 <= int(value) <= 255
                for value in color
            ):
                raise ReconstructionValidationError(
                    "color_rgb must contain three integer values within [0, 255]."
                )
            color = tuple(int(value) for value in color)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "color_rgb", color)

    def as_dict(self) -> dict[str, Any]:
        return {
            "landmark_id": self.landmark_id,
            "track_id": self.track_id,
            "position_world_m": self.position_world_m.tolist(),
            "observations": [item.as_dict() for item in self.observations],
            "color_rgb": list(self.color_rgb) if self.color_rgb else None,
        }
