"""Ordered trajectory contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.poses import CameraPose


@dataclass(frozen=True, slots=True)
class ReconstructionTrajectory:
    samples: tuple[CameraPose, ...]

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if any(not isinstance(sample, CameraPose) for sample in samples):
            raise TypeError("samples must contain CameraPose values.")
        timestamps = [sample.timestamp_ns for sample in samples]
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ReconstructionValidationError(
                "Trajectory sample timestamps must be strictly increasing."
            )
        frame_ids = [sample.frame_id for sample in samples]
        pose_ids = [sample.pose_id for sample in samples]
        if len(set(frame_ids)) != len(frame_ids):
            raise ReconstructionValidationError("Trajectory frame IDs must be unique.")
        if len(set(pose_ids)) != len(pose_ids):
            raise ReconstructionValidationError("Trajectory pose IDs must be unique.")
        object.__setattr__(self, "samples", samples)

    @classmethod
    def from_iterable(cls, samples: Iterable[CameraPose]) -> "ReconstructionTrajectory":
        return cls(tuple(samples))

    def as_dict(self) -> dict[str, Any]:
        return {"samples": [sample.as_dict() for sample in self.samples]}
