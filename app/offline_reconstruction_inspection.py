"""Offline sparse-SfM inspection helpers for Phase 8.8D2.

D2 remains presentation-only.  It derives camera inspection readouts from the
bounded D1 visualization artifact and never promotes arbitrary SfM gauge values
to metres or certified geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from app.offline_reconstruction_visualization import OfflineSparseVisualization


@dataclass(frozen=True, slots=True)
class OfflineCameraInspection:
    """Read-only inspection presentation for one registered SfM camera."""

    camera_index: int
    camera_id: str
    sampled_camera_count: int
    registered_camera_total: int
    position_reconstruction: tuple[float, float, float]
    distance_from_previous_gauge: float | None

    @property
    def ordinal_text(self) -> str:
        return f"{self.camera_index + 1}/{self.sampled_camera_count}"

    @property
    def position_text(self) -> str:
        x, y, z = self.position_reconstruction
        return f"X {x:+.3f} • Y {y:+.3f} • Z {z:+.3f} gauge"

    @property
    def delta_previous_text(self) -> str:
        if self.distance_from_previous_gauge is None:
            return "First registered camera"
        return f"{self.distance_from_previous_gauge:.3f} gauge from previous"


def build_offline_camera_inspection(
    visual: OfflineSparseVisualization,
    camera_index: int,
) -> OfflineCameraInspection:
    """Build a bounded, unit-honest camera readout from a D1 artifact."""

    index = int(camera_index)
    count = int(visual.camera_matrices_reconstruction.shape[0])
    if index < 0 or index >= count:
        raise IndexError("Offline registered camera index is out of range.")

    matrix = np.asarray(visual.camera_matrices_reconstruction[index], dtype=np.float64)
    position = tuple(float(value) for value in matrix[:3, 3])
    previous_distance: float | None = None
    if index > 0:
        previous = np.asarray(
            visual.camera_matrices_reconstruction[index - 1, :3, 3],
            dtype=np.float64,
        )
        current = np.asarray(position, dtype=np.float64)
        distance = float(np.linalg.norm(current - previous))
        if not math.isfinite(distance):
            raise ValueError("Offline camera gauge displacement is not finite.")
        previous_distance = distance

    return OfflineCameraInspection(
        camera_index=index,
        camera_id=visual.camera_ids[index],
        sampled_camera_count=count,
        registered_camera_total=int(visual.registered_camera_total),
        position_reconstruction=position,
        distance_from_previous_gauge=previous_distance,
    )


def format_offline_structure_summary(visual: OfflineSparseVisualization) -> str:
    """Compact bounded/full-count summary for the D2 inspection panel."""

    sampled_cameras = int(visual.camera_matrices_reconstruction.shape[0])
    sampled_landmarks = int(visual.landmark_points_reconstruction.shape[0])
    return (
        f"{sampled_cameras}/{visual.registered_camera_total} cameras • "
        f"{sampled_landmarks:,}/{visual.landmark_total:,} points shown"
    )


__all__ = [
    "OfflineCameraInspection",
    "build_offline_camera_inspection",
    "format_offline_structure_summary",
]
