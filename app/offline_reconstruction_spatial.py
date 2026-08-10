"""Offline sparse-SfM spatial coverage presentation for Phase 8.8D4.

D4 remains presentation-only.  It derives a robust sparse reconstruction
presentation envelope and camera-center spread from the bounded D1 artifact.
All values stay in the arbitrary reconstruction gauge.  The envelope is a
visual inspection aid, not object dimensions, metric geometry, or certified
metrology.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from app.offline_reconstruction_visualization import OfflineSparseVisualization

ROBUST_LOWER_QUANTILE = 0.02
ROBUST_UPPER_QUANTILE = 0.98


@dataclass(frozen=True, slots=True)
class OfflineSparseSpatialCoverage:
    """Unit-honest spatial presentation derived from one bounded SfM sample."""

    sampled_landmark_count: int
    landmark_total: int
    registered_camera_count: int
    robust_min_reconstruction: tuple[float, float, float] | None
    robust_max_reconstruction: tuple[float, float, float] | None
    center_reconstruction: tuple[float, float, float] | None
    extents_reconstruction: tuple[float, float, float] | None
    envelope_diagonal_gauge: float | None
    camera_span_gauge: float | None
    camera_standoff_min_gauge: float | None
    camera_standoff_median_gauge: float | None
    camera_standoff_max_gauge: float | None

    @property
    def has_envelope(self) -> bool:
        return self.robust_min_reconstruction is not None and self.robust_max_reconstruction is not None

    @property
    def envelope_text(self) -> str:
        if self.extents_reconstruction is None:
            return "Unavailable"
        x, y, z = self.extents_reconstruction
        return f"X {x:.3f} • Y {y:.3f} • Z {z:.3f} gauge"

    @property
    def camera_span_text(self) -> str:
        if self.camera_span_gauge is None:
            return "Unavailable"
        return f"{self.registered_camera_count} centers • {self.camera_span_gauge:.3f} gauge span"

    @property
    def standoff_text(self) -> str:
        values = (
            self.camera_standoff_min_gauge,
            self.camera_standoff_median_gauge,
            self.camera_standoff_max_gauge,
        )
        if any(value is None for value in values):
            return "Unavailable"
        low, median, high = values
        return f"{low:.3f} / {median:.3f} / {high:.3f} gauge min / med / max"

    @property
    def basis_text(self) -> str:
        return (
            f"{self.sampled_landmark_count:,}/{self.landmark_total:,} sparse points • "
            "presentation envelope only"
        )

    @property
    def envelope_tooltip(self) -> str:
        return (
            "Robust 2–98% axis-aligned envelope of the bounded sparse-SfM presentation sample. "
            "Values use arbitrary reconstruction gauge units and are not object dimensions or metres."
        )


def build_offline_sparse_spatial_coverage(
    visual: OfflineSparseVisualization,
) -> OfflineSparseSpatialCoverage:
    """Derive bounded spatial-coverage readouts without changing reconstruction authority."""

    if not isinstance(visual, OfflineSparseVisualization):
        raise TypeError("visual must be OfflineSparseVisualization.")

    points = np.asarray(visual.landmark_points_reconstruction, dtype=np.float64)
    cameras = np.asarray(visual.camera_matrices_reconstruction, dtype=np.float64)
    sampled_count = int(points.shape[0])
    camera_count = int(cameras.shape[0])

    robust_min: tuple[float, float, float] | None = None
    robust_max: tuple[float, float, float] | None = None
    center: tuple[float, float, float] | None = None
    extents: tuple[float, float, float] | None = None
    diagonal: float | None = None
    camera_span: float | None = None
    standoff_min: float | None = None
    standoff_median: float | None = None
    standoff_max: float | None = None

    if sampled_count >= 3:
        lower = np.quantile(points, ROBUST_LOWER_QUANTILE, axis=0)
        upper = np.quantile(points, ROBUST_UPPER_QUANTILE, axis=0)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper < lower):
            raise ValueError("Sparse-SfM presentation envelope is not finite or ordered.")
        extent = upper - lower
        midpoint = 0.5 * (lower + upper)
        diag = float(np.linalg.norm(extent))
        if not math.isfinite(diag):
            raise ValueError("Sparse-SfM presentation envelope diagonal is not finite.")
        robust_min = tuple(float(value) for value in lower)
        robust_max = tuple(float(value) for value in upper)
        center = tuple(float(value) for value in midpoint)
        extents = tuple(float(value) for value in extent)
        diagonal = diag

        if camera_count > 0:
            camera_centers = np.asarray(cameras[:, :3, 3], dtype=np.float64)
            if not np.isfinite(camera_centers).all():
                raise ValueError("Offline registered camera centers must be finite.")
            camera_extent = camera_centers.max(axis=0) - camera_centers.min(axis=0)
            span = float(np.linalg.norm(camera_extent))
            distances = np.linalg.norm(camera_centers - midpoint.reshape(1, 3), axis=1)
            if not math.isfinite(span) or not np.isfinite(distances).all():
                raise ValueError("Offline camera spatial coverage is not finite.")
            camera_span = span
            standoff_min = float(np.min(distances))
            standoff_median = float(np.median(distances))
            standoff_max = float(np.max(distances))

    return OfflineSparseSpatialCoverage(
        sampled_landmark_count=sampled_count,
        landmark_total=int(visual.landmark_total),
        registered_camera_count=camera_count,
        robust_min_reconstruction=robust_min,
        robust_max_reconstruction=robust_max,
        center_reconstruction=center,
        extents_reconstruction=extents,
        envelope_diagonal_gauge=diagonal,
        camera_span_gauge=camera_span,
        camera_standoff_min_gauge=standoff_min,
        camera_standoff_median_gauge=standoff_median,
        camera_standoff_max_gauge=standoff_max,
    )


__all__ = [
    "OfflineSparseSpatialCoverage",
    "ROBUST_LOWER_QUANTILE",
    "ROBUST_UPPER_QUANTILE",
    "build_offline_sparse_spatial_coverage",
]
