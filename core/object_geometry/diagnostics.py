"""Diagnostics for preserving image-supported thin object geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SupportRetentionDiagnostics:
    """Describe how much source-image support survives a 3D filtering stage."""

    input_points: int
    retained_points: int
    retained_fraction: float
    horizontal_span_retention: float
    vertical_span_retention: float
    occupied_cell_retention: float
    thin_support_loss: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "input_points": self.input_points,
            "retained_points": self.retained_points,
            "retained_fraction": self.retained_fraction,
            "horizontal_span_retention": self.horizontal_span_retention,
            "vertical_span_retention": self.vertical_span_retention,
            "occupied_cell_retention": self.occupied_cell_retention,
            "thin_support_loss": self.thin_support_loss,
        }


def _axis_span(values: np.ndarray) -> float:
    if values.size <= 1:
        return 1.0 if values.size else 0.0
    return max(1.0, float(np.max(values) - np.min(values) + 1.0))


def _occupied_cells(pixels: np.ndarray, reference: np.ndarray, grid_size: int) -> int:
    if len(pixels) == 0 or len(reference) == 0:
        return 0
    minimum = np.min(reference, axis=0).astype(np.float64)
    maximum = np.max(reference, axis=0).astype(np.float64)
    span = np.maximum(1.0, maximum - minimum + 1.0)
    normalized = np.clip((pixels.astype(np.float64) - minimum) / span, 0.0, 1.0 - 1e-12)
    cells = np.floor(normalized * float(grid_size)).astype(np.int64)
    cells = np.clip(cells, 0, grid_size - 1)
    return len({(int(x), int(y)) for x, y in cells})


def support_retention_diagnostics(
    source_pixel_indices_xy: np.ndarray,
    keep: np.ndarray,
    *,
    grid_size: int,
    minimum_point_retention: float,
    minimum_axis_span_retention: float,
    minimum_grid_coverage_retention: float,
) -> SupportRetentionDiagnostics:
    """Measure whether filtering removed broad image-supported object structure."""

    pixels = np.asarray(source_pixel_indices_xy, dtype=np.int64)
    keep_mask = np.asarray(keep, dtype=np.bool_)
    if pixels.ndim != 2 or pixels.shape[1:] != (2,):
        raise ValueError("source_pixel_indices_xy must have shape (N, 2).")
    if keep_mask.shape != (len(pixels),):
        raise ValueError("keep must have shape (N,).")

    input_count = len(pixels)
    retained = pixels[keep_mask]
    retained_count = len(retained)
    if input_count == 0:
        return SupportRetentionDiagnostics(0, 0, 1.0, 1.0, 1.0, 1.0, False)

    retained_fraction = retained_count / input_count
    if retained_count == 0:
        horizontal = vertical = occupied = 0.0
    else:
        horizontal = min(1.0, _axis_span(retained[:, 0]) / _axis_span(pixels[:, 0]))
        vertical = min(1.0, _axis_span(retained[:, 1]) / _axis_span(pixels[:, 1]))
        source_cells = max(1, _occupied_cells(pixels, pixels, grid_size))
        occupied = min(1.0, _occupied_cells(retained, pixels, grid_size) / source_cells)

    thin_support_loss = (
        retained_fraction < minimum_point_retention
        or horizontal < minimum_axis_span_retention
        or vertical < minimum_axis_span_retention
        or occupied < minimum_grid_coverage_retention
    )
    return SupportRetentionDiagnostics(
        input_points=input_count,
        retained_points=retained_count,
        retained_fraction=float(retained_fraction),
        horizontal_span_retention=float(horizontal),
        vertical_span_retention=float(vertical),
        occupied_cell_retention=float(occupied),
        thin_support_loss=bool(thin_support_loss),
    )
