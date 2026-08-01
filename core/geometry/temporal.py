"""Low-cost temporal smoothing for aligned monocular geometry outputs.

The smoother is deliberately UI-process only. It blends the latest depth,
point-map and normal-map arrays when consecutive frames have the same shape,
then resets automatically when a source, model, resolution or sampling stride
changes. Static images bypass it entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SmoothedGeometry:
    """One temporally filtered set of aligned geometry arrays."""

    depth: np.ndarray
    points: np.ndarray | None
    normals: np.ndarray | None


class TemporalGeometrySmoother:
    """Exponential moving average for aligned depth and geometry maps.

    ``strength`` is the retained contribution of the previous filtered frame:
    0.0 disables smoothing, while values closer to 1.0 produce a steadier but
    more delayed result. Invalid values never bleed into valid geometry.
    """

    def __init__(self) -> None:
        self._depth: np.ndarray | None = None
        self._points: np.ndarray | None = None
        self._normals: np.ndarray | None = None
        self._signature: tuple[object, ...] | None = None

    def reset(self) -> None:
        self._depth = None
        self._points = None
        self._normals = None
        self._signature = None

    def apply(
        self,
        *,
        depth: np.ndarray,
        points: np.ndarray | None,
        normals: np.ndarray | None,
        strength: float,
        signature: tuple[object, ...],
    ) -> SmoothedGeometry:
        """Return filtered arrays and update the internal history."""

        retention = float(np.clip(strength, 0.0, 0.95))
        current_depth = np.asarray(depth, dtype=np.float32)
        current_points = (
            None if points is None else np.asarray(points, dtype=np.float32)
        )
        current_normals = (
            None if normals is None else np.asarray(normals, dtype=np.float32)
        )

        compatible = (
            retention > 0.0
            and self._signature == signature
            and self._depth is not None
            and self._depth.shape == current_depth.shape
            and _same_optional_shape(self._points, current_points)
            and _same_optional_shape(self._normals, current_normals)
        )

        if compatible:
            filtered_depth = _blend_scalar_map(
                self._depth,
                current_depth,
                retention,
                positive_only=True,
            )
            filtered_points = _blend_vector_map(
                self._points,
                current_points,
                retention,
                normalize=False,
            )
            filtered_normals = _blend_vector_map(
                self._normals,
                current_normals,
                retention,
                normalize=True,
            )
        else:
            filtered_depth = current_depth.copy()
            filtered_points = None if current_points is None else current_points.copy()
            filtered_normals = None if current_normals is None else current_normals.copy()

        self._depth = filtered_depth
        self._points = filtered_points
        self._normals = filtered_normals
        self._signature = signature
        return SmoothedGeometry(filtered_depth, filtered_points, filtered_normals)


def _same_optional_shape(
    previous: np.ndarray | None,
    current: np.ndarray | None,
) -> bool:
    if previous is None or current is None:
        return previous is None and current is None
    return previous.shape == current.shape


def _blend_scalar_map(
    previous: np.ndarray,
    current: np.ndarray,
    retention: float,
    *,
    positive_only: bool,
) -> np.ndarray:
    output = current.copy()
    valid_previous = np.isfinite(previous)
    valid_current = np.isfinite(current)
    if positive_only:
        valid_previous &= previous > 0.0
        valid_current &= current > 0.0
    both = valid_previous & valid_current
    output[both] = retention * previous[both] + (1.0 - retention) * current[both]
    return output


def _blend_vector_map(
    previous: np.ndarray | None,
    current: np.ndarray | None,
    retention: float,
    *,
    normalize: bool,
) -> np.ndarray | None:
    if previous is None or current is None:
        return None if current is None else current.copy()

    output = current.copy()
    valid_previous = np.all(np.isfinite(previous), axis=-1)
    valid_current = np.all(np.isfinite(current), axis=-1)
    both = valid_previous & valid_current
    output[both] = retention * previous[both] + (1.0 - retention) * current[both]

    if normalize:
        norms = np.linalg.norm(output, axis=-1, keepdims=True)
        valid = np.isfinite(norms[..., 0]) & (norms[..., 0] > 1e-8)
        output[valid] /= norms[valid]
    return output
