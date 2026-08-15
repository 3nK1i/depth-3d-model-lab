"""Phase 9.4F presentation-only controls for interactive Gaussian appearance.

These controls only resample/scale already-persisted Gaussian display buffers.
They never change the persisted artifact, canonical geometry, metric scale,
measurement/QC authority, or certified-metrology state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.reconstruction_gaussian_interactive import InteractiveGaussianScene

GAUSSIAN_DENSITY_PERCENT_OPTIONS = (100, 75, 50, 25)
GAUSSIAN_SIZE_PERCENT_MIN = 60
GAUSSIAN_SIZE_PERCENT_MAX = 160
GAUSSIAN_OPACITY_PERCENT_MIN = 25
GAUSSIAN_OPACITY_PERCENT_MAX = 100


@dataclass(frozen=True, slots=True)
class GaussianDisplaySettings:
    density_percent: int = 100
    size_percent: int = 100
    opacity_percent: int = 100

    def __post_init__(self) -> None:
        density = int(self.density_percent)
        size = int(self.size_percent)
        opacity = int(self.opacity_percent)
        if density not in GAUSSIAN_DENSITY_PERCENT_OPTIONS:
            raise ValueError("Gaussian display density must use one of the supported percentages.")
        if not GAUSSIAN_SIZE_PERCENT_MIN <= size <= GAUSSIAN_SIZE_PERCENT_MAX:
            raise ValueError("Gaussian display size percentage is outside the supported range.")
        if not GAUSSIAN_OPACITY_PERCENT_MIN <= opacity <= GAUSSIAN_OPACITY_PERCENT_MAX:
            raise ValueError("Gaussian display opacity percentage is outside the supported range.")
        object.__setattr__(self, "density_percent", density)
        object.__setattr__(self, "size_percent", size)
        object.__setattr__(self, "opacity_percent", opacity)

    @property
    def visibility_only(self) -> bool:
        return True

    @property
    def canonical_geometry_authoritative(self) -> bool:
        return False

    @property
    def metric_scale_authoritative(self) -> bool:
        return False

    @property
    def measurement_authoritative(self) -> bool:
        return False

    @property
    def qc_authoritative(self) -> bool:
        return False

    @property
    def certified_metrology(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class GaussianDisplayBuffers:
    positions_reconstruction: np.ndarray
    colors_rgba: np.ndarray
    sizes_px: np.ndarray
    source_display_count: int
    display_count: int
    density_percent: int
    size_percent: int
    opacity_percent: int

    @property
    def status_text(self) -> str:
        return (
            f"{self.display_count:,}/{self.source_display_count:,} splats • "
            f"density {self.density_percent}% • size {self.size_percent}% • "
            f"opacity {self.opacity_percent}%"
        )

    @property
    def certified_metrology(self) -> bool:
        return False


def _stable_density_indices(total: int, density_percent: int) -> np.ndarray:
    total = int(total)
    if total < 1:
        raise ValueError("Interactive Gaussian display requires at least one splat.")
    density = int(density_percent)
    if density == 100:
        return np.arange(total, dtype=np.int64)
    target = max(1, int(round(total * density / 100.0)))
    return np.linspace(0, total - 1, num=target, dtype=np.int64)


def apply_gaussian_display_settings(
    scene: InteractiveGaussianScene,
    settings: GaussianDisplaySettings,
) -> GaussianDisplayBuffers:
    """Apply deterministic presentation-only density/size/opacity controls."""

    if not isinstance(scene, InteractiveGaussianScene):
        raise TypeError("scene must be InteractiveGaussianScene.")
    if not isinstance(settings, GaussianDisplaySettings):
        raise TypeError("settings must be GaussianDisplaySettings.")

    source_count = int(scene.display_gaussian_count)
    if source_count != int(scene.positions_reconstruction.shape[0]):
        raise ValueError("Interactive Gaussian scene display count is inconsistent.")
    indices = _stable_density_indices(source_count, settings.density_percent)

    positions = np.ascontiguousarray(scene.positions_reconstruction[indices], dtype=np.float32)
    colors = np.ascontiguousarray(scene.colors_rgba[indices], dtype=np.float32).copy()
    sizes = np.ascontiguousarray(scene.sizes_px[indices], dtype=np.float32).copy()

    colors[:, 3] = np.clip(
        colors[:, 3] * (settings.opacity_percent / 100.0),
        0.02,
        1.0,
    )
    sizes = np.clip(
        sizes * (settings.size_percent / 100.0),
        1.0,
        20.0,
    ).astype(np.float32, copy=False)

    return GaussianDisplayBuffers(
        positions_reconstruction=positions,
        colors_rgba=colors,
        sizes_px=sizes,
        source_display_count=source_count,
        display_count=int(indices.shape[0]),
        density_percent=settings.density_percent,
        size_percent=settings.size_percent,
        opacity_percent=settings.opacity_percent,
    )


__all__ = [
    "GAUSSIAN_DENSITY_PERCENT_OPTIONS",
    "GAUSSIAN_OPACITY_PERCENT_MAX",
    "GAUSSIAN_OPACITY_PERCENT_MIN",
    "GAUSSIAN_SIZE_PERCENT_MAX",
    "GAUSSIAN_SIZE_PERCENT_MIN",
    "GaussianDisplayBuffers",
    "GaussianDisplaySettings",
    "apply_gaussian_display_settings",
]
