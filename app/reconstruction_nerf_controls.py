"""Phase 9.5F presentation-only controls for interactive NeRF appearance.

The settings only influence interactive render resolution, overlay opacity and
whether a new view is requested automatically after canonical navigation. They
never modify the persisted NeRF artifact, canonical geometry, metric scale,
measurement/QC authority, or certified-metrology state.
"""

from __future__ import annotations

from dataclasses import dataclass

NERF_RENDER_PROFILES: tuple[tuple[str, str, int, int], ...] = (
    ("fast", "Fast · 320×180", 320, 180),
    ("balanced", "Balanced · 480×270", 480, 270),
    ("detail", "Detail · 640×360", 640, 360),
)
NERF_DEFAULT_RENDER_PROFILE = "detail"
NERF_OPACITY_PERCENT_MIN = 35
NERF_OPACITY_PERCENT_MAX = 100


def _profile_map() -> dict[str, tuple[str, int, int]]:
    return {key: (label, width, height) for key, label, width, height in NERF_RENDER_PROFILES}


@dataclass(frozen=True, slots=True)
class NerfDisplaySettings:
    render_profile: str = NERF_DEFAULT_RENDER_PROFILE
    opacity_percent: int = 100
    auto_refresh: bool = True

    def __post_init__(self) -> None:
        profile = str(self.render_profile).strip().lower()
        opacity = int(self.opacity_percent)
        if profile not in _profile_map():
            raise ValueError("NeRF display render profile is unsupported.")
        if not NERF_OPACITY_PERCENT_MIN <= opacity <= NERF_OPACITY_PERCENT_MAX:
            raise ValueError("NeRF display opacity percentage is outside the supported range.")
        object.__setattr__(self, "render_profile", profile)
        object.__setattr__(self, "opacity_percent", opacity)
        object.__setattr__(self, "auto_refresh", bool(self.auto_refresh))

    @property
    def render_label(self) -> str:
        return _profile_map()[self.render_profile][0]

    @property
    def render_width(self) -> int:
        return _profile_map()[self.render_profile][1]

    @property
    def render_height(self) -> int:
        return _profile_map()[self.render_profile][2]

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

    @property
    def status_text(self) -> str:
        refresh = "auto refresh" if self.auto_refresh else "manual refresh"
        return f"{self.render_label} • opacity {self.opacity_percent}% • {refresh}"


__all__ = [
    "NERF_DEFAULT_RENDER_PROFILE",
    "NERF_OPACITY_PERCENT_MAX",
    "NERF_OPACITY_PERCENT_MIN",
    "NERF_RENDER_PROFILES",
    "NerfDisplaySettings",
]
