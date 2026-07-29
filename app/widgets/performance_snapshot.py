"""Structured performance-footer data model and formatting helpers.

One dataclass carries every value the footer displays, so
:meth:`PerformanceBar.update_snapshot` can replace the whole footer in a
single call — now with static preview data, later with real values from
a runtime monitor — without the widget itself knowing which.

All value formatting (FPS decimals, thousands separators, HH:MM:SS, the
"—" placeholder for unavailable data) lives here, in one place, instead
of being repeated in signal handlers.
"""

from __future__ import annotations

from dataclasses import dataclass

PLACEHOLDER = "—"

# Runtime status states the footer's status cell must support (spec §7).
# Colors: RUNNING green, PAUSED amber, ERROR red, everything else neutral.
STATUS_COLOR_GREEN = "green"
STATUS_COLOR_AMBER = "amber"
STATUS_COLOR_RED = "red"
STATUS_COLOR_NEUTRAL = "neutral"

_STATUS_COLORS: dict[str, str] = {
    "IDLE": STATUS_COLOR_NEUTRAL,
    "CONNECTING": STATUS_COLOR_NEUTRAL,
    "READY": STATUS_COLOR_NEUTRAL,
    "RUNNING": STATUS_COLOR_GREEN,
    "PAUSED": STATUS_COLOR_AMBER,
    "STOPPING": STATUS_COLOR_NEUTRAL,
    "ERROR": STATUS_COLOR_RED,
}


def status_color(status_text: str) -> str:
    """Return the color role for a status state; unknown states are neutral."""
    return _STATUS_COLORS.get(status_text.upper(), STATUS_COLOR_NEUTRAL)


def format_fps(value: float | None) -> str:
    return PLACEHOLDER if value is None else f"{value:.1f}"


def format_latency(value_ms: float | None) -> str:
    return PLACEHOLDER if value_ms is None else f"{value_ms:.1f} ms"


def format_points(count: int | None) -> str:
    return PLACEHOLDER if count is None else f"{count:,}"


def format_resolution(size: tuple[int, int] | None) -> str:
    return PLACEHOLDER if size is None else f"{size[0]}\u00d7{size[1]}"


def format_vram(used_gb: float | None, total_gb: float | None) -> str:
    if used_gb is None or total_gb is None:
        return PLACEHOLDER
    return f"{used_gb:.1f} / {total_gb:.0f} GB"


def vram_ratio(used_gb: float | None, total_gb: float | None) -> float:
    if not used_gb or not total_gb:
        return 0.0
    return max(0.0, min(1.0, used_gb / total_gb))


def format_uptime(seconds: int | None) -> str:
    if seconds is None:
        return PLACEHOLDER
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_or_dash(value: str | None) -> str:
    return value if value else PLACEHOLDER


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    """Every value the performance footer displays, in one structure."""

    status: str
    camera_fps: float | None = None
    inference_fps: float | None = None
    latency_ms: float | None = None
    render_fps: float | None = None
    point_count: int | None = None
    resolution: tuple[int, int] | None = None
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None
    model_name: str | None = None
    precision: str | None = None
    uptime_seconds: int | None = None

    @classmethod
    def empty(cls) -> "PerformanceSnapshot":
        """The IDLE state: badge only, every metric unavailable."""
        return cls(status="IDLE")

    @classmethod
    def preview(cls) -> "PerformanceSnapshot":
        """Static, deterministic Phase-1 preview data matching the target
        mockup. Never randomized or animated (spec §8)."""
        return cls(
            status="RUNNING",
            camera_fps=30.0,
            inference_fps=21.4,
            latency_ms=46.8,
            render_fps=60.0,
            point_count=842_713,
            resolution=(768, 768),
            vram_used_gb=23.6,
            vram_total_gb=32.0,
            model_name="MoGe-2 ViT-L",
            precision="FP16",
            uptime_seconds=12 * 60 + 43,
        )