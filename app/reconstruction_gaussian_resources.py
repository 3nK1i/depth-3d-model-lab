"""Phase 9.4G interactive Gaussian performance/resource lifecycle contract.

The application-side Gaussian viewer is intentionally backend-free.  This module
chooses a conservative presentation budget from the cached hardware snapshot and
reports CPU/display-staging residency plus UI-side load/apply timings.  It does
not estimate CUDA/VRAM allocation, launch a Gaussian runtime, or change geometry,
metric, measurement, QC, or certified-metrology authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from app.reconstruction_gaussian_controls import GaussianDisplayBuffers
from app.reconstruction_gaussian_interactive import (
    MAX_INTERACTIVE_GAUSSIANS,
    InteractiveGaussianScene,
)
from core.hardware.gpu_info import HardwareInfo, detect_hardware


class GaussianViewerResidencyState(str, Enum):
    UNLOADED = "unloaded"
    RESIDENT = "resident"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class GaussianInteractiveResourcePolicy:
    device_label: str
    max_interactive_gaussians: int
    tier: str
    adaptive: bool = True
    backend_free: bool = True
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        cap = int(self.max_interactive_gaussians)
        if cap < 1 or cap > MAX_INTERACTIVE_GAUSSIANS:
            raise ValueError("Interactive Gaussian resource cap is outside the qualified range.")
        object.__setattr__(self, "device_label", str(self.device_label).strip() or "Unknown device")
        object.__setattr__(self, "max_interactive_gaussians", cap)
        object.__setattr__(self, "tier", str(self.tier).strip() or "conservative")
        object.__setattr__(self, "adaptive", True)
        object.__setattr__(self, "backend_free", True)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def budget_text(self) -> str:
        return f"{self.max_interactive_gaussians:,} splat display cap • {self.device_label}"


def build_gaussian_interactive_resource_policy(
    hardware: HardwareInfo | None = None,
) -> GaussianInteractiveResourcePolicy:
    """Return a conservative automatic display cap for the current machine.

    The cap bounds application-side VisPy point-splat presentation only.  It is
    not a CUDA allocation promise or a Gaussian training/render benchmark.
    """

    hardware = hardware or detect_hardware()
    if not isinstance(hardware, HardwareInfo):
        raise TypeError("hardware must be HardwareInfo.")

    total_gib = hardware.total_memory_gib
    if not hardware.has_cuda_device or total_gib is None:
        cap, tier = 75_000, "cpu-or-unknown"
    elif total_gib <= 6.5:
        cap, tier = 100_000, "conservative-6gb"
    elif total_gib <= 16.0:
        cap, tier = 150_000, "balanced"
    else:
        cap, tier = MAX_INTERACTIVE_GAUSSIANS, "high-memory"

    return GaussianInteractiveResourcePolicy(
        device_label=hardware.short_label,
        max_interactive_gaussians=min(cap, MAX_INTERACTIVE_GAUSSIANS),
        tier=tier,
    )


def gaussian_scene_staging_bytes(scene: InteractiveGaussianScene | None) -> int:
    if scene is None:
        return 0
    return int(
        scene.positions_reconstruction.nbytes
        + scene.colors_rgba.nbytes
        + scene.sizes_px.nbytes
    )


def gaussian_display_staging_bytes(buffers: GaussianDisplayBuffers | None) -> int:
    if buffers is None:
        return 0
    return int(
        buffers.positions_reconstruction.nbytes
        + buffers.colors_rgba.nbytes
        + buffers.sizes_px.nbytes
    )


def _mib(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


@dataclass(frozen=True, slots=True)
class GaussianViewerRuntimeHealth:
    state: GaussianViewerResidencyState
    device_label: str
    resource_tier: str
    max_display_gaussians: int
    total_gaussian_count: int
    source_display_count: int
    controlled_display_count: int
    source_staging_bytes: int
    controlled_staging_bytes: int
    load_latency_ms: float
    apply_latency_ms: float
    load_count: int
    release_count: int
    canvas_resource_generation: int
    canvas_release_count: int
    canonical_geometry_authoritative: bool
    gaussian_measurement_authoritative: bool = False
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, GaussianViewerResidencyState):
            raise TypeError("state must be GaussianViewerResidencyState.")
        for name in (
            "max_display_gaussians",
            "total_gaussian_count",
            "source_display_count",
            "controlled_display_count",
            "source_staging_bytes",
            "controlled_staging_bytes",
            "load_count",
            "release_count",
            "canvas_resource_generation",
            "canvas_release_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name in ("load_latency_ms", "apply_latency_ms"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.source_display_count > self.max_display_gaussians:
            raise ValueError("source display count exceeds the adaptive display cap.")
        if self.controlled_display_count > self.source_display_count:
            raise ValueError("controlled display count exceeds loaded source display count.")
        if self.gaussian_measurement_authoritative or self.certified_metrology:
            raise ValueError("Gaussian runtime health cannot claim measurement or certified-metrology authority.")

    @property
    def resident(self) -> bool:
        return self.state is GaussianViewerResidencyState.RESIDENT

    @property
    def resources_text(self) -> str:
        if not self.resident:
            return f"Released • cap {self.max_display_gaussians:,} splats • {self.device_label}"
        staging = self.source_staging_bytes + self.controlled_staging_bytes
        return (
            f"{self.controlled_display_count:,}/{self.max_display_gaussians:,} displayed • "
            f"{_mib(staging):.2f} MiB CPU staging"
        )

    @property
    def performance_text(self) -> str:
        if self.load_count < 1:
            return "No interactive Gaussian load measured"
        return f"Load {self.load_latency_ms:.1f} ms • display apply {self.apply_latency_ms:.1f} ms"

    @property
    def lifecycle_text(self) -> str:
        if self.resident:
            return f"Resident on demand • {self.load_count} loads • {self.release_count} releases"
        if self.release_count:
            return f"Released on hide/switch • {self.release_count} releases • reload on demand"
        return "Unloaded • no Gaussian viewer buffers resident"

    @property
    def authority_text(self) -> str:
        return "Diagnostics only • canonical picking/measurement unchanged • metrology not certified"

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "resident": self.resident,
            "device_label": self.device_label,
            "resource_tier": self.resource_tier,
            "max_display_gaussians": int(self.max_display_gaussians),
            "total_gaussian_count": int(self.total_gaussian_count),
            "source_display_count": int(self.source_display_count),
            "controlled_display_count": int(self.controlled_display_count),
            "source_staging_bytes": int(self.source_staging_bytes),
            "controlled_staging_bytes": int(self.controlled_staging_bytes),
            "load_latency_ms": float(self.load_latency_ms),
            "apply_latency_ms": float(self.apply_latency_ms),
            "load_count": int(self.load_count),
            "release_count": int(self.release_count),
            "canvas_resource_generation": int(self.canvas_resource_generation),
            "canvas_release_count": int(self.canvas_release_count),
            "canonical_geometry_authoritative": bool(self.canonical_geometry_authoritative),
            "gaussian_measurement_authoritative": False,
            "certified_metrology": False,
            "gpu_vram_measured": False,
            "timings_are_ui_side": True,
        }


def build_gaussian_viewer_runtime_health(
    *,
    policy: GaussianInteractiveResourcePolicy,
    scene: InteractiveGaussianScene | None,
    buffers: GaussianDisplayBuffers | None,
    resident: bool,
    had_release: bool,
    load_latency_ms: float,
    apply_latency_ms: float,
    load_count: int,
    release_count: int,
    canvas_resource_generation: int,
    canvas_release_count: int,
    canonical_geometry_authoritative: bool,
    last_total_gaussian_count: int = 0,
    last_source_display_count: int = 0,
) -> GaussianViewerRuntimeHealth:
    state = (
        GaussianViewerResidencyState.RESIDENT
        if resident
        else GaussianViewerResidencyState.RELEASED
        if had_release
        else GaussianViewerResidencyState.UNLOADED
    )
    return GaussianViewerRuntimeHealth(
        state=state,
        device_label=policy.device_label,
        resource_tier=policy.tier,
        max_display_gaussians=policy.max_interactive_gaussians,
        total_gaussian_count=(
            int(scene.total_gaussian_count) if scene is not None else int(last_total_gaussian_count)
        ),
        source_display_count=(
            int(scene.display_gaussian_count) if scene is not None else int(last_source_display_count)
        ),
        controlled_display_count=(int(buffers.display_count) if buffers is not None else 0),
        source_staging_bytes=gaussian_scene_staging_bytes(scene),
        controlled_staging_bytes=gaussian_display_staging_bytes(buffers),
        load_latency_ms=float(load_latency_ms),
        apply_latency_ms=float(apply_latency_ms),
        load_count=int(load_count),
        release_count=int(release_count),
        canvas_resource_generation=int(canvas_resource_generation),
        canvas_release_count=int(canvas_release_count),
        canonical_geometry_authoritative=bool(canonical_geometry_authoritative),
        gaussian_measurement_authoritative=False,
        certified_metrology=False,
    )


__all__ = [
    "GaussianInteractiveResourcePolicy",
    "GaussianViewerResidencyState",
    "GaussianViewerRuntimeHealth",
    "build_gaussian_interactive_resource_policy",
    "build_gaussian_viewer_runtime_health",
    "gaussian_display_staging_bytes",
    "gaussian_scene_staging_bytes",
]
