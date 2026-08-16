"""Phase 9.6G Digital Twin GPU/resource coordination contract.

The coordinator is deliberately dependency-light and presentation/admission only.
It observes application-owned Reconstruction execution, the generic E1 external
build supervisor, the isolated interactive Nerfacto worker, and bounded Gaussian
viewer residency.  It never imports DA3, gsplat, Nerfstudio, torch, or CUDA
libraries and never changes geometry, scale, measurement, QC, or metrology
authority.

Static previews are not runtime owners; Gaussian/NeRF preview images remain presentation-only.  Dynamic owners are
serialized so the qualification GPU cannot accidentally host incompatible
heavyweight work at the same time.  Appearance resources may be released during
an explicit handoff, but they are never silently substituted across
representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.hardware.gpu_info import HardwareInfo, detect_hardware


class DigitalTwinResourceOwner(str, Enum):
    IDLE = "idle"
    RECONSTRUCTION_RUNTIME = "reconstruction_runtime"
    DIGITAL_TWIN_BUILD = "digital_twin_build"
    INTERACTIVE_NERF = "interactive_nerf"


@dataclass(frozen=True, slots=True)
class DigitalTwinResourceCoordination:
    """Immutable resource ownership/presentation snapshot for the workspace."""

    owner: DigitalTwinResourceOwner
    device_text: str
    startup_resources_text: str
    processing_active: bool
    reconstruction_active: bool
    interactive_nerf_active: bool
    interactive_gaussian_resident: bool
    static_preview_active: bool
    build_label: str = ""
    conflict: bool = False
    conflict_reason: str = ""
    heavy_backends_external_only: bool = True
    static_previews_are_owners: bool = False
    canonical_geometry_authoritative: bool = False
    metric_scale_authoritative: bool = False
    measurement_authoritative: bool = False
    qc_authoritative: bool = False
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.owner, DigitalTwinResourceOwner):
            raise TypeError("owner must be DigitalTwinResourceOwner.")
        if not self.heavy_backends_external_only:
            raise ValueError("Phase 9.6G requires heavy Digital Twin runtimes to stay external.")
        if self.static_previews_are_owners:
            raise ValueError("Static appearance previews must not reserve heavyweight resources.")
        if any(
            (
                self.canonical_geometry_authoritative,
                self.metric_scale_authoritative,
                self.measurement_authoritative,
                self.qc_authoritative,
                self.certified_metrology,
            )
        ):
            raise ValueError("Resource coordination cannot claim scene/metrology authority.")
        object.__setattr__(self, "device_text", str(self.device_text).strip() or "Unknown device")
        object.__setattr__(self, "startup_resources_text", str(self.startup_resources_text).strip())
        object.__setattr__(self, "build_label", str(self.build_label).strip())
        object.__setattr__(self, "conflict_reason", str(self.conflict_reason).strip())

    @property
    def heavy_runtime_active(self) -> bool:
        return bool(
            self.processing_active
            or self.reconstruction_active
            or self.interactive_nerf_active
        )

    @property
    def page_resource_active(self) -> bool:
        """Resource ownership not already represented by ReconstructionController."""

        return bool(
            self.processing_active
            or self.interactive_nerf_active
            or self.interactive_gaussian_resident
        )

    @property
    def owner_text(self) -> str:
        if self.conflict:
            return "RESOURCE CONFLICT • fail closed"
        if self.owner is DigitalTwinResourceOwner.DIGITAL_TWIN_BUILD:
            suffix = f" • {self.build_label}" if self.build_label else ""
            return f"Build job owns external runtime{suffix}"
        if self.owner is DigitalTwinResourceOwner.RECONSTRUCTION_RUNTIME:
            return "Reconstruction runtime owns resources"
        if self.owner is DigitalTwinResourceOwner.INTERACTIVE_NERF:
            return "Interactive NeRF owns external GPU runtime"
        if self.interactive_gaussian_resident:
            return "Gaussian display resident • bounded viewer resources"
        return "Available • no dynamic Digital Twin owner"

    @property
    def policy_text(self) -> str:
        return "One heavy runtime owner • explicit handoff • static previews remain available"

    @property
    def details_text(self) -> str:
        values = [self.owner_text, self.startup_resources_text, self.policy_text]
        if self.conflict_reason:
            values.append(self.conflict_reason)
        return "\n".join(item for item in values if item)

    @property
    def can_start_interactive_nerf(self) -> bool:
        return bool(not self.conflict and not self.processing_active and not self.reconstruction_active)

    @property
    def can_start_interactive_gaussian(self) -> bool:
        return bool(not self.conflict and not self.processing_active and not self.reconstruction_active)

    def interactive_admission_failure(self, representation: str) -> str:
        label = str(representation).strip() or "appearance runtime"
        if self.conflict:
            return self.conflict_reason or "Digital Twin resource ownership is inconsistent; stop active work first."
        if self.processing_active:
            return f"Cancel the active Digital Twin build before starting {label}."
        if self.reconstruction_active:
            return f"Stop active Reconstruction before starting {label}."
        return ""


def _startup_resources_text(hardware: HardwareInfo) -> str:
    if not hardware.has_cuda_device:
        return "CPU/unknown accelerator • external GPU builds require their qualified runtime"
    total = hardware.total_memory_gib
    free = hardware.free_memory_gib
    if free is not None and total is not None:
        return f"{hardware.short_label} • startup {free:.1f}/{total:.1f} GB free/total VRAM"
    if total is not None:
        return f"{hardware.short_label} • {total:.1f} GB total VRAM • startup free unavailable"
    return f"{hardware.short_label} • startup VRAM snapshot unavailable"


def build_digital_twin_resource_coordination(
    *,
    processing_active: bool,
    reconstruction_active: bool,
    interactive_nerf_active: bool,
    interactive_gaussian_resident: bool,
    static_preview_active: bool = False,
    build_label: str = "",
    hardware: HardwareInfo | None = None,
) -> DigitalTwinResourceCoordination:
    """Build one fail-closed resource ownership snapshot.

    ``processing_active``, ``reconstruction_active`` and
    ``interactive_nerf_active`` are mutually exclusive heavyweight owners.  A
    Gaussian viewer is tracked as bounded display residency rather than an
    external CUDA backend owner, but it still counts as a page resource for
    application-mode admission until explicitly released.
    """

    hardware = hardware or detect_hardware()
    if not isinstance(hardware, HardwareInfo):
        raise TypeError("hardware must be HardwareInfo.")

    processing = bool(processing_active)
    reconstruction = bool(reconstruction_active)
    nerf = bool(interactive_nerf_active)
    gaussian = bool(interactive_gaussian_resident)
    preview = bool(static_preview_active)
    owners = int(processing) + int(reconstruction) + int(nerf)
    conflict = owners > 1
    conflict_reason = ""
    if conflict:
        names: list[str] = []
        if processing:
            names.append("Digital Twin build")
        if reconstruction:
            names.append("Reconstruction runtime")
        if nerf:
            names.append("interactive NeRF")
        conflict_reason = "Concurrent heavyweight owners detected: " + ", ".join(names)

    if processing:
        owner = DigitalTwinResourceOwner.DIGITAL_TWIN_BUILD
    elif reconstruction:
        owner = DigitalTwinResourceOwner.RECONSTRUCTION_RUNTIME
    elif nerf:
        owner = DigitalTwinResourceOwner.INTERACTIVE_NERF
    else:
        owner = DigitalTwinResourceOwner.IDLE

    return DigitalTwinResourceCoordination(
        owner=owner,
        device_text=hardware.short_label,
        startup_resources_text=_startup_resources_text(hardware),
        processing_active=processing,
        reconstruction_active=reconstruction,
        interactive_nerf_active=nerf,
        interactive_gaussian_resident=gaussian,
        static_preview_active=preview,
        build_label=str(build_label),
        conflict=conflict,
        conflict_reason=conflict_reason,
    )


__all__ = [
    "DigitalTwinResourceCoordination",
    "DigitalTwinResourceOwner",
    "build_digital_twin_resource_coordination",
]
