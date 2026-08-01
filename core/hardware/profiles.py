"""Hardware-aware MoGe-2 execution profiles.

Profiles are capability-driven rather than keyed to product names. That
means an RTX 2060, RTX 5090, or an unknown future NVIDIA GPU is handled by
the same VRAM/runtime rules. The profile also owns the MoGe-2 checkpoint
variant and token budget so the worker receives one coherent config.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo


class ComputeProfile(str, Enum):
    AUTO = "auto"
    EFFICIENT = "efficient"
    BALANCED = "balanced"
    ULTRA = "ultra"

    @property
    def display_name(self) -> str:
        return self.value.capitalize()


class InferencePrecision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    @property
    def display_name(self) -> str:
        labels = {
            InferencePrecision.FP32: "FP32",
            InferencePrecision.FP16: "FP16 (Half)",
            InferencePrecision.BF16: "BF16",
        }
        return labels[self]


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    profile: ComputeProfile
    model_repo: str
    model_label: str
    resolution_level: int
    num_tokens: int
    point_budget: int
    preferred_precision: InferencePrecision
    minimum_vram_gib: float


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    requested: ComputeProfile
    effective: ComputeProfile
    spec: ProfileSpec
    device: str
    precision: InferencePrecision
    downgraded: bool
    warning: str | None

    def as_model_config(self) -> dict[str, object]:
        """Picklable config ready for a future LOAD_MODEL command."""

        return {
            "profile": self.effective.value,
            "device": self.device,
            "precision": self.precision.value,
            "pretrained": self.spec.model_repo,
            "model_label": self.spec.model_label,
            "resolution_level": self.spec.resolution_level,
            "num_tokens": self.spec.num_tokens,
            "point_budget": self.spec.point_budget,
        }


PROFILE_SPECS: dict[ComputeProfile, ProfileSpec] = {
    ComputeProfile.EFFICIENT: ProfileSpec(
        profile=ComputeProfile.EFFICIENT,
        model_repo="Ruicheng/moge-2-vits-normal",
        model_label="MoGe-2 ViT-S Normal",
        resolution_level=5,
        num_tokens=1200,
        point_budget=250_000,
        preferred_precision=InferencePrecision.FP16,
        minimum_vram_gib=4.0,
    ),
    ComputeProfile.BALANCED: ProfileSpec(
        profile=ComputeProfile.BALANCED,
        model_repo="Ruicheng/moge-2-vitb-normal",
        model_label="MoGe-2 ViT-B Normal",
        resolution_level=7,
        num_tokens=1800,
        point_budget=500_000,
        preferred_precision=InferencePrecision.FP16,
        minimum_vram_gib=8.0,
    ),
    ComputeProfile.ULTRA: ProfileSpec(
        profile=ComputeProfile.ULTRA,
        model_repo="Ruicheng/moge-2-vitl-normal",
        model_label="MoGe-2 ViT-L Normal",
        resolution_level=9,
        num_tokens=2500,
        point_budget=1_000_000,
        preferred_precision=InferencePrecision.FP16,
        minimum_vram_gib=20.0,
    ),
}

_PROFILE_RANK = {
    ComputeProfile.EFFICIENT: 0,
    ComputeProfile.BALANCED: 1,
    ComputeProfile.ULTRA: 2,
}


def recommended_profile(hardware: HardwareInfo) -> ComputeProfile:
    """Return the safest useful profile for the detected environment."""

    if not hardware.cuda_ready:
        return ComputeProfile.EFFICIENT

    total_vram = hardware.total_memory_gib
    if total_vram is None:
        return ComputeProfile.EFFICIENT
    if total_vram >= PROFILE_SPECS[ComputeProfile.ULTRA].minimum_vram_gib:
        return ComputeProfile.ULTRA
    if total_vram >= PROFILE_SPECS[ComputeProfile.BALANCED].minimum_vram_gib:
        return ComputeProfile.BALANCED
    return ComputeProfile.EFFICIENT


def resolve_profile(
    requested: ComputeProfile,
    hardware: HardwareInfo,
) -> ResolvedProfile:
    """Resolve Auto/manual selection to a safe concrete profile.

    Explicit requests above the safe VRAM tier are downgraded instead of
    allowing a predictable out-of-memory failure. The UI keeps the user's
    requested value visible and exposes ``downgraded``/``warning``.
    """

    maximum = recommended_profile(hardware)
    concrete = maximum if requested is ComputeProfile.AUTO else requested

    downgraded = False
    warnings: list[str] = []
    if _PROFILE_RANK[concrete] > _PROFILE_RANK[maximum]:
        concrete = maximum
        downgraded = True
        warnings.append(
            f"{requested.display_name} exceeds the safe tier for this device; "
            f"using {concrete.display_name}."
        )

    if hardware.backend is AcceleratorBackend.CPU or not hardware.cuda_ready:
        device = "cpu"
        precision = InferencePrecision.FP32
        if hardware.warning:
            warnings.append(hardware.warning)
        else:
            warnings.append("CUDA is unavailable; using the CPU-safe profile.")
    else:
        device = hardware.device
        preferred = PROFILE_SPECS[concrete].preferred_precision
        precision = _supported_precision(preferred, hardware)
        if hardware.warning:
            warnings.append(hardware.warning)

    return ResolvedProfile(
        requested=requested,
        effective=concrete,
        spec=PROFILE_SPECS[concrete],
        device=device,
        precision=precision,
        downgraded=downgraded,
        warning=" ".join(warnings) or None,
    )


def available_precisions(hardware: HardwareInfo) -> tuple[InferencePrecision, ...]:
    values = [InferencePrecision.FP32]
    if hardware.cuda_ready and hardware.supports_fp16:
        values.insert(0, InferencePrecision.FP16)
    if hardware.cuda_ready and hardware.supports_bf16:
        values.append(InferencePrecision.BF16)
    return tuple(values)


def _supported_precision(
    requested: InferencePrecision,
    hardware: HardwareInfo,
) -> InferencePrecision:
    if requested is InferencePrecision.FP16 and hardware.supports_fp16:
        return requested
    if requested is InferencePrecision.BF16 and hardware.supports_bf16:
        return requested
    return InferencePrecision.FP32
