"""Hardware-aware MoGe-2 execution profiles.

The resolver is capability driven rather than product-name driven.  An RTX
2060, RTX 5090, or an unknown future NVIDIA GPU is mapped from its CUDA
readiness, compute capability, total VRAM and currently free VRAM.  The same
resolved profile is then passed to the isolated worker, so moving the project
to a different GPU does not require source-code or configuration changes.
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
    """One concrete, worker-ready MoGe-2 tuning tier."""

    profile: ComputeProfile
    model_repo: str
    model_label: str
    resolution_level: int
    num_tokens: int
    point_budget: int
    live_point_budget: int
    warmup_size: int
    warmup_runs: int
    preferred_precision: InferencePrecision
    minimum_total_vram_gib: float
    minimum_free_vram_gib: float
    allow_tf32: bool
    matmul_precision: str
    enable_native_sdpa: bool


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    requested: ComputeProfile
    effective: ComputeProfile
    spec: ProfileSpec
    device: str
    precision: InferencePrecision
    downgraded: bool
    warning: str | None
    memory_pressure: bool = False

    def as_model_config(self) -> dict[str, object]:
        """Return a primitive-only config suitable for multiprocessing IPC."""

        capability = self.spec.profile.value
        return {
            "profile": self.effective.value,
            "device": self.device,
            "precision": self.precision.value,
            "pretrained": self.spec.model_repo,
            "model_label": self.spec.model_label,
            "resolution_level": self.spec.resolution_level,
            "num_tokens": self.spec.num_tokens,
            "point_budget": self.spec.point_budget,
            "live_point_budget": self.spec.live_point_budget,
            "warmup_size": self.spec.warmup_size,
            "warmup_runs": self.spec.warmup_runs,
            "allow_tf32": self.spec.allow_tf32,
            "matmul_precision": self.spec.matmul_precision,
            "enable_native_sdpa": self.spec.enable_native_sdpa,
            "hardware_tier": capability,
        }


PROFILE_SPECS: dict[ComputeProfile, ProfileSpec] = {
    # RTX 2060 / 4–7 GB class.  This is intentionally conservative because the
    # desktop compositor and VisPy also consume part of the six-gigabyte card.
    ComputeProfile.EFFICIENT: ProfileSpec(
        profile=ComputeProfile.EFFICIENT,
        model_repo="Ruicheng/moge-2-vits-normal",
        model_label="MoGe-2 ViT-S Normal",
        resolution_level=5,
        num_tokens=1200,
        point_budget=250_000,
        live_point_budget=100_000,
        warmup_size=256,
        warmup_runs=1,
        preferred_precision=InferencePrecision.FP16,
        minimum_total_vram_gib=4.0,
        minimum_free_vram_gib=3.5,
        allow_tf32=False,
        matmul_precision="high",
        enable_native_sdpa=True,
    ),
    # 8–19 GB class.  It increases model capacity and geometry density while
    # retaining a live-render cap that remains practical for Qt/VisPy.
    ComputeProfile.BALANCED: ProfileSpec(
        profile=ComputeProfile.BALANCED,
        model_repo="Ruicheng/moge-2-vitb-normal",
        model_label="MoGe-2 ViT-B Normal",
        resolution_level=7,
        num_tokens=1800,
        point_budget=500_000,
        live_point_budget=250_000,
        warmup_size=384,
        warmup_runs=1,
        preferred_precision=InferencePrecision.FP16,
        minimum_total_vram_gib=8.0,
        minimum_free_vram_gib=6.0,
        allow_tf32=True,
        matmul_precision="high",
        enable_native_sdpa=True,
    ),
    # RTX 5090 / >=20 GB high-end class.  FP16 remains the automatic default
    # because it is the official MoGe execution path and therefore safer than
    # silently opting into BF16 on an untested machine.  BF16 capability is
    # still detected and exposed for future custom-profile work.
    ComputeProfile.ULTRA: ProfileSpec(
        profile=ComputeProfile.ULTRA,
        model_repo="Ruicheng/moge-2-vitl-normal",
        model_label="MoGe-2 ViT-L Normal",
        resolution_level=9,
        num_tokens=2500,
        point_budget=1_000_000,
        live_point_budget=500_000,
        warmup_size=512,
        warmup_runs=2,
        preferred_precision=InferencePrecision.FP16,
        minimum_total_vram_gib=20.0,
        minimum_free_vram_gib=16.0,
        allow_tf32=True,
        matmul_precision="high",
        enable_native_sdpa=True,
    ),
}

_PROFILE_RANK = {
    ComputeProfile.EFFICIENT: 0,
    ComputeProfile.BALANCED: 1,
    ComputeProfile.ULTRA: 2,
}
_CONCRETE_DESCENDING = (
    ComputeProfile.ULTRA,
    ComputeProfile.BALANCED,
    ComputeProfile.EFFICIENT,
)


def recommended_profile(hardware: HardwareInfo) -> ComputeProfile:
    """Return the highest safe tier for the current total and free VRAM.

    Total VRAM determines the device class.  When a trustworthy startup free
    memory reading is available, it is also used to avoid selecting an Ultra
    checkpoint while another process already occupies most of the GPU.
    """

    if not hardware.cuda_ready:
        return ComputeProfile.EFFICIENT

    total_vram = hardware.total_memory_gib
    if total_vram is None:
        return ComputeProfile.EFFICIENT

    free_vram = hardware.free_memory_gib
    for profile in _CONCRETE_DESCENDING:
        spec = PROFILE_SPECS[profile]
        if total_vram < spec.minimum_total_vram_gib:
            continue
        if free_vram is not None and free_vram < spec.minimum_free_vram_gib:
            continue
        return profile
    return ComputeProfile.EFFICIENT


def resolve_profile(
    requested: ComputeProfile,
    hardware: HardwareInfo,
) -> ResolvedProfile:
    """Resolve Auto/manual selection to a safe concrete profile.

    Manual requests above either the total-VRAM tier or current free-memory
    tier are downgraded rather than allowed to fail predictably.  The requested
    combo-box value remains visible and ``downgraded`` explains the fallback.
    """

    maximum = recommended_profile(hardware)
    concrete = maximum if requested is ComputeProfile.AUTO else requested

    downgraded = False
    memory_pressure = False
    warnings: list[str] = []
    if _PROFILE_RANK[concrete] > _PROFILE_RANK[maximum]:
        requested_spec = PROFILE_SPECS[concrete]
        total = hardware.total_memory_gib
        free = hardware.free_memory_gib
        memory_pressure = bool(
            hardware.cuda_ready
            and total is not None
            and total >= requested_spec.minimum_total_vram_gib
            and free is not None
            and free < requested_spec.minimum_free_vram_gib
        )
        concrete = maximum
        downgraded = True
        if memory_pressure:
            warnings.append(
                f"{requested.display_name} needs about "
                f"{requested_spec.minimum_free_vram_gib:.0f} GB free VRAM; "
                f"using {concrete.display_name} while the GPU is under memory pressure."
            )
        else:
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
        if precision is not preferred:
            warnings.append(
                f"{preferred.display_name} is unavailable; using {precision.display_name}."
            )
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
        memory_pressure=memory_pressure,
    )


def available_precisions(hardware: HardwareInfo) -> tuple[InferencePrecision, ...]:
    values = [InferencePrecision.FP32]
    if hardware.cuda_ready and hardware.supports_fp16:
        values.insert(0, InferencePrecision.FP16)
    if hardware.cuda_ready and hardware.supports_bf16:
        values.append(InferencePrecision.BF16)
    return tuple(values)


def profile_is_safe(profile: ComputeProfile, hardware: HardwareInfo) -> bool:
    """Return whether one concrete profile fits the current hardware snapshot."""

    if profile is ComputeProfile.AUTO:
        return True
    return _PROFILE_RANK[profile] <= _PROFILE_RANK[recommended_profile(hardware)]


def _supported_precision(
    requested: InferencePrecision,
    hardware: HardwareInfo,
) -> InferencePrecision:
    if requested is InferencePrecision.FP16 and hardware.supports_fp16:
        return requested
    if requested is InferencePrecision.BF16 and hardware.supports_bf16:
        return requested
    return InferencePrecision.FP32
