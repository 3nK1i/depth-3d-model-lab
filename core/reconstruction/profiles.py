"""Reconstruction quality profiles and bounded CPU/GPU memory policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from core.contracts.validation import require_nonempty
from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.reconstruction.errors import ReconstructionValidationError

_GIB = 1024**3


class ReconstructionQualityProfile(str, Enum):
    EFFICIENT = "efficient"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"

    @property
    def display_name(self) -> str:
        return {
            ReconstructionQualityProfile.EFFICIENT: "Efficient",
            ReconstructionQualityProfile.BALANCED: "Balanced",
            ReconstructionQualityProfile.HIGH_QUALITY: "High Quality",
        }[self]


@dataclass(frozen=True, slots=True)
class ReconstructionProfileSpec:
    profile: ReconstructionQualityProfile
    minimum_total_vram_gib: float
    minimum_free_vram_gib: float
    target_gpu_budget_gib: float
    cpu_budget_gib: float
    frame_cache_fraction: float
    keyframe_cache_fraction: float
    working_memory_fraction: float
    max_active_keyframes: int

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ReconstructionQualityProfile):
            raise TypeError("profile must be a ReconstructionQualityProfile.")
        numeric_fields = (
            "minimum_total_vram_gib",
            "minimum_free_vram_gib",
            "target_gpu_budget_gib",
            "cpu_budget_gib",
        )
        for field_name in numeric_fields:
            value = float(getattr(self, field_name))
            if not isfinite(value) or value < 0.0:
                raise ReconstructionValidationError(
                    f"{field_name} must be finite and non-negative."
                )
            object.__setattr__(self, field_name, value)
        if self.target_gpu_budget_gib <= 0.0 or self.cpu_budget_gib <= 0.0:
            raise ReconstructionValidationError(
                "Target GPU and CPU budgets must be positive."
            )
        fractions = tuple(
            float(value)
            for value in (
                self.frame_cache_fraction,
                self.keyframe_cache_fraction,
                self.working_memory_fraction,
            )
        )
        if (
            any(not isfinite(value) or value <= 0.0 for value in fractions)
            or abs(sum(fractions) - 1.0) > 1e-9
        ):
            raise ReconstructionValidationError(
                "Reconstruction profile memory fractions must be positive and sum to 1."
            )
        object.__setattr__(self, "frame_cache_fraction", fractions[0])
        object.__setattr__(self, "keyframe_cache_fraction", fractions[1])
        object.__setattr__(self, "working_memory_fraction", fractions[2])
        if (
            isinstance(self.max_active_keyframes, bool)
            or int(self.max_active_keyframes) != self.max_active_keyframes
            or int(self.max_active_keyframes) <= 0
        ):
            raise ReconstructionValidationError(
                "max_active_keyframes must be a positive integer."
            )
        object.__setattr__(self, "max_active_keyframes", int(self.max_active_keyframes))


@dataclass(frozen=True, slots=True)
class ReconstructionMemoryPolicy:
    profile: ReconstructionQualityProfile
    device: str
    cpu_budget_bytes: int
    gpu_budget_bytes: int
    gpu_reserve_bytes: int
    frame_cache_bytes: int
    keyframe_cache_bytes: int
    working_memory_bytes: int
    max_active_keyframes: int
    co_residency_reserved: bool

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ReconstructionQualityProfile):
            raise TypeError("profile must be a ReconstructionQualityProfile.")
        object.__setattr__(
            self, "device", require_nonempty(self.device, field_name="device")
        )
        if not isinstance(self.co_residency_reserved, bool):
            raise TypeError("co_residency_reserved must be a bool.")
        integer_field_names = (
            "cpu_budget_bytes",
            "gpu_budget_bytes",
            "gpu_reserve_bytes",
            "frame_cache_bytes",
            "keyframe_cache_bytes",
            "working_memory_bytes",
            "max_active_keyframes",
        )
        for field_name in integer_field_names:
            value = getattr(self, field_name)
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ReconstructionValidationError(
                    f"{field_name} must be a non-negative integer."
                )
            object.__setattr__(self, field_name, int(value))
        if self.cpu_budget_bytes <= 0 or self.max_active_keyframes <= 0:
            raise ReconstructionValidationError(
                "CPU budget and max_active_keyframes must be positive."
            )
        allocated = self.frame_cache_bytes + self.keyframe_cache_bytes + self.working_memory_bytes
        if self.gpu_budget_bytes == 0:
            if allocated != 0:
                raise ReconstructionValidationError(
                    "CPU policies cannot declare GPU cache allocations."
                )
        elif allocated != self.gpu_budget_bytes:
            raise ReconstructionValidationError(
                "GPU cache allocations must exactly match gpu_budget_bytes."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile.value,
            "device": self.device,
            "cpu_budget_bytes": self.cpu_budget_bytes,
            "gpu_budget_bytes": self.gpu_budget_bytes,
            "gpu_reserve_bytes": self.gpu_reserve_bytes,
            "frame_cache_bytes": self.frame_cache_bytes,
            "keyframe_cache_bytes": self.keyframe_cache_bytes,
            "working_memory_bytes": self.working_memory_bytes,
            "max_active_keyframes": self.max_active_keyframes,
            "co_residency_reserved": self.co_residency_reserved,
        }


@dataclass(frozen=True, slots=True)
class ResolvedReconstructionProfile:
    requested: ReconstructionQualityProfile | None
    effective: ReconstructionQualityProfile
    memory_policy: ReconstructionMemoryPolicy
    downgraded: bool
    memory_pressure: bool
    warning: str | None


PROFILE_SPECS: dict[ReconstructionQualityProfile, ReconstructionProfileSpec] = {
    ReconstructionQualityProfile.EFFICIENT: ReconstructionProfileSpec(
        ReconstructionQualityProfile.EFFICIENT,
        minimum_total_vram_gib=0.0,
        minimum_free_vram_gib=0.0,
        target_gpu_budget_gib=3.5,
        cpu_budget_gib=6.0,
        frame_cache_fraction=0.15,
        keyframe_cache_fraction=0.20,
        working_memory_fraction=0.65,
        max_active_keyframes=80,
    ),
    ReconstructionQualityProfile.BALANCED: ReconstructionProfileSpec(
        ReconstructionQualityProfile.BALANCED,
        minimum_total_vram_gib=10.0,
        minimum_free_vram_gib=8.0,
        target_gpu_budget_gib=9.0,
        cpu_budget_gib=12.0,
        frame_cache_fraction=0.18,
        keyframe_cache_fraction=0.27,
        working_memory_fraction=0.55,
        max_active_keyframes=250,
    ),
    ReconstructionQualityProfile.HIGH_QUALITY: ReconstructionProfileSpec(
        ReconstructionQualityProfile.HIGH_QUALITY,
        minimum_total_vram_gib=24.0,
        minimum_free_vram_gib=20.0,
        target_gpu_budget_gib=22.0,
        cpu_budget_gib=24.0,
        frame_cache_fraction=0.20,
        keyframe_cache_fraction=0.30,
        working_memory_fraction=0.50,
        max_active_keyframes=800,
    ),
}

_PROFILE_RANK = {
    ReconstructionQualityProfile.EFFICIENT: 0,
    ReconstructionQualityProfile.BALANCED: 1,
    ReconstructionQualityProfile.HIGH_QUALITY: 2,
}
_PROFILE_DESCENDING = (
    ReconstructionQualityProfile.HIGH_QUALITY,
    ReconstructionQualityProfile.BALANCED,
    ReconstructionQualityProfile.EFFICIENT,
)


def recommended_reconstruction_profile(hardware: HardwareInfo) -> ReconstructionQualityProfile:
    if not hardware.cuda_ready:
        return ReconstructionQualityProfile.EFFICIENT
    total = hardware.total_memory_gib
    free = hardware.free_memory_gib
    if total is None:
        return ReconstructionQualityProfile.EFFICIENT
    for profile in _PROFILE_DESCENDING:
        spec = PROFILE_SPECS[profile]
        if total < spec.minimum_total_vram_gib:
            continue
        if free is not None and free < spec.minimum_free_vram_gib:
            continue
        return profile
    return ReconstructionQualityProfile.EFFICIENT


def resolve_reconstruction_profile(
    hardware: HardwareInfo,
    requested: ReconstructionQualityProfile | None = None,
    *,
    reserve_for_co_residency: bool = True,
) -> ResolvedReconstructionProfile:
    """Resolve an automatic or requested reconstruction profile safely."""

    if not isinstance(hardware, HardwareInfo):
        raise TypeError("hardware must be a HardwareInfo.")
    if requested is not None and not isinstance(requested, ReconstructionQualityProfile):
        raise TypeError("requested must be a ReconstructionQualityProfile or None.")
    if not isinstance(reserve_for_co_residency, bool):
        raise TypeError("reserve_for_co_residency must be a bool.")

    maximum = recommended_reconstruction_profile(hardware)
    effective = maximum if requested is None else requested
    downgraded = False
    memory_pressure = False
    warnings: list[str] = []

    if _PROFILE_RANK[effective] > _PROFILE_RANK[maximum]:
        requested_spec = PROFILE_SPECS[effective]
        total = hardware.total_memory_gib
        free = hardware.free_memory_gib
        memory_pressure = bool(
            hardware.cuda_ready
            and total is not None
            and total >= requested_spec.minimum_total_vram_gib
            and free is not None
            and free < requested_spec.minimum_free_vram_gib
        )
        effective = maximum
        downgraded = True
        if memory_pressure:
            warnings.append(
                f"{requested.display_name} requires about "
                f"{requested_spec.minimum_free_vram_gib:.0f} GB free VRAM; "
                f"using {effective.display_name} under current memory pressure."
            )
        else:
            warnings.append(
                f"{requested.display_name} exceeds the safe reconstruction tier; "
                f"using {effective.display_name}."
            )

    spec = PROFILE_SPECS[effective]
    cpu_budget = int(spec.cpu_budget_gib * _GIB)
    if hardware.backend is AcceleratorBackend.CPU or not hardware.cuda_ready:
        if hardware.warning:
            warnings.append(hardware.warning)
        else:
            warnings.append("CUDA is unavailable; using an Efficient CPU reconstruction policy.")
        effective = ReconstructionQualityProfile.EFFICIENT
        spec = PROFILE_SPECS[effective]
        policy = ReconstructionMemoryPolicy(
            profile=effective,
            device="cpu",
            cpu_budget_bytes=int(spec.cpu_budget_gib * _GIB),
            gpu_budget_bytes=0,
            gpu_reserve_bytes=0,
            frame_cache_bytes=0,
            keyframe_cache_bytes=0,
            working_memory_bytes=0,
            max_active_keyframes=spec.max_active_keyframes,
            co_residency_reserved=reserve_for_co_residency,
        )
        return ResolvedReconstructionProfile(
            requested=requested,
            effective=effective,
            memory_policy=policy,
            downgraded=downgraded or requested not in {None, effective},
            memory_pressure=memory_pressure,
            warning=" ".join(warnings) or None,
        )

    free_bytes = (
        hardware.free_memory_bytes
        if hardware.free_memory_bytes is not None
        else hardware.total_memory_bytes or 0
    )
    base_reserve_gib = 1.5 if effective is ReconstructionQualityProfile.EFFICIENT else 2.0
    if effective is ReconstructionQualityProfile.HIGH_QUALITY:
        base_reserve_gib = 4.0
    if reserve_for_co_residency:
        base_reserve_gib += 1.0
    reserve_bytes = min(free_bytes, int(base_reserve_gib * _GIB))
    available_bytes = max(0, free_bytes - reserve_bytes)
    target_bytes = int(spec.target_gpu_budget_gib * _GIB)
    gpu_budget = min(target_bytes, available_bytes)
    if gpu_budget <= 0:
        raise ReconstructionValidationError(
            "No safe GPU memory remains after the reconstruction reserve."
        )

    frame_cache = int(gpu_budget * spec.frame_cache_fraction)
    keyframe_cache = int(gpu_budget * spec.keyframe_cache_fraction)
    working_memory = gpu_budget - frame_cache - keyframe_cache
    policy = ReconstructionMemoryPolicy(
        profile=effective,
        device=hardware.device,
        cpu_budget_bytes=cpu_budget,
        gpu_budget_bytes=gpu_budget,
        gpu_reserve_bytes=reserve_bytes,
        frame_cache_bytes=frame_cache,
        keyframe_cache_bytes=keyframe_cache,
        working_memory_bytes=working_memory,
        max_active_keyframes=spec.max_active_keyframes,
        co_residency_reserved=reserve_for_co_residency,
    )
    return ResolvedReconstructionProfile(
        requested=requested,
        effective=effective,
        memory_policy=policy,
        downgraded=downgraded,
        memory_pressure=memory_pressure,
        warning=" ".join(warnings) or None,
    )
