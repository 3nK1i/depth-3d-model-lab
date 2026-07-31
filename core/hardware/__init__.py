"""Hardware discovery and safe inference-profile selection."""

from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo, detect_hardware
from core.hardware.profiles import (
    PROFILE_SPECS,
    ComputeProfile,
    InferencePrecision,
    ProfileSpec,
    ResolvedProfile,
    available_precisions,
    recommended_profile,
    resolve_profile,
)

__all__ = [
    "AcceleratorBackend",
    "HardwareInfo",
    "detect_hardware",
    "PROFILE_SPECS",
    "ComputeProfile",
    "InferencePrecision",
    "ProfileSpec",
    "ResolvedProfile",
    "available_precisions",
    "recommended_profile",
    "resolve_profile",
]