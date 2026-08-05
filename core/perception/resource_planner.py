"""Advisory, allocation-free perception resource planning."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from core.hardware.gpu_info import HardwareInfo
from core.perception.metadata import CoResidencyPolicy, PerceptionDescriptor

class PerceptionExecutionMode(str, Enum):
    PRIMARY_ONLY = 'primary_only'
    CO_RESIDENT = 'co_resident'
    SEQUENTIAL_OFFLINE = 'sequential_offline'
    REJECTED = 'rejected'

@dataclass(frozen=True, slots=True)
class PerceptionResourcePlan:
    mode: PerceptionExecutionMode
    reason: str
    estimated_total_vram_mb: int | None = None
    safety_margin_mb: int | None = None

def plan_perception_resources(primary: PerceptionDescriptor, refiner: PerceptionDescriptor | None, hardware: HardwareInfo, *, live: bool, measured_primary_vram_mb: int | None=None, measured_refiner_vram_mb: int | None=None, reserved_vram_mb: int=1024) -> PerceptionResourcePlan:
    if refiner is None:
        return PerceptionResourcePlan(PerceptionExecutionMode.PRIMARY_ONLY, 'No prompted refiner is configured.', measured_primary_vram_mb)
    if not hardware.cuda_ready:
        if primary.resources.cpu_fallback_supported and refiner.resources.cpu_fallback_supported and (not live):
            return PerceptionResourcePlan(PerceptionExecutionMode.SEQUENTIAL_OFFLINE, 'CUDA unavailable; offline CPU fallback only.')
        return PerceptionResourcePlan(PerceptionExecutionMode.PRIMARY_ONLY, 'CUDA unavailable; prompted refinement is disabled.')
    if primary.resources.co_residency_policy is CoResidencyPolicy.EXCLUSIVE or refiner.resources.co_residency_policy is CoResidencyPolicy.EXCLUSIVE:
        return PerceptionResourcePlan(PerceptionExecutionMode.PRIMARY_ONLY if live else PerceptionExecutionMode.SEQUENTIAL_OFFLINE, 'An adapter requires isolated GPU use.')
    p = measured_primary_vram_mb or primary.resources.estimated_vram_mb
    r = measured_refiner_vram_mb or refiner.resources.estimated_vram_mb
    if p is None or r is None:
        return PerceptionResourcePlan(PerceptionExecutionMode.PRIMARY_ONLY if live else PerceptionExecutionMode.SEQUENTIAL_OFFLINE, 'Co-residency requires measured VRAM for both adapters.')
    total = p + r
    free = int((hardware.free_memory_bytes or hardware.total_memory_bytes or 0) / 1024 ** 2)
    margin = free - total - reserved_vram_mb
    if margin >= 0:
        return PerceptionResourcePlan(PerceptionExecutionMode.CO_RESIDENT, 'Measured models fit with the configured safety reserve.', total, margin)
    return PerceptionResourcePlan(PerceptionExecutionMode.PRIMARY_ONLY if live else PerceptionExecutionMode.SEQUENTIAL_OFFLINE, 'Insufficient measured free VRAM for safe co-residency.', total, margin)
