"""Hardware-adaptive primitive runtime configuration for perception adapters."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any
from core.hardware.gpu_info import HardwareInfo
from core.perception.metadata import PerceptionDescriptor, PerceptionTask

class PerceptionComputeProfile(str, Enum):
    AUTO = 'auto'
    EFFICIENT = 'efficient'
    BALANCED = 'balanced'
    ULTRA = 'ultra'

@dataclass(frozen=True, slots=True)
class ResolvedPerceptionRuntime:
    requested_profile: PerceptionComputeProfile
    effective_profile: PerceptionComputeProfile
    model_id: str
    checkpoint: str
    device: str
    precision: str
    imgsz: int
    confidence: float
    iou: float
    max_detections: int
    retina_masks: bool
    warning: str = ''

    def as_config(self) -> dict[str, object]:
        return {'profile': self.effective_profile.value, 'checkpoint': self.checkpoint, 'device': self.device, 'precision': self.precision, 'imgsz': self.imgsz, 'confidence': self.confidence, 'iou': self.iou, 'max_detections': self.max_detections, 'retina_masks': self.retina_masks}

def recommended_perception_profile(hardware: HardwareInfo) -> PerceptionComputeProfile:
    if not hardware.cuda_ready or hardware.total_memory_gib is None:
        return PerceptionComputeProfile.EFFICIENT
    if hardware.total_memory_gib >= 20 and (hardware.free_memory_gib is None or hardware.free_memory_gib >= 16):
        return PerceptionComputeProfile.ULTRA
    if hardware.total_memory_gib >= 8 and (hardware.free_memory_gib is None or hardware.free_memory_gib >= 6):
        return PerceptionComputeProfile.BALANCED
    return PerceptionComputeProfile.EFFICIENT

def resolve_perception_runtime(descriptor: PerceptionDescriptor, hardware: HardwareInfo, requested: PerceptionComputeProfile=PerceptionComputeProfile.AUTO) -> ResolvedPerceptionRuntime:
    maximum = recommended_perception_profile(hardware)
    rank = {PerceptionComputeProfile.EFFICIENT: 0, PerceptionComputeProfile.BALANCED: 1, PerceptionComputeProfile.ULTRA: 2}
    effective = maximum if requested is PerceptionComputeProfile.AUTO else requested
    warning = ''
    if rank.get(effective, 0) > rank[maximum]:
        warning = f'{requested.value} exceeds the current safe hardware tier; using {maximum.value}.'
        effective = maximum
    is_prompt = PerceptionTask.PROMPTED_SEGMENTATION in descriptor.capabilities.tasks
    is_seg = PerceptionTask.INSTANCE_SEGMENTATION in descriptor.capabilities.tasks
    suffix = '-seg.pt' if is_seg else '.pt'
    if is_prompt:
        checkpoints = {PerceptionComputeProfile.EFFICIENT: 'sam2.1_t.pt', PerceptionComputeProfile.BALANCED: 'sam2.1_s.pt', PerceptionComputeProfile.ULTRA: 'sam2.1_b.pt'}
        imgsz = 1024
    else:
        variants = {PerceptionComputeProfile.EFFICIENT: 'yolo11n', PerceptionComputeProfile.BALANCED: 'yolo11s', PerceptionComputeProfile.ULTRA: 'yolo11s'}
        checkpoints = {p: v + suffix for p, v in variants.items()}
        imgsz = 960 if effective is PerceptionComputeProfile.ULTRA else 640
    device = hardware.device if hardware.cuda_ready else 'cpu'
    precision = 'fp16' if hardware.cuda_ready and hardware.supports_fp16 else 'fp32'
    return ResolvedPerceptionRuntime(requested, effective, descriptor.model_id, checkpoints[effective], device, precision, imgsz, 0.25, 0.7, 100, True if is_seg else False, warning)
