"""Model-agnostic detection/segmentation orchestration."""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Mapping
from core.contracts.validation import ContractValidationError
from core.perception.base import BasePerceptionAdapter, FramePerceptionAdapter, PromptedSegmentationAdapter, UnsupportedPerceptionOperation
from core.perception.messages import FramePerceptionRequest, PromptedSegmentationRequest, SegmentationPrompt, SegmentationPromptKind
from core.perception.metadata import PerceptionDescriptor
from core.perception.normalization import border_contacts, build_box_fallback_mask, normalize_native_mask, normalize_prompted_mask, normalize_raw_detection
from core.perception.result import InstanceMask, MaskSource, ObjectObservation, PerceptionFrameResult, PerceptionResultStatus, PromptedMaskRefinement

@dataclass(frozen=True, slots=True)
class PerceptionPipelinePolicy:
    enable_prompted_refinement: bool = True
    allow_box_fallback: bool = True
    minimum_mask_pixels: int = 1

    def __post_init__(self):
        if self.minimum_mask_pixels < 1:
            raise ValueError('minimum_mask_pixels must be positive.')

class PerceptionPipeline:

    def __init__(self, primary: BasePerceptionAdapter, primary_descriptor: PerceptionDescriptor, *, refiner: BasePerceptionAdapter | None=None, refiner_descriptor: PerceptionDescriptor | None=None, policy: PerceptionPipelinePolicy=PerceptionPipelinePolicy()):
        if not isinstance(primary, FramePerceptionAdapter):
            raise TypeError('primary adapter must implement FramePerceptionAdapter.')
        if refiner is not None and (not isinstance(refiner, PromptedSegmentationAdapter)):
            raise TypeError('refiner must implement PromptedSegmentationAdapter.')
        self.primary = primary
        self.primary_descriptor = primary_descriptor
        self.refiner = refiner
        self.refiner_descriptor = refiner_descriptor
        self.policy = policy

    def process_frame(self, request: FramePerceptionRequest, *, manual_masks: Mapping[str, tuple[InstanceMask, ...]] | None=None) -> PerceptionFrameResult:
        started = time.perf_counter()
        unsupported = tuple((task for task in request.requested_tasks if task not in self.primary_descriptor.capabilities.tasks))
        if unsupported:
            raise UnsupportedPerceptionOperation(f'Primary adapter does not support requested tasks: {[task.value for task in unsupported]}')
        batch = self.primary.infer_frame(request)
        if batch.frame_reference != request.frame_reference:
            raise ContractValidationError('Adapter batch frame does not match the request.')
        if batch.pixel_space_id != request.rgb_space.space_id:
            raise ContractValidationError('Adapter batch pixel space does not match RGB space.')
        warnings = list(batch.warnings)
        observations = []
        provenances = [batch.model_provenance]
        used_refiner = False
        manual_masks = manual_masks or {}
        prefix = f'{request.frame_reference.session_id}:{request.frame_reference.frame_id}'
        for raw in batch.instances:
            detection = normalize_raw_detection(raw, request.frame_reference, request.rgb_space, batch.model_provenance, prefix=prefix)
            masks = list(manual_masks.get(raw.instance_id, ()))
            refinements = []
            try:
                native = normalize_native_mask(raw, request.frame_reference, request.rgb_space, batch.model_provenance, prefix=prefix, minimum_pixels=self.policy.minimum_mask_pixels)
                if native is not None:
                    masks.append(native)
            except ContractValidationError as exc:
                warnings.append(f'{raw.instance_id}: native mask rejected: {exc}')
            has_approved_manual = any((mask.is_approved_manual for mask in masks))
            valid_auto = any((mask.foreground_valid and mask.source is not MaskSource.MANUAL_CORRECTION for mask in masks))
            if not has_approved_manual and (not valid_auto) and self.policy.enable_prompted_refinement and (self.refiner is not None):
                prompt = SegmentationPrompt(f'{prefix}:prompt:{raw.instance_id}', SegmentationPromptKind.BOX, request.rgb_space, box_xyxy=detection.box.xyxy)
                try:
                    prompted = self.refiner.refine_mask(PromptedSegmentationRequest(request, prompt, parent_detection_id=detection.detection_id))
                    mask = normalize_prompted_mask(prompted, request.rgb_space, mask_id=f'{prefix}:prompted:{raw.instance_id}', minimum_pixels=self.policy.minimum_mask_pixels)
                    masks.append(mask)
                    refinements.append(PromptedMaskRefinement(f'{prefix}:refinement:{raw.instance_id}', prompt, mask, input_detection_id=detection.detection_id, warnings=prompted.warnings))
                    used_refiner = True
                    warnings.extend(prompted.warnings)
                except Exception as exc:
                    warnings.append(f'{raw.instance_id}: prompted refinement failed: {exc}')
            has_approved_manual = any((mask.is_approved_manual for mask in masks))
            valid_auto = any((mask.foreground_valid and mask.source is not MaskSource.MANUAL_CORRECTION for mask in masks))
            if not has_approved_manual and (not valid_auto) and self.policy.allow_box_fallback:
                masks.append(build_box_fallback_mask(detection, request.rgb_space, batch.model_provenance, mask_id=f'{prefix}:box:{raw.instance_id}'))
            observations.append(ObjectObservation(f'{prefix}:obj:{raw.instance_id}', request.frame_reference, raw.class_id, raw.class_name, detection, tuple(masks), refinements=tuple(refinements), border_contacts=border_contacts(detection, request.rgb_space), warnings=raw.warnings))
        if used_refiner and self.refiner is not None:
            meta = self.refiner.get_metadata()
            prov = meta.get('model_provenance')
            if prov is not None:
                from core.contracts.common import ModelProvenance
                if not isinstance(prov, ModelProvenance):
                    raise ContractValidationError('Refiner metadata contains invalid model provenance.')
                if prov not in provenances:
                    provenances.append(prov)
        total_ms = (time.perf_counter() - started) * 1000.0
        return PerceptionFrameResult(request.frame_reference, tuple(observations), tuple(provenances), PerceptionResultStatus.PARTIAL if warnings else PerceptionResultStatus.SUCCESS, total_ms, tuple(warnings))
