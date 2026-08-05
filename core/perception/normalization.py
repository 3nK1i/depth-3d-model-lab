"""Authoritative normalization from adapter-native arrays to public contracts."""
from __future__ import annotations
import numpy as np
from core.contracts.common import ModelProvenance
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind
from core.perception.raw import RawPerceptionInstance, RawPromptedMask
from core.perception.result import ApprovalStatus, DenseMaskPayload, Detection2D, InstanceMask, MaskSource, normalize_detection_box

def normalize_raw_detection(raw: RawPerceptionInstance, frame_reference, pixel_space, provenance: ModelProvenance, *, prefix: str) -> Detection2D:
    return Detection2D(f'{prefix}:det:{raw.instance_id}', frame_reference, raw.class_id, raw.class_name, raw.confidence, normalize_detection_box(raw.box_xyxy, pixel_space), provenance)

def normalize_native_mask(raw: RawPerceptionInstance, frame_reference, pixel_space, provenance: ModelProvenance, *, prefix: str, minimum_pixels: int=1) -> InstanceMask | None:
    if raw.mask is None:
        return None
    if raw.mask.shape != (pixel_space.size.height, pixel_space.size.width):
        raise ContractValidationError('Native mask shape does not match the declared RGB pixel space.')
    dense = np.asarray(raw.mask, dtype=np.bool_)
    valid = int(np.count_nonzero(dense)) >= minimum_pixels
    return InstanceMask(f'{prefix}:native:{raw.instance_id}', frame_reference, pixel_space, DenseMaskPayload(dense), MaskSource.NATIVE_INSTANCE, provenance, confidence=raw.mask_confidence, confidence_kind=raw.mask_confidence_kind, foreground_valid=valid)

def normalize_prompted_mask(raw: RawPromptedMask, pixel_space, *, mask_id: str, minimum_pixels: int=1) -> InstanceMask:
    if raw.mask.shape != (pixel_space.size.height, pixel_space.size.width):
        raise ContractValidationError('Prompted mask shape does not match the declared RGB pixel space.')
    dense = np.asarray(raw.mask, dtype=np.bool_)
    valid = int(np.count_nonzero(dense)) >= minimum_pixels
    return InstanceMask(mask_id, raw.frame_reference, pixel_space, DenseMaskPayload(dense), MaskSource.PROMPTED_REFINEMENT, raw.model_provenance, parent_mask_id=raw.parent_mask_id, confidence=raw.confidence, confidence_kind=raw.confidence_kind, foreground_valid=valid)

def build_box_fallback_mask(detection: Detection2D, pixel_space, provenance: ModelProvenance, *, mask_id: str) -> InstanceMask:
    x0, y0, x1, y1 = detection.box.xyxy
    ix0 = max(0, int(np.floor(x0)))
    iy0 = max(0, int(np.floor(y0)))
    ix1 = min(pixel_space.size.width, int(np.ceil(x1)))
    iy1 = min(pixel_space.size.height, int(np.ceil(y1)))
    dense = np.zeros((pixel_space.size.height, pixel_space.size.width), dtype=np.bool_)
    dense[iy0:iy1, ix0:ix1] = True
    return InstanceMask(mask_id, detection.frame_reference, pixel_space, DenseMaskPayload(dense), MaskSource.DETECTION_BOX_FALLBACK, provenance, foreground_valid=bool(np.any(dense)))

def border_contacts(detection: Detection2D, pixel_space) -> tuple[str, ...]:
    b = detection.box
    out = []
    if b.x_min <= 0:
        out.append('left')
    if b.y_min <= 0:
        out.append('top')
    if b.x_max >= pixel_space.size.width:
        out.append('right')
    if b.y_max >= pixel_space.size.height:
        out.append('bottom')
    return tuple(out)
