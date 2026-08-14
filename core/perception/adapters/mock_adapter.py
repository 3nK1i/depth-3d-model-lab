"""Deterministic dependency-free adapter for process and pipeline validation."""
from __future__ import annotations
import time
import numpy as np
from core.contracts.common import ModelProvenance
from core.models.metadata import ConfidenceKind
from core.perception.base import BasePerceptionAdapter
from core.perception.messages import FramePerceptionRequest, PromptedSegmentationRequest, SequenceResetRequest
from core.perception.raw import RawAutomaticMaskBatch, RawAutomaticMaskProposal, RawPerceptionBatch, RawPerceptionInstance, RawPromptedMask

class DevelopmentPerceptionMockAdapter(BasePerceptionAdapter):

    def __init__(self):
        self._loaded = False
        self._config = {}
        self._resets = 0

    def load(self, config):
        self._config = dict(config)
        self._loaded = True

    def warmup(self):
        if not self._loaded:
            raise RuntimeError('Mock adapter is not loaded.')

    def unload(self):
        self._loaded = False

    def reset_sequence(self, request: SequenceResetRequest):
        self._resets += 1

    def _prov(self, task):
        return ModelProvenance('development-perception-mock', 'Development Perception Mock', task, device='cpu', precision='fp32', metadata={'sequence_resets': self._resets})

    def get_metadata(self):
        return {'loaded': self._loaded, 'sequence_resets': self._resets, 'model_provenance': self._prov('prompted_segmentation')}

    def infer_frame(self, request: FramePerceptionRequest) -> RawPerceptionBatch:
        if not self._loaded:
            raise RuntimeError('Mock adapter is not loaded.')
        delay_ms = max(0.0, float(self._config.get('delay_ms', 0.0)))
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
        h, w = request.rgb.shape[:2]
        t = time.perf_counter()
        instances = []
        boxes = ((0.1 * w, 0.15 * h, 0.6 * w, 0.75 * h), (0.4 * w, 0.3 * h, 0.9 * w, 0.9 * h))
        emit = bool(self._config.get('emit_native_masks', True))
        for i, box in enumerate(boxes):
            mask = None
            if emit:
                mask = np.zeros((h, w), dtype=np.bool_)
                x0, y0, x1, y1 = map(int, box)
                mask[y0:y1, x0:x1] = True
            instances.append(RawPerceptionInstance(f'mock-{i}', i, f'mock-object-{i}', 0.9 - i * 0.1, box, mask, 0.85 if mask is not None else None, ConfidenceKind.CONFIDENCE if mask is not None else ConfidenceKind.NONE))
        return RawPerceptionBatch(request.frame_reference, request.rgb_space.space_id, tuple(instances), self._prov('instance_segmentation'), (time.perf_counter() - t) * 1000)


    def propose_masks(self, request: FramePerceptionRequest) -> RawAutomaticMaskBatch:
        if not self._loaded:
            raise RuntimeError('Mock adapter is not loaded.')
        h, w = request.rgb.shape[:2]
        proposals = []
        regions = (
            (int(0.08 * w), int(0.10 * h), int(0.48 * w), int(0.72 * h)),
            (int(0.52 * w), int(0.22 * h), int(0.92 * w), int(0.88 * h)),
        )
        for index, (x0, y0, x1, y1) in enumerate(regions):
            mask = np.zeros((h, w), dtype=np.bool_)
            mask[y0:y1, x0:x1] = True
            proposals.append(RawAutomaticMaskProposal(f'mock-auto-{index}', mask))
        return RawAutomaticMaskBatch(
            request.frame_reference,
            request.rgb_space.space_id,
            tuple(proposals),
            self._prov('automatic_segmentation'),
        )

    def refine_mask(self, request: PromptedSegmentationRequest) -> RawPromptedMask:
        h, w = request.frame_request.rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.bool_)
        if request.prompt.box_xyxy is not None:
            x0, y0, x1, y1 = map(int, request.prompt.box_xyxy)
            inset = max(1, min(x1 - x0, y1 - y0) // 10)
            mask[max(0, y0 + inset):min(h, y1 - inset), max(0, x0 + inset):min(w, x1 - inset)] = True
        return RawPromptedMask(request.frame_request.frame_reference, request.parent_detection_id, request.parent_mask_id, mask, self._prov('prompted_segmentation'), 0.8, ConfidenceKind.CONFIDENCE)
