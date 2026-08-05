"""Ultralytics SAM 2.1 prompted-segmentation adapter."""
from __future__ import annotations
import hashlib, json
from importlib.metadata import version as package_version
import numpy as np
from core.contracts.common import ModelProvenance
from core.models.metadata import ConfidenceKind
from core.perception.base import BasePerceptionAdapter, PerceptionDependencyError, PerceptionModelLoadError, UnsupportedPerceptionOperation
from core.perception.messages import PromptedSegmentationRequest, SegmentationPromptKind
from core.perception.raw import RawPromptedMask

def _ultralytics_source(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[..., ::-1])

def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)

class UltralyticsSAM2PromptedAdapter(BasePerceptionAdapter):

    def __init__(self):
        self._model = None
        self._config = {}
        self._checkpoint = ''
        self._version = ''

    def load(self, config):
        try:
            from ultralytics import SAM
        except Exception as exc:
            raise PerceptionDependencyError('Ultralytics is not installed. Install requirements/models-ultralytics-perception.txt.') from exc
        self._config = dict(config)
        self._checkpoint = str(self._config.get('checkpoint') or 'sam2.1_t.pt')
        try:
            self._model = SAM(self._checkpoint)
            self._version = package_version('ultralytics')
        except Exception as exc:
            raise PerceptionModelLoadError(f'Unable to load {self._checkpoint}: {exc}') from exc

    def warmup(self):
        if self._model is None:
            raise PerceptionModelLoadError('SAM adapter is not loaded.')

    def unload(self):
        self._model = None

    def _prov(self):
        digest = hashlib.sha256(json.dumps(self._config, sort_keys=True, default=str).encode()).hexdigest()[:16]
        return ModelProvenance('ultralytics-sam2-prompted', 'Ultralytics SAM 2.1 Prompted Segmentation', 'prompted_segmentation', self._version, self._checkpoint, str(self._config.get('device', 'auto')), str(self._config.get('precision', 'fp32')), digest)

    def get_metadata(self):
        return {'loaded': self._model is not None, 'checkpoint': self._checkpoint, 'model_provenance': self._prov()}

    def refine_mask(self, request: PromptedSegmentationRequest) -> RawPromptedMask:
        if self._model is None:
            raise PerceptionModelLoadError('SAM adapter is not loaded.')
        kwargs = {'source': _ultralytics_source(request.frame_request.rgb), 'device': self._config.get('device', 'cpu'), 'verbose': False, 'save': False}
        p = request.prompt
        if p.kind is SegmentationPromptKind.BOX:
            kwargs['bboxes'] = list(p.box_xyxy or ())
        elif p.kind is SegmentationPromptKind.POINTS:
            pts = list(p.positive_points_xy) + list(p.negative_points_xy)
            kwargs['points'] = [list(v) for v in pts]
            kwargs['labels'] = [1] * len(p.positive_points_xy) + [0] * len(p.negative_points_xy)
        else:
            raise UnsupportedPerceptionOperation('Initial-mask prompting is schema-ready but not enabled in the Step 2 Ultralytics adapter.')
        results = self._model.predict(**kwargs)
        if not results or getattr(results[0], 'masks', None) is None:
            raise PerceptionModelLoadError('SAM returned no mask for the prompt.')
        masks = _numpy(results[0].masks.data)
        if masks.ndim != 3 or len(masks) == 0:
            raise PerceptionModelLoadError('SAM returned an invalid mask tensor.')
        return RawPromptedMask(request.frame_request.frame_reference, request.parent_detection_id, request.parent_mask_id, np.asarray(masks[0] > 0.5, dtype=np.bool_), self._prov(), None, ConfidenceKind.NONE)
