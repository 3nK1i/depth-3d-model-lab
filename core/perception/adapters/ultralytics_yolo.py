"""Ultralytics YOLO detection and instance-segmentation adapters."""
from __future__ import annotations
import hashlib, json, time
from importlib.metadata import version as package_version
from typing import Any
import numpy as np
from core.contracts.common import ModelProvenance
from core.models.metadata import ConfidenceKind
from core.perception.base import BasePerceptionAdapter, PerceptionDependencyError, PerceptionModelLoadError
from core.perception.messages import FramePerceptionRequest
from core.perception.raw import RawPerceptionBatch, RawPerceptionInstance

def _ultralytics_source(rgb: np.ndarray) -> np.ndarray:
    """Convert the public RGB contract to Ultralytics/OpenCV BGR ndarray input."""
    return np.ascontiguousarray(rgb[..., ::-1])

def _numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)

class _UltralyticsYOLOBase(BasePerceptionAdapter):
    segmentation = False

    def __init__(self):
        self._model = None
        self._config = {}
        self._checkpoint = ''
        self._version = ''

    def load(self, config):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise PerceptionDependencyError('Ultralytics is not installed. Install requirements/models-ultralytics-perception.txt.') from exc
        self._config = dict(config)
        self._checkpoint = str(self._config.get('checkpoint') or ('yolo11n-seg.pt' if self.segmentation else 'yolo11n.pt'))
        try:
            self._model = YOLO(self._checkpoint)
            self._version = package_version('ultralytics')
        except Exception as exc:
            raise PerceptionModelLoadError(f'Unable to load {self._checkpoint}: {exc}') from exc

    def warmup(self):
        if self._model is None:
            raise PerceptionModelLoadError('YOLO adapter is not loaded.')

    def unload(self):
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _prov(self):
        digest = hashlib.sha256(json.dumps(self._config, sort_keys=True, default=str).encode()).hexdigest()[:16]
        return ModelProvenance('ultralytics-yolo-seg' if self.segmentation else 'ultralytics-yolo-detect', 'Ultralytics YOLO Instance Segmentation' if self.segmentation else 'Ultralytics YOLO Detection', 'instance_segmentation' if self.segmentation else 'detection', self._version, self._checkpoint, str(self._config.get('device', 'auto')), str(self._config.get('precision', 'fp32')), digest, {'retina_masks': bool(self._config.get('retina_masks', self.segmentation))})

    def get_metadata(self):
        return {'loaded': self._model is not None, 'checkpoint': self._checkpoint, 'model_provenance': self._prov()}

    def infer_frame(self, request: FramePerceptionRequest) -> RawPerceptionBatch:
        if self._model is None:
            raise PerceptionModelLoadError('YOLO adapter is not loaded.')
        kwargs = {'source': _ultralytics_source(request.rgb), 'device': self._config.get('device', 'cpu'), 'imgsz': int(self._config.get('imgsz', 640)), 'conf': float(self._config.get('confidence', 0.25)), 'iou': float(self._config.get('iou', 0.7)), 'max_det': int(self._config.get('max_detections', 100)), 'half': str(self._config.get('precision', 'fp32')).lower() == 'fp16' and str(self._config.get('device', 'cpu')) != 'cpu', 'retina_masks': bool(self._config.get('retina_masks', self.segmentation)), 'verbose': False, 'save': False}
        t = time.perf_counter()
        results = self._model.predict(**kwargs)
        elapsed = (time.perf_counter() - t) * 1000
        if not results:
            return RawPerceptionBatch(request.frame_reference, request.rgb_space.space_id, (), self._prov(), elapsed)
        result = results[0]
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return RawPerceptionBatch(request.frame_reference, request.rgb_space.space_id, (), self._prov(), elapsed)
        xyxy = _numpy(boxes.xyxy)
        confs = _numpy(boxes.conf).reshape(-1)
        classes = _numpy(boxes.cls).reshape(-1).astype(int)
        names = getattr(result, 'names', {})
        masks_obj = getattr(result, 'masks', None)
        masks = _numpy(getattr(masks_obj, 'data', None)) if masks_obj is not None else np.asarray([])
        items = []
        warnings = []
        for i, box in enumerate(xyxy):
            mask = None
            if self.segmentation and masks.ndim == 3 and (i < len(masks)):
                mask = np.asarray(masks[i] > 0.5, dtype=np.bool_)
            elif self.segmentation:
                warnings.append(f'instance {i}: segmentation mask missing')
            cid = int(classes[i])
            cname = str(names.get(cid, cid)) if isinstance(names, dict) else str(cid)
            items.append(RawPerceptionInstance(str(i), cid, cname, float(confs[i]), tuple((float(v) for v in box)), mask, None, ConfidenceKind.NONE))
        return RawPerceptionBatch(request.frame_reference, request.rgb_space.space_id, tuple(items), self._prov(), elapsed, tuple(warnings))

class UltralyticsYOLOSegmentationAdapter(_UltralyticsYOLOBase):
    segmentation = True

class UltralyticsYOLODetectionAdapter(_UltralyticsYOLOBase):
    segmentation = False
