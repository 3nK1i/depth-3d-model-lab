#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import cv2
from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.hardware.gpu_info import detect_hardware
from core.perception.messages import FramePerceptionRequest
from core.perception.metadata import PerceptionTask
from core.perception.pipeline import PerceptionPipeline
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.perception.runtime_config import PerceptionComputeProfile, resolve_perception_runtime

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--model', default='ultralytics-yolo-seg')
    p.add_argument('--refiner', default='')
    p.add_argument('--profile', choices=[v.value for v in PerceptionComputeProfile], default='auto')
    p.add_argument('--overlay', '--output', dest='overlay', default='')
    p.add_argument('--allow-unprofiled-co-residency', action='store_true')
    a = p.parse_args()
    bgr = cv2.imread(a.image)
    if bgr is None:
        raise SystemExit(f'Unable to read {a.image}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    frame = FrameReference(1, 1, 0.0, ImageSize(w, h))
    space = PixelCoordinateSpace('rgb', frame, frame.source_size, PixelSpaceRole.RGB)
    hardware = detect_hardware()
    desc = PERCEPTION_MODEL_REGISTRY.get(a.model)
    adapter = PERCEPTION_MODEL_REGISTRY.build(a.model)
    runtime = resolve_perception_runtime(desc, hardware, PerceptionComputeProfile(a.profile))
    adapter.load(runtime.as_config())
    adapter.warmup()
    refiner = None
    rdesc = None
    if a.refiner:
        if not a.allow_unprofiled_co_residency:
            raise SystemExit('Refiner loading requires measured VRAM or the explicit --allow-unprofiled-co-residency validation override.')
        rdesc = PERCEPTION_MODEL_REGISTRY.get(a.refiner)
        refiner = PERCEPTION_MODEL_REGISTRY.build(a.refiner)
        refiner.load(resolve_perception_runtime(rdesc, hardware, PerceptionComputeProfile(a.profile)).as_config())
        refiner.warmup()
    try:
        tasks = tuple((task for task in (PerceptionTask.DETECTION, PerceptionTask.INSTANCE_SEGMENTATION) if task in desc.capabilities.tasks))
        result = PerceptionPipeline(adapter, desc, refiner=refiner, refiner_descriptor=rdesc).process_frame(FramePerceptionRequest(frame, rgb, space, tasks))
        print(f'status={result.status.value} observations={len(result.observations)} processing_ms={result.processing_ms:.2f}')
        if a.overlay:
            out = rgb.copy()
            for obs in result.observations:
                x0, y0, x1, y1 = map(int, obs.detection.box.xyxy)
                cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
                if obs.selected_mask is not None:
                    m = obs.selected_mask.payload.data
                    out[m] = ((out[m].astype('uint16') + [0, 255, 0]) // 2).astype('uint8')
            cv2.imwrite(a.overlay, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    finally:
        if refiner:
            refiner.unload()
        adapter.unload()
if __name__ == '__main__':
    main()
