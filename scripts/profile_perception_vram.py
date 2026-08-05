#!/usr/bin/env python3
"""Measure one perception adapter's CUDA allocation on the current hardware."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.hardware.gpu_info import detect_hardware
from core.perception.base import FramePerceptionAdapter, PromptedSegmentationAdapter
from core.perception.messages import (
    FramePerceptionRequest,
    PromptedSegmentationRequest,
    SegmentationPrompt,
    SegmentationPromptKind,
)
from core.perception.metadata import PerceptionTask
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.perception.runtime_config import (
    PerceptionComputeProfile,
    resolve_perception_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ultralytics-yolo-seg")
    parser.add_argument(
        "--profile",
        default="auto",
        choices=[value.value for value in PerceptionComputeProfile],
    )
    args = parser.parse_args()

    import torch

    hardware = detect_hardware()
    if not hardware.cuda_ready:
        raise SystemExit("CUDA is not ready.")

    descriptor = PERCEPTION_MODEL_REGISTRY.get(args.model)
    adapter = PERCEPTION_MODEL_REGISTRY.build(args.model)
    runtime = resolve_perception_runtime(
        descriptor,
        hardware,
        PerceptionComputeProfile(args.profile),
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()

    adapter.load(runtime.as_config())
    try:
        adapter.warmup()
        allocated_after_load = torch.cuda.memory_allocated()

        size = runtime.imgsz
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        frame = FrameReference(1, 1, 0.0, ImageSize(size, size))
        space = PixelCoordinateSpace(
            "rgb",
            frame,
            frame.source_size,
            PixelSpaceRole.RGB,
        )
        if isinstance(adapter, FramePerceptionAdapter):
            requested_tasks = tuple(
                task
                for task in (
                    PerceptionTask.DETECTION,
                    PerceptionTask.INSTANCE_SEGMENTATION,
                )
                if task in descriptor.capabilities.tasks
            )
        elif isinstance(adapter, PromptedSegmentationAdapter):
            requested_tasks = (PerceptionTask.PROMPTED_SEGMENTATION,)
        else:
            raise SystemExit("The selected adapter exposes no profileable Step 2 capability.")

        frame_request = FramePerceptionRequest(
            frame,
            rgb,
            space,
            requested_tasks,
        )

        if isinstance(adapter, FramePerceptionAdapter):
            adapter.infer_frame(frame_request)
        elif isinstance(adapter, PromptedSegmentationAdapter):
            prompt = SegmentationPrompt(
                "profile-box",
                SegmentationPromptKind.BOX,
                space,
                box_xyxy=(size * 0.25, size * 0.25, size * 0.75, size * 0.75),
            )
            adapter.refine_mask(
                PromptedSegmentationRequest(
                    frame_request,
                    prompt,
                    parent_detection_id="profile-detection",
                )
            )
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        print(
            json.dumps(
                {
                    "model": args.model,
                    "checkpoint": runtime.checkpoint,
                    "profile": runtime.effective_profile.value,
                    "allocated_before_mb": allocated_before / 1024**2,
                    "allocated_after_load_mb": allocated_after_load / 1024**2,
                    "peak_mb": peak / 1024**2,
                },
                indent=2,
            )
        )
    finally:
        adapter.unload()


if __name__ == "__main__":
    main()
