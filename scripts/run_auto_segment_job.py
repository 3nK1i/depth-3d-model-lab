#!/usr/bin/env python3
"""Isolated Phase 9.2D Auto Segment worker used by the desktop UI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import FrameReference, ImageSize, PixelCoordinateSpace, PixelSpaceRole
from core.hardware.gpu_info import detect_hardware
from core.perception import AutoSegmentEngine, AutoSegmentPolicy, PerceptionTask
from core.perception.messages import FramePerceptionRequest
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.perception.runtime_config import PerceptionComputeProfile, resolve_perception_runtime


def _event(progress: int, stage: str, message: str) -> None:
    print("EVENT " + json.dumps({"progress": int(progress), "stage": stage, "message": message}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated Auto Segment proposal generation")
    parser.add_argument("--input", required=True, help="Input BGR frame saved with numpy.save")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="ultralytics-sam2-prompted")
    parser.add_argument("--profile", choices=[value.value for value in PerceptionComputeProfile], default="efficient")
    parser.add_argument("--minimum-mask-pixels", type=int, default=64)
    parser.add_argument("--max-proposals", type=int, default=64)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _event(5, "loading", "Loading Auto Segment input")

    bgr = np.asarray(np.load(source), dtype=np.uint8)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise SystemExit("Auto Segment input must be an HxWx3 BGR image.")
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    height, width = rgb.shape[:2]
    frame = FrameReference(0, 0, 0.0, ImageSize(width, height))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    request = FramePerceptionRequest(frame, rgb, space, (PerceptionTask.AUTOMATIC_SEGMENTATION,))

    descriptor = PERCEPTION_MODEL_REGISTRY.get(args.model)
    adapter = PERCEPTION_MODEL_REGISTRY.build(args.model)
    runtime = resolve_perception_runtime(
        descriptor,
        detect_hardware(),
        PerceptionComputeProfile(args.profile),
    )
    _event(20, "model", f"Loading {descriptor.display_name}")
    adapter.load(runtime.as_config())
    adapter.warmup()
    try:
        _event(40, "segmenting", "Generating automatic mask proposals")
        result = AutoSegmentEngine(
            adapter,
            descriptor,
            policy=AutoSegmentPolicy(
                minimum_mask_pixels=args.minimum_mask_pixels,
                max_proposals=args.max_proposals,
            ),
        ).generate(request, workspace_id="phase9-2d-ui-workspace")
    finally:
        adapter.unload()

    masks: dict[str, np.ndarray] = {}
    proposals: list[dict[str, object]] = []
    for index, item in enumerate(result.workspace.objects):
        assert item.proposal_mask is not None
        key = f"mask_{index:04d}"
        mask = np.asarray(item.proposal_mask.payload.data, dtype=bool)
        masks[key] = mask
        proposals.append(
            {
                "object_id": item.object_id,
                "mask_id": item.proposal_mask.mask_id,
                "mask_key": key,
                "area_pixels": int(np.count_nonzero(mask)),
                "confidence": item.proposal_mask.confidence,
                "confidence_kind": item.proposal_mask.confidence_kind.value,
            }
        )

    np.savez_compressed(output / "masks.npz", **masks)
    manifest = {
        "schema_version": 1,
        "model_id": result.model_id,
        "status": result.status.value,
        "generated_proposals": result.generated_proposals,
        "kept_proposals": result.kept_proposals,
        "dropped_small": result.dropped_small,
        "dropped_duplicates": result.dropped_duplicates,
        "dropped_over_limit": result.dropped_over_limit,
        "processing_ms": result.processing_ms,
        "warnings": list(result.warnings),
        "image_size": {"width": width, "height": height},
        "proposals": proposals,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _event(100, "complete", f"Generated {result.kept_proposals} proposal(s)")
    print(f"RESULT {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
