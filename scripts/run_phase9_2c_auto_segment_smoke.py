#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import FrameReference, ImageSize, PixelCoordinateSpace, PixelSpaceRole
from core.hardware.gpu_info import detect_hardware
from core.perception import AutoSegmentEngine, AutoSegmentPolicy, PerceptionTask, create_segmentation_project
from core.perception.messages import FramePerceptionRequest
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.perception.runtime_config import PerceptionComputeProfile, resolve_perception_runtime


def _synthetic_rgb() -> np.ndarray:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[15:90, 12:75] = (190, 190, 190)
    image[35:110, 90:150] = (110, 170, 210)
    return image


def _overlay(rgb: np.ndarray, workspace) -> np.ndarray:
    out = rgb.copy()
    palette = np.asarray(
        [
            [255, 128, 0],
            [0, 210, 255],
            [160, 100, 255],
            [90, 220, 120],
            [255, 90, 160],
        ],
        dtype=np.uint16,
    )
    for index, item in enumerate(workspace.objects):
        mask = np.asarray(item.proposal_mask.payload.data, dtype=bool)
        tint = palette[index % len(palette)]
        out[mask] = ((out[mask].astype(np.uint16) + tint) // 2).astype(np.uint8)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9.2C Auto Segment proposal smoke")
    parser.add_argument("--image", default="")
    parser.add_argument("--model", default="development-perception-mock")
    parser.add_argument(
        "--profile",
        choices=[value.value for value in PerceptionComputeProfile],
        default="efficient",
    )
    parser.add_argument("--minimum-mask-pixels", type=int, default=64)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--output-dir", default="/tmp/eia_phase9_2c_auto_segment")
    args = parser.parse_args()

    if args.image:
        bgr = cv2.imread(str(Path(args.image).expanduser()))
        if bgr is None:
            raise SystemExit(f"Unable to read image: {args.image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        source = str(Path(args.image).expanduser().resolve())
    else:
        if args.model != "development-perception-mock":
            raise SystemExit("A real --image is required for a production Auto Segment model.")
        rgb = _synthetic_rgb()
        source = "synthetic"

    height, width = rgb.shape[:2]
    frame = FrameReference(92, 1, 0.0, ImageSize(width, height))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    request = FramePerceptionRequest(
        frame,
        rgb,
        space,
        (PerceptionTask.AUTOMATIC_SEGMENTATION,),
    )

    descriptor = PERCEPTION_MODEL_REGISTRY.get(args.model)
    adapter = PERCEPTION_MODEL_REGISTRY.build(args.model)
    hardware = detect_hardware()
    runtime = resolve_perception_runtime(
        descriptor,
        hardware,
        PerceptionComputeProfile(args.profile),
    )
    adapter.load(runtime.as_config())
    adapter.warmup()
    try:
        project = create_segmentation_project(
            "Phase 9.2C Smoke",
            project_id="phase9-2c-smoke-project",
        )
        result = AutoSegmentEngine(
            adapter,
            descriptor,
            policy=AutoSegmentPolicy(
                minimum_mask_pixels=args.minimum_mask_pixels,
                max_proposals=args.max_proposals,
            ),
        ).generate(request, workspace_id="phase9-2c-smoke-workspace", project=project)
    finally:
        adapter.unload()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "auto_segment_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(_overlay(rgb, result.workspace), cv2.COLOR_RGB2BGR))

    summary = {
        "phase": "9.2C",
        "status": "PASS" if result.kept_proposals > 0 else "FAIL",
        "engine_status": result.status.value,
        "source": source,
        "model_id": result.model_id,
        "effective_profile": runtime.effective_profile.value,
        "generated_proposals": result.generated_proposals,
        "kept_proposals": result.kept_proposals,
        "unlabeled_proposals": sum(item.class_id is None for item in result.workspace.objects),
        "dropped_small": result.dropped_small,
        "dropped_duplicates": result.dropped_duplicates,
        "dropped_over_limit": result.dropped_over_limit,
        "geometry_authority": result.is_geometry_authority,
        "metric_authority": result.is_metric_authority,
        "measurement_authority": result.is_measurement_authority,
        "overlay": str(overlay_path),
        "warnings": list(result.warnings),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
