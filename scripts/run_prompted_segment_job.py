#!/usr/bin/env python3
"""Isolated Phase 9.2E SAM2 point-prompt segmentation worker."""
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
from core.perception.base import PromptedSegmentationAdapter
from core.perception.messages import (
    FramePerceptionRequest,
    PromptedSegmentationRequest,
    SegmentationPrompt,
    SegmentationPromptKind,
)
from core.perception.metadata import PerceptionTask
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.perception.runtime_config import PerceptionComputeProfile, resolve_perception_runtime


def _event(progress: int, stage: str, message: str) -> None:
    print("EVENT " + json.dumps({"progress": int(progress), "stage": stage, "message": message}), flush=True)


def _points(values: object, *, field_name: str) -> tuple[tuple[float, float], ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise SystemExit(f"{field_name} must be a JSON list of [x, y] pairs.")
    output: list[tuple[float, float]] = []
    for item in values:
        if not isinstance(item, list) or len(item) != 2:
            raise SystemExit(f"{field_name} must contain [x, y] pairs.")
        x, y = float(item[0]), float(item[1])
        if not np.isfinite([x, y]).all():
            raise SystemExit(f"{field_name} coordinates must be finite.")
        output.append((x, y))
    return tuple(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated SAM2 point-prompt segmentation")
    parser.add_argument("--input", required=True, help="Input BGR frame saved with numpy.save")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True, help="JSON prompt specification")
    parser.add_argument("--model", default="ultralytics-sam2-prompted")
    parser.add_argument("--profile", choices=[value.value for value in PerceptionComputeProfile], default="efficient")
    parser.add_argument("--minimum-mask-pixels", type=int, default=64)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    prompt_path = Path(args.prompt).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _event(5, "loading", "Loading interactive segmentation input")

    bgr = np.asarray(np.load(source), dtype=np.uint8)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise SystemExit("Prompted segmentation input must be an HxWx3 BGR image.")
    spec = json.loads(prompt_path.read_text(encoding="utf-8"))
    operation = str(spec.get("operation", "")).strip().lower()
    if operation not in {"create", "refine"}:
        raise SystemExit("Prompt operation must be 'create' or 'refine'.")
    object_id = str(spec.get("object_id", "")).strip()
    if not object_id:
        raise SystemExit("Prompt object_id must not be empty.")
    parent_mask_id = str(spec.get("parent_mask_id", "")).strip()
    revision = int(spec.get("revision", 0))
    if operation == "create":
        if parent_mask_id or revision != 0:
            raise SystemExit("New interactive objects must start at revision 0 without a parent mask.")
    else:
        if not parent_mask_id or revision < 1:
            raise SystemExit("Refinement requires a parent mask and revision >= 1.")

    positive = _points(spec.get("positive_points"), field_name="positive_points")
    negative = _points(spec.get("negative_points"), field_name="negative_points")
    if not positive:
        raise SystemExit("Interactive SAM2 prompting requires at least one positive point.")
    minimum_pixels = int(args.minimum_mask_pixels)
    if minimum_pixels < 1:
        raise SystemExit("minimum-mask-pixels must be positive.")

    rgb = np.ascontiguousarray(bgr[..., ::-1])
    height, width = rgb.shape[:2]
    for x, y in positive + negative:
        if not (0.0 <= x < width and 0.0 <= y < height):
            raise SystemExit("Prompt point lies outside the source image.")

    frame = FrameReference(0, 0, 0.0, ImageSize(width, height))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    frame_request = FramePerceptionRequest(frame, rgb, space, (PerceptionTask.PROMPTED_SEGMENTATION,))
    prompt = SegmentationPrompt(
        prompt_id=f"phase9-2e:{object_id}:r{revision}",
        kind=SegmentationPromptKind.POINTS,
        pixel_space=space,
        positive_points_xy=positive,
        negative_points_xy=negative,
    )
    request = PromptedSegmentationRequest(
        frame_request,
        prompt,
        parent_mask_id=parent_mask_id,
        creates_new_instance=(operation == "create"),
    )

    descriptor = PERCEPTION_MODEL_REGISTRY.get(args.model)
    adapter = PERCEPTION_MODEL_REGISTRY.build(args.model)
    if not isinstance(adapter, PromptedSegmentationAdapter):
        raise SystemExit(f"Perception model {args.model!r} does not support prompted segmentation.")
    runtime = resolve_perception_runtime(descriptor, detect_hardware(), PerceptionComputeProfile(args.profile))
    _event(20, "model", f"Loading {descriptor.display_name}")
    adapter.load(runtime.as_config())
    adapter.warmup()
    try:
        _event(45, "segmenting", "Running SAM2 point prompt")
        raw = adapter.refine_mask(request)
    finally:
        adapter.unload()

    mask = np.asarray(raw.mask, dtype=bool)
    if mask.shape != (height, width):
        raise SystemExit("Prompted segmentation returned a mask with an unexpected shape.")
    area = int(np.count_nonzero(mask))
    if area < minimum_pixels:
        raise SystemExit(
            f"Prompted segmentation mask has {area} foreground pixels; minimum is {minimum_pixels}."
        )

    np.save(output / "mask.npy", mask)
    mask_id = f"{object_id}:sam2:r{revision}"
    manifest = {
        "schema_version": 1,
        "operation": operation,
        "object_id": object_id,
        "mask_id": mask_id,
        "parent_mask_id": parent_mask_id,
        "revision": revision,
        "model_id": raw.model_provenance.model_id,
        "confidence": raw.confidence,
        "confidence_kind": raw.confidence_kind.value,
        "area_pixels": area,
        "positive_points": [list(point) for point in positive],
        "negative_points": [list(point) for point in negative],
        "image_size": {"width": width, "height": height},
        "warnings": list(raw.warnings),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _event(100, "complete", "Interactive SAM2 mask ready")
    print(f"RESULT {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
