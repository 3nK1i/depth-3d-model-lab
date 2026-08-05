#!/usr/bin/env python3
"""Standalone Phase 7.5 Step 3 alignment smoke test.

This script intentionally uses synthetic depth and a deterministic dense mask.
It validates the alignment layer without loading a depth or perception model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.alignment import (
    AlignmentRequest,
    DepthAlignmentInput,
    align_frame,
    declare_full_frame_resize,
)
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import (
    CalibrationStatus,
    CameraIntrinsics,
    PixelCoordinateSpace,
    PixelSpaceRole,
)
from core.models.metadata import ConfidenceKind, DepthScale, IntrinsicsSource
from core.perception.result import (
    DenseMaskPayload,
    Detection2D,
    InstanceMask,
    MaskSource,
    ObjectObservation,
    PerceptionFrameResult,
    normalize_detection_box,
)


def _overlay(image: np.ndarray, mask: np.ndarray, box=None) -> np.ndarray:
    output = image.copy()
    green = np.zeros_like(output)
    green[..., 1] = 255
    output[mask] = cv2.addWeighted(output, 0.35, green, 0.65, 0)[mask]
    if box is not None:
        x0, y0, x1, y1 = box
        cv2.rectangle(
            output,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            (255, 255, 255),
            2,
        )
    return output


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="/tmp/phase7_5_step3_alignment.png")
    parser.add_argument("--depth-width", type=int, default=0)
    parser.add_argument("--depth-height", type=int, default=0)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Unable to read image: {image_path}")
    height, width = image.shape[:2]
    depth_width = args.depth_width or max(1, width // 2)
    depth_height = args.depth_height or max(1, height // 2)
    if depth_width <= 0 or depth_height <= 0:
        raise SystemExit("Depth dimensions must be positive.")

    frame = FrameReference(1, 1, 0.0, ImageSize(width, height))
    rgb_space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    depth_space = PixelCoordinateSpace(
        "depth", frame, ImageSize(depth_width, depth_height), PixelSpaceRole.DEPTH
    )

    x0, y0 = int(width * 0.25), int(height * 0.25)
    x1, y1 = int(width * 0.75), int(height * 0.75)
    source_mask = np.zeros((height, width), dtype=np.bool_)
    source_mask[y0:y1, x0:x1] = True
    perception_provenance = ModelProvenance(
        "alignment-smoke-mask", "Alignment Smoke Mask", "instance_segmentation"
    )
    detection = Detection2D(
        "smoke:det",
        frame,
        0,
        "validation-region",
        1.0,
        normalize_detection_box((float(x0), float(y0), float(x1), float(y1)), rgb_space),
        perception_provenance,
    )
    mask = InstanceMask(
        "smoke:mask",
        frame,
        rgb_space,
        DenseMaskPayload(source_mask),
        MaskSource.NATIVE_INSTANCE,
        perception_provenance,
        confidence=1.0,
        confidence_kind=ConfidenceKind.CONFIDENCE,
        foreground_valid=True,
    )
    observation = ObjectObservation(
        "smoke:object",
        frame,
        0,
        "validation-region",
        detection,
        (mask,),
    )
    perception = PerceptionFrameResult(
        frame, (observation,), (perception_provenance,)
    )

    x = np.linspace(0.5, 3.0, depth_width, dtype=np.float32)
    depth = np.repeat(x[None, :], depth_height, axis=0)
    intrinsics = CameraIntrinsics(
        np.array(
            [
                [float(depth_width), 0.0, depth_width / 2.0],
                [0.0, float(depth_height), depth_height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        "depth",
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    depth_input = DepthAlignmentInput(
        frame,
        depth_space,
        depth,
        np.ones(depth.shape, bool),
        None,
        ConfidenceKind.NONE,
        intrinsics,
        DepthScale.METRIC,
        ModelProvenance("alignment-smoke-depth", "Alignment Smoke Depth", "depth"),
    )
    transform = declare_full_frame_resize(rgb_space, depth_space)
    result = align_frame(
        AlignmentRequest(
            frame,
            image,
            rgb_space,
            depth_input,
            perception,
            (transform,),
        )
    )
    aligned = result.aligned_observations[0]
    assert aligned.aligned_mask is not None

    source_overlay = _overlay(image, source_mask, detection.box.xyxy)
    aligned_overlay = _overlay(
        result.aligned_rgb,
        aligned.aligned_mask,
        aligned.aligned_detection_box.xyxy if aligned.aligned_detection_box else None,
    )
    depth_valid = np.where(result.depth_valid_mask, result.depth, np.nan)
    finite = np.isfinite(depth_valid)
    normalized = np.zeros_like(result.depth, dtype=np.uint8)
    if np.any(finite):
        low, high = np.nanmin(depth_valid), np.nanmax(depth_valid)
        if high > low:
            normalized[finite] = np.clip(
                (depth_valid[finite] - low) * 255.0 / (high - low), 0, 255
            ).astype(np.uint8)
    depth_preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    mask_preview = np.zeros_like(result.aligned_rgb)
    mask_preview[aligned.aligned_mask] = (0, 255, 0)

    panel_size = (max(480, width // 2), max(320, height // 2))
    panels = [
        _label(cv2.resize(source_overlay, panel_size), "Original RGB + selected mask"),
        _label(cv2.resize(aligned_overlay, panel_size), "Aligned RGB + depth-grid mask"),
        _label(cv2.resize(depth_preview, panel_size), "Native depth grid (not resampled)"),
        _label(cv2.resize(mask_preview, panel_size, interpolation=cv2.INTER_NEAREST), "Aligned boolean mask"),
    ]
    canvas = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise SystemExit(f"Failed to write output: {output_path}")

    print(
        f"status={result.status.value} observations={len(result.aligned_observations)} "
        f"target={depth_width}x{depth_height} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
