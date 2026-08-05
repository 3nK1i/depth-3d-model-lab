#!/usr/bin/env python3
"""Standalone deterministic Phase 7.5 Step 4 extraction smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.alignment.contracts import AlignedFrameBundle, AlignedObjectObservation, AlignmentStatus
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import CalibrationStatus, CameraIntrinsics, PixelCoordinateSpace, PixelSpaceRole
from core.models.metadata import ConfidenceKind, DepthScale, IntrinsicsSource
from core.object_geometry import (
    BackgroundPlanePolicy,
    ClusteringPolicy,
    DepthDiscontinuityPolicy,
    MaskPreprocessingPolicy,
    ObjectPointExtractionPolicy,
    ObjectPointExtractionRequest,
    StatisticalOutlierPolicy,
    extract_object_point_clouds,
)
from core.perception.result import ApprovalStatus, MaskSource


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, np.uint8)
    if np.any(finite):
        low, high = np.percentile(depth[finite], (1, 99))
        if high > low:
            normalized[finite] = np.clip((depth[finite] - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def _top_down(points: np.ndarray, size: tuple[int, int] = (640, 400)) -> np.ndarray:
    width, height = size
    canvas = np.zeros((height, width, 3), np.uint8)
    if not len(points):
        return canvas
    xz = points[:, (0, 2)]
    low = np.min(xz, axis=0)
    high = np.max(xz, axis=0)
    span = np.maximum(high - low, 1e-9)
    uv = (xz - low) / span
    px = (40 + uv[:, 0] * (width - 80)).astype(int)
    py = (height - 40 - uv[:, 1] * (height - 80)).astype(int)
    for x, y in zip(px, py, strict=True):
        cv2.circle(canvas, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.line(canvas, (40, height - 40), (width - 20, height - 40), (180, 180, 180), 1)
    cv2.line(canvas, (40, 20), (40, height - 40), (180, 180, 180), 1)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/phase7_5_step4_extraction.png")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()
    width, height = args.width, args.height
    if width < 96 or height < 72:
        raise SystemExit("Smoke dimensions must be at least 96x72.")

    frame = FrameReference(1, 1, 0.0, ImageSize(width, height))
    space = PixelCoordinateSpace("depth", frame, frame.source_size, PixelSpaceRole.DEPTH)
    rgb = np.full((height, width, 3), (35, 45, 55), np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    rgb[..., 0] = np.clip(35 + xx * 80 / width, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(45 + yy * 100 / height, 0, 255).astype(np.uint8)

    mask = np.zeros((height, width), bool)
    x0, x1 = int(width * 0.24), int(width * 0.76)
    y0, y1 = int(height * 0.22), int(height * 0.78)
    mask[y0:y1, x0:x1] = True

    depth = np.full((height, width), 2.50, np.float32)
    object_depth = 1.20 + (xx - width / 2) * 0.0004 + (yy - height / 2) * 0.0003
    depth[mask] = object_depth[mask]
    # A background-contamination patch and isolated depth spikes.
    depth[y0 + 8:y0 + 30, x1 - 24:x1 - 5] = 2.50
    spikes = ((y0 + 20, x0 + 20), (y1 - 25, x0 + 40), (y0 + 40, x1 - 45))
    for y, x in spikes:
        depth[y, x] = 8.0

    provenance = ModelProvenance("step4-smoke-mask", "Step 4 Smoke Mask", "instance_segmentation")
    observation = AlignedObjectObservation(
        "smoke-object", 0, "synthetic-object", space,
        source_mask_id="smoke-mask", source_mask_space_id="depth",
        aligned_mask=mask,
        source_foreground_pixels=int(np.count_nonzero(mask)),
        aligned_foreground_pixels=int(np.count_nonzero(mask)),
        foreground_retention_ratio=1.0,
        mask_source=MaskSource.NATIVE_INSTANCE,
        approval_status=ApprovalStatus.NOT_REQUIRED,
        mask_model_provenance=provenance,
    )
    intrinsics = CameraIntrinsics(
        np.array([[280.0, 0.0, width / 2], [0.0, 280.0, height / 2], [0.0, 0.0, 1.0]]),
        "depth", IntrinsicsSource.ESTIMATED, CalibrationStatus.NOT_CALIBRATED,
    )
    bundle = AlignedFrameBundle(
        frame, space, rgb, depth, np.ones(depth.shape, bool), np.ones(depth.shape, bool),
        None, np.ones(depth.shape, bool), ConfidenceKind.NONE, intrinsics,
        DepthScale.METRIC, 1,
        ModelProvenance("step4-smoke-depth", "Step 4 Smoke Depth", "depth"),
        (provenance,), (observation,), (), AlignmentStatus.SUCCESS,
    )
    policy = ObjectPointExtractionPolicy(
        mask=MaskPreprocessingPolicy(erosion_pixels=1, minimum_mask_pixels=64),
        discontinuity=DepthDiscontinuityPolicy(enabled=True, minimum_neighbor_samples=16),
        outliers=StatisticalOutlierPolicy(enabled=True, minimum_points=64),
        background_plane=BackgroundPlanePolicy(
            enabled=True, minimum_ring_points=64, distance_threshold=0.025,
            minimum_behind_separation=0.15, maximum_removal_fraction=0.45,
        ),
        clustering=ClusteringPolicy(enabled=True, voxel_size=0.025, minimum_cluster_points=32),
        minimum_retained_points=64,
    )
    result = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=policy))
    item = result.object_results[0]
    if item.filtered_cloud is None or item.raw_cloud is None:
        raise SystemExit(f"Extraction rejected: {[reason.value for reason in item.rejection_reasons]}")

    candidate = rgb.copy()
    candidate[mask] = cv2.addWeighted(candidate, 0.35, np.full_like(candidate, (0, 180, 255)), 0.65, 0)[mask]
    raw_pixels = np.zeros_like(rgb)
    raw_xy = item.raw_cloud.source_pixel_indices_xy
    raw_pixels[raw_xy[:, 1], raw_xy[:, 0]] = (0, 0, 255)
    filtered_pixels = np.zeros_like(rgb)
    filtered_xy = item.filtered_cloud.source_pixel_indices_xy
    filtered_pixels[filtered_xy[:, 1], filtered_xy[:, 0]] = (0, 255, 0)

    panel_size = (640, 400)
    panels = (
        _label(cv2.resize(candidate, panel_size), "Synthetic RGB + selected object mask"),
        _label(cv2.resize(_depth_preview(depth), panel_size), "Metric depth with background contamination + spikes"),
        _label(cv2.resize(raw_pixels, panel_size, interpolation=cv2.INTER_NEAREST), "Raw projected object pixels"),
        _label(_top_down(item.filtered_cloud.points_xyz_m, panel_size), "Filtered object points - top-down X/Z"),
    )
    canvas = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise SystemExit(f"Failed to write {output}")

    print(
        f"status={item.status.value} raw_points={item.raw_cloud.retained_point_count} "
        f"filtered_points={item.filtered_cloud.retained_point_count} "
        f"metric_ready={str(item.filtered_cloud.metric_ready).lower()} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
