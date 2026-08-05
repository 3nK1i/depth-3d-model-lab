#!/usr/bin/env python3
"""Validation-only real image pipeline for Phase 7.5 Steps 2-6.

This script deliberately runs perception and depth sequentially so an RTX 2060
never needs to keep both model families resident. It does not touch the Qt UI,
production controllers, workers, or export implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.alignment import (
    AlignmentRequest,
    align_frame,
    declare_full_frame_resize,
    declare_identity_transform,
    depth_input_from_inference_result,
    make_depth_space,
)
from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.hardware.gpu_info import detect_hardware
from core.hardware.profiles import ComputeProfile, resolve_profile
from core.measurement import MeasurementStatus, compute_measurements, make_measurement_request
from core.models.registry import MODEL_REGISTRY
from core.models.runtime_config import ModelRuntimeContext
from core.models.validation import normalize_geometry_result
from core.object_geometry import (
    extract_object_point_clouds,
    fit_object_geometry,
    make_extraction_request,
    make_fitting_request,
)
from core.perception.messages import FramePerceptionRequest
from core.perception.metadata import PerceptionTask
from core.perception.pipeline import PerceptionPipeline
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.perception.runtime_config import PerceptionComputeProfile, resolve_perception_runtime
from core.runtime.messages import FrameRequest, InferenceResult

_BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7), (6, 7),
)
_BOX_STYLES = {
    "axis_aligned": ((210, 210, 210), 1),
    "camera_aligned": ((255, 220, 0), 2),
    "oriented": ((0, 150, 255), 2),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--depth-model", default="depth-pro")
    parser.add_argument("--perception-model", default="ultralytics-yolo-seg")
    parser.add_argument("--depth-profile", choices=[p.value for p in ComputeProfile], default="efficient")
    parser.add_argument("--perception-profile", choices=[p.value for p in PerceptionComputeProfile], default="efficient")
    parser.add_argument("--output-dir", default="/tmp/phase7_5_real_measurement")
    parser.add_argument("--max-objects", type=int, default=5)
    parser.add_argument("--minimum-mask-pixels", type=int, default=64)
    return parser


def _to_inference_result(result) -> InferenceResult:
    return InferenceResult(
        session_id=result.session_id,
        frame_id=result.frame_id,
        model_id=result.model_id,
        depth=result.depth,
        latency_ms=result.latency_ms,
        points=result.points,
        normals=result.normals,
        intrinsics=result.intrinsics,
        valid_mask=result.valid_mask,
        point_stride=result.point_stride,
        is_metric=result.is_metric,
        model_name=result.model_name,
        precision=result.precision,
        preprocess_ms=result.preprocess_ms,
        postprocess_ms=result.postprocess_ms,
        worker_total_ms=result.worker_total_ms,
        queue_wait_ms=result.queue_wait_ms,
        vram_used_gib=result.vram_used_gib,
        vram_allocated_gib=result.vram_allocated_gib,
        vram_reserved_gib=result.vram_reserved_gib,
        vram_peak_reserved_gib=result.vram_peak_reserved_gib,
        vram_device_used_gib=result.vram_device_used_gib,
        vram_total_gib=result.vram_total_gib,
        confidence=result.confidence,
        colors=result.colors,
        depth_scale=result.depth_scale,
        intrinsics_source=result.intrinsics_source,
        confidence_kind=result.confidence_kind,
    )


def _run_perception(rgb: np.ndarray, frame: FrameReference, space: PixelCoordinateSpace, model_id: str, profile: str):
    descriptor = PERCEPTION_MODEL_REGISTRY.get(model_id)
    adapter = PERCEPTION_MODEL_REGISTRY.build(model_id)
    runtime = resolve_perception_runtime(descriptor, detect_hardware(), PerceptionComputeProfile(profile))
    adapter.load(runtime.as_config())
    adapter.warmup()
    try:
        tasks = tuple(
            task
            for task in (PerceptionTask.DETECTION, PerceptionTask.INSTANCE_SEGMENTATION)
            if task in descriptor.capabilities.tasks
        )
        return PerceptionPipeline(adapter, descriptor).process_frame(
            FramePerceptionRequest(frame, rgb, space, tasks)
        )
    finally:
        adapter.unload()


def _run_depth(bgr: np.ndarray, model_id: str, profile: str):
    descriptor = MODEL_REGISTRY.get(model_id)
    if descriptor.capabilities.depth_scale.value != "metric":
        print(f"WARNING: {model_id} is not metric; metric dimensions will be suppressed.")
    hardware = detect_hardware()
    resolved = resolve_profile(ComputeProfile(profile), hardware)
    config = MODEL_REGISTRY.resolve_runtime_config(
        model_id,
        ModelRuntimeContext(hardware_info=hardware, resolved_profile=resolved),
    )
    adapter = MODEL_REGISTRY.build(model_id)
    adapter.load(
        device=str(config.get("device", resolved.device)),
        precision=str(config.get("precision", resolved.precision.value)),
        config=config,
    )
    adapter.warmup()
    request = FrameRequest(1, 1, bgr, source_timestamp=0.0)
    try:
        raw = adapter.infer(request)
        return normalize_geometry_result(raw, descriptor, request)
    finally:
        adapter.unload()


def _depth_preview(depth: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    array = np.asarray(depth, dtype=np.float32)
    mask = np.isfinite(array) & (array > 0)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    preview = np.zeros(array.shape, dtype=np.uint8)
    values = array[mask]
    if values.size:
        lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        preview[mask] = np.clip(
            (array[mask] - lo) * 255.0 / (hi - lo), 0, 255
        ).astype(np.uint8)
    return cv2.applyColorMap(255 - preview, cv2.COLORMAP_TURBO)


def _color_for(index: int) -> tuple[int, int, int]:
    palette = ((0, 220, 0), (255, 180, 0), (0, 180, 255), (220, 0, 220), (255, 80, 80))
    return palette[index % len(palette)]


def _annotate(rgb: np.ndarray, observations) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for index, obs in enumerate(observations):
        color = _color_for(index)
        if obs.selected_mask is not None and hasattr(obs.selected_mask.payload, "data"):
            mask = np.asarray(obs.selected_mask.payload.data, dtype=bool)
            overlay = out.copy()
            overlay[mask] = color
            out = cv2.addWeighted(out, 0.68, overlay, 0.32, 0)
        if obs.detection is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in obs.detection.box.xyxy]
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            label = f"{index}: {obs.class_name} {obs.detection.confidence:.2f}"
            cv2.putText(
                out,
                label,
                (x0, max(18, y0 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def _boxes_for_fit(fit_result) -> tuple[tuple[str, Any], ...]:
    values = []
    for name, box in (
        ("axis_aligned", fit_result.axis_aligned_box),
        ("camera_aligned", fit_result.camera_aligned_box),
        ("oriented", fit_result.oriented_box),
    ):
        if box is not None:
            values.append((name, box))
    return tuple(values)


def _box_corners(box) -> np.ndarray:
    signs = np.array(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    local = signs * (np.asarray(box.extents_xyz_m, dtype=np.float64) * 0.5)
    rotation = np.asarray(box.orientation_matrix, dtype=np.float64)
    centre = np.asarray(box.centre_xyz_m, dtype=np.float64)
    return local @ rotation.T + centre


def _project_camera_points(points_xyz: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-8)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = k[0, 0] * points[valid, 0] / z[valid] + k[0, 2]
    uv[valid, 1] = k[1, 1] * points[valid, 1] / z[valid] + k[1, 2]
    return uv, valid


def _draw_projected_box(
    image: np.ndarray,
    box,
    intrinsics: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    label: str,
) -> None:
    corners = _box_corners(box)
    uv, valid = _project_camera_points(corners, intrinsics)
    h, w = image.shape[:2]
    for first, second in _BOX_EDGES:
        if not (valid[first] and valid[second]):
            continue
        p0 = tuple(np.rint(uv[first]).astype(int))
        p1 = tuple(np.rint(uv[second]).astype(int))
        if (
            max(p0[0], p1[0]) < 0
            or max(p0[1], p1[1]) < 0
            or min(p0[0], p1[0]) >= w
            or min(p0[1], p1[1]) >= h
        ):
            continue
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)
    finite_uv = uv[valid]
    if finite_uv.size:
        anchor = tuple(np.rint(finite_uv[np.argmin(finite_uv[:, 1])]).astype(int))
        cv2.putText(
            image,
            label,
            (max(0, anchor[0]), max(16, anchor[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def _annotate_aligned_geometry(aligned, fitting) -> np.ndarray:
    out = cv2.cvtColor(np.asarray(aligned.aligned_rgb), cv2.COLOR_RGB2BGR)
    fits = {item.observation_id: item for item in fitting.object_results}
    for index, observation in enumerate(aligned.aligned_observations):
        color = _color_for(index)
        if observation.aligned_mask is not None:
            overlay = out.copy()
            overlay[np.asarray(observation.aligned_mask, dtype=bool)] = color
            out = cv2.addWeighted(out, 0.82, overlay, 0.18, 0)
        fit = fits.get(observation.observation_id)
        if fit is None:
            continue
        for name, box in _boxes_for_fit(fit):
            box_color, thickness = _BOX_STYLES[name]
            preferred = box.box_id == fit.preferred_box_id
            _draw_projected_box(
                out,
                box,
                aligned.intrinsics.matrix,
                color if preferred else box_color,
                thickness + (1 if preferred else 0),
                f"{index}:{name}{'*' if preferred else ''}",
            )
    cv2.putText(out, "3D boxes projected on aligned depth grid", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(out, "gray=AABB cyan=camera orange=OBB * preferred", (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (225, 225, 225), 1, cv2.LINE_AA)
    return out


def _measurement_lines(computation) -> list[str]:
    lines: list[str] = []
    for estimate in computation.measurement_result.estimates:
        if estimate.status is MeasurementStatus.AVAILABLE:
            uncertainty = estimate.uncertainty.expanded_uncertainty_si if estimate.uncertainty else 0.0
            lines.append(
                f"{estimate.kind.value}: {estimate.value_si:.5g} {estimate.unit_si} +/- "
                f"{uncertainty:.3g} [{estimate.lower_bound_si:.5g}, "
                f"{estimate.upper_bound_si:.5g}] {estimate.reliability.level.value}"
            )
        else:
            reasons = ",".join(reason.value for reason in estimate.rejection_reasons) or estimate.status.value
            lines.append(f"{estimate.kind.value}: {estimate.status.value} ({reasons})")
    return lines


def _diagnostic_lines(extraction_result, fit_result) -> list[str]:
    extraction_reasons = ",".join(value.value for value in extraction_result.rejection_reasons) or "none"
    fit_reasons = ",".join(value.value for value in fit_result.rejection_reasons) or "none"
    counts = extraction_result.stage_counts
    lines = [
        f"extraction={extraction_result.status.value} reasons={extraction_reasons}",
        (
            "counts mask={0} depth={1} confidence={2} discontinuity={3} "
            "cluster={4} output={5}"
        ).format(
            counts.eroded_mask_pixels,
            counts.depth_valid_pixels,
            counts.confidence_valid_pixels,
            counts.discontinuity_retained_pixels,
            counts.cluster_retained_points,
            counts.output_points,
        ),
        f"fitting={fit_result.status.value} reasons={fit_reasons}",
    ]
    if fit_result.preferred_box_id:
        lines.append(
            f"preferred_box={fit_result.preferred_box_id} reason={fit_result.preferred_box_reason}"
        )
    for name, box in _boxes_for_fit(fit_result):
        extents = ", ".join(f"{value:.4g}" for value in box.extents_xyz_m)
        lines.append(f"{name}_extents=[{extents}] fit={box.fit_method}")
    return lines


def _text_panel(lines: list[str], width: int, height: int, title: str) -> np.ndarray:
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.putText(panel, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    y = 58
    for line in lines:
        for start in range(0, len(line), 82):
            cv2.putText(panel, line[start:start + 82], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 230, 210), 1, cv2.LINE_AA)
            y += 20
            if y > height - 14:
                return panel
    return panel


def _box_xz_hull(box) -> np.ndarray:
    """Return the convex X/Z silhouette of a 3D box for uncluttered diagnostics."""
    corners = np.asarray(_box_corners(box)[:, [0, 2]], dtype=np.float64)
    hull = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)
    return np.asarray(hull, dtype=np.float64)


def _draw_xz_box(
    panel: np.ndarray,
    box,
    lo: np.ndarray,
    span: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    # A full 3D wireframe can self-cross after orthographic X/Z projection and
    # look like an invalid fit. Draw the convex projected silhouette instead;
    # the RGB panel still renders the complete 12-edge 3D box.
    hull = _box_xz_hull(box)
    normalized = (hull - lo) / span
    px = np.clip((normalized[:, 0] * (panel.shape[1] - 50) + 25).astype(int), 0, panel.shape[1] - 1)
    py = np.clip(((1.0 - normalized[:, 1]) * (panel.shape[0] - 65) + 40).astype(int), 0, panel.shape[0] - 1)
    polygon = np.column_stack((px, py)).astype(np.int32).reshape((-1, 1, 2))
    if len(polygon) >= 2:
        cv2.polylines(panel, [polygon], True, color, thickness, cv2.LINE_AA)


def _points_panel(extraction_result, fit_result, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 15, dtype=np.uint8)
    cloud = extraction_result.filtered_cloud
    if cloud is None or not len(cloud.points_xyz_m):
        cv2.putText(panel, "No filtered object points", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        reasons = ",".join(value.value for value in extraction_result.rejection_reasons)
        cv2.putText(panel, f"extraction={extraction_result.status.value} {reasons}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 180, 255), 1, cv2.LINE_AA)
        counts = extraction_result.stage_counts
        cv2.putText(panel, f"mask={counts.eroded_mask_pixels} depth={counts.depth_valid_pixels} confidence={counts.confidence_valid_pixels}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return panel

    points = np.asarray(cloud.points_xyz_m)
    xz = points[:, [0, 2]]
    box_points = [
        _box_corners(box)[:, [0, 2]]
        for _, box in _boxes_for_fit(fit_result)
    ]
    combined = np.vstack((xz, *box_points)) if box_points else xz
    lo = np.percentile(combined, 1, axis=0)
    hi = np.percentile(combined, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    normalized = (xz - lo) / span
    px = np.clip((normalized[:, 0] * (width - 50) + 25).astype(int), 0, width - 1)
    py = np.clip(((1.0 - normalized[:, 1]) * (height - 65) + 40).astype(int), 0, height - 1)
    panel[py, px] = (0, 220, 0)
    for name, box in _boxes_for_fit(fit_result):
        color, thickness = _BOX_STYLES[name]
        if box.box_id == fit_result.preferred_box_id:
            color = (0, 255, 120)
            thickness += 1
        _draw_xz_box(panel, box, lo, span, color, thickness)
    cv2.putText(panel, "Filtered object points + fitted boxes: camera X/Z", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, "gray=AABB cyan=camera orange=OBB green=preferred", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    if fit_result.preferred_box_id:
        cv2.putText(panel, f"preferred={fit_result.preferred_box_id} status={fit_result.status.value}", (16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1, cv2.LINE_AA)
    return panel


def _box_payload(fit_result) -> list[dict[str, Any]]:
    payload = []
    for name, box in _boxes_for_fit(fit_result):
        payload.append(
            {
                "name": name,
                "box_id": box.box_id,
                "kind": box.kind.value,
                "centre_xyz": box.centre_xyz_m.tolist(),
                "extents_xyz": box.extents_xyz_m.tolist(),
                "orientation_matrix": box.orientation_matrix.tolist(),
                "fit_method": box.fit_method,
                "support_point_count": box.support_point_count,
                "fit_residual": box.fit_residual_m,
                "preferred": box.box_id == fit_result.preferred_box_id,
            }
        )
    return payload


def _serialize(batch, extraction, fitting, metadata: dict[str, Any]) -> dict[str, Any]:
    extraction_by_id = {item.observation_id: item for item in extraction.object_results}
    fitting_by_id = {item.observation_id: item for item in fitting.object_results}
    objects = []
    for object_index, item in enumerate(batch.object_results):
        ext = extraction_by_id[item.observation_id]
        fit = fitting_by_id[item.observation_id]
        objects.append(
            {
                "object_index": object_index,
                "observation_id": item.observation_id,
                "status": item.status.value,
                "warnings": list(item.warnings),
                "extraction": {
                    "status": ext.status.value,
                    "rejection_reasons": [value.value for value in ext.rejection_reasons],
                    "warnings": list(ext.warnings),
                    "stage_counts": ext.stage_counts.as_dict(),
                },
                "fitting": {
                    "status": fit.status.value,
                    "rejection_reasons": [value.value for value in fit.rejection_reasons],
                    "warnings": list(fit.warnings),
                    "preferred_box_id": fit.preferred_box_id,
                    "preferred_box_reason": fit.preferred_box_reason,
                    "boxes": _box_payload(fit),
                },
                "measurements": [
                    {
                        "kind": estimate.kind.value,
                        "status": estimate.status.value,
                        "value_si": estimate.value_si,
                        "unit_si": estimate.unit_si,
                        "expanded_uncertainty_si": (
                            estimate.uncertainty.expanded_uncertainty_si
                            if estimate.uncertainty
                            else None
                        ),
                        "lower_bound_si": estimate.lower_bound_si,
                        "upper_bound_si": estimate.upper_bound_si,
                        "reliability": estimate.reliability.level.value,
                        "rejection_reasons": [value.value for value in estimate.rejection_reasons],
                        "calibration_status": estimate.calibration_status.value,
                        "intrinsics_source": estimate.intrinsics_source.value,
                        "depth_scale": estimate.depth_scale.value,
                        "uncertainty_qualification": (
                            estimate.uncertainty.display_qualification
                            if estimate.uncertainty
                            else ""
                        ),
                    }
                    for estimate in item.measurement_result.estimates
                ],
            }
        )
    return {"validation_only": True, "metadata": metadata, "objects": objects}


def _resize_panel(image: np.ndarray, size: tuple[int, int], title: str) -> np.ndarray:
    width, height = size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (width, 34), (15, 15, 15), -1)
    cv2.putText(resized, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
    return resized


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(Path(args.image).expanduser()))
    if bgr is None:
        raise SystemExit(f"Unable to read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    frame = FrameReference(1, 1, 0.0, ImageSize(w, h))
    rgb_space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)

    started = time.perf_counter()
    perception = _run_perception(rgb, frame, rgb_space, args.perception_model, args.perception_profile)
    observations = sorted(
        perception.observations,
        key=lambda observation: int(np.count_nonzero(observation.selected_mask.payload.data))
        if observation.selected_mask is not None and hasattr(observation.selected_mask.payload, "data")
        else 0,
        reverse=True,
    )
    observations = [
        observation
        for observation in observations
        if observation.selected_mask is not None
        and hasattr(observation.selected_mask.payload, "data")
        and np.count_nonzero(observation.selected_mask.payload.data) >= args.minimum_mask_pixels
    ][: max(1, args.max_objects)]
    if not observations:
        raise SystemExit("No usable segmented objects were produced.")
    perception = type(perception)(
        frame_reference=perception.frame_reference,
        observations=tuple(observations),
        model_provenance=perception.model_provenance,
        status=perception.status,
        processing_ms=perception.processing_ms,
        warnings=perception.warnings,
        failure=perception.failure,
    )

    depth_result = _run_depth(bgr, args.depth_model, args.depth_profile)
    inference = _to_inference_result(depth_result)
    depth_input = depth_input_from_inference_result(inference, frame)
    depth_space = make_depth_space(frame, inference.depth)
    transform = (
        declare_identity_transform(rgb_space, depth_space)
        if rgb_space.size == depth_space.size
        else declare_full_frame_resize(rgb_space, depth_space)
    )
    aligned = align_frame(
        AlignmentRequest(frame, rgb, rgb_space, depth_input, perception, (transform,))
    )
    extraction = extract_object_point_clouds(make_extraction_request(aligned))
    fitting = fit_object_geometry(make_fitting_request(extraction))
    measurements = compute_measurements(
        make_measurement_request(aligned, extraction, fitting)
    )

    annotated = _annotate(rgb, observations)
    cv2.imwrite(str(output / "rgb_detections_masks.png"), annotated)
    depth_preview = _depth_preview(inference.depth, inference.valid_mask)
    cv2.imwrite(str(output / "depth_preview.png"), depth_preview)
    aligned_geometry = _annotate_aligned_geometry(aligned, fitting)
    cv2.imwrite(str(output / "aligned_geometry_boxes.png"), aligned_geometry)

    extraction_by_id = {item.observation_id: item for item in extraction.object_results}
    fitting_by_id = {item.observation_id: item for item in fitting.object_results}
    for index, computation in enumerate(measurements.object_results):
        ext = extraction_by_id[computation.observation_id]
        fit = fitting_by_id[computation.observation_id]
        lines = _diagnostic_lines(ext, fit) + [""] + _measurement_lines(computation)
        text = _text_panel(
            lines + ["", "Estimated Uncertainty - Not Calibrated"],
            900,
            520,
            f"Object {index}: {computation.observation_id}",
        )
        points = _points_panel(ext, fit, 900, 460)
        object_view = np.vstack((points, text))
        cv2.imwrite(
            str(output / f"object_{index:03d}_geometry_measurements.png"),
            object_view,
        )
        available = sum(
            estimate.status is MeasurementStatus.AVAILABLE
            for estimate in computation.measurement_result.estimates
        )
        print(
            f"object={index} id={computation.observation_id} "
            f"extraction={ext.status.value} points={ext.stage_counts.output_points} "
            f"fitting={fit.status.value} preferred={fit.preferred_box_id or '-'} "
            f"measurements={available}/{len(computation.measurement_result.estimates)}"
        )
        if ext.rejection_reasons:
            print(
                "  extraction_reasons="
                + ",".join(value.value for value in ext.rejection_reasons)
                + " counts="
                + json.dumps(ext.stage_counts.as_dict(), sort_keys=True)
            )

    panel_size = (600, 520)
    overview_top = np.hstack(
        (
            _resize_panel(annotated, panel_size, "RGB detections + masks"),
            _resize_panel(depth_preview, panel_size, "Depth preview"),
            _resize_panel(aligned_geometry, panel_size, "Aligned 3D boxes"),
        )
    )
    summary_lines = [
        f"depth_model={args.depth_model} depth_scale={inference.depth_scale} intrinsics={inference.intrinsics_source} confidence={inference.confidence_kind}",
        f"perception_model={args.perception_model} objects={len(measurements.object_results)}",
        f"alignment={aligned.status.value} extraction={extraction.status.value} fitting={fitting.status.value} measurements={measurements.status.value}",
        "Estimated Uncertainty - Not Calibrated; no coverage probability is claimed.",
    ]
    summary = _text_panel(
        summary_lines,
        1800,
        170,
        "Phase 7.5 real end-to-end validation (not production export)",
    )
    overview = np.vstack((overview_top, summary))
    cv2.imwrite(str(output / "overview.png"), overview)

    metadata = {
        "image": str(Path(args.image).expanduser()),
        "depth_model": args.depth_model,
        "perception_model": args.perception_model,
        "depth_scale": inference.depth_scale,
        "intrinsics_source": inference.intrinsics_source,
        "confidence_kind": inference.confidence_kind,
        "elapsed_seconds": time.perf_counter() - started,
        "object_count": len(measurements.object_results),
    }
    payload = _serialize(measurements, extraction, fitting, metadata)
    (output / "measurements.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    available = sum(
        estimate.status is MeasurementStatus.AVAILABLE
        for obj in measurements.object_results
        for estimate in obj.measurement_result.estimates
    )
    total = sum(
        len(obj.measurement_result.estimates)
        for obj in measurements.object_results
    )
    print(
        f"status={measurements.status.value} objects={len(measurements.object_results)} "
        f"available={available}/{total} output={output}"
    )
    print(f"overview={output / 'overview.png'}")
    print(f"aligned_boxes={output / 'aligned_geometry_boxes.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
