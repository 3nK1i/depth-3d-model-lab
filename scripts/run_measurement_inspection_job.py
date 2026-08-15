#!/usr/bin/env python3
"""Isolated Phase 7.5 Step 8 measurement-inspection analysis job.

Initial runs execute perception and depth sequentially. Recompute runs load the
trusted temporary state produced by this script, apply one approved manual mask
revision and/or an operator-selected box method, then rerun only Steps 4-6.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import pickle
from pathlib import Path
import shutil
import sys
import tempfile
import time
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
    resample_boolean_mask,
    transform_intrinsics,
)
from core.alignment.calibration import (
    rectify_bgr_with_camera_calibration,
    rectify_boolean_mask_with_applied_calibration,
)
from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.inspection import ARRAYS_FILENAME, BUNDLE_FILENAME, STATE_FILENAME, InspectionBoxMethod
from core.geometry.projection import intrinsics_from_virtual_fov
from core.measurement import compute_measurements, make_measurement_request
from core.measurement.contracts import BoundingBox3D, BoundingBoxKind, MeasurementStatus
from core.measurement.policy import MeasurementComputationPolicy, MeasurementSelectionPolicy
from core.hardware.gpu_info import detect_hardware
from core.hardware.profiles import ComputeProfile, resolve_profile
from core.models.metadata import DepthScale
from core.models.registry import MODEL_REGISTRY
from core.models.runtime_config import ModelRuntimeContext
from core.models.validation import normalize_geometry_result
from core.reconstruction.calibration_profile import load_camera_calibration_profile
from core.runtime.messages import FrameRequest
from core.object_geometry import (
    extract_object_point_clouds,
    fit_object_geometry,
    make_extraction_request,
    make_fitting_request,
)
from core.object_geometry.policy import DepthDiscontinuityPolicy, ObjectPointExtractionPolicy
from core.perception.result import ApprovalStatus, MaskSource, PerceptionFrameResult
from core.perception.reviewed_analysis import (
    load_reviewed_semantic_selection,
    perception_from_reviewed_selection,
    source_image_sha256,
)
from scripts.run_real_measurement_validation import (
    _box_corners,
    _depth_preview,
    _run_depth,
    _run_perception,
    _to_inference_result,
)

_SCHEMA_VERSION = 1
_BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)


def _event(stage: str, progress: int, message: str, **extra: Any) -> None:
    payload = {"stage": stage, "progress": int(progress), "message": message, **extra}
    print("EVENT " + json.dumps(payload, separators=(",", ":")), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="BGR uint8 frame saved as .npy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth-model", default="depth-pro")
    parser.add_argument("--depth-config")
    parser.add_argument("--calibration-profile", help="Phase 9 camera-calibration profile JSON")
    parser.add_argument("--reviewed-semantics", help="Transient accepted Auto Segment semantic handoff JSON")
    parser.add_argument("--perception-model", default="ultralytics-yolo-seg")
    parser.add_argument("--depth-profile", default="efficient")
    parser.add_argument("--perception-profile", default="efficient")
    parser.add_argument("--max-objects", type=int, default=5)
    parser.add_argument("--minimum-mask-pixels", type=int, default=64)
    parser.add_argument("--box-method", choices=[value.value for value in InspectionBoxMethod], default="auto")
    parser.add_argument("--base-state", help="Trusted state file produced by a prior inspection run")
    parser.add_argument("--manual-mask", help="Aligned boolean mask saved as .npy")
    parser.add_argument("--observation-id", default="")
    parser.add_argument("--manual-revision", type=int, default=1)
    parser.add_argument("--approved-by", default="EIA TECH operator")
    parser.add_argument("--reset-manual", action="store_true", help="Restore the original automatic mask before recomputing")
    return parser


def _measurement_policy(box_method: InspectionBoxMethod) -> MeasurementComputationPolicy:
    return MeasurementComputationPolicy(
        selection=MeasurementSelectionPolicy(forced_box_kind=box_method.bounding_box_kind)
    )


def _project_camera_points(points_xyz: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-8)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = k[0, 0] * points[valid, 0] / z[valid] + k[0, 2]
    uv[valid, 1] = k[1, 1] * points[valid, 1] / z[valid] + k[1, 2]
    return uv, valid


def _box_payload(box: BoundingBox3D, intrinsics: np.ndarray) -> dict[str, Any]:
    corners = _box_corners(box)
    uv, valid = _project_camera_points(corners, intrinsics)
    segments = []
    for first, second in _BOX_EDGES:
        if valid[first] and valid[second]:
            segments.append([
                [float(uv[first, 0]), float(uv[first, 1])],
                [float(uv[second, 0]), float(uv[second, 1])],
            ])
    return {
        "box_id": box.box_id,
        "kind": box.kind.value,
        "centre_xyz_m": np.asarray(box.centre_xyz_m).tolist(),
        "extents_xyz_m": np.asarray(box.extents_xyz_m).tolist(),
        "orientation_matrix": np.asarray(box.orientation_matrix).tolist(),
        "fit_method": box.fit_method,
        "fit_residual_m": box.fit_residual_m,
        "support_point_count": box.support_point_count,
        "coordinate_frame_id": box.coordinate_frame_id,
        "corners_xyz_m": corners.tolist(),
        "projected_segments": segments,
    }


def _measurement_payload(estimate) -> dict[str, Any]:
    uncertainty = estimate.uncertainty
    return {
        "kind": estimate.kind.value,
        "status": estimate.status.value,
        "value_si": estimate.value_si,
        "unit_si": estimate.unit_si,
        "expanded_uncertainty_si": uncertainty.expanded_uncertainty_si if uncertainty else None,
        "lower_bound_si": estimate.lower_bound_si,
        "upper_bound_si": estimate.upper_bound_si,
        "reliability": estimate.reliability.level.value,
        "reliability_score": estimate.reliability.score,
        "valid_observation_count": estimate.valid_observation_count,
        "method": estimate.method,
        "rejection_reasons": [value.value for value in estimate.rejection_reasons],
        "warnings": list(estimate.warnings),
        "uncertainty_basis": uncertainty.basis.value if uncertainty else "unknown",
        "uncertainty_qualification": uncertainty.display_qualification if uncertainty else "",
        "uncertainty_components": [
            {
                "kind": component.kind.value,
                "contribution_si": component.contribution_si,
                "lower_contribution_si": component.lower_contribution_si,
                "upper_contribution_si": component.upper_contribution_si,
                "method": component.method,
                "sample_count": component.sample_count,
                "note": component.note,
            }
            for component in (uncertainty.components if uncertainty else ())
        ],
    }


def _observation_metadata(perception: PerceptionFrameResult) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for observation in perception.observations:
        detection = observation.detection
        result[observation.observation_id] = {
            "class_id": observation.class_id,
            "class_name": observation.class_name,
            "detection_confidence": detection.confidence if detection is not None else None,
            "detection_box_xyxy": list(detection.box.xyxy) if detection is not None else None,
            "track_id": observation.track_id,
        }
    return result


def _apply_manual_mask(aligned, observation_id: str, mask: np.ndarray, revision: int):
    if not observation_id:
        raise ValueError("A manual mask requires observation_id.")
    expected = (aligned.target_depth_space.size.height, aligned.target_depth_space.size.width)
    dense = np.asarray(mask, dtype=bool)
    if dense.shape != expected:
        raise ValueError(f"Manual mask shape {dense.shape} does not match aligned image {expected}.")
    if not np.any(dense):
        raise ValueError("Manual mask must contain foreground pixels.")
    found = False
    observations = []
    for observation in aligned.aligned_observations:
        if observation.observation_id != observation_id:
            observations.append(observation)
            continue
        found = True
        source_pixels = max(observation.source_foreground_pixels, int(np.count_nonzero(dense)))
        observations.append(
            replace(
                observation,
                source_mask_id=f"{observation.source_mask_id or observation.observation_id}:manual:r{revision}",
                aligned_mask=dense,
                source_foreground_pixels=source_pixels,
                aligned_foreground_pixels=int(np.count_nonzero(dense)),
                foreground_retention_ratio=float(np.count_nonzero(dense)) / max(source_pixels, 1),
                mask_source=MaskSource.MANUAL_CORRECTION,
                mask_revision=int(revision),
                approval_status=ApprovalStatus.APPROVED,
                warnings=tuple(dict.fromkeys((*observation.warnings, "approved_manual_mask_revision"))),
            )
        )
    if not found:
        raise ValueError(f"Unknown observation_id for manual correction: {observation_id}")
    return replace(aligned, aligned_observations=tuple(observations))


def _compute_from_aligned(aligned, box_method: InspectionBoxMethod):
    extraction_policy = (
        ObjectPointExtractionPolicy(discontinuity=DepthDiscontinuityPolicy(enabled=False))
        if aligned.depth_scale is DepthScale.SYNTHETIC
        else None
    )
    extraction = extract_object_point_clouds(
        make_extraction_request(aligned, policy=extraction_policy)
    )
    fitting = fit_object_geometry(make_fitting_request(extraction))
    measurements = compute_measurements(
        make_measurement_request(
            aligned,
            extraction,
            fitting,
            policy=_measurement_policy(box_method),
        )
    )
    return extraction, fitting, measurements



def _run_depth_configured(bgr: np.ndarray, model_id: str, profile: str, config_path: str | None):
    if not config_path:
        return _run_depth(bgr, model_id, profile)
    descriptor = MODEL_REGISTRY.get(model_id)
    hardware = detect_hardware()
    resolved = resolve_profile(ComputeProfile(profile), hardware)
    config = MODEL_REGISTRY.resolve_runtime_config(
        model_id,
        ModelRuntimeContext(hardware_info=hardware, resolved_profile=resolved),
    )
    supplied = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("depth-config must contain a JSON object.")
    config.update(supplied)
    adapter = MODEL_REGISTRY.build(model_id)
    adapter.load(
        device=str(config.get("device", resolved.device)),
        precision=str(config.get("precision", resolved.precision.value)),
        config=config,
    )
    adapter.warmup()
    request = FrameRequest(1, 1, bgr, source_timestamp=0.0)
    try:
        return normalize_geometry_result(adapter.infer(request), descriptor, request)
    finally:
        adapter.unload()

def _initial_analysis(args):
    if not args.input:
        raise ValueError("Initial inspection analysis requires --input.")
    bgr = np.load(args.input, allow_pickle=False)
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Inspection input must be an HxWx3 BGR image.")
    source_frame_sha256 = source_image_sha256(bgr)

    reviewed_selection = None
    if args.reviewed_semantics:
        reviewed_selection = load_reviewed_semantic_selection(args.reviewed_semantics)
        if reviewed_selection.source_image_sha256 != source_image_sha256(bgr):
            raise ValueError("Reviewed semantic masks belong to a different source frame.")
        if len(reviewed_selection.objects) > int(args.max_objects):
            raise ValueError(
                f"Reviewed semantic input contains {len(reviewed_selection.objects)} object(s), "
                f"exceeding --max-objects={int(args.max_objects)}."
            )

    calibration_meta: dict[str, Any] = {
        "calibration_profile_id": "",
        "calibration_record_id": "",
        "calibration_method": "",
        "calibration_validation_state": "",
        "calibration_application_mode": "",
        "calibration_rectified": False,
        "calibration_warnings": [],
    }
    rectification_support = None
    applied_calibration = None
    if args.calibration_profile:
        profile = load_camera_calibration_profile(args.calibration_profile)
        if profile.validation_state.value == "rejected":
            raise ValueError("Rejected camera-calibration profiles cannot be used for measurement inspection.")
        prepared = rectify_bgr_with_camera_calibration(
            bgr,
            profile,
            allow_same_aspect_full_frame_resize=True,
            target_pixel_space_id="rgb-rectified",
        )
        bgr = np.ascontiguousarray(prepared.image_bgr, dtype=np.uint8)
        rectification_support = np.asarray(prepared.support_mask, dtype=np.bool_)
        applied_calibration = prepared.application
        calibration_meta.update({
            "calibration_profile_id": profile.profile_id,
            "calibration_record_id": profile.calibration_record_id,
            "calibration_method": profile.method.value,
            "calibration_validation_state": profile.validation_state.value,
            "calibration_application_mode": prepared.application.mode.value,
            "calibration_rectified": bool(prepared.rectified),
            "calibration_warnings": list(prepared.warnings),
        })
        _event(
            "calibration",
            4,
            "Applied camera calibration to the captured measurement frame",
            profile_id=profile.profile_id,
            validation_state=profile.validation_state.value,
            application_mode=prepared.application.mode.value,
            rectified=bool(prepared.rectified),
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    frame = FrameReference(1, 1, 0.0, ImageSize(width, height))
    rgb_space_id = (
        applied_calibration.camera.intrinsics.pixel_space_id
        if applied_calibration is not None
        else "rgb"
    )
    rgb_space = PixelCoordinateSpace(rgb_space_id, frame, frame.source_size, PixelSpaceRole.RGB)

    if reviewed_selection is not None:
        reviewed_masks = {item.object_id: np.asarray(item.mask, dtype=np.bool_) for item in reviewed_selection.objects}
        transformed = False
        if applied_calibration is not None and bool(calibration_meta["calibration_rectified"]):
            reviewed_masks = {
                item.object_id: np.logical_and(
                    rectify_boolean_mask_with_applied_calibration(item.mask, applied_calibration),
                    rectification_support,
                )
                for item in reviewed_selection.objects
            }
            transformed = True
        too_small = [
            item.object_id for item in reviewed_selection.objects
            if int(np.count_nonzero(reviewed_masks[item.object_id])) < int(args.minimum_mask_pixels)
        ]
        if too_small:
            raise RuntimeError(
                "Reviewed semantic mask(s) became too small for analysis: " + ", ".join(too_small)
            )
        perception, metadata = perception_from_reviewed_selection(
            reviewed_selection,
            frame,
            rgb_space,
            masks_by_object_id=reviewed_masks,
            transformed=transformed,
        )
        observations = list(perception.observations)
        _event(
            "perception",
            8,
            "Using accepted reviewed Auto Segment masks; perception model inference skipped",
            object_count=len(observations),
            workspace_id=reviewed_selection.workspace_id,
        )
    else:
        _event("perception", 8, "Loading the perception model")
        perception = _run_perception(rgb, frame, rgb_space, args.perception_model, args.perception_profile)
        observations = sorted(
            perception.observations,
            key=lambda observation: int(np.count_nonzero(observation.selected_mask.payload.data))
            if observation.selected_mask is not None and hasattr(observation.selected_mask.payload, "data")
            else 0,
            reverse=True,
        )
        observations = [
            observation for observation in observations
            if observation.selected_mask is not None
            and hasattr(observation.selected_mask.payload, "data")
            and np.count_nonzero(observation.selected_mask.payload.data) >= args.minimum_mask_pixels
        ][: max(1, args.max_objects)]
        if not observations:
            raise RuntimeError("No usable segmented objects were produced.")
        perception = replace(perception, observations=tuple(observations))
        metadata = _observation_metadata(perception)

    _event("depth", 30, "Semantic input ready; loading the depth model", object_count=len(observations))
    depth_result = _run_depth_configured(bgr, args.depth_model, args.depth_profile, args.depth_config)
    inference = _to_inference_result(depth_result)
    depth_space = make_depth_space(frame, inference.depth)
    transform = (
        declare_identity_transform(rgb_space, depth_space)
        if rgb_space.size == depth_space.size
        else declare_full_frame_resize(rgb_space, depth_space)
    )

    calibrated_depth_intrinsics = None
    if applied_calibration is not None:
        calibrated_depth_intrinsics = transform_intrinsics(
            applied_calibration.camera.intrinsics,
            transform,
        )
    elif inference.intrinsics is None:
        depth_height, depth_width = inference.depth.shape
        inference = replace(
            inference,
            intrinsics=intrinsics_from_virtual_fov(depth_width, depth_height),
            intrinsics_source="estimated",
        )

    depth_input = depth_input_from_inference_result(
        inference,
        frame,
        intrinsics_override=calibrated_depth_intrinsics,
    )

    _event("alignment", 55, "Aligning RGB, masks, depth and intrinsics")
    aligned = align_frame(AlignmentRequest(frame, rgb, rgb_space, depth_input, perception, (transform,)))
    if rectification_support is not None:
        aligned_support = resample_boolean_mask(rectification_support, transform)
        aligned = replace(
            aligned,
            rgb_support_mask=np.logical_and(aligned.rgb_support_mask, aligned_support),
            warnings=tuple(dict.fromkeys((*aligned.warnings, *calibration_meta["calibration_warnings"]))),
        )
    preview = cv2.cvtColor(_depth_preview(inference.depth, inference.valid_mask), cv2.COLOR_BGR2RGB)
    return aligned, metadata, preview, {
        "depth_model_id": args.depth_model,
        "depth_model_name": aligned.depth_model_provenance.display_name,
        "perception_model_id": ("reviewed-auto-segment" if reviewed_selection is not None else args.perception_model),
        "perception_model_name": perception.model_provenance[0].display_name if perception.model_provenance else args.perception_model,
        "semantic_source": ("reviewed_auto_segment" if reviewed_selection is not None else "perception_model"),
        "semantic_workspace_id": reviewed_selection.workspace_id if reviewed_selection is not None else "",
        "semantic_workspace_revision": reviewed_selection.workspace_revision if reviewed_selection is not None else 0,
        "source_frame_sha256": source_frame_sha256,
        **calibration_meta,
    }


def _load_recompute_state(args):
    if not args.base_state:
        raise ValueError("Recompute requires --base-state.")
    state_path = Path(args.base_state).expanduser().resolve()
    with state_path.open("rb") as handle:
        state = pickle.load(handle)  # trusted file created by this application
    aligned = state.get("automatic_aligned", state["aligned"]) if args.reset_manual else state["aligned"]
    if args.manual_mask:
        _event("manual_mask", 18, "Applying approved manual mask revision")
        aligned = _apply_manual_mask(
            aligned,
            args.observation_id,
            np.load(args.manual_mask, allow_pickle=False),
            args.manual_revision,
        )
    return aligned, state["observation_metadata"], state["depth_preview"], state["models"], state.get("automatic_aligned", aligned)


def _write_bundle(output: Path, aligned, extraction, fitting, measurements, observation_metadata, depth_preview, models, box_method, elapsed: float, *, automatic_aligned=None) -> None:
    arrays: dict[str, np.ndarray] = {
        "aligned_rgb": np.asarray(aligned.aligned_rgb, dtype=np.uint8),
        "depth_preview": np.asarray(depth_preview, dtype=np.uint8),
    }
    extraction_by_id = {value.observation_id: value for value in extraction.object_results}
    fitting_by_id = {value.observation_id: value for value in fitting.object_results}
    measurement_by_id = {value.observation_id: value for value in measurements.object_results}
    objects: list[dict[str, Any]] = []

    for index, observation in enumerate(aligned.aligned_observations):
        oid = observation.observation_id
        ext = extraction_by_id[oid]
        fit = fitting_by_id[oid]
        computation = measurement_by_id[oid]
        meta = observation_metadata.get(oid, {})
        prefix = f"object_{index:03d}"
        mask_key = f"{prefix}_mask"
        raw_points_key = f"{prefix}_raw_points"
        filtered_points_key = f"{prefix}_filtered_points"
        raw_colors_key = f"{prefix}_raw_colors"
        filtered_colors_key = f"{prefix}_filtered_colors"
        arrays[mask_key] = np.asarray(observation.aligned_mask, dtype=bool) if observation.aligned_mask is not None else np.zeros((aligned.target_depth_space.size.height, aligned.target_depth_space.size.width), dtype=bool)
        raw_cloud = ext.raw_cloud
        filtered_cloud = ext.filtered_cloud
        arrays[raw_points_key] = np.asarray(raw_cloud.points_xyz_m if raw_cloud is not None else np.zeros((0, 3)), dtype=np.float32)
        arrays[filtered_points_key] = np.asarray(filtered_cloud.points_xyz_m if filtered_cloud is not None else np.zeros((0, 3)), dtype=np.float32)
        arrays[raw_colors_key] = np.asarray(raw_cloud.colors_rgb if raw_cloud is not None and raw_cloud.colors_rgb is not None else np.zeros((len(arrays[raw_points_key]), 3), dtype=np.uint8), dtype=np.uint8)
        arrays[filtered_colors_key] = np.asarray(filtered_cloud.colors_rgb if filtered_cloud is not None and filtered_cloud.colors_rgb is not None else np.zeros((len(arrays[filtered_points_key]), 3), dtype=np.uint8), dtype=np.uint8)

        boxes: dict[str, Any] = {}
        for name, box in (
            ("axis_aligned", fit.axis_aligned_box),
            ("camera_aligned", fit.camera_aligned_box),
            ("oriented", fit.oriented_box),
        ):
            if box is not None:
                boxes[name] = _box_payload(box, aligned.intrinsics.matrix)
        measurements_payload = [_measurement_payload(value) for value in computation.measurement_result.estimates]
        objects.append({
            "object_index": index,
            "observation_id": oid,
            "class_id": int(meta.get("class_id", observation.class_id)),
            "class_name": str(meta.get("class_name", observation.class_name)),
            "instance_name": str(meta.get("instance_name", "")),
            "semantic_source": str(meta.get("semantic_source", models.get("semantic_source", "perception_model"))),
            "semantic_review_status": str(meta.get("semantic_review_status", "")),
            "detection_confidence": meta.get("detection_confidence"),
            "detection_box_xyxy": meta.get("detection_box_xyxy"),
            "track_id": meta.get("track_id", ""),
            "mask_key": mask_key,
            "raw_points_key": raw_points_key,
            "filtered_points_key": filtered_points_key,
            "raw_colors_key": raw_colors_key,
            "filtered_colors_key": filtered_colors_key,
            "mask_source": observation.mask_source.value if observation.mask_source is not None else "",
            "mask_revision": observation.mask_revision,
            "approval_status": observation.approval_status.value,
            "status": computation.status.value,
            "extraction_status": ext.status.value,
            "fitting_status": fit.status.value,
            "measurement_status": computation.status.value,
            "preferred_box_id": fit.preferred_box_id,
            "preferred_box_reason": fit.preferred_box_reason,
            "boxes": boxes,
            "measurements": measurements_payload,
            "diagnostics": {
                "extraction_stage_counts": ext.stage_counts.as_dict(),
                "extraction_rejection_reasons": [value.value for value in ext.rejection_reasons],
                "fitting_rejection_reasons": [value.value for value in fit.rejection_reasons],
                "measurement_available_kinds": [value.value for value in computation.diagnostics.available_kinds],
                "measurement_unavailable_kinds": [value.value for value in computation.diagnostics.unavailable_kinds],
                "measurement_rejected_kinds": [value.value for value in computation.diagnostics.rejected_kinds],
                "perturbation_sample_count": computation.diagnostics.perturbation_sample_count,
            },
            "warnings": list(dict.fromkeys((*observation.warnings, *ext.warnings, *fit.warnings, *computation.warnings))),
        })

    np.savez_compressed(output / ARRAYS_FILENAME, **arrays)
    state = {
        "aligned": aligned,
        "automatic_aligned": automatic_aligned if automatic_aligned is not None else aligned,
        "extraction": extraction,
        "fitting": fitting,
        "measurements": measurements,
        "observation_metadata": observation_metadata,
        "depth_preview": np.asarray(depth_preview, dtype=np.uint8),
        "models": models,
        "box_method": box_method.value,
    }
    with (output / STATE_FILENAME).open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    status = measurements.status.value
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "status": status,
        "arrays_file": ARRAYS_FILENAME,
        "state_file": STATE_FILENAME,
        "image_size": {"width": aligned.target_depth_space.size.width, "height": aligned.target_depth_space.size.height},
        "models": {
            "depth": {"model_id": models["depth_model_id"], "display_name": models["depth_model_name"]},
            "perception": {"model_id": models["perception_model_id"], "display_name": models["perception_model_name"]},
        },
        "depth_scale": aligned.depth_scale.value,
        "intrinsics_source": aligned.intrinsics.source.value,
        "calibration_status": aligned.intrinsics.calibration_status.value,
        "confidence_kind": aligned.confidence_kind.value,
        "box_method": box_method.value,
        "objects": objects,
        "warnings": list(aligned.warnings) + list(measurements.warnings),
        "elapsed_seconds": elapsed,
        "uncertainty_notice": (
            "Estimated Uncertainty — Calibrated camera geometry; depth/scale uncertainty still applies"
            if aligned.intrinsics.calibration_status.value == "calibrated"
            else "Estimated Uncertainty — Not Calibrated"
        ),
        "calibration_profile_id": str(models.get("calibration_profile_id", "")),
        "calibration_record_id": str(models.get("calibration_record_id", "")),
        "calibration_method": str(models.get("calibration_method", "")),
        "calibration_validation_state": str(models.get("calibration_validation_state", "")),
        "calibration_application_mode": str(models.get("calibration_application_mode", "")),
        "calibration_rectified": bool(models.get("calibration_rectified", False)),
        "semantic_source": str(models.get("semantic_source", "perception_model")),
        "semantic_workspace_id": str(models.get("semantic_workspace_id", "")),
        "semantic_workspace_revision": int(models.get("semantic_workspace_revision", 0)),
        "source_frame_sha256": str(models.get("source_frame_sha256", "")),
    }
    (output / BUNDLE_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _atomic_output(target: Path):
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(parent)))
    return temporary


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    target = Path(args.output).expanduser().resolve()
    temporary = _atomic_output(target)
    try:
        box_method = InspectionBoxMethod(args.box_method)
        if args.base_state:
            aligned, metadata, depth_preview, models, automatic_aligned = _load_recompute_state(args)
        else:
            aligned, metadata, depth_preview, models = _initial_analysis(args)
            automatic_aligned = aligned
        _event("geometry", 68, "Extracting object-specific 3D points")
        extraction, fitting, measurements = _compute_from_aligned(aligned, box_method)
        _event("bundle", 90, "Building the professional inspection bundle")
        _write_bundle(
            temporary,
            aligned,
            extraction,
            fitting,
            measurements,
            metadata,
            depth_preview,
            models,
            box_method,
            time.perf_counter() - started,
            automatic_aligned=automatic_aligned,
        )
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
        _event("complete", 100, "Inspection analysis completed", output=str(target), object_count=len(measurements.object_results))
        print(f"RESULT {target}", flush=True)
        return 0
    except Exception as exc:
        _event("failed", 100, str(exc))
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
