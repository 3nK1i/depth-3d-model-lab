"""Read and validate temporary Step 8 inspection bundles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts.validation import ContractValidationError
from core.inspection.contracts import (
    InspectionBoxMethod,
    InspectionBundleSummary,
    InspectionObjectSummary,
    InspectionRunStatus,
)

BUNDLE_FILENAME = "inspection_bundle.json"
ARRAYS_FILENAME = "inspection_arrays.npz"
STATE_FILENAME = "inspection_state.pkl"


def _pairs(value: dict[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    return tuple((str(key), item) for key, item in (value or {}).items())


def load_inspection_bundle(bundle_dir: str | Path) -> tuple[InspectionBundleSummary, dict[str, np.ndarray]]:
    directory = Path(bundle_dir).expanduser().resolve()
    payload_path = directory / BUNDLE_FILENAME
    if not payload_path.is_file():
        raise ContractValidationError(f"Inspection bundle metadata is missing: {payload_path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    objects = []
    for value in payload.get("objects", []):
        box = value.get("detection_box_xyxy")
        objects.append(
            InspectionObjectSummary(
                observation_id=value["observation_id"],
                object_index=value["object_index"],
                class_id=value["class_id"],
                class_name=value["class_name"],
                detection_confidence=value.get("detection_confidence"),
                detection_box_xyxy=tuple(box) if box is not None else None,
                mask_key=value["mask_key"],
                raw_points_key=value["raw_points_key"],
                filtered_points_key=value["filtered_points_key"],
                raw_colors_key=value.get("raw_colors_key", ""),
                filtered_colors_key=value.get("filtered_colors_key", ""),
                mask_source=value.get("mask_source", ""),
                mask_revision=value.get("mask_revision", 0),
                approval_status=value.get("approval_status", ""),
                track_id=value.get("track_id", ""),
                status=value.get("status", ""),
                extraction_status=value.get("extraction_status", ""),
                fitting_status=value.get("fitting_status", ""),
                measurement_status=value.get("measurement_status", ""),
                preferred_box_id=value.get("preferred_box_id", ""),
                preferred_box_reason=value.get("preferred_box_reason", ""),
                boxes=_pairs(value.get("boxes")),
                measurements=tuple(value.get("measurements", ())),
                diagnostics=_pairs(value.get("diagnostics")),
                instance_name=str(value.get("instance_name", "")),
                semantic_source=str(value.get("semantic_source", "perception_model")),
                semantic_review_status=str(value.get("semantic_review_status", "")),
                warnings=tuple(value.get("warnings", ())),
            )
        )
    summary = InspectionBundleSummary(
        bundle_dir=directory,
        schema_version=payload.get("schema_version", 0),
        status=InspectionRunStatus(payload.get("status", "failed")),
        arrays_file=payload.get("arrays_file", ARRAYS_FILENAME),
        state_file=payload.get("state_file", STATE_FILENAME),
        image_width=payload["image_size"]["width"],
        image_height=payload["image_size"]["height"],
        depth_model_id=payload["models"]["depth"]["model_id"],
        depth_model_name=payload["models"]["depth"].get("display_name", payload["models"]["depth"]["model_id"]),
        perception_model_id=payload["models"]["perception"]["model_id"],
        perception_model_name=payload["models"]["perception"].get("display_name", payload["models"]["perception"]["model_id"]),
        depth_scale=payload.get("depth_scale", "unknown"),
        intrinsics_source=payload.get("intrinsics_source", "none"),
        calibration_status=payload.get("calibration_status", "unknown"),
        confidence_kind=payload.get("confidence_kind", "none"),
        box_method=InspectionBoxMethod(payload.get("box_method", "auto")),
        objects=tuple(objects),
        warnings=tuple(payload.get("warnings", ())),
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        calibration_profile_id=str(payload.get("calibration_profile_id", "")),
        calibration_record_id=str(payload.get("calibration_record_id", "")),
        calibration_method=str(payload.get("calibration_method", "")),
        calibration_validation_state=str(payload.get("calibration_validation_state", "")),
        calibration_application_mode=str(payload.get("calibration_application_mode", "")),
        calibration_rectified=bool(payload.get("calibration_rectified", False)),
        semantic_source=str(payload.get("semantic_source", "perception_model")),
        semantic_workspace_id=str(payload.get("semantic_workspace_id", "")),
        semantic_workspace_revision=int(payload.get("semantic_workspace_revision", 0)),
    )
    if not summary.arrays_path.is_file():
        raise ContractValidationError(f"Inspection array archive is missing: {summary.arrays_path}")
    with np.load(summary.arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    required = {"aligned_rgb", "depth_preview"}
    for obj in summary.objects:
        required.update({obj.mask_key, obj.raw_points_key, obj.filtered_points_key})
    missing = sorted(required.difference(arrays))
    if missing:
        raise ContractValidationError("Inspection bundle arrays are incomplete: " + ", ".join(missing))
    if arrays["aligned_rgb"].shape != (summary.image_height, summary.image_width, 3):
        raise ContractValidationError("aligned_rgb dimensions disagree with bundle metadata.")
    for obj in summary.objects:
        mask = arrays[obj.mask_key]
        if mask.shape != (summary.image_height, summary.image_width):
            raise ContractValidationError(f"Mask dimensions disagree for {obj.observation_id}.")
        arrays[obj.mask_key] = np.asarray(mask, dtype=bool)
    return summary, arrays
