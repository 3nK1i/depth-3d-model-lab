from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def write_inspection_bundle(directory: Path, *, manual: bool = False, status: str = "partial") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    height, width = 48, 64
    mask = np.zeros((height, width), dtype=bool)
    mask[10:38, 18:50] = True
    points = np.array(
        [[-0.2, -0.1, 2.0], [0.2, -0.1, 2.0], [-0.2, 0.1, 2.2], [0.2, 0.1, 2.2]],
        dtype=np.float32,
    )
    colors = np.tile(np.array([[40, 180, 90]], dtype=np.uint8), (len(points), 1))
    arrays = {
        "aligned_rgb": np.full((height, width, 3), (28, 42, 58), dtype=np.uint8),
        "depth_preview": np.full((height, width, 3), (80, 120, 180), dtype=np.uint8),
        "object_000_mask": mask,
        "object_000_raw_points": points,
        "object_000_filtered_points": points,
        "object_000_raw_colors": colors,
        "object_000_filtered_colors": colors,
    }
    np.savez_compressed(directory / "inspection_arrays.npz", **arrays)
    (directory / "inspection_state.pkl").write_bytes(b"test-state")
    box = {
        "box_id": "object-0:oriented",
        "kind": "oriented",
        "centre_xyz_m": [0.0, 0.0, 2.1],
        "extents_xyz_m": [0.4, 0.2, 0.2],
        "orientation_matrix": np.eye(3).tolist(),
        "fit_method": "test",
        "fit_residual_m": 0.001,
        "support_point_count": 4,
        "corners_xyz_m": [
            [-0.2, -0.1, 2.0], [0.2, -0.1, 2.0], [-0.2, 0.1, 2.0], [0.2, 0.1, 2.0],
            [-0.2, -0.1, 2.2], [0.2, -0.1, 2.2], [-0.2, 0.1, 2.2], [0.2, 0.1, 2.2],
        ],
        "projected_segments": [[[18.0, 10.0], [50.0, 10.0]]],
    }
    measurement = {
        "kind": "width",
        "status": "available",
        "value_si": 0.4,
        "unit_si": "m",
        "expanded_uncertainty_si": 0.01,
        "lower_bound_si": 0.39,
        "upper_bound_si": 0.41,
        "reliability": "high",
        "reliability_score": 0.9,
        "valid_observation_count": 1,
        "method": "oriented box",
        "rejection_reasons": [],
        "warnings": [],
        "uncertainty_basis": "heuristic_perturbation",
        "uncertainty_qualification": "Estimated Uncertainty — Not Calibrated",
        "uncertainty_components": [],
    }
    payload = {
        "schema_version": 1,
        "status": status,
        "arrays_file": "inspection_arrays.npz",
        "state_file": "inspection_state.pkl",
        "image_size": {"width": width, "height": height},
        "models": {
            "depth": {"model_id": "depth-pro", "display_name": "Depth Pro"},
            "perception": {"model_id": "ultralytics-yolo-seg", "display_name": "YOLO Segmentation"},
        },
        "depth_scale": "metric",
        "intrinsics_source": "model_predicted",
        "calibration_status": "not_calibrated",
        "confidence_kind": "none",
        "box_method": "auto",
        "objects": [{
            "object_index": 0,
            "observation_id": "object-0",
            "class_id": 1,
            "class_name": "box",
            "detection_confidence": 0.91,
            "detection_box_xyxy": [18.0, 10.0, 50.0, 38.0],
            "track_id": "track-000001",
            "mask_key": "object_000_mask",
            "raw_points_key": "object_000_raw_points",
            "filtered_points_key": "object_000_filtered_points",
            "raw_colors_key": "object_000_raw_colors",
            "filtered_colors_key": "object_000_filtered_colors",
            "mask_source": "manual_correction" if manual else "native_instance",
            "mask_revision": 2 if manual else 0,
            "approval_status": "approved" if manual else "not_required",
            "status": "partial",
            "extraction_status": "success",
            "fitting_status": "success",
            "measurement_status": "partial",
            "preferred_box_id": "object-0:oriented",
            "preferred_box_reason": "stable oriented box",
            "boxes": {"oriented": box},
            "measurements": [measurement],
            "diagnostics": {"extraction_stage_counts": {"output_points": 4}},
            "warnings": [],
        }],
        "warnings": [],
        "elapsed_seconds": 0.25,
        "uncertainty_notice": "Estimated Uncertainty — Not Calibrated",
    }
    (directory / "inspection_bundle.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory
