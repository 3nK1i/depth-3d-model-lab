from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_forward_motion_kitti import (
    build_i5_4a_forward_motion_kitti_admission,
    build_i5_4a_source_identity,
)
from tests.i5_4a_helpers import (
    I53C_SHA,
    RECEIPT_54A_SHA,
    kitti_source_dir,
    passing_i5_3c,
    passing_i5_4a_receipt,
)

I54A_SHA = "4" * 64
RECEIPT_54B_SHA = "5" * 64


def clone(value):
    return deepcopy(value)


def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def passing_i5_4a(tmp_path: Path) -> dict:
    source = kitti_source_dir(tmp_path, count=80)
    return build_i5_4a_forward_motion_kitti_admission(
        i5_3c_result=passing_i5_3c(tmp_path),
        i5_3c_binding=binding("/evidence/i5_3c.json", I53C_SHA),
        source_identity=build_i5_4a_source_identity(source),
        source_label="KITTI Raw synthetic image_02 left color",
        kitti_style_forward_motion_confirmation=True,
        admission_receipt=passing_i5_4a_receipt(),
        admission_receipt_binding=binding("/evidence/i5_4a_receipt.json", RECEIPT_54A_SHA),
    )


def passing_i5_4b_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.4B_REPLAY_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_4b_tests": True,
            "predecessor_i5_4a_regression": True,
            "production_native_sfm_contract_regression": True,
            "forward_motion_geometry_regression": True,
            "i2_4_scale_invariant_geometry_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "real_sfm_executed_by_test_receipt": False,
            "dense_provider_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 18,
            "predecessor_unit_tests": 42,
            "predecessor_integration_tests": 10,
            "predecessor_stress_tests": 10,
            "predecessor_total_tests": 62,
            "combined_exercised_tests": 80,
        },
        "return_codes": {
            "focused": 0,
            "predecessor_unit": 0,
            "predecessor_integration": 0,
            "predecessor_stress": 0,
            "import_probe": 0,
            "runner_help_probe": 0,
        },
    }


def synthetic_i54b_sfm_report(
    admission: dict,
    *,
    forward_step: float = 0.25,
    lateral_step: float = 0.02,
    registered_count: int | None = None,
    quality_status: str = "accepted",
    matching_strategy: str = "sequential",
) -> dict:
    plan = admission["stress_plan"]["forward_motion_window"]
    names = list(plan["names"])
    count = len(names)
    reg = count if registered_count is None else int(registered_count)
    registered_names = names[:reg]
    keyframes = {name: f"kf-{i:03d}" for i, name in enumerate(names)}

    workspace_images = [
        {
            "order": i,
            "keyframe_id": keyframes[name],
            "frame_id": f"frame-{i:03d}",
            "source_sequence_index": int(plan["indices"][i]),
            "timestamp_ns": i * 100_000_000,
            "source_locator": f"/scenario/{name}",
            "staged_image_path": f"images/{name}",
            "staged_sha256": f"{i + 1:064x}"[-64:],
            "image_size": {"width": 1242, "height": 375},
        }
        for i, name in enumerate(names)
    ]
    cameras = []
    for i, name in enumerate(registered_names):
        matrix = [
            [1.0, 0.0, 0.0, float(i * lateral_step)],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, float(i * forward_step)],
            [0.0, 0.0, 0.0, 1.0],
        ]
        cameras.append({
            "estimate_id": f"cam-{i:03d}",
            "camera_id": "camera-0",
            "keyframe_id": keyframes[name],
            "frame_id": f"frame-{i:03d}",
            "backend_image_id": i + 1,
            "T_reconstruction_from_camera": matrix,
        })

    landmarks = []
    lid = 0
    for i in range(max(0, reg - 1)):
        for j in range(8):
            z = 40.0 + i * forward_step + j * 0.05
            x = i * lateral_step + (j - 4) * 0.02
            lm = f"lm-{lid:05d}"
            landmarks.append({
                "landmark_id": lm,
                "track_id": lm,
                "position_reconstruction": [x, 0.2, z],
                "observations": [
                    {"observation_id": f"{lm}-a", "landmark_id": lm, "keyframe_id": keyframes[registered_names[i]], "feature_xy": [500.0, 180.0], "feature_index": j, "backend_point2d_index": j},
                    {"observation_id": f"{lm}-b", "landmark_id": lm, "keyframe_id": keyframes[registered_names[i + 1]], "feature_xy": [502.0, 180.0], "feature_index": j, "backend_point2d_index": j},
                ],
                "color_rgb": [180, 180, 180],
            })
            lid += 1
    observations = sum(len(item["observations"]) for item in landmarks)
    qa = {
        "status": quality_status,
        "usable": quality_status != "rejected",
        "registration_ratio": reg / count,
        "landmarks_per_registered_keyframe": (len(landmarks) / reg) if reg else 0.0,
        "mean_track_length": (observations / len(landmarks)) if landmarks else 0.0,
        "mean_reprojection_error_px": 0.7,
        "reasons": [],
    }
    return {
        "phase": "8.3D",
        "real_sfm_executed": True,
        "source_kind": "image_sequence",
        "registration_ratio": reg / count,
        "quality_assessment": deepcopy(qa),
        "sfm_result": {
            "schema_version": 1,
            "backend_id": "pycolmap-offline-sfm",
            "backend_version": "4.1.1",
            "gauge": {"scale_status": "arbitrary", "alignment_status": "arbitrary", "unit_to_metre": None, "scale_source": "", "relative_scale_uncertainty": None},
            "cameras": cameras,
            "camera_models": [{"camera_id": "camera-0", "backend_camera_id": 1, "backend_model_name": "PINHOLE", "image_size": {"width": 1242, "height": 375}, "parameters": [700.0, 700.0, 621.0, 187.0]}],
            "landmarks": landmarks,
            "statistics": {"input_keyframes": count, "registered_keyframes": reg, "landmarks": len(landmarks), "observations": observations, "elapsed_ns": 1_000_000, "mean_reprojection_error_px": 0.7},
            "workspace": {"schema_version": 1, "session_id": "00000000-0000-4000-8000-000000000001", "source": {"kind": "image_sequence"}, "workspace_root": "/scenario/workspace", "paths": {"images_directory": "images", "database_path": "database.db", "sparse_directory": "sparse", "checkpoints_directory": "checkpoints", "reports_directory": "reports", "logs_directory": "logs"}, "images": workspace_images},
            "warnings": [],
            "metadata": {"real_sfm": True, "matching_strategy": matching_strategy, "resume_policy": "reuse_completed_stages", "resume_from_stage": "none", "quality_assessment": deepcopy(qa)},
        },
    }
