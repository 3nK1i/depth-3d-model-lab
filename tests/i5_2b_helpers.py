from __future__ import annotations

from copy import deepcopy
from math import radians, tan
from pathlib import Path
from typing import Iterable

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import (
    build_i5_2a_low_overlap_weak_parallax_admission,
    build_ordered_source_identity,
)
from tests.i5_2a_helpers import (
    G2A3_DOC_SHA,
    G2B_DOC_SHA,
    G2C_DOC_SHA,
    G2_FINAL_SHA,
    I51C_SHA,
    RECEIPT_52A_SHA,
    g2a3_doc,
    g2_final,
    g2b_doc,
    g2c_doc,
    make_ordered_dataset,
    passing_i5_1c,
    passing_receipt,
)

I52A_SHA = "7" * 64
RECEIPT_52B_SHA = "8" * 64


def clone(value):
    return deepcopy(value)


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def passing_i5_2a(tmp_path: Path) -> dict:
    dataset = make_ordered_dataset(tmp_path / "castle-P19")
    return build_i5_2a_low_overlap_weak_parallax_admission(
        i5_1c_result=passing_i5_1c(),
        i5_1c_binding=_binding("/evidence/i5_1c.json", I51C_SHA),
        i4_3g2_final_result=g2_final(),
        i4_3g2_final_binding=_binding("/evidence/i4_3g2_final.json", G2_FINAL_SHA),
        i4_3g2a3_doc=g2a3_doc(),
        i4_3g2a3_binding=_binding("/repo/docs/i4_3g2a3.md", G2A3_DOC_SHA),
        i4_3g2b_doc=g2b_doc(),
        i4_3g2b_binding=_binding("/repo/docs/i4_3g2b.md", G2B_DOC_SHA),
        i4_3g2c_doc=g2c_doc(),
        i4_3g2c_binding=_binding("/repo/docs/i4_3g2c.md", G2C_DOC_SHA),
        source_identity=build_ordered_source_identity(dataset),
        semantic_confirmation=True,
        admission_receipt=passing_receipt(),
        admission_receipt_binding=_binding("/evidence/i5_2a_receipt.json", RECEIPT_52A_SHA),
    )


def passing_i5_2b_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.2B_REPLAY_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_2b_tests": True,
            "predecessor_i5_2a_regression": True,
            "production_native_sfm_contract_regression": True,
            "scale_invariant_geometry_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "real_sfm_executed_by_test_receipt": False,
            "dense_provider_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 30,
            "predecessor_unit_tests": 35,
            "predecessor_integration_tests": 10,
            "predecessor_stress_tests": 10,
            "predecessor_total_tests": 55,
            "combined_exercised_tests": 85,
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


def _pose(center_x: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, float(center_x)],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _scenario_names(i5_2a: dict, scenario_key: str) -> tuple[str, ...]:
    return tuple(i5_2a["stress_plan"][scenario_key]["names"])


def synthetic_sfm_report(
    i5_2a: dict,
    scenario_key: str,
    *,
    camera_step: float,
    common_landmarks_per_adjacent_pair: int,
    registered_count: int | None = None,
    quality_status: str = "accepted",
) -> dict:
    names = _scenario_names(i5_2a, scenario_key)
    count = len(names)
    reg = count if registered_count is None else int(registered_count)
    registered_names = names[:reg]
    keyframes = {name: f"kf-{scenario_key}-{i}" for i, name in enumerate(names)}

    workspace_images = [
        {
            "order": i,
            "keyframe_id": keyframes[name],
            "frame_id": f"frame-{scenario_key}-{i}",
            "source_sequence_index": i,
            "timestamp_ns": i * 100_000_000,
            "source_locator": f"/scenario/{name}",
            "staged_image_path": f"images/{name}",
            "staged_sha256": f"{i + 1:064x}"[-64:],
            "image_size": {"width": 96, "height": 72},
        }
        for i, name in enumerate(names)
    ]
    cameras = [
        {
            "estimate_id": f"cam-{i}",
            "camera_id": "camera-0",
            "keyframe_id": keyframes[name],
            "frame_id": f"frame-{scenario_key}-{i}",
            "backend_image_id": i + 1,
            "T_reconstruction_from_camera": _pose(i * camera_step),
        }
        for i, name in enumerate(registered_names)
    ]

    landmarks = []
    landmark_id = 0
    # Pair-specific landmarks make adjacent co-visibility directly controllable.
    for i in range(max(0, reg - 1)):
        for j in range(common_landmarks_per_adjacent_pair):
            z = 8.0 + 0.01 * j
            x = (i + 0.5) * camera_step
            lid = f"lm-{scenario_key}-{landmark_id}"
            landmarks.append(
                {
                    "landmark_id": lid,
                    "track_id": lid,
                    "position_reconstruction": [x, 0.0, z],
                    "observations": [
                        {
                            "observation_id": f"obs-{lid}-a",
                            "landmark_id": lid,
                            "keyframe_id": keyframes[registered_names[i]],
                            "feature_xy": [20.0, 20.0],
                            "feature_index": j,
                            "backend_point2d_index": j,
                        },
                        {
                            "observation_id": f"obs-{lid}-b",
                            "landmark_id": lid,
                            "keyframe_id": keyframes[registered_names[i + 1]],
                            "feature_xy": [22.0, 20.0],
                            "feature_index": j,
                            "backend_point2d_index": j,
                        },
                    ],
                    "color_rgb": [180, 180, 180],
                }
            )
            landmark_id += 1

    observations = sum(len(item["observations"]) for item in landmarks)
    metadata = {
        "real_sfm": True,
        "matching_strategy": "exhaustive",
        "resume_policy": "reuse_completed_stages",
        "resume_from_stage": "none",
        "quality_assessment": {
            "status": quality_status,
            "usable": quality_status != "rejected",
            "registration_ratio": reg / count,
            "landmarks_per_registered_keyframe": (len(landmarks) / reg) if reg else 0.0,
            "mean_track_length": (observations / len(landmarks)) if landmarks else 0.0,
            "mean_reprojection_error_px": 0.8,
            "reasons": [],
        },
    }
    return {
        "phase": "8.3D",
        "real_sfm_executed": True,
        "source_kind": "image_sequence",
        "registration_ratio": reg / count,
        "quality_assessment": deepcopy(metadata["quality_assessment"]),
        "sfm_result": {
            "schema_version": 1,
            "backend_id": "pycolmap-offline-sfm",
            "backend_version": "4.1.1",
            "gauge": {
                "scale_status": "arbitrary",
                "alignment_status": "arbitrary",
                "unit_to_metre": None,
                "scale_source": "",
                "relative_scale_uncertainty": None,
            },
            "cameras": cameras,
            "camera_models": [
                {
                    "camera_id": "camera-0",
                    "backend_camera_id": 1,
                    "backend_model_name": "SIMPLE_RADIAL",
                    "image_size": {"width": 96, "height": 72},
                    "parameters": [100.0, 48.0, 36.0, 0.0],
                }
            ],
            "landmarks": landmarks,
            "statistics": {
                "input_keyframes": count,
                "registered_keyframes": reg,
                "landmarks": len(landmarks),
                "observations": observations,
                "elapsed_ns": 1_000_000,
                "mean_reprojection_error_px": 0.8,
            },
            "workspace": {
                "schema_version": 1,
                "session_id": "00000000-0000-4000-8000-000000000001",
                "source": {"kind": "image_sequence"},
                "workspace_root": "/scenario/workspace",
                "paths": {
                    "images_directory": "images",
                    "database_path": "database.db",
                    "sparse_directory": "sparse",
                    "checkpoints_directory": "checkpoints",
                    "reports_directory": "reports",
                    "logs_directory": "logs",
                },
                "images": workspace_images,
            },
            "warnings": [],
            "metadata": metadata,
        },
    }
