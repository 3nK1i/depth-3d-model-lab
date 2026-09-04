from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from core.reconstruction.i5_dynamic_scene_handling import (
    build_i5_5a_dynamic_scene_admission,
    build_i5_5a_source_identity,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import analyze_i5_2b_sfm_report
from tests.i5_2b_helpers import synthetic_sfm_report
from tests.i5_5a_helpers import (
    I54C_SHA,
    RECEIPT_55A_SHA,
    passing_i5_4c,
    passing_i5_5a_receipt,
)

I55A_SHA = "c" * 64
RECEIPT_55B_SHA = "d" * 64


def clone(value):
    return deepcopy(value)


def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def image_source_dir(tmp_path: Path, *, count: int = 19, width: int = 160, height: int = 120) -> Path:
    root = tmp_path / "castle-P19"
    root.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:height, 0:width]
    for index in range(count):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = ((xx * 3 + index * 7) % 256).astype(np.uint8)
        image[..., 1] = ((yy * 5 + index * 11) % 256).astype(np.uint8)
        image[..., 2] = (((xx + yy) * 2 + index * 13) % 256).astype(np.uint8)
        # High-contrast source-derived texture around the frozen donor crop.
        for y in range(44, 72, 7):
            for x in range(62, 98, 9):
                cv2.circle(image, (x, y), 3, (255, 255, 255), -1)
                cv2.rectangle(image, (x + 2, y + 2), (x + 6, y + 6), (0, 0, 0), -1)
        assert cv2.imwrite(str(root / f"{index:04d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 98])
    (root / "K.txt").write_text("ignored calibration metadata\n", encoding="utf-8")
    return root


def passing_i5_5a(tmp_path: Path) -> dict:
    source = image_source_dir(tmp_path)
    return build_i5_5a_dynamic_scene_admission(
        i5_4c_result=passing_i5_4c(),
        i5_4c_binding=binding("/evidence/i5_4c.json", I54C_SHA),
        source_identity=build_i5_5a_source_identity(source),
        source_label="Castle-P19 static multiview control for source-derived dynamic foreground proxy",
        static_control_confirmation=True,
        admission_receipt=passing_i5_5a_receipt(),
        admission_receipt_binding=binding("/evidence/i5_5a_receipt.json", RECEIPT_55A_SHA),
    )


def passing_i5_5b_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.5B_REPLAY_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_5b_tests": True,
            "predecessor_i5_5a_regression": True,
            "production_native_sfm_contract_regression": True,
            "dynamic_materialization_regression": True,
            "dynamic_comparison_regression": True,
            "authority_guard_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "real_sfm_executed_by_test_receipt": False,
            "real_stress_images_materialized_by_test_receipt": False,
            "dense_provider_rerun": False,
            "dynamic_masking_added": False,
            "semantic_segmentation_added": False,
            "motion_outlier_filter_added": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 18,
            "predecessor_unit_tests": 48,
            "predecessor_integration_tests": 10,
            "predecessor_stress_tests": 12,
            "predecessor_total_tests": 70,
            "combined_exercised_tests": 88,
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


def synthetic_i55b_sfm_report(
    admission: dict,
    scenario_key: str,
    *,
    registered_count: int | None = None,
    quality_status: str = "accepted",
    camera_step: float = 1.0,
    common_landmarks_per_adjacent_pair: int = 12,
) -> dict:
    plan = deepcopy(admission["stress_plan"][scenario_key])
    count = int(plan["image_count"])
    plan["indices"] = list(range(count))
    plan["names"] = [f"{index:04d}.png" for index in range(count)]
    fake = {"stress_plan": {scenario_key: plan}}
    return synthetic_sfm_report(
        fake,
        scenario_key,
        camera_step=camera_step,
        common_landmarks_per_adjacent_pair=common_landmarks_per_adjacent_pair,
        registered_count=registered_count,
        quality_status=quality_status,
    )


def synthetic_i55b_analysis(
    admission: dict,
    scenario_key: str,
    *,
    registered_count: int | None = None,
    quality_status: str = "accepted",
    sparse_component_count: int = 1,
) -> dict:
    plan = deepcopy(admission["stress_plan"][scenario_key])
    count = int(plan["image_count"])
    plan["indices"] = list(range(count))
    plan["names"] = [f"{index:04d}.png" for index in range(count)]
    report = synthetic_i55b_sfm_report(
        admission,
        scenario_key,
        registered_count=registered_count,
        quality_status=quality_status,
    )
    analysis = analyze_i5_2b_sfm_report(report, scenario_plan=plan)
    analysis["registration_attempted"] = True
    analysis["sfm_executed"] = True
    analysis["pre_sfm_ingestion_rejected"] = False
    analysis["rejection_stage"] = None
    analysis["rejection_reason"] = None
    analysis["mapping_produced_sparse_model"] = True
    analysis["sparse_component_count"] = sparse_component_count
    return analysis


def passing_materialization_revalidation() -> dict:
    return {
        "same_decoded_source_pixels_for_control_and_stress": True,
        "control_pixel_values_preserved": True,
        "donor_patch_pixel_identity_preserved": True,
        "target_centroid_trajectory_matches_frozen_recipe": True,
        "changed_pixels_confined_to_frozen_target_boxes": True,
        "dynamic_region_feature_bearing_revalidated": True,
        "source_dimensions_constant": True,
        "donor_patch_orb_keypoints": 12,
        "stress_target_orb_keypoints_min": 4,
        "stress_target_orb_keypoints_p50": 6.0,
        "changed_pixel_fraction_p50": 0.05,
        "natural_dynamic_scene_ground_truth": False,
        "controlled_independent_foreground_motion_proxy": True,
    }
