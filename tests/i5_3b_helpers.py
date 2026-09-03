from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.reconstruction.i5_blur_exposure_clutter_symmetry import (
    build_i5_3a_blur_exposure_clutter_symmetry_admission,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import (
    analyze_i5_2b_sfm_report,
    build_ordered_source_identity,
)
from tests.i5_2b_helpers import synthetic_sfm_report
from tests.i5_3a_helpers import I52C_SHA, RECEIPT_53A_SHA, passing_i5_2c, passing_i5_3a_receipt, source_dir

I53A_SHA = "d" * 64
RECEIPT_53B_SHA = "e" * 64


def clone(value):
    return deepcopy(value)


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def passing_i5_3a(tmp_path: Path) -> dict:
    return build_i5_3a_blur_exposure_clutter_symmetry_admission(
        i5_2c_result=passing_i5_2c(tmp_path),
        i5_2c_binding=_binding("/evidence/i5_2c.json", I52C_SHA),
        source_identity=build_ordered_source_identity(source_dir(tmp_path)),
        controlled_source_confirmation=True,
        admission_receipt=passing_i5_3a_receipt(),
        admission_receipt_binding=_binding("/evidence/i5_3a_receipt.json", RECEIPT_53A_SHA),
    )


def passing_i5_3b_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.3B_REPLAY_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_3b_tests": True,
            "predecessor_i5_3a_regression": True,
            "production_native_sfm_contract_regression": True,
            "image_domain_stress_regression": True,
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
            "focused_tests": 20,
            "predecessor_unit_tests": 30,
            "predecessor_integration_tests": 8,
            "predecessor_stress_tests": 12,
            "predecessor_total_tests": 50,
            "combined_exercised_tests": 70,
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


def synthetic_i53_sfm_analysis(
    admission: dict,
    scenario_key: str,
    *,
    registered_count: int | None = None,
    quality_status: str = "accepted",
    camera_step: float = 1.0,
    common_landmarks_per_adjacent_pair: int = 12,
) -> dict:
    plan = deepcopy(admission["stress_plan"][scenario_key])
    plan["indices"] = list(range(int(plan["image_count"])))
    fake = {"stress_plan": {scenario_key: plan}}
    report = synthetic_sfm_report(
        fake,
        scenario_key,
        camera_step=camera_step,
        common_landmarks_per_adjacent_pair=common_landmarks_per_adjacent_pair,
        registered_count=registered_count,
        quality_status=quality_status,
    )
    return analyze_i5_2b_sfm_report(report, scenario_plan=plan)


def passing_image_domain_comparison() -> dict:
    return {
        "directional_only": True,
        "threshold_relaxation_used": False,
        "blur": {
            "control_blur_variance_p50": 100.0,
            "stress_blur_variance_p50": 20.0,
            "stress_to_control_ratio": 0.2,
            "semantic_stress_revalidated": True,
        },
        "underexposure": {
            "control_mean_luma_p50": 120.0,
            "stress_mean_luma_p50": 42.0,
            "control_underexposed_fraction_p50": 0.01,
            "stress_underexposed_fraction_p50": 0.12,
            "control_quality_score_p50": 0.8,
            "stress_quality_score_p50": 0.45,
            "semantic_stress_revalidated": True,
        },
        "overexposure": {
            "control_mean_luma_p50": 120.0,
            "stress_mean_luma_p50": 210.0,
            "control_overexposed_fraction_p50": 0.01,
            "stress_overexposed_fraction_p50": 0.18,
            "control_quality_score_p50": 0.8,
            "stress_quality_score_p50": 0.5,
            "semantic_stress_revalidated": True,
        },
        "clutter": {
            "control_bottom_band_orb_feature_density_p50": 1000.0,
            "stress_bottom_band_orb_feature_density_p50": 1500.0,
            "stress_to_control_ratio": 1.5,
            "semantic_stress_revalidated": True,
            "natural_scene_ground_truth": False,
        },
        "symmetry": {
            "control_mirrored_side_similarity_p50": 0.35,
            "stress_mirrored_side_similarity_p50": 0.98,
            "semantic_stress_revalidated": True,
            "natural_scene_ground_truth": False,
        },
        "all_semantic_stresses_revalidated": True,
    }
