from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_dynamic_scene_handling import (
    build_i5_5b_dynamic_scene_replay,
    compare_i5_5b_sfm_outcomes,
)
from tests.i5_5b_helpers import (
    I55A_SHA,
    RECEIPT_55B_SHA,
    binding,
    passing_i5_5a,
    passing_i5_5b_receipt,
    passing_materialization_revalidation,
    synthetic_i55b_analysis,
)

I55B_SHA = "e" * 64
RECEIPT_55C_SHA = "f" * 64


def clone(value):
    return deepcopy(value)


def i55c_binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def passing_i5_5b(tmp_path: Path, *, directional_penalty: bool = True) -> dict:
    admission = passing_i5_5a(tmp_path)
    control = synthetic_i55b_analysis(admission, "static_control")
    stress = synthetic_i55b_analysis(admission, "dynamic_foreground_proxy")
    if directional_penalty:
        stress["statistics"]["landmarks"] = 120
        stress["statistics"]["observations"] = 260
        stress["statistics"]["mean_reprojection_error_px"] = 1.1
        stress["quality_assessment"]["mean_reprojection_error_px"] = 1.1
        stress["quality_assessment"]["mean_track_length"] = 1.8
    materialization = passing_materialization_revalidation()
    comparison = compare_i5_5b_sfm_outcomes(
        control=control,
        dynamic_proxy=stress,
        materialization_revalidation=materialization,
    )
    return build_i5_5b_dynamic_scene_replay(
        i5_5a_result=admission,
        i5_5a_binding=binding("/evidence/i5_5a.json", I55A_SHA),
        replay_receipt=passing_i5_5b_receipt(),
        replay_receipt_binding=binding("/evidence/i5_5b_receipt.json", RECEIPT_55B_SHA),
        materialization_revalidation=materialization,
        control_analysis=control,
        dynamic_proxy_analysis=stress,
        paired_comparison=comparison,
        source_reverified_before=True,
        source_reverified_after=True,
        staged_inputs_reverified=True,
    )


def passing_i5_5c_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.5C_QUALIFICATION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_5c_tests": True,
            "predecessor_i5_5b_regression": True,
            "predecessor_i5_5a_regression": True,
            "evidence_only_closure_contract_regression": True,
            "dynamic_quality_penalty_preservation_regression": True,
            "authority_guard_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "sfm_rerun": False,
            "source_rematerialized": False,
            "stress_images_rematerialized": False,
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
            "focused_tests": 20,
            "predecessor_unit_tests": 60,
            "predecessor_integration_tests": 12,
            "predecessor_stress_tests": 16,
            "predecessor_total_tests": 88,
            "combined_exercised_tests": 108,
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
