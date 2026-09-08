from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_remaining_depth_provider_robustness import (
    I59A_G4C_FROZEN_SHA256,
    I59A_I56A_FROZEN_SHA256,
    I59A_I58C_FROZEN_SHA256,
)
from tests.i4_3g5a_helpers import passing_g4
from tests.i5_6b_helpers import admission as i56a_admission


def clone(value):
    return deepcopy(value)


def binding(path: str, sha256: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha256)


def i58c_closure() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.8C",
        "status": "video_depth_anything_robustness_qualified_and_closed",
        "pass": True,
        "quality_status": "degraded",
        "qualification": {
            "clean_quality_pass_claimed": False,
            "universal_video_depth_anything_robustness_claimed": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_8_complete": True,
            "i5_9_authorized": True,
            "i5_9_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "mp_sfm_authorized": False,
            "fusion_capacity_increase_authorized": False,
            "production_default_modified": False,
        },
        "next": "I5.9 Remaining depth-provider robustness.",
    }


def g4c_closure() -> dict:
    payload = passing_g4()
    payload["all_required_single_image_scenarios_accounted"] = True
    payload["all_required_manual_reviews_complete"] = True
    payload["all_six_provider_outcomes_accounted_per_scenario"] = True
    return payload


def receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.9A_ADMISSION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_9a_tests": True,
            "single_image_provider_regression": True,
            "historical_g4_regression": True,
            "i5_6a_source_binding_regression": True,
            "i5_8c_authorization_regression": True,
            "authority_guard_regression": True,
            "roadmap_guard_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "single_image_provider_inference": False,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "provider_comparison_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {"focused_tests": 1, "predecessor_total_tests": 1, "combined_exercised_tests": 2},
        "return_codes": {
            "focused": 0,
            "predecessor_unit": 0,
            "predecessor_integration": 0,
            "predecessor_stress": 0,
            "import_probe": 0,
            "runner_help_probe": 0,
        },
    }


def build_args() -> dict:
    return {
        "i5_8c_closure": i58c_closure(),
        "i5_8c_binding": binding("/evidence/i5_8c.json", I59A_I58C_FROZEN_SHA256),
        "i5_6a_admission": i56a_admission(),
        "i5_6a_binding": binding("/evidence/i5_6a.json", I59A_I56A_FROZEN_SHA256),
        "i4_3g4c_closure": g4c_closure(),
        "i4_3g4c_binding": binding("/evidence/i4_3g4c.json", I59A_G4C_FROZEN_SHA256),
        "admission_receipt": receipt(),
        "admission_receipt_binding": binding("/evidence/i5_9a_receipt.json", "d" * 64),
    }
