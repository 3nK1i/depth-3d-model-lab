from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_video_depth_anything_robustness import (
    I58A_G3C_FROZEN_SHA256,
    I58A_I56A_FROZEN_SHA256,
    I58A_I57C_FROZEN_SHA256,
)
from tests.i4_3g5a_helpers import passing_g3
from tests.i5_7a_helpers import i56a_admission


def clone(value):
    return deepcopy(value)


def binding(path: str, sha256: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha256)


def i57c_closure() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.7C",
        "status": "rolling_depth_robustness_qualified_and_closed",
        "pass": True,
        "quality_status": "pass",
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "rolling_depth_inference": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "mvs_rerun": False,
            "provider_comparison_started": False,
            "video_depth_anything_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "qualification": {
            "universal_rolling_depth_robustness_claimed": False,
            "provider_ranking_claimed": False,
            "metric_scale_claimed": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
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
            "i5_7_complete": True,
            "i5_8_authorized": True,
            "i5_8_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "mp_sfm_authorized": False,
            "threshold_relaxation_authorized": False,
            "production_default_modified": False,
        },
        "roadmap_guard": {
            "i5_8_video_depth_anything_authorized_not_started": True,
        },
        "next": "I5.8 Video Depth Anything robustness.",
    }


def g3c_closure() -> dict:
    payload = passing_g3()
    payload["schema_version"] = 1
    payload["scenario_count"] = 3
    payload["all_required_temporal_scenarios_accounted"] = True
    payload["all_required_manual_reviews_complete"] = True
    return payload


def receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.8A_ADMISSION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_8a_tests": True,
            "vda_provider_regression": True,
            "temporal_g3_regression": True,
            "i5_6a_source_binding_regression": True,
            "i5_7c_authorization_regression": True,
            "authority_guard_regression": True,
            "roadmap_guard_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "video_depth_anything_inference": False,
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
        "counts": {
            "focused_tests": 15,
            "predecessor_unit_tests": 20,
            "predecessor_integration_tests": 10,
            "predecessor_stress_tests": 10,
            "predecessor_total_tests": 40,
            "combined_exercised_tests": 55,
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


def build_args() -> dict:
    return {
        "i5_7c_closure": i57c_closure(),
        "i5_7c_binding": binding("/evidence/i5_7c.json", I58A_I57C_FROZEN_SHA256),
        "i5_6a_admission": i56a_admission(),
        "i5_6a_binding": binding("/evidence/i5_6a.json", I58A_I56A_FROZEN_SHA256),
        "i4_3g3c_closure": g3c_closure(),
        "i4_3g3c_binding": binding("/evidence/i4_3g3c.json", I58A_G3C_FROZEN_SHA256),
        "admission_receipt": receipt(),
        "admission_receipt_binding": binding("/evidence/i5_8a_receipt.json", "d" * 64),
    }
