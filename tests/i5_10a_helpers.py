from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding

I59C_SHA = "48148d22de7e67cfa3b5891b230dfd64e69888a224e4e930e6d765b1c3383364"


def synthetic_i59c_closure() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "I5.9C",
        "status": "remaining_depth_provider_robustness_qualified_and_closed",
        "pass": True,
        "quality_status": "mixed_terminal_quality",
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "provider_inference": False,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
        },
        "qualification": {
            "provider_count": 6,
            "scenario_count": 6,
            "combined_terminal_runs": 36,
            "combined_status_counts": {"pass": 35, "degraded": 1, "rejected": 0},
            "all_36_current_provider_scenario_attempts_terminal": True,
            "provider_comparison_performed": False,
            "provider_ranking_claimed": False,
            "universal_provider_winner_claimed": False,
            "universal_single_image_provider_robustness_claimed": False,
        },
        "historical_g4_anchors": {
            "low_texture": {
                "metric3d_v2": "pass",
                "unidepth_v2": "pass",
                "moge_2": "pass",
                "depth_pro": "pass",
                "depth_anything_v2": "rejected",
                "marigold": "rejected",
            }
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_9_complete": True,
            "i5_10_authorized": True,
            "i5_10_started": False,
            "i5_11_started": False,
            "i5_12_started": False,
            "i5_13_started": False,
            "i5_14_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "mp_sfm_authorized": True,
            "mp_sfm_started": False,
            "threshold_relaxation_authorized": False,
            "fusion_capacity_increase_authorized": False,
            "production_default_modified": False,
        },
        "roadmap_guard": {
            "i5_10_mp_sfm_authorized_not_started": True,
            "i5_11_oom_disk_crash_recovery_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_auto_deferred": True,
            "i5_14_final_i5_closure_deferred": True,
        },
        "next": "I5.10 MP-SfM evaluation.",
    }


def synthetic_i510a_receipt() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "I5.10A_ADMISSION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_10a_tests": True,
            "sfm_contract_regression": True,
            "i5_4_forward_motion_regression": True,
            "i5_9c_authorization_regression": True,
            "authority_guard_regression": True,
            "roadmap_guard_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "mp_sfm_execution": False,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "provider_inference": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 20,
            "predecessor_unit_tests": 40,
            "predecessor_integration_tests": 10,
            "predecessor_stress_tests": 10,
            "predecessor_total_tests": 60,
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


def i59c_binding() -> I51EvidenceBinding:
    return I51EvidenceBinding("/evidence/i5_9c_remaining_depth_provider_robustness_closure.json", I59C_SHA)


def receipt_binding() -> I51EvidenceBinding:
    return I51EvidenceBinding("/evidence/i5_10a_admission_receipt.json", "a" * 64)


def mutate(payload: dict[str, Any], *path: str, value: Any) -> dict[str, Any]:
    out = deepcopy(payload)
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return out
