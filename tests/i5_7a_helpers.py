from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_rolling_depth_robustness import (
    I57A_I56A_FROZEN_SHA256,
    I57A_I56C_FROZEN_SHA256,
)
from core.reconstruction.single_image_provider_qualification import I43G3C_FROZEN_CLOSURE_SHA256
from tests.i5_6b_helpers import admission as i56a_admission


def clone(payload: dict) -> dict:
    return deepcopy(payload)


def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def i56c_closure() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.6C",
        "status": "mvs_robustness_qualified_and_closed",
        "pass": True,
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "sfm_rerun": False, "bundle_adjustment_rerun": False, "mvs_rerun": False,
            "patch_match_rerun": False, "stereo_fusion_rerun": False,
            "provider_comparison_started": False, "da3_rerun": False,
            "thresholds_modified": False, "production_default_modified": False,
            "result_driven_retuning": False, "retry_until_pass": False,
            "mp_sfm_started": False, "auto_policy_implemented": False, "hybrid_authorized": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm", "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True, "provider_promoted": False,
            "provider_ranking_created": False, "metric_scale_claimed": False,
            "measurement_authoritative": False, "certified_metrology": False,
            "surface_mesh_generated": False, "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_6_complete": True, "i5_7_authorized": True, "i5_7_started": False,
            "provider_promoted": False, "provider_ranking_created": False,
            "universal_provider_winner_declared": False, "global_scalar_score_created": False,
            "auto_policy_implemented": False, "hybrid_authorized": False, "mp_sfm_authorized": False,
            "threshold_relaxation_authorized": False, "production_default_modified": False,
        },
        "roadmap_guard": {
            "i5_7_rollingdepth_authorized": True, "i5_8_video_depth_anything_not_started": True,
            "i5_10_mp_sfm_deferred": True, "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_auto_deferred": True,
        },
    }


def g3c_closure() -> dict:
    statuses = {
        "g3_temporal_static_sweep": "degraded",
        "g3_temporal_motion": "rejected",
        "g3_temporal_low_texture": "pass",
    }
    return {
        "schema_version": 1, "phase": "I4.3G3C", "status": "formally_closed", "pass": True,
        "scenario_count": 3, "all_required_temporal_scenarios_accounted": True,
        "all_required_manual_reviews_complete": True,
        "scenario_synthesis": {
            sid: {
                "production_statuses": {"rolling_depth": status, "video_depth_anything": "pass"},
                "provider_preferences": {}, "manual_review_complete": True,
                "matched_view_diagnostic_secondary": True, "production_rejection_overridden": False,
                "global_provider_winner_declared": False,
            }
            for sid, status in statuses.items()
        },
        "decision": {
            "production_native_evidence_primary": True, "matched_view_diagnostic_secondary": True,
            "scenario_specific_verdicts_preserved": True, "quality_failures_preserved": True,
            "provider_selected": False, "provider_promoted": False,
            "universal_provider_winner_declared": False, "global_scalar_score_created": False,
            "auto_policy_implemented": False, "hybrid_authorized": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm", "authoritative_sfm_modified": False,
            "provider_selected": False, "provider_promoted": False, "auto_policy_implemented": False,
            "hybrid_authorized": False, "metric_scale_claimed": False,
            "measurement_authoritative": False, "certified_metrology": False,
            "surface_mesh_generated": False, "permanent_global_view_cap": None,
        },
    }


def receipt() -> dict:
    return {
        "schema_version": 1, "phase": "I5.7A_ADMISSION_RECEIPT", "status": "pass", "pass": True,
        "scope": {
            "focused_i5_7a_tests": True, "rolling_depth_provider_regression": True,
            "temporal_g3_regression": True, "i5_6_predecessor_regression": True,
            "authority_guard_regression": True, "roadmap_guard_regression": True,
            "import_probe": True, "runner_help_probe": True,
            "rolling_depth_inference": False, "sfm_rerun": False, "mvs_rerun": False,
            "provider_comparison_started": False, "thresholds_modified": False,
            "production_default_modified": False, "result_driven_retuning": False,
            "retry_until_pass": False, "mp_sfm_started": False,
            "auto_policy_implemented": False, "hybrid_authorized": False,
        },
        "counts": {"focused_tests": 12, "predecessor_total_tests": 30, "combined_exercised_tests": 42},
        "return_codes": {"focused": 0, "predecessor_unit": 0, "predecessor_integration": 0,
                         "predecessor_stress": 0, "import_probe": 0, "runner_help_probe": 0},
    }


def build_args() -> dict:
    return {
        "i5_6a_admission": i56a_admission(),
        "i5_6a_binding": binding("/evidence/i5_6a.json", I57A_I56A_FROZEN_SHA256),
        "i5_6c_closure": i56c_closure(),
        "i5_6c_binding": binding("/evidence/i5_6c.json", I57A_I56C_FROZEN_SHA256),
        "i4_3g3c_closure": g3c_closure(),
        "i4_3g3c_binding": binding("/evidence/i4_3g3c.json", I43G3C_FROZEN_CLOSURE_SHA256),
        "admission_receipt": receipt(),
        "admission_receipt_binding": binding("/evidence/i5_7a_receipt.json", "d" * 64),
    }
