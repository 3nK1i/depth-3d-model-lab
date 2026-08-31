from __future__ import annotations

from core.reconstruction.cross_provider_qualification import qualification_scenario_matrix


def passing_g1() -> dict:
    specs = [item.as_dict() for item in qualification_scenario_matrix()]
    return {
        "phase": "I4.3G1",
        "status": "pass",
        "pass": True,
        "scenario_matrix": specs,
        "coverage": {
            "scenario_count": 11,
            "provider_implementation_count": 10,
            "g2_scenarios": 5,
            "g3_scenarios": 3,
            "g4_scenarios": 3,
        },
        "decision_rules": {
            "production_native_plan_is_primary_operating_envelope": True,
            "matched_view_diagnostic_is_secondary_causal_diagnostic": True,
            "scenario_specific_verdicts_required": True,
            "global_scalar_score_forbidden": True,
            "auto_policy_implemented": False,
            "hybrid_experiment_authorized": False,
            "provider_promotion_performed": False,
            "universal_provider_winner_assumed": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
    }


def passing_g2(*, with_scenario_detail: bool = False) -> dict:
    synthesis = {
        "camera_pose_authority_owner": "pycolmap_sfm",
        "auto_provider_selection_implemented": False,
        "hybrid_provider_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "provider_promoted": False,
        "provider_selected": False,
        "surface_mesh_generated": False,
        "universal_provider_winner_declared": False,
        "all_five_scenarios_accounted": True,
        "all_required_manual_reviews_complete": True,
    }
    payload = {
        "phase": "I4.3G2",
        "status": "formally_closed",
        "i4_3g2_closed": True,
        "i4_3g3_authorized": True,
        "synthesis": synthesis,
    }
    if with_scenario_detail:
        payload["scenario_synthesis"] = {
            item.scenario_id: {"status": "frozen_g2_result"}
            for item in qualification_scenario_matrix()
            if item.substep == "I4.3G2"
        }
    return payload


def _authority() -> dict:
    return {
        "camera_pose_authority_owner": "pycolmap_sfm",
        "authoritative_sfm_modified": False,
        "provider_selected": False,
        "provider_promoted": False,
        "auto_policy_implemented": False,
        "hybrid_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "permanent_global_view_cap": None,
    }


def passing_g3() -> dict:
    statuses = {
        "g3_temporal_static_sweep": {"rolling_depth": "degraded", "video_depth_anything": "pass"},
        "g3_temporal_motion": {"rolling_depth": "rejected", "video_depth_anything": "degraded"},
        "g3_temporal_low_texture": {"rolling_depth": "pass", "video_depth_anything": "pass"},
    }
    return {
        "phase": "I4.3G3C",
        "status": "formally_closed",
        "pass": True,
        "scenario_count": 3,
        "i4_3g3_closed": True,
        "i4_3g4_authorized": True,
        "scenario_synthesis": {
            sid: {
                "production_statuses": values,
                "provider_preferences": {"example": ["video_depth_anything"]},
                "manual_review_complete": True,
                "matched_view_diagnostic_secondary": True,
                "production_rejection_overridden": False,
                "global_provider_winner_declared": False,
            }
            for sid, values in statuses.items()
        },
        "decision": {
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "scenario_specific_verdicts_preserved": True,
            "quality_failures_preserved": True,
            "provider_selected": False,
            "provider_promoted": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "authority": _authority(),
    }


def passing_g4() -> dict:
    providers = ("metric3d_v2", "unidepth_v2", "moge_2", "depth_pro", "depth_anything_v2", "marigold")
    statuses = {
        "g4_single_image_complete_room": {p: "pass" for p in providers},
        "g4_single_image_low_texture": {
            **{p: "pass" for p in providers},
            "depth_anything_v2": "rejected",
            "marigold": "rejected",
        },
        "g4_single_image_weak_overlap": {p: "pass" for p in providers},
    }
    return {
        "phase": "I4.3G4C",
        "status": "formally_closed",
        "pass": True,
        "scenario_count": 3,
        "i4_3g4_closed": True,
        "i4_3g5_authorized": True,
        "scenario_synthesis": {
            sid: {
                "production_statuses": values,
                "provider_preferences": {"example": ["moge_2"]},
                "provider_not_assessable": {},
                "manual_review_complete": True,
                "matched_view_diagnostic_secondary": True,
                "production_rejection_overridden": False,
                "provider_ranking_created": False,
                "global_provider_winner_declared": False,
            }
            for sid, values in statuses.items()
        },
        "decision": {
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "scenario_specific_verdicts_preserved": True,
            "quality_rejections_preserved": True,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "authority": _authority(),
    }
