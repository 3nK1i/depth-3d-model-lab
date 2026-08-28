from __future__ import annotations

from core.reconstruction.dense_provider_architecture import dense_provider_implementation_contracts


def implementation_ids() -> list[str]:
    return [item.implementation.value for item in dense_provider_implementation_contracts()]


def passing_i4_3a_snapshot() -> dict:
    return {
        "phase": "I4.3A",
        "status": "pass",
        "pass": True,
        "architecture": {
            "provider_implementations": implementation_ids(),
            "scenario_specific_qualification_required": True,
            "universal_provider_winner_assumed": False,
        },
        "qualification_plan": {
            "i4_3g_required": True,
            "complete_room_mvs_vs_da3_required": True,
            "temple_ring_mvs_vs_ai_where_useful": True,
            "house_mvs_vs_da3_evidence_retained": True,
            "temporal_video_comparison_required": True,
            "low_texture_comparison_required": True,
            "weak_overlap_comparison_required": True,
            "hybrid_implementation_authorized": False,
        },
        "authority": {
            "backend_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_implementation_authorized": False,
            "authoritative_sparse_modified": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
    }


def passing_i4_3f3_snapshot() -> dict:
    return {
        "phase": "I4.3F3",
        "status": "pass",
        "pass": True,
        "validated_provider_implementations": implementation_ids(),
        "decision": {
            "i4_3f_closed": True,
            "i4_3g_cross_provider_qualification_authorized": True,
            "permanent_global_view_cap": None,
            "provider_selected": False,
            "auto_policy_implemented": False,
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
