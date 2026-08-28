from __future__ import annotations


def d1_snapshot() -> dict:
    return {
        "phase": "I4.3D1", "status": "pass", "pass": True,
        "provider": {
            "provider_id": "video-depth-anything-temporal", "implementation": "video_depth_anything",
            "family": "temporal_depth", "qualification_state": "candidate", "production_qualified": False,
            "available": True, "backend_version": "vda-rev",
            "implementation_contract": {"family": "temporal_depth", "frame_binding": "align_to_authoritative_sfm", "native_output": "depth_evidence", "integration_stage": "I4.3D"},
        },
        "architecture": {
            "camera_pose_authority_owner": "pycolmap_sfm", "frame_binding": "align_to_authoritative_sfm",
            "native_output": "depth_evidence", "temporal_mode": "official_streaming_cache", "stateful_sequence": True,
            "permanent_global_view_cap": None, "adaptive_view_selection_deferred_to": "I4.3F", "progressive_status_supported": True,
        },
        "authority": {
            "backend_started": False, "provider_selected": False, "provider_promotion_performed": False,
            "estimated_model_intrinsics_consumed_for_canonical_projection": False,
            "model_metric_depth_promoted_to_canonical_metric_scale": False,
            "metric_scale_claimed": False, "measurement_authoritative": False, "certified_metrology": False, "surface_mesh_generated": False,
        },
        "qualification": {"qualification_state": "candidate", "rolling_depth_binding_pending_i4_3d2": True, "temporal_cross_provider_comparison_pending_i4_3g_i5": True, "universal_provider_winner_assumed": False},
    }


def d2_snapshot() -> dict:
    return {
        "phase": "I4.3D2", "status": "pass", "pass": True,
        "provider": {
            "provider_id": "rolling-depth-temporal", "implementation": "rolling_depth", "family": "temporal_depth",
            "qualification_state": "candidate", "production_qualified": False, "available": True, "backend_version": "rd-rev",
            "implementation_contract": {"family": "temporal_depth", "frame_binding": "align_to_authoritative_sfm", "native_output": "depth_evidence", "integration_stage": "I4.3D"},
        },
        "architecture": {
            "camera_pose_authority_owner": "pycolmap_sfm", "frame_binding": "align_to_authoritative_sfm", "native_output": "depth_evidence",
            "temporal_mode": "multi_dilation_snippet_coalignment", "sequence_level_alignment": "global_affine_to_authoritative_sfm",
            "per_frame_alignment": False, "permanent_global_view_cap": None, "adaptive_view_selection_deferred_to": "I4.3F", "progressive_status_supported": True,
        },
        "authority": {
            "backend_started": False, "provider_selected": False, "provider_promotion_performed": False,
            "rolling_depth_pose_authority_promoted": False, "sequence_affine_establishes_physical_metric_scale": False,
            "metric_scale_claimed": False, "measurement_authoritative": False, "certified_metrology": False, "surface_mesh_generated": False,
        },
        "qualification": {
            "qualification_state": "candidate", "video_depth_anything_binding_retained": True,
            "temporal_provider_closure_pending_i4_3d3": True, "temporal_cross_provider_comparison_pending_i4_3g_i5": True,
            "universal_provider_winner_assumed": False,
        },
    }
