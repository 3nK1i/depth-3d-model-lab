from __future__ import annotations

from copy import deepcopy

SOURCE_SHA = "a" * 64
BASELINE_SHA = "b" * 64


def clone(value):
    return deepcopy(value)


def passing_i5_1a() -> dict:
    return {
        "schema_version": 2,
        "phase": "I5.1A",
        "status": "historical_baseline_admitted",
        "pass": True,
        "scenario": {
            "scenario_id": "i5_1_horse_difficult_handheld_baseline",
            "capture_class": "difficult_handheld_static_world_capture",
            "source_role": "historical_robustness_target",
            "fresh_source_identity_required_for_i5_1b": True,
        },
        "historical_baseline": {
            "historical_capture_regime_difficult": True,
            "historical_local_temporal_density_materially_affects_dense_consistency": True,
            "historical_full_span_da3_sfm_alignment": "36/40",
            "historical_dense_path_collapsed_to_views": 2,
            "historical_decoupled_sampling_materially_improved_real_dense_consistency": True,
            "historical_dense_gate_reached": False,
            "historical_dense_gate_required_connected_views": 10,
            "historical_horse_preserved_for_i5": True,
            "historical_outcome": "degraded_robustness_case",
            "historical_failure_is_preserved_evidence": True,
            "historical_failure_reclassified_as_pass": False,
            "historical_counts_are_not_provider_scores": True,
        },
        "provenance": {
            "historical_horse_standalone_summary_json_bound": False,
            "historical_raw_video_content_rehashed_by_i5_1a": False,
            "no_stronger_historical_source_identity_claim_created": True,
            "current_replay_must_establish_fresh_source_identity": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "permanent_global_view_cap": None,
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "authority_audit": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "default_reconstruction_scale_status": "arbitrary",
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i4_closed": True,
            "i5_authorized": True,
            "i5_1_started": True,
            "i5_1a_complete": True,
            "i5_1_current_replay_authorized": True,
            "i5_1_complete": False,
            "i5_2_authorized": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "next": "I5.1B current frozen-foundation difficult-handheld replay.",
    }


def source_identity() -> dict:
    return {
        "path": "/data/i4-horse-orbit-qualified.mp4",
        "sha256": SOURCE_SHA,
        "size_bytes": 123456789,
        "total_frames": 1800,
        "fps": 30.0,
        "duration_seconds": 60.0,
        "width": 1920,
        "height": 1080,
    }


def i1_summary(*, selected_views: int = 5, status: str = "degraded") -> dict:
    payload = {
        "phase": "I1",
        "schema_version": 1,
        "status": status,
        "selected_path": {
            "orders": list(range(selected_views)),
            "view_count": selected_views,
            "span": max(0, selected_views - 1),
            "worst_consistency_ratio": 0.71,
            "worst_median_relative_residual": 0.07,
        },
        "authority": {
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
    }
    if status == "degraded":
        payload.update({
            "failed_stage": "dense_pair_qualification",
            "reason": "insufficient_connected_qualified_views",
            "minimum_qualified_views": 10,
        })
    return payload


def localization_summary(*, gate: bool = True) -> dict:
    return {
        "phase": "I4.1.6A",
        "schema_version": 1,
        "status": "localization_ready" if gate else "localization_degraded",
        "source": {
            "video": "/data/i4-horse-orbit-qualified.mp4",
            "source_total_frames": 1800,
            "source_fps": 30.0,
            "source_duration_seconds": 60.0,
            "source_sha256": SOURCE_SHA,
            "sfm_session_id": "sfm-1",
            "registered_global_anchors": 40,
        },
        "plans": [
            {
                "plan_name": "uniform_localized_dense_pool",
                "planned_candidates": 102,
                "extra_candidates": 62,
                "accepted_extra_candidates": 58,
                "extra_localization_fraction": 58 / 62,
                "full_span": True,
                "localization_gate_pass": gate,
            },
            {
                "plan_name": "global_anchors_plus_local_neighbors",
                "planned_candidates": 118,
                "extra_candidates": 78,
                "accepted_extra_candidates": 73,
                "extra_localization_fraction": 73 / 78,
                "full_span": True,
                "localization_gate_pass": gate,
            },
        ],
        "decision": {
            "single_authoritative_sfm_frame": True,
            "global_full_span_sampling_preserved": True,
            "extra_dense_candidates_are_observations_not_second_map": True,
            "extra_dense_candidates_require_pose_localization": True,
            "dense_ab_execution_authorized": gate,
            "production_sampling_integration_authorized": False,
            "production_default_changed": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "execution": {
            "pycolmap_rerun": False,
            "da3_rerun": False,
            "dense_pair_rerun": False,
            "fusion_rerun": False,
        },
    }


def dense_ab_summary(*, dense_gate: bool = True, selected: int = 12) -> dict:
    return {
        "phase": "I4.1.6B",
        "schema_version": 1,
        "status": "real_dense_ab_ready",
        "source": {
            "video": "/data/i4-horse-orbit-qualified.mp4",
            "sfm_session_id": "sfm-1",
        },
        "baseline_global_only": {
            "candidate_mode": "global_anchors_only",
            "pair_edges": 100,
            "pair_pass": 30,
            "pair_degraded": 70,
            "pair_inconclusive": 0,
            "pair_pass_fraction": 0.3,
            "selected_qualified_views": 5,
            "selected_path": {"view_count": 5, "orders": [0, 1, 2, 3, 4]},
            "rerun": False,
        },
        "plans": [
            {
                "plan_name": "uniform_localized_dense_pool",
                "planned_candidates": 102,
                "alignment_accepted": 92,
                "alignment_acceptance_fraction": 92 / 102,
                "pair_edges": 130,
                "pair_pass": 40,
                "pair_degraded": 90,
                "pair_inconclusive": 0,
                "pair_pass_fraction": 40 / 130,
                "selected_qualified_views": selected - 1,
                "full_span": True,
                "dense_gate_pass": bool(dense_gate and selected - 1 >= 10),
            },
            {
                "plan_name": "global_anchors_plus_local_neighbors",
                "planned_candidates": 118,
                "alignment_accepted": 107,
                "alignment_acceptance_fraction": 107 / 118,
                "pair_edges": 140,
                "pair_pass": 50,
                "pair_degraded": 90,
                "pair_inconclusive": 0,
                "pair_pass_fraction": 50 / 140,
                "selected_qualified_views": selected,
                "full_span": True,
                "dense_gate_pass": dense_gate,
            },
        ],
        "decision": {
            "recommended_plan": "global_anchors_plus_local_neighbors",
            "real_dense_ab_validation_pass": dense_gate,
            "i4_1_6c_production_binding_authorized": dense_gate,
            "production_sampling_integration_authorized": False,
            "single_authoritative_sfm_frame": True,
            "global_full_span_sampling_preserved": True,
            "extra_dense_candidates_are_observations_not_second_map": True,
            "extra_dense_candidates_pose_localized": True,
            "production_default_changed": False,
            "filter_policy_changed": False,
            "persistence_changed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "execution": {
            "pycolmap_rerun": False,
            "da3_extra_candidates_executed": True,
            "global_anchor_da3_reused": True,
            "dense_pair_rerun": True,
            "fusion_rerun": False,
            "filtering_rerun": False,
            "persistence_rerun": False,
        },
        "thresholds": {
            "dense_consistency": {
                "consistency_relative_tolerance": 0.12,
                "occlusion_relative_margin": 0.20,
                "minimum_comparable_samples": 1000,
                "minimum_consistent_ratio": 0.50,
                "maximum_median_relative_residual": 0.10,
                "maximum_p90_relative_residual": 0.25,
                "maximum_front_conflict_ratio": 0.25,
                "nn_support_scene_fraction": 0.05,
            },
            "threshold_relaxation": False,
            "max_bridge_gap": 3,
        },
    }
