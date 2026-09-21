from __future__ import annotations

from copy import deepcopy


def binding(path: str, char: str) -> dict[str, str]:
    return {"path": path, "sha256": char * 64}


def g5b() -> dict:
    providers = {
        "mvs_da3": ["colmap_mvs", "da3"],
        "temporal_depth": ["rolling_depth", "video_depth_anything"],
        "single_image_depth": [
            "metric3d_v2", "unidepth_v2", "moge_2", "depth_pro", "depth_anything_v2", "marigold"
        ],
    }
    roles = {
        family: {
            "provider_implementations": ids,
            "formal_closure_valid": True,
            "direct_rank_comparison_with_other_families_valid": False,
        }
        for family, ids in providers.items()
    }
    events = [
        {"family_id": "temporal_depth", "scenario_id": "g3_temporal_static_sweep", "provider_implementation": "rolling_depth", "production_status": "degraded"},
        {"family_id": "temporal_depth", "scenario_id": "g3_temporal_motion", "provider_implementation": "rolling_depth", "production_status": "rejected"},
        {"family_id": "temporal_depth", "scenario_id": "g3_temporal_motion", "provider_implementation": "video_depth_anything", "production_status": "degraded"},
        {"family_id": "single_image_depth", "scenario_id": "g4_single_image_low_texture", "provider_implementation": "depth_anything_v2", "production_status": "rejected"},
        {"family_id": "single_image_depth", "scenario_id": "g4_single_image_low_texture", "provider_implementation": "marigold", "production_status": "rejected"},
    ]
    return {
        "phase": "I4.3G5B",
        "status": "formally_closed",
        "pass": True,
        "scope": {
            "evidence_only": True,
            "provider_execution_started": False,
            "sfm_rerun": False,
            "matched_view_rerun": False,
            "manual_review_rerun": False,
            "formal_i4_3g_closure_performed": True,
            "i4_3h_work_started": False,
        },
        "closure_chain": {
            "g1_framework_bound": True,
            "g2_formally_closed": True,
            "g3_formally_closed": True,
            "g4_formally_closed": True,
            "g5a_synthesis_complete": True,
            "g5a_deterministically_rebuilt_from_predecessors": True,
            "all_predecessor_sha256_bindings_verified": True,
        },
        "coverage": {
            "qualification_family_count": 3,
            "scenario_count": 11,
            "provider_implementation_count": 10,
            "g2_scenarios": 5,
            "g3_scenarios": 3,
            "g4_scenarios": 3,
        },
        "operating_envelope_summary": {
            "scenario_specific_operating_envelopes_preserved": True,
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "terminal_quality_events": events,
            "terminal_quality_event_count": 5,
            "quality_event_counts_are_not_provider_scores": True,
            "direct_cross_family_provider_ranking_valid": False,
            "family_roles": roles,
        },
        "decision": {
            "g5a_cross_provider_synthesis_complete": True,
            "i4_3g_closed": True,
            "i4_3h_authorized": True,
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
        "evidence_bindings": {
            "i4_3g1_snapshot": binding("/e/g1.json", "1"),
            "i4_3g2_closure": binding("/e/g2.json", "2"),
            "i4_3g3c_closure": binding("/e/g3.json", "3"),
            "i4_3g4c_closure": binding("/e/g4.json", "4"),
            "i4_3g5a_synthesis": binding("/e/g5a.json", "5"),
        },
    }


def i51c() -> dict:
    return {
        "phase": "I5.1C", "status": "difficult_handheld_robustness_qualified", "pass": True,
        "qualification": {
            "quality_status": "degraded",
            "operating_envelope_classification": "outside_current_production_dense_robustness_envelope",
            "production_dense_gate_reached": False,
            "primary_robustness_bottleneck": "local_dense_consistency_connectivity",
        },
        "decision": {},
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "permanent_global_view_cap": None},
    }


def i52c() -> dict:
    return {
        "phase": "I5.2C", "status": "low_overlap_weak_parallax_robustness_qualified", "pass": True,
        "qualification": {
            "quality_status": "pass",
            "operating_envelope": "qualified_on_admitted_castle_p19_sparse_sfm_stress_cases",
            "partial_registration_preserved": True,
            "low_overlap_authoritative_revalidation": True,
            "weak_parallax_authoritative_revalidation": True,
        },
        "decision": {},
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "permanent_global_view_cap": None},
    }


def _stress(status: str, conclusion: str) -> dict:
    return {"scenario_quality_status": status, "robustness_conclusion": conclusion}


def i53c() -> dict:
    return {
        "phase": "I5.3C", "status": "blur_exposure_clutter_symmetry_robustness_qualified", "pass": True,
        "qualification": {"quality_status": "degraded", "operating_envelope": "mixed_visual_stress_envelope"},
        "stress_conclusions": {
            "blur": _stress("rejected", "outside_current_sparse_sfm_ingestion_envelope"),
            "underexposure": _stress("rejected", "outside_current_sparse_sfm_ingestion_envelope"),
            "overexposure": _stress("accepted", "survived_admitted_stress_case"),
            "clutter": _stress("accepted", "inconclusive_proxy_not_revalidated"),
            "symmetry": _stress("degraded", "degraded_partial_registration"),
        },
        "decision": {},
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "permanent_global_view_cap": None},
    }


def i54c() -> dict:
    return {
        "phase": "I5.4C", "status": "forward_motion_kitti_robustness_qualified_and_closed", "pass": True,
        "quality_status": "rejected",
        "qualification": {
            "quality_status": "rejected",
            "operating_envelope": "outside_current_production_sparse_sfm_envelope_on_admitted_kitti_style_forward_motion_case",
            "primary_bottleneck": "incremental_mapping_no_sparse_model",
            "mapping_produced_sparse_model": False,
            "registered_images": 0,
        },
        "decision": {},
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "permanent_global_view_cap": None},
    }


def i55c() -> dict:
    return {
        "phase": "I5.5C", "status": "dynamic_scene_handling_robustness_qualified_and_closed", "pass": True,
        "quality_status": "pass",
        "qualification": {
            "quality_status": "pass",
            "operating_envelope": "qualified_on_admitted_controlled_dynamic_foreground_proxy",
            "natural_dynamic_scene_ground_truth": False,
            "universal_dynamic_scene_robustness_claimed": False,
        },
        "decision": {},
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "permanent_global_view_cap": None},
    }


def i56c() -> dict:
    return {
        "phase": "I5.6C", "status": "mvs_robustness_qualified_and_closed", "pass": True,
        "quality_status": "pass",
        "qualification": {
            "current_production_colmap_mvs_path_qualified_within_admitted_envelope": True,
            "provider_ranking_claimed": False,
        },
        "limitations": {"runtime_performance_envelope_complete": False, "resource_envelope_complete": False},
        "decision": {},
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "permanent_global_view_cap": None},
    }


def i57c() -> dict:
    return {
        "phase": "I5.7C", "status": "rolling_depth_robustness_qualified_and_closed", "pass": True,
        "quality_status": "pass",
        "qualification": {
            "current_frozen_foundation_rolling_depth_path_qualified_within_admitted_envelope": True,
            "historical_g3_motion_rejected_preserved": True,
            "historical_g3_static_sweep_degraded_preserved": True,
        },
        "decision": {},
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "permanent_global_view_cap": None},
    }


def i58c() -> dict:
    return {
        "phase": "I5.8C", "status": "video_depth_anything_robustness_qualified_and_closed", "pass": True,
        "quality_status": "degraded",
        "qualification": {
            "current_frozen_foundation_vda_path_execution_qualified_within_admitted_envelope": True,
            "current_frozen_foundation_vda_quality_clean_pass_claimed": False,
            "fusion_capacity_limitation_prevents_clean_quality_pass_claim": True,
        },
        "decision": {},
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "permanent_global_view_cap": None},
    }


def i59c() -> dict:
    providers = {
        "metric3d_v2": {}, "unidepth_v2": {}, "moge_2": {}, "depth_pro": {}, "depth_anything_v2": {}, "marigold": {}
    }
    return {
        "phase": "I5.9C", "status": "remaining_depth_provider_robustness_qualified_and_closed", "pass": True,
        "quality_status": "mixed_terminal_quality",
        "qualification": {
            "combined_status_counts": {"pass": 35, "degraded": 1, "rejected": 0},
            "current_castle_degraded_provider": "marigold",
            "current_castle_degraded_scenario": "castle_dynamic_proxy",
        },
        "provider_qualification": providers,
        "historical_g4_anchors": {
            "complete_room": {}, "low_texture": {"depth_anything_v2": "rejected", "marigold": "rejected"}, "weak_overlap": {}
        },
        "observed_limitations": {
            "historical_depth_anything_v2_low_texture_rejection_remains_active": True,
            "historical_marigold_low_texture_rejection_remains_active": True,
        },
        "decision": {},
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "permanent_global_view_cap": None},
    }


def i510d() -> dict:
    return {
        "phase": "I5.10D", "status": "mp_sfm_evidence_qualified_and_closed", "pass": True,
        "qualification": {
            "conclusion": "matching_policy_sensitivity_with_partial_sparse_recovery_but_no_production_quality_recovery",
        },
        "candidate_quality": {"production_quality_status": "rejected"},
        "decision": {"auto_policy_implemented": False, "hybrid_authorized": False, "production_default_modified": False},
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "permanent_global_view_cap": None},
    }


def i511d() -> dict:
    return {
        "phase": "I5.11D", "status": "bounded_failure_recovery_evidence_qualified_and_closed", "pass": True,
        "quality_status": "bounded_fault_recovery_pass",
        "qualification": {
            "bounded_fault_recovery_pass": True,
            "all_expected_fault_signatures_observed": True,
            "all_process_groups_clean": True,
            "all_recoveries_completed": True,
            "all_repository_integrity_preserved": True,
            "runtime_integrity_failure": False,
        },
        "decision": {"auto_policy_implemented": False, "hybrid_authorized": False, "production_default_modified": False},
    }


def i512a() -> dict:
    return {
        "phase": "I5.12A", "status": "hybrid_evidence_justification_review_complete", "pass": True,
        "quality_status": "hybrid_not_justified",
        "decision": {
            "i5_12_complete": True,
            "i5_13_authorized": True,
            "i5_13_started": False,
            "hybrid_authorized": False,
            "hybrid_execution_authorized": False,
            "auto_policy_implemented": False,
            "production_default_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
        },
    }


def receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.13A_ADMISSION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_13a_tests": True,
            "dense_architecture_regression": True,
            "cross_provider_closure_regression": True,
            "i5_10_to_i5_12_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "auto_policy_engine_implemented": False,
            "auto_provider_selection_performed": False,
            "production_auto_integration": False,
            "production_default_modified": False,
            "hybrid_execution": False,
        },
        "counts": {"focused_tests": 30, "predecessor_total_tests": 70, "combined_exercised_tests": 100},
        "return_codes": {
            "focused": 0, "predecessor_unit": 0, "predecessor_integration": 0, "predecessor_stress": 0,
            "import_probe": 0, "runner_help_probe": 0,
        },
    }


def build_args() -> dict:
    artifacts = [g5b(), i51c(), i52c(), i53c(), i54c(), i55c(), i56c(), i57c(), i58c(), i59c(), i510d(), i511d(), i512a()]
    names = ["i4_3g5b", "i5_1c", "i5_2c", "i5_3c", "i5_4c", "i5_5c", "i5_6c", "i5_7c", "i5_8c", "i5_9c", "i5_10d", "i5_11d", "i5_12a"]
    args: dict = {}
    for idx, (name, payload) in enumerate(zip(names, artifacts), start=1):
        suffix = "closure"
        args[f"{name}_{suffix}"] = payload
        args[f"{name}_binding"] = binding(f"/evidence/{name}.json", format(idx % 16, "x"))
    args["post_i5_12a_checkpoint_binding"] = binding("/checkpoints/post_i5_12a_delta.tar.gz", "d")
    args["admission_receipt"] = receipt()
    args["admission_receipt_binding"] = binding("/evidence/i5_13a_receipt.json", "e")
    return args


def clone(value):
    return deepcopy(value)
