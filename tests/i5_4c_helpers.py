from __future__ import annotations
from copy import deepcopy
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding

I54B_SHA = "7" * 64
RECEIPT_54C_SHA = "8" * 64

def clone(value):
    return deepcopy(value)

def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)

def rejected_i5_4b() -> dict:
    empty_q = {"p10":None,"p25":None,"p50":None,"p75":None,"p90":None}
    motion = {
        "adjacent_registered_motion_pairs":0,
        "records":[],
        "forward_component_fraction_quantiles":deepcopy(empty_q),
        "lateral_component_fraction_quantiles":deepcopy(empty_q),
        "signed_forward_component_fraction_quantiles":deepcopy(empty_q),
        "median_forward_component_fraction":None,
        "median_lateral_component_fraction":None,
        "authoritative_forward_motion_geometry_proven":False,
        "forward_motion_label":"inconclusive_forward_motion_proxy_not_revalidated",
        "absolute_translation_magnitude_cross_run_comparison_permitted":False,
        "arbitrary_gauge":True,
    }
    return {
        "schema_version":1,"phase":"I5.4B",
        "status":"current_frozen_foundation_forward_motion_kitti_replayed","pass":True,"quality_status":"rejected",
        "scope":{
            "current_frozen_foundation_real_replay":True,"i5_4a_admission_reused":True,"frozen_60_frame_window_reused":True,
            "production_native_pycolmap_replay":True,"sequential_matching_reused":True,"source_images_transformed":False,
            "window_retuned":False,"retry_until_quality_pass":False,"dense_provider_rerun":False,
            "provider_specific_robustness_started":False,"mp_sfm_strategy_evaluation_started":False,"thresholds_modified":False,
            "production_default_modified":False,"auto_policy_work_started":False,"hybrid_experiment_started":False,
        },
        "evidence_bindings":{
            "i5_4a_admission":{"path":"/evidence/i5_4a.json","sha256":"4"*64},
            "i5_4b_replay_receipt":{"path":"/evidence/i5_4b_receipt.json","sha256":"5"*64},
        },
        "source_binding":{
            "source_asset_set_sha256":"a"*64,"window_asset_set_sha256":"b"*64,"window_images":60,
            "source_reverified_before":True,"source_window_reverified_after":True,"staged_inputs_reverified":True,
        },
        "replay":{
            "input_images":60,"registered_images":0,"registration_ratio":0.0,"full_input_scope_registered":False,
            "partial_registration_preserved":False,"registered_names":[],"quality_status":"rejected",
            "quality_assessment":{"status":"rejected","usable":False,"reasons":["PyCOLMAP incremental mapping produced no sparse model."]},
            "statistics":{"input_keyframes":60,"registered_keyframes":0,"landmarks":0,"observations":0,"mean_reprojection_error_px":None},
            "geometry":{
                "camera_motion":deepcopy(motion),
                "covisibility":{"registered_graph_largest_component":0,"registered_graph_connected":False,"adjacent_pair_common_landmarks_quantiles":deepcopy(empty_q),"adjacent_selected_pair_count":0},
                "parallax":{"landmarks_with_two_or_more_registered_observations":0,"maximum_pair_parallax_degrees_quantiles":deepcopy(empty_q),"maximum_baseline_to_range_ratio_quantiles":deepcopy(empty_q),"scale_invariant":True,"absolute_baseline_cross_scenario_comparison_permitted":False,"methodology_lineage":"I2.4 maximum-supporting-pair parallax and baseline/range formulation"},
            },
            "pre_sfm_ingestion_rejected":False,"rejection_stage":"incremental_mapping",
            "rejection_reason":"PyCOLMAP incremental mapping produced no sparse model.",
            "mapping_produced_sparse_model":False,"registration_attempted":True,"sfm_executed":True,
            "authority":{"camera_pose_authority_owner":"pycolmap_sfm","authoritative_sfm_modified":False,"metric_scale_claimed":False,"measurement_authoritative":False,"certified_metrology":False,"surface_mesh_generated":False,"permanent_global_view_cap":None},
        },
        "forward_motion_revalidation":deepcopy(motion),
        "qualification":{
            "quality_status":"rejected","forward_motion_robustness_conclusion":"rejected_before_forward_motion_geometry_revalidation",
            "authoritative_forward_motion_geometry_proven":False,"registration":"0/60","retry_until_quality_pass":False,"result_driven_window_tuning":False,
            "universal_forward_motion_robustness_claimed":False,"kitti_dataset_universal_robustness_claimed":False,
            "dense_reconstruction_robustness_claimed":False,"provider_robustness_claimed":False,
        },
        "regression_summary":{"focused_i5_4b_tests":23,"predecessor_tests":51,"combined_exercised_tests":74,"all_required_suites_passed":True,"test_counts_are_not_runtime_quality_scores":True},
        "authority":{"camera_pose_authority_owner":"pycolmap_sfm","authoritative_sfm_modified":False,"provider_selected":False,"provider_promoted":False,"auto_policy_implemented":False,"hybrid_authorized":False,"metric_scale_claimed":False,"measurement_authoritative":False,"certified_metrology":False,"surface_mesh_generated":False,"permanent_global_view_cap":None},
        "decision":{
            "i4_closed":True,"i5_authorized":True,"i5_1_complete":True,"i5_2_complete":True,"i5_3_complete":True,
            "i5_4_started":True,"i5_4a_complete":True,"i5_4b_complete":True,"i5_4c_qualification_closure_authorized":True,
            "i5_4c_complete":False,"i5_4_complete":False,"i5_5_authorized":False,"provider_specific_robustness_started":False,
            "mp_sfm_authorized":False,"provider_selected":False,"provider_promoted":False,"provider_ranking_created":False,
            "universal_provider_winner_declared":False,"global_scalar_score_created":False,"auto_policy_implemented":False,"hybrid_authorized":False,
            "threshold_relaxation_authorized":False,"production_default_modified":False,"metric_scale_claimed":False,
            "measurement_authoritative":False,"certified_metrology":False,"surface_mesh_generated":False,
        },
        "roadmap_guard":{"i5_4c_qualification_closure_authorized_not_started":True,"i5_5_dynamic_scene_not_started":True,"provider_specific_robustness_not_started":True,"mp_sfm_deferred_to_i5_10":True,"hybrid_deferred_to_i5_12_if_justified":True,"auto_policy_deferred_to_i5_13":True},
        "next":"I5.4C forward-motion / KITTI robustness qualification and closure.",
    }

def passing_i5_4c_receipt() -> dict:
    return {
        "schema_version":1,"phase":"I5.4C_CLOSURE_RECEIPT","status":"pass","pass":True,
        "scope":{"focused_i5_4c_tests":True,"predecessor_i5_4b_regression":True,"predecessor_i5_4a_regression":True,"evidence_only_closure_contract_regression":True,"negative_quality_preservation_regression":True,"authority_guard_regression":True,"import_probe":True,"runner_help_probe":True,"real_sfm_executed_by_test_receipt":False,"source_rematerialized":False,"dense_provider_rerun":False,"thresholds_modified":False,"production_default_modified":False,"mp_sfm_started":False,"auto_policy_implemented":False,"hybrid_authorized":False},
        "counts":{"focused_tests":16,"predecessor_unit_tests":20,"predecessor_integration_tests":8,"predecessor_stress_tests":6,"predecessor_total_tests":34,"combined_exercised_tests":50},
        "return_codes":{"focused":0,"predecessor_unit":0,"predecessor_integration":0,"predecessor_stress":0,"import_probe":0,"runner_help_probe":0},
    }
