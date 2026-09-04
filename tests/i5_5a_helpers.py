from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding

I54C_SHA = "a" * 64
RECEIPT_55A_SHA = "b" * 64


def clone(value):
    return deepcopy(value)


def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def castle_source_dir(tmp_path: Path, *, count: int = 19) -> Path:
    root = tmp_path / "castle-P19"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        # Admission hashes source bytes only; decoding/materialization belongs to I5.5B.
        (root / f"{index:04d}.jpg").write_bytes((f"castle-{index:04d}-" * 8).encode("ascii"))
    (root / "K.txt").write_text("ignored calibration metadata\n", encoding="utf-8")
    return root


def passing_i5_4c() -> dict:
    authority = {
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
    }
    return {
        "schema_version": 1,
        "phase": "I5.4C",
        "status": "forward_motion_kitti_robustness_qualified_and_closed",
        "pass": True,
        "quality_status": "rejected",
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "i5_4b_real_evidence_reused": True,
            "negative_quality_result_preserved": True,
            "sfm_rerun": False,
            "source_rematerialized": False,
            "window_retuned": False,
            "quality_replay_rerun": False,
            "dense_provider_rerun": False,
            "provider_specific_robustness_started": False,
            "mp_sfm_strategy_evaluation_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
        },
        "qualification": {
            "quality_status": "rejected",
            "operating_envelope": "outside_current_production_sparse_sfm_envelope_on_admitted_kitti_style_forward_motion_case",
            "primary_bottleneck": "incremental_mapping_no_sparse_model",
            "rejection_stage": "incremental_mapping",
            "registered_images": 0,
            "input_images": 60,
            "mapping_produced_sparse_model": False,
            "registration_attempted": True,
            "sfm_executed": True,
            "forward_motion_geometry_revalidation_status": "unavailable_no_sparse_model",
            "authoritative_forward_motion_geometry_proven": False,
            "i2_4_geometry_available": False,
            "retry_until_quality_pass": False,
            "result_driven_window_tuning": False,
            "universal_forward_motion_robustness_claimed": False,
            "kitti_dataset_universal_robustness_claimed": False,
            "dense_reconstruction_robustness_claimed": False,
            "provider_robustness_claimed": False,
            "metric_scale_claimed": False,
        },
        "authority": authority,
        "decision": {
            "i4_closed": True,
            "i5_authorized": True,
            "i5_1_complete": True,
            "i5_2_complete": True,
            "i5_3_complete": True,
            "i5_4_started": True,
            "i5_4a_complete": True,
            "i5_4b_complete": True,
            "i5_4c_complete": True,
            "i5_4_complete": True,
            "i5_5_authorized": True,
            "i5_5_started": False,
            "provider_specific_robustness_started": False,
            "mp_sfm_authorized": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "threshold_relaxation_authorized": False,
            "production_default_modified": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "roadmap_guard": {
            "i5_5_dynamic_scene_authorized_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.5 dynamic-scene handling robustness.",
    }


def passing_i5_5a_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.5A_ADMISSION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_5a_tests": True,
            "predecessor_i5_4_regression": True,
            "source_identity_regression": True,
            "controlled_dynamic_proxy_design_regression": True,
            "authority_guard_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "sfm_rerun": False,
            "stress_images_materialized": False,
            "dense_provider_rerun": False,
            "dynamic_masking_added": False,
            "semantic_segmentation_added": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 18,
            "predecessor_unit_tests": 20,
            "predecessor_integration_tests": 8,
            "predecessor_stress_tests": 8,
            "predecessor_total_tests": 36,
            "combined_exercised_tests": 54,
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
