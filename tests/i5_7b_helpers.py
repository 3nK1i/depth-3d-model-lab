from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_rolling_depth_robustness import build_i5_7a_rolling_depth_robustness_admission
from core.reconstruction.i5_rolling_depth_robustness_replay import I57B_I57A_FROZEN_SHA256, I57B_SCENARIO_IDS
from tests.i5_7a_helpers import build_args as i57a_build_args


def clone(payload: dict) -> dict:
    return deepcopy(payload)


def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def admission() -> dict:
    payload = build_i5_7a_rolling_depth_robustness_admission(**i57a_build_args())
    return payload


def receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.7B_REPLAY_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_7b_tests": True,
            "rolling_depth_provider_regression": True,
            "i5_7a_predecessor_regression": True,
            "authority_guard_regression": True,
            "runtime_runner_help_probe": True,
            "rolling_depth_inference": False,
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
            "focused_tests": 12,
            "predecessor_total_tests": 30,
            "combined_exercised_tests": 42,
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


def runtime_preflight() -> dict:
    return {
        "rolling_depth_runtime_ready": True,
        "provider_id": "rolling-depth-temporal",
        "source_revision": "a" * 40,
        "runtime_lock_valid": True,
        "checkpoint_ready": True,
        "cuda_available": True,
        "device_names": ["NVIDIA GeForce RTX 2060"],
        "device": "cuda:0",
        "precision": "fp16",
        "production_config_unchanged": True,
    }


def provider_summary(status: str = "pass", *, views: int = 19, residual: float = 0.10, accepted_views: int | None = None) -> dict:
    if status not in {"pass", "degraded", "rejected"}:
        raise ValueError(status)
    rejected = status == "rejected"
    accepted = 0 if rejected else (views if accepted_views is None else accepted_views)
    result_status = "failed" if rejected else ("degraded" if status == "degraded" else "completed")
    failure = "global_sequence_alignment_rejected:relative_residual_too_large" if rejected else ""
    return {
        "schema_version": 1,
        "phase": "I4.3D2",
        "status": status,
        "pass": not rejected,
        "provider": {
            "provider_id": "rolling-depth-temporal",
            "implementation": "rolling_depth",
            "qualification_state": "candidate",
        },
        "selection_request": {
            "mode": "explicit_provider",
            "available_view_count": views,
            "authoritative_sfm_available": True,
            "authoritative_sfm_frame_id": "frame",
            "requested_provider_id": "rolling-depth-temporal",
            "allow_fallback": False,
            "engineer_override_provider_id": "",
        },
        "selection_decision": {
            "request_mode": "explicit_provider",
            "requested_provider_id": "rolling-depth-temporal",
            "selected_provider_id": "rolling-depth-temporal",
            "selected_implementation": "rolling_depth",
            "considered_provider_ids": ["rolling-depth-temporal"],
            "automatic": False,
            "engineer_override_applied": False,
            "fallback_used": False,
            "reason_codes": ["i5_7b_frozen_explicit_rolling_depth_replay"],
        },
        "execution_result": {
            "provider_id": "rolling-depth-temporal",
            "implementation": "rolling_depth",
            "status": result_status,
            "source_view_count": views,
            "selected_view_count": views,
            "adaptive_view_selection": False,
            "view_selection_reason": "all_registered_authoritative_sfm_views_in_temporal_order",
            "permanent_global_view_cap": None,
            "canonical_geometry_ref": "" if rejected else "/tmp/canonical.npz",
            "failure_reason": failure,
            "authority": {
                "canonical_geometry_authoritative": not rejected,
                "single_authoritative_sfm_frame_preserved": True,
                "camera_pose_authority_owner": "pycolmap_sfm",
                "metric_scale_claimed": False,
                "measurement_authoritative": False,
                "certified_metrology": False,
                "surface_mesh_generated": False,
                "production_filter_policy_changed": False,
                "provider_promotion_performed": False,
            },
        },
        "config": {
            "minimum_sequence_correspondences": 24,
            "minimum_contributing_views": 2,
            "maximum_relative_residual": 0.35,
            "voxel_scene_fraction": 0.004,
            "pixel_stride": 2,
            "max_voxels": 1500000,
            "minimum_accepted_views": 2,
            "processing_resolution": 512,
            "dilations": [1, 25],
            "snippet_lengths": [3],
            "refine_step": 0,
            "max_vae_batch_size": 1,
            "unload_snippet": True,
            "restore_resolution": True,
            "seed": 42,
        },
        "runtime": {
            "source_revision": "a" * 40,
            "runtime_lock_valid": True,
            "checkpoint_ready": True,
            "cuda_available": True,
        },
        "worker": {
            "source_revision": "a" * 40,
            "depth_semantics": "temporally_coaligned_relative_depth",
            "per_frame_alignment": False,
        },
        "depth_canonicalization": {
            "native_depth_polarity": "decreasing_with_canonical_camera_depth",
            "canonical_depth_polarity": "increasing_with_canonical_camera_depth",
            "multiplier": -1.0,
            "sequence_order_changed": False,
        },
        "temporal_sequence": {
            "ordered_keyframe_ids": [f"kf-{i:03d}" for i in range(views)],
            "ordered_source_sequence_indices": list(range(views)),
            "ordered_timestamps_ns": [i * 100000000 for i in range(views)],
            "sequence_reordered_by": "source_sequence_index_then_timestamp",
            "native_temporal_coalignment_preserved": True,
            "canonicalization_preserves_sequence_order": True,
        },
        "alignment": {
            "alignment_scope": "one_global_affine_for_entire_temporal_sequence",
            "accepted": not rejected,
            "initial_correspondences": 100,
            "retained_correspondences": 80 if not rejected else 20,
            "contributing_views": views if not rejected else max(1, views // 2),
            "scale_to_reconstruction": None if rejected else 1.5,
            "shift_to_reconstruction": None if rejected else -0.2,
            "median_relative_residual": residual,
            "p90_relative_residual": residual * 1.5,
            "maximum_relative_residual": 0.35,
            "message": "accepted" if not rejected else "relative_residual_too_large",
        },
        "fusion": {
            "performed": not rejected,
            "accepted_views": accepted,
            "rejected_views": [] if rejected else [f"kf-{i:03d}" for i in range(accepted, views)],
            "point_cloud": None if rejected else {"point_count": 10000 + accepted},
            "statistics": {} if rejected else {"input_frames": accepted},
        },
        "integrity": {
            "authoritative_sfm_unchanged": True,
            "registered_images_unchanged": True,
            "authoritative_sfm_digest": "b" * 64,
            "registered_image_sha256": {f"kf-{i:03d}": "c" * 64 for i in range(views)},
            "canonical_pose_layer": "offline_sfm_bundle_adjusted",
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "rolling_depth_pose_authority_promoted": False,
            "sequence_affine_establishes_physical_metric_scale": False,
            "per_frame_alignment_performed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_promotion_performed": False,
        },
    }


def scenario_record(scenario_id: str, status: str = "pass", *, views: int | None = None, residual: float = 0.10) -> dict:
    if scenario_id not in I57B_SCENARIO_IDS:
        raise ValueError(scenario_id)
    default_views = {
        "castle_control": 19,
        "castle_low_overlap": 6,
        "castle_weak_parallax": 7,
        "castle_overexposure": 19,
        "castle_symmetry": 12,
        "castle_dynamic_proxy": 19,
    }[scenario_id]
    count = default_views if views is None else views
    summary = provider_summary(status, views=count, residual=residual)
    return {
        "scenario_id": scenario_id,
        "runtime_status": status,
        "provider_execution_terminal": True,
        "provider_execution_completed": status != "rejected",
        "source_registered_views": count,
        "processed_views": count,
        "scientific_interpretation": {
            "castle_control": "temporal_sequence_control",
            "castle_low_overlap": "temporal_support_spacing_stress_not_mvs_overlap_equivalence",
            "castle_weak_parallax": "narrow_baseline_temporal_geometry_stress",
            "castle_overexposure": "photometric_overexposure_temporal_stress",
            "castle_symmetry": "registered_subset_symmetry_temporal_stress",
            "castle_dynamic_proxy": "dynamic_motion_boundary_temporal_stress",
        }[scenario_id],
        "native_metrics": {
            "processed_views": count,
            "alignment_accepted": status != "rejected",
            "initial_correspondences": summary["alignment"]["initial_correspondences"],
            "retained_correspondences": summary["alignment"]["retained_correspondences"],
            "contributing_views": summary["alignment"]["contributing_views"],
            "alignment_median_relative_residual": summary["alignment"]["median_relative_residual"],
            "alignment_p90_relative_residual": summary["alignment"]["p90_relative_residual"],
            "accepted_fusion_views": summary["fusion"]["accepted_views"],
            "canonical_point_count": None if status == "rejected" else summary["fusion"]["point_cloud"]["point_count"],
            "wall_time_s": 12.5,
            "peak_vram_mib": 2048.0,
            "peak_system_ram_used_mib": 8192.0,
        },
        "resource_observation": {
            "gpu_measurement_scope": "device_wide_nvidia_smi_peak_used_memory",
            "host_measurement_scope": "system_wide_used_memory",
            "peak_vram_mib": 2048.0,
            "peak_system_ram_used_mib": 8192.0,
        },
        "artifact_bindings": {
            "frozen_sfm_report": {"path": f"/{scenario_id}/sfm.json", "sha256": "d" * 64},
            "provider_summary": {"path": f"/{scenario_id}/summary.json", "sha256": "e" * 64},
        },
        "failure": None if status != "rejected" else {
            "quality_terminal": True,
            "reason": "global_sequence_alignment_rejected:relative_residual_too_large",
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "source_rematerialized": False,
            "per_frame_alignment_performed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_promotion_performed": False,
            "permanent_global_view_cap": None,
        },
    }


def build_args() -> dict:
    return {
        "i5_7a_admission": admission(),
        "i5_7a_binding": binding("/evidence/i5_7a.json", I57B_I57A_FROZEN_SHA256),
        "replay_receipt": receipt(),
        "replay_receipt_binding": binding("/evidence/i5_7b_receipt.json", "f" * 64),
        "runtime_preflight": runtime_preflight(),
        "scenario_records": {sid: scenario_record(sid) for sid in I57B_SCENARIO_IDS},
    }


def recovery_artifact_bindings() -> dict:
    names = (
        "provider_summary", "provider_input_manifest", "worker_result_manifest",
        "worker_evidence_npz", "canonical_npz", "canonical_ply",
        "original_attempt_failure", "original_scenario_failure", "original_runtime_preflight",
    )
    return {name: {"path": f"/recovery/{name}", "sha256": "a" * 64} for name in names}


def recovery_provenance() -> dict:
    return {
        "scenario_id": "castle_control",
        "original_attempt_path": "/evidence/i5_7b_rolling_depth_production_native_20260908_110816",
        "original_manifest": {"path": "/evidence/original.sha256_manifest.txt", "sha256": "b" * 64},
        "original_console_log": {"path": "/evidence/original.console.log", "sha256": "c" * 64},
        "original_wrapper_failure_after_provider_execution": True,
        "original_exception_message": "RollingDepth execution result.metric_scale_claimed must be False; found None.",
        "provider_execution_rerun": False,
        "retry_until_pass": False,
        "thresholds_modified": False,
        "production_default_modified": False,
        "sfm_rerun": False,
        "mvs_rerun": False,
        "manifest_verified": True,
        "console_log_verified": True,
        "missing_performance_fields": ["wall_time_s", "peak_vram_mib", "peak_system_ram_used_mib"],
    }
