from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_video_depth_anything_robustness import build_i5_8a_video_depth_anything_robustness_admission
from core.reconstruction.i5_video_depth_anything_robustness_replay import I58B_I58A_FROZEN_SHA256, I58B_SCENARIO_IDS
from tests.i5_8a_helpers import build_args as i58a_build_args


def clone(payload):
    return deepcopy(payload)


def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def admission() -> dict:
    return build_i5_8a_video_depth_anything_robustness_admission(**i58a_build_args())


def receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.8B_REPLAY_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_8b_tests": True,
            "video_depth_anything_provider_regression": True,
            "i5_8a_predecessor_regression": True,
            "authority_guard_regression": True,
            "runtime_runner_help_probe": True,
            "video_depth_anything_inference": False,
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


def runtime_config(profile: str = "efficient") -> dict:
    variants = {
        "efficient": ("vits", "metric_video_depth_anything_vits.pth", "Metric Video Depth Anything Small", 392, 960),
        "balanced": ("vitb", "metric_video_depth_anything_vitb.pth", "Metric Video Depth Anything Base", 448, 1280),
        "ultra": ("vitl", "metric_video_depth_anything_vitl.pth", "Metric Video Depth Anything Large", 518, 1920),
    }
    encoder, filename, label, input_size, max_resolution = variants[profile]
    return {
        "source_dir": "/opt/depth3d/Video-Depth-Anything",
        "checkpoint": f"/opt/depth3d/checkpoints/{filename}",
        "encoder": encoder,
        "model_label": label,
        "input_size": input_size,
        "max_resolution": max_resolution,
        "virtual_fov_degrees": 60.0,
        "max_depth_m": 300.0,
        "point_budget": 250000,
        "live_point_budget": 100000,
        "warmup_size": 256,
        "warmup_runs": 1,
        "allow_tf32": False,
        "matmul_precision": "high",
        "hardware_tier": profile,
        "profile": profile,
        "device": "cuda:0",
        "precision": "fp16",
    }


def runtime_preflight(profile: str = "efficient") -> dict:
    cfg = runtime_config(profile)
    return {
        "video_depth_anything_runtime_ready": True,
        "provider_id": "video-depth-anything-temporal",
        "implementation": "video_depth_anything",
        "source_revision": "a" * 40,
        "source_ready": True,
        "checkpoint_ready": True,
        "checkpoint_sha256": "b" * 64,
        "cuda_available": True,
        "device_names": ["NVIDIA GeForce RTX 2060"],
        "resolved_profile": profile,
        "device": "cuda:0",
        "precision": "fp16",
        "resolved_checkpoint": cfg["checkpoint"],
        "resolved_encoder": cfg["encoder"],
        "runtime_config": cfg,
        "production_config_unchanged": True,
        "production_config": {
            "minimum_correspondences": 8,
            "maximum_relative_residual": 0.35,
            "voxel_scene_fraction": 0.004,
            "pixel_stride": 2,
            "max_voxels": 1500000,
            "minimum_accepted_views": 2,
        },
    }


def _alignment_record(index: int, accepted: bool = True) -> dict:
    return {
        "schema_version": 1,
        "session_id": "session",
        "keyframe_id": f"kf-{index:03d}",
        "frame_id": f"frame-{index:03d}",
        "camera_id": "camera-0",
        "camera_model_name": "PINHOLE",
        "depth_provenance": "Video Depth Anything",
        "status": "accepted" if accepted else "rejected",
        "rejection_code": None if accepted else "robust_scale_fit_rejected",
        "message": "accepted" if accepted else "rejected",
        "sfm_scale_status": "arbitrary",
        "sfm_world_alignment_status": "not_aligned",
        "depth_image_size": {"width": 64, "height": 48},
        "minimum_correspondences": 8,
        "maximum_relative_residual": 0.35,
        "candidate_observations": 20,
        "behind_camera_observations": 0,
        "out_of_bounds_observations": 0,
        "invalid_depth_observations": 0,
        "valid_correspondences": 20,
        "inlier_correspondences": 18 if accepted else 4,
        "robust_relative_residual_limit": 0.35 if accepted else None,
        "median_relative_residual": 0.08 + index * 0.001 if accepted else None,
        "p90_relative_residual": 0.12 + index * 0.001 if accepted else None,
        "alignment": {"mode": "sparse_correspondence_scale", "scale": 1.2} if accepted else None,
        "metric_scale_claimed": False,
        "source_sequence_index": index,
        "temporal_order_index": index,
        "estimated_model_intrinsics_consumed_for_canonical_projection": False,
        "model_metric_depth_promoted_to_canonical_metric_scale": False,
    }


def provider_summary(status: str = "pass", *, views: int = 19, rejected_views: int = 0, profile: str = "efficient") -> dict:
    if status not in {"pass", "degraded"}:
        raise ValueError(status)
    if status == "pass":
        rejected_views = 0
    cfg = runtime_config(profile)
    accepted = views - rejected_views
    records = [_alignment_record(i, i < accepted) for i in range(views)]
    degradation_reasons = []
    if rejected_views:
        degradation_reasons.append(f"alignment_rejected_views:{rejected_views}")
    if status == "degraded" and not degradation_reasons:
        degradation_reasons.append("fusion_capacity_drops:3")
    return {
        "schema_version": 1,
        "phase": "I4.3D1",
        "status": status,
        "pass": True,
        "provider": {
            "provider_id": "video-depth-anything-temporal",
            "implementation": "video_depth_anything",
            "qualification_state": "candidate",
        },
        "selection_request": {
            "mode": "explicit_provider",
            "available_view_count": views,
            "authoritative_sfm_available": True,
            "authoritative_sfm_frame_id": "frame",
            "requested_provider_id": "video-depth-anything-temporal",
            "allow_fallback": False,
            "engineer_override_provider_id": "",
        },
        "selection_decision": {
            "request_mode": "explicit_provider",
            "requested_provider_id": "video-depth-anything-temporal",
            "selected_provider_id": "video-depth-anything-temporal",
            "selected_implementation": "video_depth_anything",
            "considered_provider_ids": ["video-depth-anything-temporal"],
            "automatic": False,
            "engineer_override_applied": False,
            "fallback_used": False,
            "reason_codes": ["i5_8b_frozen_explicit_video_depth_anything_replay"],
        },
        "execution_result": {
            "provider_id": "video-depth-anything-temporal",
            "implementation": "video_depth_anything",
            "provider_family": "temporal_depth",
            "status": "completed" if status == "pass" else "degraded",
            "source_view_count": views,
            "selected_view_count": views,
            "adaptive_view_selection": False,
            "view_selection_reason": "all_registered_authoritative_sfm_views_in_temporal_order",
            "permanent_global_view_cap": None,
            "canonical_geometry_ref": "/tmp/canonical_vda_fused.npz",
            "canonical_frame_id": "frame",
            "native_artifact_refs": ["/tmp/vda_result_manifest.json", "/tmp/vda_depth_evidence.npz", "/tmp/canonical_vda_fused.ply"],
            "degradation_reasons": degradation_reasons,
            "failure_reason": "",
            "progress_event_count": views + 5,
            "final_preview_revision": 0,
            "authority": {
                "canonical_geometry_authoritative": True,
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
            "minimum_correspondences": 8,
            "maximum_relative_residual": 0.35,
            "voxel_scene_fraction": 0.004,
            "pixel_stride": 2,
            "max_voxels": 1500000,
            "minimum_accepted_views": 2,
        },
        "runtime_config": cfg,
        "runtime": {
            "schema_version": 1,
            "python_executable": "/opt/depth3d/python",
            "project_root": "/opt/depth3d/project",
            "source_directory": "/opt/depth3d/Video-Depth-Anything",
            "ready_marker": "/opt/depth3d/ready.json",
            "checkpoint_path": cfg["checkpoint"],
            "source_revision": "a" * 40,
            "source_ready": True,
            "checkpoint_ready": True,
            "adapter_importable": True,
            "upstream_importable": True,
            "torch_available": True,
            "cuda_available": True,
            "device_names": ["NVIDIA GeForce RTX 2060"],
            "available": True,
            "error": "",
        },
        "worker": {
            "schema_version": 1,
            "phase": "I4.3D1",
            "status": "pass",
            "model_id": "video-depth-anything",
            "source_revision": "a" * 40,
            "frame_count": views,
            "depth_npz_path": "/tmp/vda_depth_evidence.npz",
            "frame_records": [
                {
                    "order": i,
                    "keyframe_id": f"kf-{i:03d}",
                    "source_sequence_index": i,
                    "timestamp_ns": (i + 1) * 100000000,
                    "latency_ms": 20.0 + i,
                    "depth_scale_native": "metric",
                    "intrinsics_source_native": "estimated",
                    "native_is_metric": True,
                }
                for i in range(views)
            ],
            "elapsed_s": 2.5,
            "authority": {
                "native_metric_depth_preserved_as_evidence": True,
                "native_estimated_intrinsics_preserved_as_evidence": True,
                "native_metric_depth_promoted_to_canonical_metric_scale": False,
                "native_estimated_intrinsics_promoted_to_canonical_projection": False,
                "camera_pose_authority": "pycolmap_sfm",
            },
        },
        "temporal_sequence": {
            "ordered_keyframe_ids": [f"kf-{i:03d}" for i in range(views)],
            "ordered_source_sequence_indices": list(range(views)),
            "ordered_timestamps_ns": [(i + 1) * 100000000 for i in range(views)],
            "stateful_sequence_required": True,
            "sequence_reordered_by": "source_sequence_index_then_timestamp",
        },
        "alignment": {
            "accepted_views": accepted,
            "rejected_views": rejected_views,
            "rejected_keyframe_ids": [f"kf-{i:03d}" for i in range(accepted, views)],
            "records": records,
        },
        "fusion": {
            "robust_sparse_scene_diagonal_reconstruction_units": 3.0,
            "voxel_size_reconstruction_units": 0.012,
            "statistics": {
                "input_frames": accepted,
                "dropped_new_voxels_capacity": 3 if status == "degraded" and rejected_views == 0 else 0,
            },
            "point_cloud": {"point_count": 15000 + accepted},
            "canonical_npz": "/tmp/canonical_vda_fused.npz",
            "canonical_ply": "/tmp/canonical_vda_fused.ply",
        },
        "integrity": {
            "authoritative_sfm_unchanged": True,
            "registered_images_unchanged": True,
            "authoritative_sfm_digest": "c" * 64,
            "registered_image_sha256": {f"kf-{i:03d}": "d" * 64 for i in range(views)},
            "canonical_pose_layer": "offline_sfm_bundle_adjusted",
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "estimated_model_intrinsics_consumed_for_canonical_projection": False,
            "model_metric_depth_promoted_to_canonical_metric_scale": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_promotion_performed": False,
        },
    }


def scenario_record(scenario_id: str, status: str = "pass", *, profile: str = "efficient", rejected_views: int = 0) -> dict:
    if scenario_id not in I58B_SCENARIO_IDS:
        raise ValueError(scenario_id)
    views = {
        "castle_control": 19,
        "castle_low_overlap": 6,
        "castle_weak_parallax": 7,
        "castle_overexposure": 19,
        "castle_symmetry": 12,
        "castle_dynamic_proxy": 19,
    }[scenario_id]
    summary = provider_summary(status, views=views, rejected_views=rejected_views, profile=profile)
    latencies = [item["latency_ms"] for item in summary["worker"]["frame_records"]]
    return {
        "scenario_id": scenario_id,
        "runtime_status": status,
        "provider_execution_terminal": True,
        "provider_execution_completed": True,
        "source_registered_views": views,
        "processed_views": views,
        "scientific_interpretation": {
            "castle_control": "streaming_cache_registered_sequence_control",
            "castle_low_overlap": "registered_keyframe_spacing_and_cache_continuity_stress_not_mvs_overlap_equivalence",
            "castle_weak_parallax": "narrow_baseline_streaming_temporal_stability_stress",
            "castle_overexposure": "photometric_overexposure_streaming_temporal_stress",
            "castle_symmetry": "registered_subset_repetitive_structure_streaming_temporal_stress",
            "castle_dynamic_proxy": "dynamic_motion_boundary_streaming_cache_stress",
        }[scenario_id],
        "native_metrics": {
            "processed_views": views,
            "alignment_accepted_views": summary["alignment"]["accepted_views"],
            "alignment_rejected_views": summary["alignment"]["rejected_views"],
            "alignment_accepted_fraction": summary["alignment"]["accepted_views"] / float(views),
            "canonical_point_count": summary["fusion"]["point_cloud"]["point_count"],
            "worker_elapsed_s": summary["worker"]["elapsed_s"],
            "frame_latency_count": views,
            "frame_latency_mean_ms": sum(latencies) / len(latencies),
            "frame_latency_median_ms": latencies[len(latencies)//2],
            "frame_latency_p95_ms": latencies[-1],
            "wall_time_s": 10.0,
            "peak_vram_mib": 2500.0,
            "peak_system_ram_used_mib": 9000.0,
        },
        "resource_observation": {
            "peak_vram_mib": 2500.0,
            "peak_system_ram_used_mib": 9000.0,
        },
        "runtime_profile": {
            "resolved_profile": profile,
            "resolved_checkpoint": runtime_config(profile)["checkpoint"],
            "resolved_encoder": runtime_config(profile)["encoder"],
        },
        "artifact_bindings": {
            "frozen_sfm_report": {"path": f"/{scenario_id}/sfm.json", "sha256": "e" * 64},
            "provider_summary": {"path": f"/{scenario_id}/summary.json", "sha256": "f" * 64},
        },
        "failure": None,
        "execution_result": summary["execution_result"],
        "input_integrity_revalidated_after": True,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "source_rematerialized": False,
            "per_frame_alignment_performed": True,
            "sequence_global_alignment_performed": False,
            "native_estimated_intrinsics_are_canonical_authority": False,
            "native_metric_depth_establishes_physical_metric_scale": False,
            "permanent_global_view_cap": None,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_promotion_performed": False,
        },
    }


def build_args() -> dict:
    return {
        "i5_8a_admission": admission(),
        "i5_8a_binding": binding("/evidence/i5_8a.json", I58B_I58A_FROZEN_SHA256),
        "replay_receipt": receipt(),
        "replay_receipt_binding": binding("/evidence/i5_8b_receipt.json", "f" * 64),
        "runtime_preflight": runtime_preflight(),
        "scenario_records": {sid: scenario_record(sid) for sid in I58B_SCENARIO_IDS},
    }
