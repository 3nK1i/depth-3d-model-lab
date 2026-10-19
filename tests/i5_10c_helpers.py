from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

I510B_SHA = "0db52edcf9805d66656f874256b5219ded26afe2b25666024c7b9bd3ed02dae4"
I510B_DELTA_SHA = "5fc21a70bc9825036037c5aa953aa67f44dc94f154a795282a50a590388bcf3a"
I510C_RECEIPT_SHA = "c" * 64


def synthetic_preflight(tmp_path: Path | None = None) -> dict[str, Any]:
    root = Path('/dataset') if tmp_path is None else tmp_path / 'dataset'
    selected = []
    for i in range(60):
        name = f"{i:010d}.png"
        selected.append({"name": name, "order": i, "sha256": f"{i+1:064x}"[-64:], "size_bytes": 100 + i})
    return {
        "schema_version": 1,
        "phase": "I5.10B",
        "status": "mp_sfm_runtime_and_evidence_preflight_frozen",
        "pass": True,
        "historical_baseline": {
            "baseline_matching_strategy": "sequential",
            "baseline_sequential_overlap": 10,
            "input_images": 60,
            "registered_images": 0,
            "mapping_produced_sparse_model": False,
            "primary_bottleneck": "incremental_mapping_no_sparse_model",
            "quality_status": "rejected",
            "authoritative_forward_motion_geometry_proven": False,
        },
        "source_revalidation": {
            "dataset_directory": str(root),
            "full_source_image_count": 4544,
            "full_source_asset_set_sha256": "a" * 64,
            "window_image_count": 60,
            "window_asset_set_sha256": "b" * 64,
            "full_source_identity_verified": True,
            "exact_60_image_window_verified": True,
            "selected_source_sha256_verified": True,
            "selected_files": selected,
        },
        "runtime_preflight": {
            "available": True,
            "backend_ready": True,
            "cpu_fallback": False,
            "cuda_devices": 1,
            "cuda_ready": True,
            "device_mode": "cuda",
            "distribution": "pycolmap-cuda12",
            "distribution_version": "4.1.1",
            "error": "",
            "pycolmap_version": "4.1.1",
            "python_executable": "/opt/depth3d-pycolmap-probe/bin/python3.11",
            "recommended_version": True,
            "required_apis_available": True,
            "apis": {"match_sequential": True, "match_exhaustive": True, "incremental_mapping": True},
        },
        "candidate_plan": {
            "attempt_count": 1,
            "baseline_rerun": False,
            "camera_mode": "single",
            "camera_model": "SIMPLE_RADIAL",
            "cpu_fallback": False,
            "device_mode": "cuda",
            "exhaustive_pair_count": 1770,
            "feature_strategy": "colmap_sift",
            "image_sequence_fps": 10.0,
            "input_images": 60,
            "matching_strategy": "exhaustive",
            "max_keyframes": 60,
            "output_directory_must_be_new_and_immutable": True,
            "profile": "efficient",
            "scenario_id": "kitti_forward_motion",
            "timeout_minutes": 60.0,
        },
        "resource_policy": {
            "adaptive_image_reduction": False,
            "cpu_fallback_must_be_explicit": True,
            "cuda_preferred": True,
            "disk_failure_experiment": False,
            "exhaustive_pair_count": 1770,
            "i5_11_owns_failure_injection": True,
            "input_images": 60,
            "oom_recovery_experiment": False,
            "timeout_minutes": 60.0,
        },
        "i5_10c_evidence_schema": {
            "attempt_directory_must_be_new_and_immutable": True,
            "no_retry_artifact_overwrite": True,
            "real_sfm_report_required_when_sparse_model_exists": True,
            "recognized_no_sparse_model_rejection_preserved_without_report": True,
            "required_execution_records": ["candidate_command", "attempt_started_utc", "attempt_finished_utc", "return_code", "console_log_sha256", "source_integrity_after", "worker_integrity_after"],
            "required_pre_execution_bindings": ["i5_10b_preflight", "post_i5_10a_source_checkpoint", "exact_60_image_source_manifest", "runtime_probe", "repository_anchors"],
            "runtime_integrity_failure_is_not_quality_rejection": True,
            "schema_version": 1,
            "terminal_outcomes": ["pass", "degraded", "rejected", "runtime_integrity_failure"],
        },
        "i5_10c_execution_freeze": {
            "authorized": True,
            "baseline_rerun": False,
            "candidate_attempt_count": 1,
            "matching_policy_is_primary_changed_variable": True,
            "production_default_modified": False,
            "real_candidate_run_count": 1,
            "recognized_quality_rejection_is_scientific_evidence": True,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "runtime_integrity_failure_is_experiment_incomplete": True,
            "same_exact_60_image_window_required": True,
            "same_exact_source_bytes_required": True,
            "sequential_overlap_not_applicable_to_exhaustive": True,
            "threshold_relaxation": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "candidate_can_replace_production_default": False,
            "candidate_reconstruction_is_evaluation_only": True,
            "candidate_reconstruction_uses_pycolmap_authority": True,
            "certified_metrology": False,
            "independent_sfm_gauges_preserved": True,
            "measurement_authoritative": False,
            "metric_scale_claimed": False,
            "permanent_global_view_cap": None,
            "surface_mesh_generated": False,
        },
        "decision": {
            "i5_10a_complete": True,
            "i5_10b_complete": True,
            "i5_10c_authorized": True,
            "i5_10c_started": False,
            "i5_10d_authorized": False,
            "i5_10d_started": False,
            "i5_11_started": False,
            "i5_12_started": False,
            "i5_13_started": False,
            "i5_14_started": False,
            "production_default_modified": False,
            "real_mp_sfm_execution_started": False,
            "threshold_relaxation_authorized": False,
            "universal_matching_strategy_winner_declared": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "repository_anchors": {
            "scripts/run_reconstruction_pycolmap_smoke.py": {"path": "/repo/scripts/run_reconstruction_pycolmap_smoke.py", "sha256": "d" * 64, "verified": True},
            "scripts/reconstruction_pycolmap_worker.py": {"path": "/repo/scripts/reconstruction_pycolmap_worker.py", "sha256": "e" * 64, "verified": True},
        },
        "evidence_bindings": {
            "post_i5_10a_source_checkpoint": {"path": "/checkpoints/post_i5_10a.tar.gz", "sha256": "f" * 64}
        },
    }


def synthetic_receipt() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "I5.10C_EVALUATION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {"focused_i5_10c_tests": True, "predecessor_i5_10b_regression": True, "real_sfm_executed_by_test_receipt": False},
        "counts": {"focused_tests": 25, "predecessor_total_tests": 80, "combined_exercised_tests": 105},
        "return_codes": {"focused": 0, "predecessor_unit": 0, "predecessor_integration": 0, "predecessor_stress": 0, "import_probe": 0, "runner_help_probe": 0},
    }


def synthetic_report(*, quality: str = "accepted", registered: int = 60) -> dict[str, Any]:
    names = [f"{i:010d}.png" for i in range(60)]
    workspace_images = [
        {"keyframe_id": f"kf{i}", "source_locator": f"/stage/{name}", "staged_image_path": f"/workspace/{name}"}
        for i, name in enumerate(names)
    ]
    cameras = []
    for i in range(registered):
        matrix = [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,float(i)],[0.0,0.0,0.0,1.0]]
        cameras.append({"keyframe_id": f"kf{i}", "T_reconstruction_from_camera": matrix})
    landmarks = []
    for j in range(max(1, registered * 3)):
        obs = [{"keyframe_id": f"kf{i}"} for i in range(min(registered, 3))]
        landmarks.append({"landmark_id": f"lm{j}", "position_reconstruction": [float(j), 0.0, 10.0], "observations": obs})
    q = {"status": quality, "usable": quality != "rejected", "reasons": [], "mean_track_length": 3.0}
    return {
        "phase": "8.3D",
        "real_sfm_executed": True,
        "quality_assessment": q,
        "sfm_result": {
            "backend_id": "pycolmap-offline-sfm",
            "gauge": {"scale_status": "arbitrary", "unit_to_metre": None},
            "metadata": {"real_sfm": True, "matching_strategy": "exhaustive", "quality_assessment": q},
            "workspace": {"images": workspace_images},
            "statistics": {"input_keyframes": 60, "registered_keyframes": registered, "landmarks": len(landmarks), "observations": len(landmarks)*3, "mean_reprojection_error_px": 1.2},
            "cameras": cameras,
            "landmarks": landmarks,
        },
    }


def clone(value: Any) -> Any:
    return deepcopy(value)
