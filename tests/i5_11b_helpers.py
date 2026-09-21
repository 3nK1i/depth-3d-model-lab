from __future__ import annotations

from copy import deepcopy

I511A_SHA = "9d1f02e4cc0cc90eb909cf17ea5621a3aaf580a75ddf2a2269ca4402d5cdf1e6"
POST_I510D_DELTA_SHA = "773f63c31d27d5746fe788f9a0463792a6d5061676fea6f44cb004f99e4a6247"


def binding(path: str, sha: str) -> dict[str, str]:
    return {"path": path, "sha256": sha}


def synthetic_i511a_admission() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.11A",
        "status": "failure_mode_inventory_and_experiment_admitted",
        "pass": True,
        "fault_matrix": {
            "cuda_oom": {
                "admitted": True,
                "injection": "synthetic_cuda_oom_signature_in_isolated_fault_worker",
                "consumes_real_gpu_memory": False,
            },
            "disk_enospc": {
                "admitted": True,
                "injection": "synthetic_errno_enospc_in_isolated_owned_workspace",
                "fills_real_filesystem": False,
                "generic_failure_classifier_has_first_class_enospc": False,
            },
            "crash_orphan_descendant": {
                "admitted": True,
                "existing_g1_control": True,
            },
        },
        "i5_11b_preflight_requirements": {
            "bind_i5_10d_closure": True,
            "bind_i5_11a_admission": True,
            "bind_post_i5_10d_delta_checkpoint": True,
            "execute_fault_injection": False,
            "freeze_attempt_directory_schema": True,
            "freeze_clean_recovery_after_each": True,
            "freeze_expected_failure_signatures": True,
            "freeze_three_fault_mechanisms": True,
            "preflight_oom_classifier_with_synthetic_exception_only": True,
            "real_disk_fill": False,
            "real_gpu_oom": False,
            "record_enospc_classifier_gap_without_production_change": True,
            "reverify_repository_anchors": True,
            "verify_disk_free_space_without_pressure": True,
            "verify_no_fault_processes_are_running": True,
        },
        "i5_11c_campaign_contract": {
            "fault_scenarios": ["cuda_oom", "disk_enospc", "crash_orphan_descendant"],
            "fault_attempts_total": 3,
            "recovery_attempts_total": 3,
            "fault_attempts_each": 1,
            "recovery_attempts_each": 1,
            "attempt_directories_new_and_immutable": True,
            "fault_workspace_supervisor_owned": True,
            "clean_recovery_after_each": True,
            "retry_until_pass": False,
            "result_driven_retuning": False,
            "production_default_modified": False,
            "real_resource_exhaustion": False,
        },
        "decision": {
            "i5_10_complete": True,
            "i5_11_started": True,
            "i5_11a_complete": True,
            "i5_11b_authorized": True,
            "i5_11b_started": False,
            "i5_11c_authorized": False,
            "i5_11c_started": False,
            "i5_11d_authorized": False,
            "i5_11d_started": False,
            "i5_12_started": False,
            "i5_13_started": False,
            "i5_14_started": False,
            "production_default_modified": False,
            "production_failure_classifier_change_authorized": False,
            "real_resource_exhaustion_authorized": False,
        },
    }


def synthetic_i511b_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.11B_PREFLIGHT_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_11b_tests": True,
            "engine_hardening_regression": True,
            "failure_classification_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "synthetic_exception_classification_probe": True,
            "fault_injection": False,
            "supervised_fault_worker_execution": False,
            "real_gpu_oom": False,
            "real_disk_fill": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "production_default_modified": False,
            "production_failure_classifier_modified": False,
            "production_recovery_engine_modified": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
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


def synthetic_anchors() -> dict:
    return {"x": {"path": "/repo/x", "sha256": "a" * 64, "verified": True}}


def synthetic_classifier_probe() -> dict:
    return {
        "probe_is_pure_in_process_classification_only": True,
        "fault_worker_executed": False,
        "cuda_oom": {
            "observed_kind": "cuda_out_of_memory",
            "expected_kind": "cuda_out_of_memory",
            "first_class": True,
            "requires_worker_restart": True,
            "technical_details": "RuntimeError: simulated cuda out-of-memory",
        },
        "disk_enospc": {
            "observed_kind": "unknown_model_error",
            "first_class": False,
            "expected_current_kind": "unknown_model_error",
            "gap_preserved": True,
            "raw_signature_preserved": True,
            "technical_details": "OSError: [Errno 28] No space left on device",
        },
    }


def synthetic_disk_observation(*, free_bytes: int = 8 * 1024**3) -> dict:
    return {
        "path": "/tmp",
        "total_bytes": 100 * 1024**3,
        "used_bytes": 92 * 1024**3,
        "free_bytes": free_bytes,
        "minimum_free_bytes_for_i5_11c": 4 * 1024**3,
        "safety_floor_met": free_bytes >= 4 * 1024**3,
        "pressure_performed": False,
        "files_created_for_probe": False,
    }


def mutate(payload: dict, *path: str, value) -> dict:
    out = deepcopy(payload)
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return out
