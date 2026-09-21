from __future__ import annotations

from copy import deepcopy

I511A_SHA = "9d1f02e4cc0cc90eb909cf17ea5621a3aaf580a75ddf2a2269ca4402d5cdf1e6"
I511B_SHA = "e72c7d19ef403aed55e80a9ab3ff6a6f76b2b7ba17706bd3524fda4b8bf3f275"
POST_I510D_DELTA_SHA = "773f63c31d27d5746fe788f9a0463792a6d5061676fea6f44cb004f99e4a6247"
FAULT_WORKER_SHA = "5d3557e1c07b87816800008de0569e1f86d388657f2fbedf1200b8518c58fae7"


def binding(path: str, sha: str) -> dict[str, str]:
    return {"path": path, "sha256": sha}


def synthetic_i511b_preflight() -> dict:
    scenarios = ["cuda_oom", "disk_enospc", "crash_orphan_descendant"]
    plans = {}
    signatures = {
        "cuda_oom": "simulated cuda out-of-memory",
        "disk_enospc": "No space left on device",
        "crash_orphan_descendant": "I511_FAULT=crash_orphan_descendant",
    }
    for index, scenario in enumerate(scenarios, 1):
        plans[scenario] = {
            "order": index,
            "fault_attempt_directory": f"{index:02d}_{scenario}_fault_attempt_001",
            "recovery_attempt_directory": f"{index:02d}_{scenario}_recovery_attempt_001",
            "fault_mode": scenario,
            "recovery_mode": "complete",
            "fault_worker": "scripts/i5_11_bounded_fault_worker.py",
            "fault_command_template": [
                "${CURRENT_PYTHON}", "scripts/i5_11_bounded_fault_worker.py",
                "--workspace", "${FAULT_WORKSPACE}", "--mode", scenario,
                "--child-sleep-s", "30.0",
            ],
            "recovery_command_template": [
                "${CURRENT_PYTHON}", "scripts/i5_11_bounded_fault_worker.py",
                "--workspace", "${RECOVERY_WORKSPACE}", "--mode", "complete",
                "--child-sleep-s", "30.0",
            ],
            "fault_timeout_s": 10.0,
            "recovery_timeout_s": 10.0,
            "terminate_grace_s": 0.25,
            "expected_fault_state": "crashed",
            "expected_fault_exit_code": 17 if scenario == "crash_orphan_descendant" else "nonzero",
            "expected_fault_signature": signatures[scenario],
            "expected_fault_process_group_clean": True,
            "expected_failed_workspace_removed": True,
            "expected_recovery_state": "completed",
            "expected_recovery_process_group_clean": True,
            "expected_recovery_workspace_preserved": True,
            "real_gpu_exhaustion": False,
            "real_disk_fill": False,
            "attempts": 1,
            "recovery_attempts": 1,
        }
    return {
        "schema_version": 1,
        "phase": "I5.11B",
        "status": "failure_injection_runtime_and_evidence_preflight_frozen",
        "pass": True,
        "evidence_bindings": {
            "i5_11a_admission": binding("/evidence/a.json", I511A_SHA),
            "post_i5_10d_delta_checkpoint": binding("/checkpoint/d.tar.gz", POST_I510D_DELTA_SHA),
            "i5_11b_preflight_receipt": binding("/evidence/b_receipt.json", "a" * 64),
        },
        "repository_anchors": {
            "scripts/i5_11_bounded_fault_worker.py": {
                "path": "/repo/scripts/i5_11_bounded_fault_worker.py",
                "sha256": FAULT_WORKER_SHA,
                "verified": True,
            }
        },
        "classifier_preflight": {
            "cuda_oom": {"observed_kind": "cuda_out_of_memory", "first_class": True},
            "disk_enospc": {"observed_kind": "unknown_model_error", "first_class": False, "gap_preserved": True},
            "fault_worker_executed": False,
            "probe_is_pure_in_process_classification_only": True,
        },
        "disk_preflight": {
            "free_bytes": 8 * 1024**3,
            "minimum_free_bytes_for_i5_11c": 4 * 1024**3,
            "safety_floor_met": True,
            "pressure_performed": False,
            "files_created_for_probe": False,
        },
        "process_preflight": {"live_fault_workers": [], "no_fault_processes_running": True, "process_probe_is_read_only": True},
        "fault_plan": plans,
        "campaign_contract": {
            "fault_scenarios": scenarios,
            "fault_attempts_each": 1,
            "fault_attempts_total": 3,
            "recovery_attempts_each": 1,
            "recovery_attempts_total": 3,
            "clean_recovery_after_each": True,
            "scenario_order_frozen": True,
            "attempt_directories_new_and_immutable": True,
            "fault_worker_hash_frozen": True,
            "real_resource_exhaustion": False,
            "real_gpu_oom": False,
            "real_disk_fill": False,
            "retry_until_pass": False,
            "result_driven_retuning": False,
            "production_default_modified": False,
            "production_failure_classifier_modified": False,
            "production_recovery_engine_modified": False,
        },
        "i5_11c_evidence_schema": {
            "schema_version": 1,
            "required_campaign_bindings": [
                "i5_11a_admission", "i5_11b_preflight", "post_i5_10d_delta_checkpoint",
                "repository_anchors", "fault_worker",
            ],
            "required_per_scenario_records": [
                "fault_command", "fault_started_utc", "fault_finished_utc", "fault_outcome",
                "fault_signature_observed", "fault_console_sha256", "recovery_command",
                "recovery_started_utc", "recovery_finished_utc", "recovery_outcome",
                "recovery_result_sha256", "process_group_clean", "failed_workspace_removed",
                "no_leaked_descendants", "repository_integrity_after", "disk_free_after",
            ],
            "campaign_terminal_states": ["pass", "robustness_failure", "runtime_integrity_failure"],
            "expected_fault_is_not_reconstruction_quality": True,
            "failed_recovery_is_robustness_failure": True,
            "source_or_anchor_drift_is_runtime_integrity_failure": True,
        },
        "abort_rules": {
            "abort_before_campaign_if_disk_free_below_safety_floor": True,
            "abort_before_campaign_if_fault_worker_already_running": True,
            "abort_on_repository_anchor_drift": True,
            "abort_on_unexpected_fault_signature": True,
            "abort_on_process_group_leak": True,
            "abort_on_workspace_cleanup_failure": True,
            "abort_on_recovery_failure": True,
            "no_retry_after_any_terminal_scenario": True,
            "do_not_fill_disk_to_continue": True,
            "do_not_allocate_gpu_memory_to_force_oom": True,
        },
        "scope": {
            "execution_free_preflight": True,
            "synthetic_exception_classification_probe": True,
            "fault_injection": False,
            "supervised_fault_worker_execution": False,
            "real_gpu_oom": False,
            "real_disk_fill": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "provider_inference": False,
            "production_default_modified": False,
            "production_failure_classifier_modified": False,
            "production_recovery_engine_modified": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "decision": {
            "i5_10_complete": True,
            "i5_11_started": True,
            "i5_11a_complete": True,
            "i5_11b_complete": True,
            "i5_11c_authorized": True,
            "i5_11c_started": False,
            "i5_11d_authorized": False,
            "i5_11d_started": False,
            "i5_12_started": False,
            "i5_13_started": False,
            "i5_14_started": False,
            "production_default_modified": False,
            "production_failure_classifier_change_authorized": False,
            "production_recovery_engine_change_authorized": False,
            "real_resource_exhaustion_authorized": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
    }


def synthetic_i511c_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.11C_CAMPAIGN_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_11c_tests": True,
            "engine_hardening_regression": True,
            "i5_11b_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "real_campaign_execution": False,
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
    return {"x": {"path": "/repo/x", "sha256": "b" * 64, "verified": True}}


def synthetic_disk(*, free_bytes: int = 8 * 1024**3) -> dict:
    return {
        "path": "/tmp",
        "free_bytes": free_bytes,
        "minimum_free_bytes_for_i5_11c": 4 * 1024**3,
        "safety_floor_met": free_bytes >= 4 * 1024**3,
        "pressure_performed": False,
    }


def fault_outcome(*, scenario: str, signature: str | None = None, exit_code: int | None = None) -> dict:
    expected_sig = {
        "cuda_oom": "simulated cuda out-of-memory",
        "disk_enospc": "No space left on device",
        "crash_orphan_descendant": "I511_FAULT=crash_orphan_descendant",
    }[scenario]
    return {
        "state": "crashed",
        "exit_code": (17 if scenario == "crash_orphan_descendant" else 1) if exit_code is None else exit_code,
        "stdout": signature if signature is not None else expected_sig,
        "process_group_clean": True,
        "workspace_exists_after": False,
        "failed_workspace_removed": True,
        "safe_for_recovery": True,
    }


def recovery_outcome(*, completed: bool = True) -> dict:
    return {
        "state": "completed" if completed else "crashed",
        "exit_code": 0 if completed else 1,
        "process_group_clean": True,
        "workspace_exists_after": completed,
        "failed_workspace_removed": not completed,
        "safe_for_recovery": True,
    }


def mutate(payload: dict, *path: str, value) -> dict:
    out = deepcopy(payload)
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return out
