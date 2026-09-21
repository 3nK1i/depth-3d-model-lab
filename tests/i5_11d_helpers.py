from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_failure_robustness_campaign import (
    build_i5_11c_campaign_header,
    finalize_i5_11c_campaign,
)
from tests.i5_11b_helpers import synthetic_i511a_admission
from tests.i5_11c_helpers import (
    I511A_SHA,
    I511B_SHA,
    POST_I510D_DELTA_SHA,
    binding,
    synthetic_anchors,
    synthetic_disk,
    synthetic_i511b_preflight,
    synthetic_i511c_receipt,
)

I511C_SHA = "c0d4f34bc9c145fe3f606df7f69714027fc5ff49372e9ec0f427cccd29927bf9"


def _scenario_record(scenario: str, *, disk_free_after: int = 8 * 1024**3) -> dict:
    signatures = {
        "cuda_oom": "simulated cuda out-of-memory",
        "disk_enospc": "No space left on device",
        "crash_orphan_descendant": "I511_FAULT=crash_orphan_descendant",
    }
    exit_code = 17 if scenario == "crash_orphan_descendant" else 1
    return {
        "scenario": scenario,
        "fault_attempt_executed": True,
        "recovery_attempt_executed": True,
        "fault_command": ["python", "scripts/i5_11_bounded_fault_worker.py", "--mode", scenario],
        "fault_started_utc": "2026-09-14T10:38:20Z",
        "fault_finished_utc": "2026-09-14T10:38:21Z",
        "fault_outcome": {
            "state": "crashed",
            "exit_code": exit_code,
            "process_group_clean": True,
            "failed_workspace_removed": True,
            "workspace_exists_after": False,
            "safe_for_recovery": True,
            "stdout": signatures[scenario],
        },
        "fault_evaluation": {
            "terminal_kind": "expected_fault_contained",
            "expected_fault_observed": True,
            "containment_ok": True,
            "state_ok": True,
            "signature_ok": True,
            "exit_ok": True,
        },
        "fault_signature_observed": True,
        "fault_console_sha256": "1" * 64,
        "process_group_clean": True,
        "failed_workspace_removed": True,
        "no_leaked_descendants": True,
        "recovery_command": ["python", "scripts/i5_11_bounded_fault_worker.py", "--mode", "complete"],
        "recovery_started_utc": "2026-09-14T10:38:21Z",
        "recovery_finished_utc": "2026-09-14T10:38:22Z",
        "recovery_outcome": {
            "state": "completed",
            "exit_code": 0,
            "process_group_clean": True,
            "failed_workspace_removed": False,
            "workspace_exists_after": True,
            "safe_for_recovery": True,
            "stdout": "",
        },
        "recovery_result_sha256": "2" * 64,
        "recovery_result": {
            "phase": "I5.11",
            "mode": "complete",
            "pass": True,
            "child_reaped_by_worker": True,
            "real_gpu_exhaustion": False,
            "real_filesystem_fill": False,
        },
        "recovery_evaluation": {
            "terminal_kind": "recovery_pass",
            "completed": True,
            "containment_ok": True,
            "result_ok": True,
        },
        "repository_integrity_after": True,
        "repository_anchors_after": {"x": {"path": "/repo/x", "sha256": "b" * 64, "verified": True}},
        "disk_free_before": disk_free_after,
        "disk_free_after": disk_free_after,
        "terminal_status": "pass",
        "terminal_reason": "expected_fault_contained_and_clean_recovery_completed",
    }


def synthetic_i511c_pass() -> dict:
    b = synthetic_i511b_preflight()
    header = build_i5_11c_campaign_header(
        i5_11a_binding=binding("/evidence/a.json", I511A_SHA),
        i5_11b_preflight=b,
        i5_11b_binding=binding("/evidence/b.json", I511B_SHA),
        post_i5_10d_delta_binding=binding("/checkpoint/d.tar.gz", POST_I510D_DELTA_SHA),
        post_i5_11b_delta_binding=binding("/checkpoint/b.tar.gz", "c" * 64),
        campaign_receipt=synthetic_i511c_receipt(),
        campaign_receipt_binding=binding("/evidence/c_receipt.json", "d" * 64),
        repository_anchors_before=synthetic_anchors(),
        disk_before=synthetic_disk(),
    )
    records = [_scenario_record(s) for s in ("cuda_oom", "disk_enospc", "crash_orphan_descendant")]
    return finalize_i5_11c_campaign(
        header,
        scenario_records=records,
        terminal_status="pass",
        terminal_reason="all_three_bounded_fault_recovery_pairs_passed",
    )


def synthetic_i511d_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.11D_CLOSURE_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_11d_tests": True,
            "predecessor_i5_11c_regression": True,
            "evidence_only": True,
            "fault_injection": False,
            "supervised_fault_worker_execution": False,
            "real_gpu_oom": False,
            "real_disk_fill": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "provider_inference": False,
        },
        "counts": {
            "focused_tests": 1,
            "predecessor_total_tests": 1,
            "combined_exercised_tests": 2,
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


def clone(value):
    return deepcopy(value)
