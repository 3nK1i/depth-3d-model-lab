"""I5.11D evidence-only qualification and closure for bounded failure recovery.

This step consumes the frozen I5.11A admission, I5.11B runtime/evidence
preflight, and the single authoritative I5.11C bounded failure/recovery
campaign. It does not execute fault workers, consume additional fault attempts,
apply resource pressure, alter failure classification, alter recovery behavior,
or modify production defaults.

The admitted conclusion is intentionally bounded: the existing supervisor
contained the three explicitly admitted synthetic/control fault contracts
(CUDA-OOM signature, ENOSPC signature, and orphan-descendant crash) and a fresh
clean recovery completed after each. This does not qualify arbitrary failures,
real VRAM exhaustion, real filesystem exhaustion, or universal recovery.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError

I511D_SCHEMA_VERSION = 1
I511D_PHASE = "I5.11D"
I511D_RECEIPT_PHASE = "I5.11D_CLOSURE_RECEIPT"
I511D_I511A_FROZEN_SHA256 = "9d1f02e4cc0cc90eb909cf17ea5621a3aaf580a75ddf2a2269ca4402d5cdf1e6"
I511D_I511B_FROZEN_SHA256 = "e72c7d19ef403aed55e80a9ab3ff6a6f76b2b7ba17706bd3524fda4b8bf3f275"
I511D_I511C_FROZEN_SHA256 = "c0d4f34bc9c145fe3f606df7f69714027fc5ff49372e9ec0f427cccd29927bf9"
I511D_SCENARIOS = ("cuda_oom", "disk_enospc", "crash_orphan_descendant")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.11D {label} must be an object.")
    return value


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = node.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.11D {label}.{key} contract drift: expected={expected!r} actual={actual!r}."
        )


def _valid_sha(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _binding(value: Mapping[str, Any], *, label: str, expected_sha256: str | None = None) -> dict[str, str]:
    node = _mapping(value, label=label)
    path = str(node.get("path", "")).strip()
    sha = str(node.get("sha256", "")).strip().lower()
    if not path or not _valid_sha(sha):
        raise ReconstructionValidationError(f"I5.11D {label} binding is malformed.")
    if expected_sha256 is not None and sha != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"I5.11D {label} SHA256 drift: expected={expected_sha256} actual={sha}."
        )
    return {"path": path, "sha256": sha}


def validate_i5_11a_for_i5_11d(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.11A admission")
    _require(root, "schema_version", 1, label="I5.11A admission")
    _require(root, "phase", "I5.11A", label="I5.11A admission")
    _require(root, "status", "failure_mode_inventory_and_experiment_admitted", label="I5.11A admission")
    _require(root, "pass", True, label="I5.11A admission")
    matrix = _mapping(root.get("fault_matrix"), label="I5.11A fault matrix")
    if set(matrix) != set(I511D_SCENARIOS):
        raise ReconstructionValidationError("I5.11D requires exactly the frozen three I5.11A fault scenarios.")
    for scenario in I511D_SCENARIOS:
        _require(_mapping(matrix.get(scenario), label=f"I5.11A {scenario}"), "admitted", True, label=f"I5.11A {scenario}")
    contract = _mapping(root.get("i5_11c_campaign_contract"), label="I5.11A C campaign contract")
    _require(contract, "fault_scenarios", list(I511D_SCENARIOS), label="I5.11A C campaign contract")
    _require(contract, "fault_attempts_total", 3, label="I5.11A C campaign contract")
    _require(contract, "recovery_attempts_total", 3, label="I5.11A C campaign contract")
    _require(contract, "retry_until_pass", False, label="I5.11A C campaign contract")
    _require(contract, "real_resource_exhaustion", False, label="I5.11A C campaign contract")
    decision = _mapping(root.get("decision"), label="I5.11A decision")
    _require(decision, "i5_11a_complete", True, label="I5.11A decision")
    _require(decision, "i5_11b_authorized", True, label="I5.11A decision")
    _require(decision, "real_resource_exhaustion_authorized", False, label="I5.11A decision")
    return deepcopy(dict(root))


def validate_i5_11b_for_i5_11d(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.11B preflight")
    _require(root, "schema_version", 1, label="I5.11B preflight")
    _require(root, "phase", "I5.11B", label="I5.11B preflight")
    _require(root, "status", "failure_injection_runtime_and_evidence_preflight_frozen", label="I5.11B preflight")
    _require(root, "pass", True, label="I5.11B preflight")
    classifier = _mapping(root.get("classifier_preflight"), label="I5.11B classifier preflight")
    cuda = _mapping(classifier.get("cuda_oom"), label="I5.11B CUDA OOM classifier")
    enospc = _mapping(classifier.get("disk_enospc"), label="I5.11B ENOSPC classifier")
    _require(cuda, "observed_kind", "cuda_out_of_memory", label="I5.11B CUDA OOM classifier")
    _require(cuda, "first_class", True, label="I5.11B CUDA OOM classifier")
    _require(enospc, "observed_kind", "unknown_model_error", label="I5.11B ENOSPC classifier")
    _require(enospc, "first_class", False, label="I5.11B ENOSPC classifier")
    _require(enospc, "gap_preserved", True, label="I5.11B ENOSPC classifier")
    _require(classifier, "fault_worker_executed", False, label="I5.11B classifier preflight")
    disk = _mapping(root.get("disk_preflight"), label="I5.11B disk preflight")
    _require(disk, "safety_floor_met", True, label="I5.11B disk preflight")
    _require(disk, "pressure_performed", False, label="I5.11B disk preflight")
    process = _mapping(root.get("process_preflight"), label="I5.11B process preflight")
    _require(process, "no_fault_processes_running", True, label="I5.11B process preflight")
    campaign = _mapping(root.get("campaign_contract"), label="I5.11B campaign contract")
    _require(campaign, "fault_scenarios", list(I511D_SCENARIOS), label="I5.11B campaign contract")
    _require(campaign, "fault_attempts_total", 3, label="I5.11B campaign contract")
    _require(campaign, "recovery_attempts_total", 3, label="I5.11B campaign contract")
    _require(campaign, "retry_until_pass", False, label="I5.11B campaign contract")
    _require(campaign, "real_resource_exhaustion", False, label="I5.11B campaign contract")
    _require(campaign, "production_failure_classifier_modified", False, label="I5.11B campaign contract")
    _require(campaign, "production_recovery_engine_modified", False, label="I5.11B campaign contract")
    decision = _mapping(root.get("decision"), label="I5.11B decision")
    _require(decision, "i5_11b_complete", True, label="I5.11B decision")
    _require(decision, "i5_11c_authorized", True, label="I5.11B decision")
    _require(decision, "real_resource_exhaustion_authorized", False, label="I5.11B decision")
    return deepcopy(dict(root))


def _validate_scenario_record(record: Mapping[str, Any], *, scenario: str, expected_exit_code: int) -> dict[str, Any]:
    node = _mapping(record, label=f"I5.11C {scenario} record")
    _require(node, "scenario", scenario, label=f"I5.11C {scenario} record")
    _require(node, "terminal_status", "pass", label=f"I5.11C {scenario} record")
    _require(node, "terminal_reason", "expected_fault_contained_and_clean_recovery_completed", label=f"I5.11C {scenario} record")
    for key in (
        "fault_attempt_executed", "recovery_attempt_executed", "fault_signature_observed", "process_group_clean",
        "failed_workspace_removed", "no_leaked_descendants", "repository_integrity_after",
    ):
        _require(node, key, True, label=f"I5.11C {scenario} record")
    if int(node.get("disk_free_after", 0)) <= 0:
        raise ReconstructionValidationError(f"I5.11D {scenario} disk_free_after must be positive.")

    fault = _mapping(node.get("fault_outcome"), label=f"I5.11C {scenario} fault outcome")
    _require(fault, "state", "crashed", label=f"I5.11C {scenario} fault outcome")
    _require(fault, "exit_code", expected_exit_code, label=f"I5.11C {scenario} fault outcome")
    _require(fault, "process_group_clean", True, label=f"I5.11C {scenario} fault outcome")
    _require(fault, "failed_workspace_removed", True, label=f"I5.11C {scenario} fault outcome")
    _require(fault, "workspace_exists_after", False, label=f"I5.11C {scenario} fault outcome")
    _require(fault, "safe_for_recovery", True, label=f"I5.11C {scenario} fault outcome")

    recovery = _mapping(node.get("recovery_outcome"), label=f"I5.11C {scenario} recovery outcome")
    _require(recovery, "state", "completed", label=f"I5.11C {scenario} recovery outcome")
    _require(recovery, "exit_code", 0, label=f"I5.11C {scenario} recovery outcome")
    _require(recovery, "process_group_clean", True, label=f"I5.11C {scenario} recovery outcome")
    _require(recovery, "workspace_exists_after", True, label=f"I5.11C {scenario} recovery outcome")
    _require(recovery, "safe_for_recovery", True, label=f"I5.11C {scenario} recovery outcome")

    result = _mapping(node.get("recovery_result"), label=f"I5.11C {scenario} recovery result")
    _require(result, "phase", "I5.11", label=f"I5.11C {scenario} recovery result")
    _require(result, "mode", "complete", label=f"I5.11C {scenario} recovery result")
    _require(result, "pass", True, label=f"I5.11C {scenario} recovery result")
    _require(result, "child_reaped_by_worker", True, label=f"I5.11C {scenario} recovery result")
    _require(result, "real_gpu_exhaustion", False, label=f"I5.11C {scenario} recovery result")
    _require(result, "real_filesystem_fill", False, label=f"I5.11C {scenario} recovery result")
    return deepcopy(dict(node))


def validate_i5_11c_for_i5_11d(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.11C campaign")
    _require(root, "schema_version", 1, label="I5.11C campaign")
    _require(root, "phase", "I5.11C", label="I5.11C campaign")
    _require(root, "status", "bounded_failure_recovery_campaign_completed", label="I5.11C campaign")
    _require(root, "pass", True, label="I5.11C campaign")
    _require(root, "terminal_status", "pass", label="I5.11C campaign")
    _require(root, "terminal_reason", "all_three_bounded_fault_recovery_pairs_passed", label="I5.11C campaign")
    _require(root, "campaign_quality_status", "all_bounded_faults_contained_and_recovered", label="I5.11C campaign")

    accounting = _mapping(root.get("attempt_accounting"), label="I5.11C attempt accounting")
    _require(accounting, "authorized_fault_attempts_total", 3, label="I5.11C attempt accounting")
    _require(accounting, "authorized_recovery_attempts_total", 3, label="I5.11C attempt accounting")
    _require(accounting, "fault_attempts_executed", 3, label="I5.11C attempt accounting")
    _require(accounting, "recovery_attempts_executed", 3, label="I5.11C attempt accounting")
    _require(accounting, "retry_attempts_executed", 0, label="I5.11C attempt accounting")

    records = root.get("scenario_records")
    if not isinstance(records, list) or len(records) != 3:
        raise ReconstructionValidationError("I5.11D requires exactly three I5.11C scenario records.")
    observed_order = [str(_mapping(item, label="I5.11C scenario record").get("scenario", "")) for item in records]
    if observed_order != list(I511D_SCENARIOS):
        raise ReconstructionValidationError(
            f"I5.11D I5.11C scenario order drift: expected={list(I511D_SCENARIOS)!r} actual={observed_order!r}."
        )
    validated_records = [
        _validate_scenario_record(records[0], scenario="cuda_oom", expected_exit_code=1),
        _validate_scenario_record(records[1], scenario="disk_enospc", expected_exit_code=1),
        _validate_scenario_record(records[2], scenario="crash_orphan_descendant", expected_exit_code=17),
    ]

    scope = _mapping(root.get("scope"), label="I5.11C scope")
    _require(scope, "real_bounded_campaign", True, label="I5.11C scope")
    _require(scope, "synthetic_fault_injection", True, label="I5.11C scope")
    _require(scope, "supervised_fault_worker_execution", True, label="I5.11C scope")
    for key in (
        "real_gpu_oom", "real_disk_fill", "real_resource_exhaustion", "sfm_execution", "mvs_execution",
        "provider_inference", "reconstruction_quality_comparison", "production_default_modified",
        "production_failure_classifier_modified", "production_recovery_engine_modified", "auto_policy_implemented",
        "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.11C scope")

    claims = _mapping(root.get("claim_boundaries"), label="I5.11C claim boundaries")
    _require(claims, "enospc_classifier_gap_preserved", True, label="I5.11C claim boundaries")
    _require(claims, "expected_fault_is_not_reconstruction_quality", True, label="I5.11C claim boundaries")

    decision = _mapping(root.get("decision"), label="I5.11C decision")
    _require(decision, "i5_11c_complete", True, label="I5.11C decision")
    _require(decision, "i5_11d_authorized", True, label="I5.11C decision")
    _require(decision, "i5_11d_started", False, label="I5.11C decision")
    _require(decision, "production_failure_classifier_change_authorized", False, label="I5.11C decision")
    _require(decision, "production_recovery_engine_change_authorized", False, label="I5.11C decision")
    _require(decision, "real_resource_exhaustion_authorized", False, label="I5.11C decision")

    out = deepcopy(dict(root))
    out["scenario_records"] = validated_records
    return out


def validate_i5_11d_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.11D closure receipt")
    _require(root, "schema_version", 1, label="I5.11D closure receipt")
    _require(root, "phase", I511D_RECEIPT_PHASE, label="I5.11D closure receipt")
    _require(root, "status", "pass", label="I5.11D closure receipt")
    _require(root, "pass", True, label="I5.11D closure receipt")
    scope = _mapping(root.get("scope"), label="I5.11D receipt scope")
    _require(scope, "evidence_only", True, label="I5.11D receipt scope")
    for key in (
        "fault_injection", "supervised_fault_worker_execution", "real_gpu_oom", "real_disk_fill",
        "sfm_execution", "mvs_execution", "provider_inference",
    ):
        _require(scope, key, False, label="I5.11D receipt scope")
    codes = _mapping(root.get("return_codes"), label="I5.11D receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        _require(codes, key, 0, label="I5.11D receipt return codes")
    return deepcopy(dict(root))


def build_i5_11d_failure_robustness_closure(
    *,
    i5_11a_admission: Mapping[str, Any],
    i5_11a_binding: Mapping[str, Any],
    i5_11b_preflight: Mapping[str, Any],
    i5_11b_binding: Mapping[str, Any],
    i5_11c_campaign: Mapping[str, Any],
    i5_11c_binding: Mapping[str, Any],
    closure_receipt: Mapping[str, Any],
    closure_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    a = validate_i5_11a_for_i5_11d(i5_11a_admission)
    b = validate_i5_11b_for_i5_11d(i5_11b_preflight)
    c = validate_i5_11c_for_i5_11d(i5_11c_campaign)
    receipt = validate_i5_11d_receipt(closure_receipt)

    a_bind = _binding(i5_11a_binding, label="I5.11A admission", expected_sha256=I511D_I511A_FROZEN_SHA256)
    b_bind = _binding(i5_11b_binding, label="I5.11B preflight", expected_sha256=I511D_I511B_FROZEN_SHA256)
    c_bind = _binding(i5_11c_binding, label="I5.11C campaign", expected_sha256=I511D_I511C_FROZEN_SHA256)
    receipt_bind = _binding(closure_receipt_binding, label="I5.11D closure receipt")

    c_bindings = _mapping(c.get("evidence_bindings"), label="I5.11C evidence bindings")
    c_b = _mapping(c_bindings.get("i5_11b_preflight"), label="I5.11C I5.11B binding")
    if str(c_b.get("sha256", "")).lower() != I511D_I511B_FROZEN_SHA256:
        raise ReconstructionValidationError("I5.11D I5.11C does not bind the frozen I5.11B preflight SHA256.")

    records = list(c["scenario_records"])
    all_signatures = all(item["fault_signature_observed"] is True for item in records)
    all_process_groups = all(item["process_group_clean"] is True for item in records)
    all_workspaces = all(item["failed_workspace_removed"] is True for item in records)
    all_no_leaks = all(item["no_leaked_descendants"] is True for item in records)
    all_repo_integrity = all(item["repository_integrity_after"] is True for item in records)
    all_recoveries = all(
        _mapping(item.get("recovery_outcome"), label="I5.11C recovery outcome").get("state") == "completed"
        and _mapping(item.get("recovery_result"), label="I5.11C recovery result").get("pass") is True
        for item in records
    )

    classifier = _mapping(b.get("classifier_preflight"), label="I5.11B classifier preflight")
    enospc = _mapping(classifier.get("disk_enospc"), label="I5.11B ENOSPC classifier")

    qualification = {
        "conclusion": "all_three_admitted_bounded_faults_contained_and_clean_recovery_completed",
        "bounded_fault_recovery_pass": True,
        "fault_scenarios_qualified": list(I511D_SCENARIOS),
        "fault_attempts_executed": int(c["attempt_accounting"]["fault_attempts_executed"]),
        "recovery_attempts_executed": int(c["attempt_accounting"]["recovery_attempts_executed"]),
        "retry_attempts_executed": int(c["attempt_accounting"]["retry_attempts_executed"]),
        "all_expected_fault_signatures_observed": all_signatures,
        "all_process_groups_clean": all_process_groups,
        "all_failed_workspaces_removed": all_workspaces,
        "all_no_leaked_descendants": all_no_leaks,
        "all_repository_integrity_preserved": all_repo_integrity,
        "all_recoveries_completed": all_recoveries,
        "runtime_integrity_failure": False,
        "robustness_failure_observed": False,
    }
    if not all(qualification[key] is True for key in (
        "all_expected_fault_signatures_observed", "all_process_groups_clean", "all_failed_workspaces_removed",
        "all_no_leaked_descendants", "all_repository_integrity_preserved", "all_recoveries_completed",
    )):
        raise ReconstructionValidationError("I5.11D cannot close PASS because the I5.11C bounded recovery evidence is incomplete.")

    return {
        "schema_version": I511D_SCHEMA_VERSION,
        "phase": I511D_PHASE,
        "status": "bounded_failure_recovery_evidence_qualified_and_closed",
        "pass": True,
        "quality_status": "bounded_fault_recovery_pass",
        "qualification": qualification,
        "scenario_summary": {
            item["scenario"]: {
                "terminal_status": item["terminal_status"],
                "fault_signature_observed": item["fault_signature_observed"],
                "process_group_clean": item["process_group_clean"],
                "failed_workspace_removed": item["failed_workspace_removed"],
                "no_leaked_descendants": item["no_leaked_descendants"],
                "repository_integrity_after": item["repository_integrity_after"],
                "recovery_state": _mapping(item["recovery_outcome"], label="recovery outcome")["state"],
                "recovery_pass": _mapping(item["recovery_result"], label="recovery result")["pass"],
            }
            for item in records
        },
        "observed_gaps": {
            "disk_enospc_classifier": {
                "observed_kind": str(enospc.get("observed_kind")),
                "first_class_before_campaign": bool(enospc.get("first_class")),
                "gap_preserved": bool(enospc.get("gap_preserved")),
                "raw_enospc_signature_was_contained": True,
                "production_classifier_change_authorized": False,
            },
        },
        "observed_limitations": {
            "only_three_admitted_fault_contracts_qualified": True,
            "real_gpu_oom_was_not_executed": True,
            "real_disk_fill_was_not_executed": True,
            "arbitrary_failure_recovery_not_proven": True,
            "cancel_and_timeout_remain_historical_context_not_reexecuted_here": True,
            "checkpoint_corruption_not_evaluated": True,
            "i5_11_does_not_measure_reconstruction_quality": True,
            "i5_11_does_not_by_itself_justify_hybrid": True,
        },
        "claim_boundaries": {
            "universal_failure_recovery_claim": False,
            "real_resource_exhaustion_qualified": False,
            "runtime_performance_ranking": False,
            "reconstruction_quality_comparison": False,
            "provider_ranking": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "enospc_classifier_fix_claimed": False,
        },
        "evidence_bindings": {
            "i5_11a_admission": a_bind,
            "i5_11b_preflight": b_bind,
            "i5_11c_bounded_campaign": c_bind,
            "i5_11d_closure_receipt": receipt_bind,
            "post_i5_10d_delta_checkpoint": deepcopy(dict(_mapping(c_bindings.get("post_i5_10d_delta_checkpoint"), label="post-I5.10D delta"))),
            "post_i5_11b_delta_checkpoint": deepcopy(dict(_mapping(c_bindings.get("post_i5_11b_delta_checkpoint"), label="post-I5.11B delta"))),
            "fault_worker": deepcopy(dict(_mapping(c_bindings.get("fault_worker"), label="fault worker"))),
        },
        "validation_receipt": receipt,
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "fault_injection": False,
            "supervised_fault_worker_execution": False,
            "real_gpu_oom": False,
            "real_disk_fill": False,
            "real_resource_exhaustion": False,
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
            "i5_11c_complete": True,
            "i5_11d_complete": True,
            "i5_11_complete": True,
            "i5_12_admission_review_authorized": True,
            "i5_12_started": False,
            "i5_13_started": False,
            "i5_14_started": False,
            "production_default_modified": False,
            "production_failure_classifier_change_authorized": False,
            "production_recovery_engine_change_authorized": False,
            "real_resource_exhaustion_authorized": False,
            "hybrid_authorized": False,
            "auto_policy_implemented": False,
        },
        "next": "I5.12A hybrid experiment evidence-justification and admission review (hybrid execution not yet authorized).",
    }


__all__ = [
    "I511D_SCHEMA_VERSION", "I511D_PHASE", "I511D_RECEIPT_PHASE",
    "I511D_I511A_FROZEN_SHA256", "I511D_I511B_FROZEN_SHA256", "I511D_I511C_FROZEN_SHA256",
    "I511D_SCENARIOS", "validate_i5_11a_for_i5_11d", "validate_i5_11b_for_i5_11d",
    "validate_i5_11c_for_i5_11d", "validate_i5_11d_receipt", "build_i5_11d_failure_robustness_closure",
]
