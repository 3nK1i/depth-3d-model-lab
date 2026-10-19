"""I5.11C real bounded OOM / ENOSPC / crash + recovery campaign.

I5.11C executes only the three synthetic failure modes frozen by I5.11A/B and
runs each attempt through the existing whole-engine process-group supervisor.
The campaign never deliberately exhausts GPU memory or disk space. A fresh
clean recovery follows each successfully-contained expected fault.

A valid robustness failure is scientific evidence and is terminal: there is no
retry-until-pass. Repository/disk/process/fault-signature integrity drift is a
runtime-integrity failure and does not complete I5.11C.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

from core.reconstruction.engine_hardening import (
    ReconstructionEngineProcessOutcome,
    ReconstructionEngineProcessSpec,
    ReconstructionEngineRunState,
    run_supervised_engine_process,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_failure_robustness_preflight import (
    I511B_MIN_FREE_BYTES_FOR_C,
    discover_live_i5_11_fault_workers,
)

I511C_SCHEMA_VERSION = 1
I511C_PHASE = "I5.11C"
I511C_RECEIPT_PHASE = "I5.11C_CAMPAIGN_RECEIPT"
I511C_I511A_FROZEN_SHA256 = "9d1f02e4cc0cc90eb909cf17ea5621a3aaf580a75ddf2a2269ca4402d5cdf1e6"
I511C_I511B_FROZEN_SHA256 = "e72c7d19ef403aed55e80a9ab3ff6a6f76b2b7ba17706bd3524fda4b8bf3f275"
I511C_POST_I510D_DELTA_FROZEN_SHA256 = "773f63c31d27d5746fe788f9a0463792a6d5061676fea6f44cb004f99e4a6247"
I511C_FAULT_WORKER_SHA256 = "3a94eb740230f9b2d9d2698bae1170883c0a8b4d662521dfb9d783dcc7c13836"
I511C_FAULT_SCENARIOS = ("cuda_oom", "disk_enospc", "crash_orphan_descendant")
I511C_FAULT_WORKER = "scripts/i5_11_bounded_fault_worker.py"
I511C_FINAL_ARTIFACT = "i5_11c_bounded_failure_recovery_campaign.json"

I511C_REPOSITORY_ANCHORS: dict[str, str] = {
    "core/reconstruction/engine_hardening.py": "541bdd9016db0dced5a06bf6104c4138e3c84eda6b6874034e896b852cbcbdf1",
    "core/reconstruction/i5_failure_robustness.py": "6156f88c9c9bf7f677deeaad0c0090197d731604cdfb82de2aa7b4112d03f03e",
    "core/reconstruction/i5_failure_robustness_preflight.py": "0c83a922195fabfff089e6b0cdbe3f46343677f150e215198809982a4b68c08a",
    "core/runtime/failures.py": "ba05a440c51d4c5ef6816a586acd44504ddc9ead720243791c4bfe57db4229c4",
    "scripts/i5_11_bounded_fault_worker.py": I511C_FAULT_WORKER_SHA256,
    "scripts/reconstruction_engine_fault_worker.py": "463181895302c89703c9ef83ba99c5f11fce6b4bd94a311cd29291f44bf33d17",
    "app/reconstruction_failure_recovery.py": "4da8cc6a3a1fff30e208c034ff0af99613e3ed3eb43e370a76199c01392e408a",
    "app/reconstruction_processing_job_controller.py": "b60a949b0ddb7814f828d48bb06a02ebf4cf9f3b46b42f87ff03ca3eed2bfa77",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.11C {label} must be an object.")
    return value


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = node.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.11C {label}.{key} contract drift: expected={expected!r} actual={actual!r}."
        )


def _valid_sha(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _binding(value: Mapping[str, Any], *, label: str, expected_sha256: str | None = None) -> dict[str, str]:
    node = _mapping(value, label=label)
    path = str(node.get("path", "")).strip()
    sha = str(node.get("sha256", "")).strip().lower()
    if not path or not _valid_sha(sha):
        raise ReconstructionValidationError(f"I5.11C {label} binding is malformed.")
    if expected_sha256 is not None and sha != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"I5.11C {label} SHA256 drift: expected={expected_sha256} actual={sha}."
        )
    return {"path": path, "sha256": sha}


def verify_i5_11c_repository_anchors(repository_root: Path) -> dict[str, dict[str, Any]]:
    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"I5.11C repository root missing: {root}")
    verified: dict[str, dict[str, Any]] = {}
    for relative, expected in I511C_REPOSITORY_ANCHORS.items():
        path = root / relative
        if not path.is_file():
            raise ReconstructionValidationError(f"I5.11C repository anchor missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ReconstructionValidationError(
                f"I5.11C repository anchor drifted: {relative} expected={expected} actual={actual}"
            )
        verified[relative] = {"path": str(path), "sha256": actual, "verified": True}
    return verified


def validate_i5_11b_for_i5_11c(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.11B preflight")
    _require(root, "schema_version", 1, label="I5.11B preflight")
    _require(root, "phase", "I5.11B", label="I5.11B preflight")
    _require(root, "status", "failure_injection_runtime_and_evidence_preflight_frozen", label="I5.11B preflight")
    _require(root, "pass", True, label="I5.11B preflight")

    classifier = _mapping(root.get("classifier_preflight"), label="I5.11B classifier")
    _require(_mapping(classifier.get("cuda_oom"), label="I5.11B cuda_oom classifier"), "first_class", True, label="I5.11B cuda_oom classifier")
    disk_classifier = _mapping(classifier.get("disk_enospc"), label="I5.11B ENOSPC classifier")
    _require(disk_classifier, "first_class", False, label="I5.11B ENOSPC classifier")
    _require(disk_classifier, "gap_preserved", True, label="I5.11B ENOSPC classifier")
    _require(classifier, "fault_worker_executed", False, label="I5.11B classifier")

    disk = _mapping(root.get("disk_preflight"), label="I5.11B disk preflight")
    _require(disk, "safety_floor_met", True, label="I5.11B disk preflight")
    _require(disk, "pressure_performed", False, label="I5.11B disk preflight")

    process = _mapping(root.get("process_preflight"), label="I5.11B process preflight")
    _require(process, "no_fault_processes_running", True, label="I5.11B process preflight")

    plans = _mapping(root.get("fault_plan"), label="I5.11B fault plan")
    if set(plans) != set(I511C_FAULT_SCENARIOS):
        raise ReconstructionValidationError("I5.11C requires exactly the frozen I5.11B fault-plan members.")
    for order, scenario in enumerate(I511C_FAULT_SCENARIOS, 1):
        plan = _mapping(plans.get(scenario), label=f"I5.11B fault plan {scenario}")
        _require(plan, "order", order, label=f"I5.11B fault plan {scenario}")
        _require(plan, "fault_mode", scenario, label=f"I5.11B fault plan {scenario}")
        _require(plan, "recovery_mode", "complete", label=f"I5.11B fault plan {scenario}")
        _require(plan, "fault_worker", I511C_FAULT_WORKER, label=f"I5.11B fault plan {scenario}")
        _require(plan, "attempts", 1, label=f"I5.11B fault plan {scenario}")
        _require(plan, "recovery_attempts", 1, label=f"I5.11B fault plan {scenario}")
        _require(plan, "real_gpu_exhaustion", False, label=f"I5.11B fault plan {scenario}")
        _require(plan, "real_disk_fill", False, label=f"I5.11B fault plan {scenario}")

    campaign = _mapping(root.get("campaign_contract"), label="I5.11B campaign contract")
    _require(campaign, "fault_scenarios", list(I511C_FAULT_SCENARIOS), label="I5.11B campaign contract")
    _require(campaign, "fault_attempts_each", 1, label="I5.11B campaign contract")
    _require(campaign, "fault_attempts_total", 3, label="I5.11B campaign contract")
    _require(campaign, "recovery_attempts_each", 1, label="I5.11B campaign contract")
    _require(campaign, "recovery_attempts_total", 3, label="I5.11B campaign contract")
    for key in ("clean_recovery_after_each", "scenario_order_frozen", "attempt_directories_new_and_immutable", "fault_worker_hash_frozen"):
        _require(campaign, key, True, label="I5.11B campaign contract")
    for key in (
        "real_resource_exhaustion", "real_gpu_oom", "real_disk_fill", "retry_until_pass",
        "result_driven_retuning", "production_default_modified", "production_failure_classifier_modified",
        "production_recovery_engine_modified",
    ):
        _require(campaign, key, False, label="I5.11B campaign contract")

    schema = _mapping(root.get("i5_11c_evidence_schema"), label="I5.11B C evidence schema")
    _require(schema, "campaign_terminal_states", ["pass", "robustness_failure", "runtime_integrity_failure"], label="I5.11B C evidence schema")
    _require(schema, "expected_fault_is_not_reconstruction_quality", True, label="I5.11B C evidence schema")
    _require(schema, "failed_recovery_is_robustness_failure", True, label="I5.11B C evidence schema")
    _require(schema, "source_or_anchor_drift_is_runtime_integrity_failure", True, label="I5.11B C evidence schema")

    abort = _mapping(root.get("abort_rules"), label="I5.11B abort rules")
    for key in (
        "abort_before_campaign_if_disk_free_below_safety_floor", "abort_before_campaign_if_fault_worker_already_running",
        "abort_on_repository_anchor_drift", "abort_on_unexpected_fault_signature", "abort_on_process_group_leak",
        "abort_on_workspace_cleanup_failure", "abort_on_recovery_failure", "no_retry_after_any_terminal_scenario",
        "do_not_fill_disk_to_continue", "do_not_allocate_gpu_memory_to_force_oom",
    ):
        _require(abort, key, True, label="I5.11B abort rules")

    scope = _mapping(root.get("scope"), label="I5.11B scope")
    _require(scope, "execution_free_preflight", True, label="I5.11B scope")
    for key in (
        "fault_injection", "supervised_fault_worker_execution", "real_gpu_oom", "real_disk_fill",
        "sfm_execution", "mvs_execution", "production_default_modified", "production_failure_classifier_modified",
        "production_recovery_engine_modified", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.11B scope")

    decision = _mapping(root.get("decision"), label="I5.11B decision")
    _require(decision, "i5_11b_complete", True, label="I5.11B decision")
    _require(decision, "i5_11c_authorized", True, label="I5.11B decision")
    for key in (
        "i5_11c_started", "i5_11d_authorized", "i5_11d_started", "i5_12_started", "i5_13_started", "i5_14_started",
        "production_default_modified", "production_failure_classifier_change_authorized",
        "production_recovery_engine_change_authorized", "real_resource_exhaustion_authorized",
    ):
        _require(decision, key, False, label="I5.11B decision")
    return deepcopy(dict(root))


def validate_i5_11c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.11C campaign receipt")
    _require(root, "schema_version", 1, label="I5.11C campaign receipt")
    _require(root, "phase", I511C_RECEIPT_PHASE, label="I5.11C campaign receipt")
    _require(root, "status", "pass", label="I5.11C campaign receipt")
    _require(root, "pass", True, label="I5.11C campaign receipt")
    scope = _mapping(root.get("scope"), label="I5.11C receipt scope")
    for key in (
        "real_campaign_execution", "fault_injection", "supervised_fault_worker_execution", "real_gpu_oom", "real_disk_fill",
        "sfm_execution", "mvs_execution", "production_default_modified", "production_failure_classifier_modified",
        "production_recovery_engine_modified", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.11C receipt scope")
    codes = _mapping(root.get("return_codes"), label="I5.11C receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        _require(codes, key, 0, label="I5.11C receipt return codes")
    return deepcopy(dict(root))


def observe_i5_11c_disk(path: Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    usage = shutil.disk_usage(target)
    return {
        "path": str(target),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "minimum_free_bytes_for_i5_11c": I511B_MIN_FREE_BYTES_FOR_C,
        "safety_floor_met": int(usage.free) >= I511B_MIN_FREE_BYTES_FOR_C,
        "pressure_performed": False,
    }


def classify_fault_attempt(*, scenario: str, plan: Mapping[str, Any], outcome: Mapping[str, Any]) -> dict[str, Any]:
    if scenario not in I511C_FAULT_SCENARIOS:
        raise ReconstructionValidationError(f"I5.11C unknown scenario: {scenario}")
    p = _mapping(plan, label=f"fault plan {scenario}")
    out = _mapping(outcome, label=f"fault outcome {scenario}")
    stdout = str(out.get("stdout", ""))
    signature = str(p.get("expected_fault_signature", ""))
    expected_exit = p.get("expected_fault_exit_code")
    state_ok = out.get("state") == p.get("expected_fault_state") == "crashed"
    signature_ok = bool(signature) and signature in stdout
    exit_code = out.get("exit_code")
    exit_ok = (isinstance(expected_exit, int) and exit_code == expected_exit) or (expected_exit == "nonzero" and isinstance(exit_code, int) and exit_code != 0)
    if not (state_ok and signature_ok and exit_ok):
        return {
            "terminal_kind": "runtime_integrity_failure",
            "expected_fault_observed": False,
            "state_ok": state_ok,
            "signature_ok": signature_ok,
            "exit_ok": exit_ok,
        }
    containment_ok = (
        out.get("process_group_clean") is True
        and out.get("failed_workspace_removed") is True
        and out.get("workspace_exists_after") is False
        and out.get("safe_for_recovery") is True
    )
    if not containment_ok:
        return {
            "terminal_kind": "robustness_failure",
            "expected_fault_observed": True,
            "containment_ok": False,
        }
    return {
        "terminal_kind": "expected_fault_contained",
        "expected_fault_observed": True,
        "containment_ok": True,
        "state_ok": True,
        "signature_ok": True,
        "exit_ok": True,
    }


def classify_recovery_attempt(*, outcome: Mapping[str, Any], result_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    out = _mapping(outcome, label="recovery outcome")
    payload = {} if result_payload is None else dict(_mapping(result_payload, label="recovery result"))
    completed = out.get("state") == "completed" and out.get("exit_code") == 0
    contained = out.get("process_group_clean") is True and out.get("workspace_exists_after") is True and out.get("failed_workspace_removed") is False
    result_ok = (
        payload.get("phase") == "I5.11"
        and payload.get("mode") == "complete"
        and payload.get("pass") is True
        and payload.get("child_reaped_by_worker") is True
        and payload.get("real_gpu_exhaustion") is False
        and payload.get("real_filesystem_fill") is False
    )
    if completed and contained and result_ok:
        return {"terminal_kind": "recovery_pass", "completed": True, "containment_ok": True, "result_ok": True}
    return {
        "terminal_kind": "robustness_failure",
        "completed": completed,
        "containment_ok": contained,
        "result_ok": result_ok,
    }


def build_i5_11c_campaign_header(
    *,
    i5_11a_binding: Mapping[str, Any],
    i5_11b_preflight: Mapping[str, Any],
    i5_11b_binding: Mapping[str, Any],
    post_i5_10d_delta_binding: Mapping[str, Any],
    post_i5_11b_delta_binding: Mapping[str, Any],
    campaign_receipt: Mapping[str, Any],
    campaign_receipt_binding: Mapping[str, Any],
    repository_anchors_before: Mapping[str, Any],
    disk_before: Mapping[str, Any],
) -> dict[str, Any]:
    b = validate_i5_11b_for_i5_11c(i5_11b_preflight)
    validate_i5_11c_receipt(campaign_receipt)
    a_bind = _binding(i5_11a_binding, label="I5.11A admission", expected_sha256=I511C_I511A_FROZEN_SHA256)
    b_bind = _binding(i5_11b_binding, label="I5.11B preflight", expected_sha256=I511C_I511B_FROZEN_SHA256)
    d_delta = _binding(post_i5_10d_delta_binding, label="post-I5.10D delta", expected_sha256=I511C_POST_I510D_DELTA_FROZEN_SHA256)
    b_delta = _binding(post_i5_11b_delta_binding, label="post-I5.11B delta")
    receipt_bind = _binding(campaign_receipt_binding, label="I5.11C campaign receipt")
    anchors = _mapping(repository_anchors_before, label="repository anchors before")
    if not anchors or not all(_mapping(v, label=f"anchor {k}").get("verified") is True for k, v in anchors.items()):
        raise ReconstructionValidationError("I5.11C repository anchors must all verify before campaign start.")
    disk = _mapping(disk_before, label="disk before")
    if disk.get("safety_floor_met") is not True or int(disk.get("free_bytes", 0)) < I511B_MIN_FREE_BYTES_FOR_C:
        raise ReconstructionValidationError("I5.11C disk safety floor is not met; campaign must not start.")
    return {
        "schema_version": I511C_SCHEMA_VERSION,
        "phase": I511C_PHASE,
        "status": "bounded_failure_recovery_campaign_running",
        "pass": False,
        "terminal_status": "running",
        "terminal_reason": "",
        "evidence_bindings": {
            "i5_11a_admission": a_bind,
            "i5_11b_preflight": b_bind,
            "post_i5_10d_delta_checkpoint": d_delta,
            "post_i5_11b_delta_checkpoint": b_delta,
            "i5_11c_campaign_receipt": receipt_bind,
            "fault_worker": {"path": I511C_FAULT_WORKER, "sha256": I511C_FAULT_WORKER_SHA256},
        },
        "repository_anchors_before": deepcopy(dict(anchors)),
        "disk_before": deepcopy(dict(disk)),
        "fault_plan": deepcopy(dict(b["fault_plan"])),
        "campaign_contract": {
            **deepcopy(dict(b["campaign_contract"])),
            "no_retry_after_terminal_scenario": True,
        },
        "attempt_accounting": {
            "authorized_fault_attempts_total": 3,
            "authorized_recovery_attempts_total": 3,
            "fault_attempts_executed": 0,
            "recovery_attempts_executed": 0,
        },
        "scenario_records": [],
        "scope": {
            "real_bounded_campaign": True,
            "synthetic_fault_injection": True,
            "supervised_fault_worker_execution": True,
            "real_gpu_oom": False,
            "real_disk_fill": False,
            "real_resource_exhaustion": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "provider_inference": False,
            "reconstruction_quality_comparison": False,
            "production_default_modified": False,
            "production_failure_classifier_modified": False,
            "production_recovery_engine_modified": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "claim_boundaries": {
            "expected_fault_is_not_reconstruction_quality": True,
            "runtime_performance_ranking": False,
            "provider_ranking": False,
            "metric_scale_claimed": False,
            "certified_metrology": False,
            "enospc_classifier_gap_preserved": True,
        },
        "i5_11b_snapshot": {
            "classifier_preflight": deepcopy(dict(b["classifier_preflight"])),
            "abort_rules": deepcopy(dict(b["abort_rules"])),
            "i5_11c_evidence_schema": deepcopy(dict(b["i5_11c_evidence_schema"])),
        },
    }


def finalize_i5_11c_campaign(
    header: Mapping[str, Any],
    *,
    scenario_records: list[Mapping[str, Any]],
    terminal_status: str,
    terminal_reason: str,
) -> dict[str, Any]:
    if terminal_status not in {"pass", "robustness_failure", "runtime_integrity_failure"}:
        raise ReconstructionValidationError(f"I5.11C invalid terminal status: {terminal_status}")
    out = deepcopy(dict(_mapping(header, label="campaign header")))
    records = [deepcopy(dict(_mapping(item, label="scenario record"))) for item in scenario_records]
    fault_count = sum(1 for item in records if item.get("fault_attempt_executed") is True)
    recovery_count = sum(1 for item in records if item.get("recovery_attempt_executed") is True)
    evidence_complete = terminal_status in {"pass", "robustness_failure"}
    out.update({
        "status": "bounded_failure_recovery_campaign_completed" if evidence_complete else "bounded_failure_recovery_campaign_incomplete",
        "pass": evidence_complete,
        "terminal_status": terminal_status,
        "terminal_reason": str(terminal_reason),
        "campaign_quality_status": "all_bounded_faults_contained_and_recovered" if terminal_status == "pass" else (
            "robustness_failure_observed" if terminal_status == "robustness_failure" else "runtime_integrity_failure"
        ),
        "scenario_records": records,
        "attempt_accounting": {
            **deepcopy(dict(out["attempt_accounting"])),
            "fault_attempts_executed": fault_count,
            "recovery_attempts_executed": recovery_count,
            "retry_attempts_executed": 0,
        },
        "decision": {
            "i5_10_complete": True,
            "i5_11_started": True,
            "i5_11a_complete": True,
            "i5_11b_complete": True,
            "i5_11c_complete": evidence_complete,
            "i5_11d_authorized": evidence_complete,
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
        "next": "I5.11D evidence-only qualification and closure." if evidence_complete else "Diagnose I5.11C runtime-integrity failure before any further campaign execution.",
    })
    return out


def _outcome_dict(outcome: ReconstructionEngineProcessOutcome) -> dict[str, Any]:
    value = outcome.as_dict()
    value["stdout"] = outcome.stdout
    return value


def _attempt_spec(
    *,
    repository_root: Path,
    attempt_root: Path,
    mode: str,
    timeout_s: float,
    terminate_grace_s: float,
    child_sleep_s: float,
) -> ReconstructionEngineProcessSpec:
    workspace = (attempt_root / "workspace").resolve()
    command = (
        sys.executable,
        I511C_FAULT_WORKER,
        "--workspace",
        str(workspace),
        "--mode",
        mode,
        "--child-sleep-s",
        str(float(child_sleep_s)),
    )
    return ReconstructionEngineProcessSpec(
        command=command,
        owned_root=str(attempt_root.resolve()),
        workspace=str(workspace),
        timeout_s=float(timeout_s),
        terminate_grace_s=float(terminate_grace_s),
        cwd=str(repository_root.resolve()),
    )


def _write_text_and_sha(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    path.write_bytes(data)
    return _sha256_bytes(data)


def _write_json_and_sha(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    return _write_text_and_sha(path, text)


def execute_i5_11c_campaign(
    *,
    header: Mapping[str, Any],
    repository_root: Path,
    output_dir: Path,
    disk_path: Path,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ReconstructionValidationError(f"I5.11C output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, Any]] = []
    plan_root = _mapping(header.get("fault_plan"), label="campaign fault plan")

    def runtime_failure(reason: str) -> dict[str, Any]:
        return finalize_i5_11c_campaign(header, scenario_records=records, terminal_status="runtime_integrity_failure", terminal_reason=reason)

    try:
        anchors_before = verify_i5_11c_repository_anchors(root)
    except ReconstructionValidationError as exc:
        return runtime_failure(f"repository_anchor_drift_before_campaign:{exc}")
    if anchors_before != header.get("repository_anchors_before"):
        return runtime_failure("repository_anchor_binding_changed_before_campaign")
    disk_before = observe_i5_11c_disk(disk_path)
    if disk_before["safety_floor_met"] is not True:
        return runtime_failure("disk_safety_floor_not_met_before_campaign")
    if discover_live_i5_11_fault_workers():
        return runtime_failure("fault_worker_already_running_before_campaign")

    for scenario in I511C_FAULT_SCENARIOS:
        plan = dict(_mapping(plan_root.get(scenario), label=f"fault plan {scenario}"))
        record: dict[str, Any] = {
            "scenario": scenario,
            "fault_attempt_executed": False,
            "recovery_attempt_executed": False,
            "terminal_status": "running",
        }
        records.append(record)

        try:
            verify_i5_11c_repository_anchors(root)
        except ReconstructionValidationError as exc:
            record["terminal_status"] = "runtime_integrity_failure"
            record["terminal_reason"] = f"repository_anchor_drift_before_{scenario}:{exc}"
            return runtime_failure(record["terminal_reason"])
        disk_pre = observe_i5_11c_disk(disk_path)
        record["disk_free_before"] = disk_pre["free_bytes"]
        if disk_pre["safety_floor_met"] is not True:
            record["terminal_status"] = "runtime_integrity_failure"
            record["terminal_reason"] = "disk_safety_floor_not_met_before_scenario"
            return runtime_failure(f"{scenario}:disk_safety_floor_not_met_before_scenario")
        if discover_live_i5_11_fault_workers():
            record["terminal_status"] = "runtime_integrity_failure"
            record["terminal_reason"] = "fault_worker_already_running_before_scenario"
            return runtime_failure(f"{scenario}:fault_worker_already_running_before_scenario")

        fault_root = output / str(plan["fault_attempt_directory"])
        if fault_root.exists():
            record["terminal_status"] = "runtime_integrity_failure"
            return runtime_failure(f"{scenario}:fault_attempt_directory_collision")
        fault_root.mkdir(parents=True, exist_ok=False)
        fault_spec = _attempt_spec(
            repository_root=root,
            attempt_root=fault_root,
            mode=scenario,
            timeout_s=float(plan["fault_timeout_s"]),
            terminate_grace_s=float(plan["terminate_grace_s"]),
            child_sleep_s=30.0,
        )
        record["fault_command"] = list(fault_spec.command)
        record["fault_started_utc"] = _utc_now()
        fault_outcome = run_supervised_engine_process(fault_spec)
        record["fault_finished_utc"] = _utc_now()
        record["fault_attempt_executed"] = True
        fault_dict = _outcome_dict(fault_outcome)
        record["fault_outcome"] = fault_dict
        record["fault_console_sha256"] = _write_text_and_sha(fault_root / "console.log", fault_outcome.stdout)
        _write_json_and_sha(fault_root / "fault_outcome.json", fault_dict)
        fault_eval = classify_fault_attempt(scenario=scenario, plan=plan, outcome=fault_dict)
        record["fault_evaluation"] = fault_eval
        record["fault_signature_observed"] = bool(fault_eval.get("signature_ok", fault_eval.get("expected_fault_observed")))
        record["process_group_clean"] = fault_outcome.process_group_clean
        record["failed_workspace_removed"] = fault_outcome.failed_workspace_removed
        record["no_leaked_descendants"] = fault_outcome.process_group_clean

        if fault_eval["terminal_kind"] == "runtime_integrity_failure":
            record["recovery_command"] = []
            record["recovery_started_utc"] = None
            record["recovery_finished_utc"] = None
            record["recovery_outcome"] = {"state": "not_run_due_abort"}
            record["recovery_result_sha256"] = None
            record["repository_integrity_after"] = False
            record["disk_free_after"] = observe_i5_11c_disk(disk_path)["free_bytes"]
            record["terminal_status"] = "runtime_integrity_failure"
            record["terminal_reason"] = "unexpected_fault_signature_state_or_exit"
            return runtime_failure(f"{scenario}:unexpected_fault_signature_state_or_exit")
        if fault_eval["terminal_kind"] == "robustness_failure":
            record["recovery_command"] = []
            record["recovery_started_utc"] = None
            record["recovery_finished_utc"] = None
            record["recovery_outcome"] = {"state": "not_run_due_abort"}
            record["recovery_result_sha256"] = None
            record["repository_integrity_after"] = True
            record["disk_free_after"] = observe_i5_11c_disk(disk_path)["free_bytes"]
            record["terminal_status"] = "robustness_failure"
            record["terminal_reason"] = "fault_containment_failed"
            return finalize_i5_11c_campaign(header, scenario_records=records, terminal_status="robustness_failure", terminal_reason=f"{scenario}:fault_containment_failed")

        try:
            verify_i5_11c_repository_anchors(root)
        except ReconstructionValidationError as exc:
            record["terminal_status"] = "runtime_integrity_failure"
            record["terminal_reason"] = f"repository_anchor_drift_after_fault:{exc}"
            return runtime_failure(f"{scenario}:repository_anchor_drift_after_fault")
        if discover_live_i5_11_fault_workers():
            record["terminal_status"] = "robustness_failure"
            record["terminal_reason"] = "leaked_fault_descendant"
            return finalize_i5_11c_campaign(header, scenario_records=records, terminal_status="robustness_failure", terminal_reason=f"{scenario}:leaked_fault_descendant")

        recovery_root = output / str(plan["recovery_attempt_directory"])
        if recovery_root.exists():
            record["terminal_status"] = "runtime_integrity_failure"
            return runtime_failure(f"{scenario}:recovery_attempt_directory_collision")
        recovery_root.mkdir(parents=True, exist_ok=False)
        recovery_spec = _attempt_spec(
            repository_root=root,
            attempt_root=recovery_root,
            mode="complete",
            timeout_s=float(plan["recovery_timeout_s"]),
            terminate_grace_s=float(plan["terminate_grace_s"]),
            child_sleep_s=30.0,
        )
        record["recovery_command"] = list(recovery_spec.command)
        record["recovery_started_utc"] = _utc_now()
        recovery_outcome = run_supervised_engine_process(recovery_spec)
        record["recovery_finished_utc"] = _utc_now()
        record["recovery_attempt_executed"] = True
        recovery_dict = _outcome_dict(recovery_outcome)
        record["recovery_outcome"] = recovery_dict
        _write_text_and_sha(recovery_root / "console.log", recovery_outcome.stdout)

        recovery_result_path = Path(recovery_outcome.workspace) / "i5_11_recovery_result.json"
        recovery_payload: dict[str, Any] | None = None
        recovery_sha: str | None = None
        if recovery_result_path.is_file():
            recovery_sha = _sha256(recovery_result_path)
            value = json.loads(recovery_result_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                recovery_payload = value
        record["recovery_result_sha256"] = recovery_sha
        record["recovery_result"] = recovery_payload
        recovery_eval = classify_recovery_attempt(outcome=recovery_dict, result_payload=recovery_payload)
        record["recovery_evaluation"] = recovery_eval

        disk_after = observe_i5_11c_disk(disk_path)
        record["disk_free_after"] = disk_after["free_bytes"]
        try:
            anchors_after = verify_i5_11c_repository_anchors(root)
            record["repository_integrity_after"] = True
            record["repository_anchors_after"] = anchors_after
        except ReconstructionValidationError as exc:
            record["repository_integrity_after"] = False
            record["terminal_status"] = "runtime_integrity_failure"
            record["terminal_reason"] = f"repository_anchor_drift_after_recovery:{exc}"
            return runtime_failure(f"{scenario}:repository_anchor_drift_after_recovery")
        if disk_after["safety_floor_met"] is not True:
            record["terminal_status"] = "runtime_integrity_failure"
            record["terminal_reason"] = "disk_safety_floor_not_met_after_recovery"
            return runtime_failure(f"{scenario}:disk_safety_floor_not_met_after_recovery")
        if discover_live_i5_11_fault_workers():
            record["terminal_status"] = "robustness_failure"
            record["terminal_reason"] = "leaked_descendant_after_recovery"
            return finalize_i5_11c_campaign(header, scenario_records=records, terminal_status="robustness_failure", terminal_reason=f"{scenario}:leaked_descendant_after_recovery")
        if recovery_eval["terminal_kind"] != "recovery_pass":
            record["terminal_status"] = "robustness_failure"
            record["terminal_reason"] = "clean_recovery_failed"
            return finalize_i5_11c_campaign(header, scenario_records=records, terminal_status="robustness_failure", terminal_reason=f"{scenario}:clean_recovery_failed")

        record["terminal_status"] = "pass"
        record["terminal_reason"] = "expected_fault_contained_and_clean_recovery_completed"

    return finalize_i5_11c_campaign(
        header,
        scenario_records=records,
        terminal_status="pass",
        terminal_reason="all_three_bounded_fault_recovery_pairs_passed",
    )
