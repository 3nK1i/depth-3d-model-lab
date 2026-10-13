"""I5.11B controlled fault-injection runtime/evidence preflight.

B is deliberately fault-execution-free. It binds I5.11A, re-verifies the
current lifecycle/failure-classification implementation, probes only pure
in-process exception classification, observes free disk space without pressure,
verifies that no I5.11 fault worker is currently running, and freezes the exact
I5.11C synthetic fault/recovery plan and evidence schema.

No supervised fault worker is launched here. No GPU memory is exhausted. No
filesystem is filled. No production classifier/recovery/default is modified.
"""
from __future__ import annotations

from copy import deepcopy
import errno
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_failure_robustness import I511A_FAULT_SCENARIOS
from core.runtime.failures import FailureKind, FailureStage, classify_failure

I511B_SCHEMA_VERSION = 1
I511B_PHASE = "I5.11B"
I511B_RECEIPT_PHASE = "I5.11B_PREFLIGHT_RECEIPT"
I511B_I511A_FROZEN_SHA256 = "9d1f02e4cc0cc90eb909cf17ea5621a3aaf580a75ddf2a2269ca4402d5cdf1e6"
I511B_POST_I510D_DELTA_FROZEN_SHA256 = "773f63c31d27d5746fe788f9a0463792a6d5061676fea6f44cb004f99e4a6247"
I511B_FAULT_SCENARIOS = I511A_FAULT_SCENARIOS
I511B_MIN_FREE_BYTES_FOR_C = 4 * 1024 * 1024 * 1024
I511B_FAULT_TIMEOUT_S = 10.0
I511B_RECOVERY_TIMEOUT_S = 10.0
I511B_TERMINATE_GRACE_S = 0.25
I511B_FAULT_WORKER = "scripts/i5_11_bounded_fault_worker.py"

I511B_REPOSITORY_ANCHORS: dict[str, str] = {
    "core/reconstruction/engine_hardening.py": "b441890a45afd93137a3ccfed1ac9b610e1c3f394658e397825d92ae68f7e6d2",
    "core/reconstruction/i5_failure_robustness.py": "6156f88c9c9bf7f677deeaad0c0090197d731604cdfb82de2aa7b4112d03f03e",
    "core/runtime/failures.py": "ba05a440c51d4c5ef6816a586acd44504ddc9ead720243791c4bfe57db4229c4",
    "scripts/reconstruction_engine_fault_worker.py": "463181895302c89703c9ef83ba99c5f11fce6b4bd94a311cd29291f44bf33d17",
    "scripts/i5_11_bounded_fault_worker.py": "5d3557e1c07b87816800008de0569e1f86d388657f2fbedf1200b8518c58fae7",
    "app/reconstruction_failure_recovery.py": "4da8cc6a3a1fff30e208c034ff0af99613e3ed3eb43e370a76199c01392e408a",
    "app/reconstruction_processing_job_controller.py": "b60a949b0ddb7814f828d48bb06a02ebf4cf9f3b46b42f87ff03ca3eed2bfa77",
    "docs/phase_history/I5_11A_OOM_DISK_CRASH_RECOVERY_INVENTORY_AND_ADMISSION.md": "2eeb26facf540a94c4ee3e3a0967f45bca17459febb6bb9cb8fa59a72ed86402",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.11B {label} must be an object.")
    return value


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = node.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.11B {label}.{key} contract drift: expected={expected!r} actual={actual!r}."
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


def _binding(value: Mapping[str, Any], *, label: str, expected_sha256: str | None = None) -> dict[str, str]:
    node = _mapping(value, label=label)
    path = str(node.get("path", "")).strip()
    sha = str(node.get("sha256", "")).strip().lower()
    if not path or not _valid_sha(sha):
        raise ReconstructionValidationError(f"I5.11B {label} binding is malformed.")
    if expected_sha256 is not None and sha != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"I5.11B {label} SHA256 drift: expected={expected_sha256} actual={sha}."
        )
    return {"path": path, "sha256": sha}


def verify_i5_11b_repository_anchors(repository_root: Path) -> dict[str, dict[str, Any]]:
    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"I5.11B repository root missing: {root}")
    verified: dict[str, dict[str, Any]] = {}
    for relative, expected in I511B_REPOSITORY_ANCHORS.items():
        path = root / relative
        if not path.is_file():
            raise ReconstructionValidationError(f"I5.11B repository anchor missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ReconstructionValidationError(
                f"I5.11B repository anchor drifted: {relative} expected={expected} actual={actual}"
            )
        verified[relative] = {"path": str(path), "sha256": actual, "verified": True}
    return verified


def validate_i5_11a_for_i5_11b(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.11A admission")
    _require(root, "schema_version", 1, label="I5.11A admission")
    _require(root, "phase", "I5.11A", label="I5.11A admission")
    _require(root, "status", "failure_mode_inventory_and_experiment_admitted", label="I5.11A admission")
    _require(root, "pass", True, label="I5.11A admission")

    matrix = _mapping(root.get("fault_matrix"), label="I5.11A fault matrix")
    if set(matrix) != set(I511B_FAULT_SCENARIOS):
        raise ReconstructionValidationError("I5.11B requires exactly the frozen OOM/ENOSPC/crash fault matrix members.")

    campaign = _mapping(root.get("i5_11c_campaign_contract"), label="I5.11A C campaign contract")
    _require(campaign, "fault_scenarios", list(I511B_FAULT_SCENARIOS), label="I5.11A C campaign contract")
    _require(campaign, "fault_attempts_total", 3, label="I5.11A C campaign contract")
    _require(campaign, "recovery_attempts_total", 3, label="I5.11A C campaign contract")
    _require(campaign, "fault_attempts_each", 1, label="I5.11A C campaign contract")
    _require(campaign, "recovery_attempts_each", 1, label="I5.11A C campaign contract")
    for key in (
        "attempt_directories_new_and_immutable", "fault_workspace_supervisor_owned", "clean_recovery_after_each",
    ):
        _require(campaign, key, True, label="I5.11A C campaign contract")
    for key in (
        "retry_until_pass", "result_driven_retuning", "production_default_modified", "real_resource_exhaustion",
    ):
        _require(campaign, key, False, label="I5.11A C campaign contract")

    oom = _mapping(matrix.get("cuda_oom"), label="I5.11A cuda_oom")
    _require(oom, "injection", "synthetic_cuda_oom_signature_in_isolated_fault_worker", label="I5.11A cuda_oom")
    _require(oom, "consumes_real_gpu_memory", False, label="I5.11A cuda_oom")
    disk = _mapping(matrix.get("disk_enospc"), label="I5.11A disk_enospc")
    _require(disk, "injection", "synthetic_errno_enospc_in_isolated_owned_workspace", label="I5.11A disk_enospc")
    _require(disk, "fills_real_filesystem", False, label="I5.11A disk_enospc")
    _require(disk, "generic_failure_classifier_has_first_class_enospc", False, label="I5.11A disk_enospc")
    crash = _mapping(matrix.get("crash_orphan_descendant"), label="I5.11A crash")
    _require(crash, "existing_g1_control", True, label="I5.11A crash")

    requirements = _mapping(root.get("i5_11b_preflight_requirements"), label="I5.11A B requirements")
    for key in (
        "bind_i5_10d_closure", "bind_i5_11a_admission", "bind_post_i5_10d_delta_checkpoint",
        "freeze_attempt_directory_schema", "freeze_clean_recovery_after_each",
        "freeze_expected_failure_signatures", "freeze_three_fault_mechanisms",
        "preflight_oom_classifier_with_synthetic_exception_only", "record_enospc_classifier_gap_without_production_change",
        "reverify_repository_anchors", "verify_disk_free_space_without_pressure", "verify_no_fault_processes_are_running",
    ):
        _require(requirements, key, True, label="I5.11A B requirements")
    for key in ("execute_fault_injection", "real_disk_fill", "real_gpu_oom"):
        _require(requirements, key, False, label="I5.11A B requirements")

    decision = _mapping(root.get("decision"), label="I5.11A decision")
    _require(decision, "i5_11a_complete", True, label="I5.11A decision")
    _require(decision, "i5_11b_authorized", True, label="I5.11A decision")
    for key in (
        "i5_11b_started", "i5_11c_authorized", "i5_11c_started", "i5_11d_authorized", "i5_11d_started",
        "i5_12_started", "i5_13_started", "i5_14_started", "production_default_modified",
        "production_failure_classifier_change_authorized", "real_resource_exhaustion_authorized",
    ):
        _require(decision, key, False, label="I5.11A decision")
    return deepcopy(dict(root))


def validate_i5_11b_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.11B preflight receipt")
    _require(root, "schema_version", 1, label="I5.11B preflight receipt")
    _require(root, "phase", I511B_RECEIPT_PHASE, label="I5.11B preflight receipt")
    _require(root, "status", "pass", label="I5.11B preflight receipt")
    _require(root, "pass", True, label="I5.11B preflight receipt")
    scope = _mapping(root.get("scope"), label="I5.11B receipt scope")
    for key in (
        "fault_injection", "supervised_fault_worker_execution", "real_gpu_oom", "real_disk_fill",
        "sfm_execution", "mvs_execution", "production_default_modified", "production_failure_classifier_modified",
        "production_recovery_engine_modified", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.11B receipt scope")
    _require(scope, "synthetic_exception_classification_probe", True, label="I5.11B receipt scope")
    codes = _mapping(root.get("return_codes"), label="I5.11B receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        _require(codes, key, 0, label="I5.11B receipt return codes")
    return deepcopy(dict(root))


def probe_failure_classifier_without_fault_execution() -> dict[str, Any]:
    oom_exc = RuntimeError("simulated cuda out-of-memory")
    oom = classify_failure(oom_exc, stage=FailureStage.WORKER, model_id="i5_11_synthetic")
    disk_exc = OSError(errno.ENOSPC, "No space left on device")
    disk = classify_failure(disk_exc, stage=FailureStage.WORKER, model_id="i5_11_synthetic")
    return {
        "probe_is_pure_in_process_classification_only": True,
        "fault_worker_executed": False,
        "cuda_oom": {
            "observed_kind": oom.kind.value,
            "expected_kind": FailureKind.CUDA_OUT_OF_MEMORY.value,
            "first_class": oom.kind is FailureKind.CUDA_OUT_OF_MEMORY,
            "requires_worker_restart": oom.requires_worker_restart,
            "technical_details": oom.technical_details,
        },
        "disk_enospc": {
            "observed_kind": disk.kind.value,
            "first_class": False,
            "expected_current_kind": FailureKind.UNKNOWN_MODEL_ERROR.value,
            "gap_preserved": disk.kind is FailureKind.UNKNOWN_MODEL_ERROR,
            "raw_signature_preserved": "No space left on device" in disk.technical_details,
            "technical_details": disk.technical_details,
        },
    }


def observe_disk_space(path: Path) -> dict[str, Any]:
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
        "files_created_for_probe": False,
    }


def discover_live_i5_11_fault_workers(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    root = Path(proc_root)
    if not root.is_dir():
        return []
    current_pid = os.getpid()
    found: list[dict[str, Any]] = []
    needles = ("i5_11_bounded_fault_worker.py", "reconstruction_engine_fault_worker.py")
    for entry in root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        cmdline = entry / "cmdline"
        try:
            raw = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        except (OSError, PermissionError):
            continue
        if raw and any(needle in raw for needle in needles):
            found.append({"pid": int(entry.name), "cmdline": raw})
    return sorted(found, key=lambda item: item["pid"])


def _fault_plan() -> dict[str, Any]:
    plans: dict[str, Any] = {}
    for index, scenario in enumerate(I511B_FAULT_SCENARIOS, start=1):
        mode = scenario
        fault_dir = f"{index:02d}_{scenario}_fault_attempt_001"
        recovery_dir = f"{index:02d}_{scenario}_recovery_attempt_001"
        expected_signature = {
            "cuda_oom": "simulated cuda out-of-memory",
            "disk_enospc": "No space left on device",
            "crash_orphan_descendant": "I511_FAULT=crash_orphan_descendant",
        }[scenario]
        expected_exit = 17 if scenario == "crash_orphan_descendant" else "nonzero"
        plans[scenario] = {
            "order": index,
            "fault_attempt_directory": fault_dir,
            "recovery_attempt_directory": recovery_dir,
            "fault_mode": mode,
            "recovery_mode": "complete",
            "fault_worker": I511B_FAULT_WORKER,
            "fault_command_template": [
                "${CURRENT_PYTHON}", I511B_FAULT_WORKER,
                "--workspace", "${FAULT_WORKSPACE}", "--mode", mode, "--child-sleep-s", "30.0",
            ],
            "recovery_command_template": [
                "${CURRENT_PYTHON}", I511B_FAULT_WORKER,
                "--workspace", "${RECOVERY_WORKSPACE}", "--mode", "complete", "--child-sleep-s", "30.0",
            ],
            "fault_timeout_s": I511B_FAULT_TIMEOUT_S,
            "recovery_timeout_s": I511B_RECOVERY_TIMEOUT_S,
            "terminate_grace_s": I511B_TERMINATE_GRACE_S,
            "expected_fault_state": "crashed",
            "expected_fault_exit_code": expected_exit,
            "expected_fault_signature": expected_signature,
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
    return plans


def build_i5_11b_preflight(
    *,
    i5_11a_admission: Mapping[str, Any],
    i5_11a_binding: Mapping[str, Any],
    post_i5_10d_delta_binding: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    preflight_receipt_binding: Mapping[str, Any],
    repository_anchors: Mapping[str, Any],
    classifier_probe: Mapping[str, Any],
    disk_observation: Mapping[str, Any],
    live_fault_workers: list[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_i5_11a_for_i5_11b(i5_11a_admission)
    receipt = validate_i5_11b_receipt(preflight_receipt)
    a_bind = _binding(i5_11a_binding, label="I5.11A admission", expected_sha256=I511B_I511A_FROZEN_SHA256)
    delta_bind = _binding(
        post_i5_10d_delta_binding,
        label="post-I5.10D delta checkpoint",
        expected_sha256=I511B_POST_I510D_DELTA_FROZEN_SHA256,
    )
    receipt_bind = _binding(preflight_receipt_binding, label="I5.11B preflight receipt")

    anchors = _mapping(repository_anchors, label="repository anchors")
    if not anchors:
        raise ReconstructionValidationError("I5.11B repository anchors cannot be empty.")
    for name, value in anchors.items():
        node = _mapping(value, label=f"repository anchor {name}")
        _require(node, "verified", True, label=f"repository anchor {name}")

    classifier = _mapping(classifier_probe, label="classifier probe")
    _require(classifier, "probe_is_pure_in_process_classification_only", True, label="classifier probe")
    _require(classifier, "fault_worker_executed", False, label="classifier probe")
    oom = _mapping(classifier.get("cuda_oom"), label="classifier cuda_oom")
    _require(oom, "first_class", True, label="classifier cuda_oom")
    _require(oom, "observed_kind", FailureKind.CUDA_OUT_OF_MEMORY.value, label="classifier cuda_oom")
    disk_class = _mapping(classifier.get("disk_enospc"), label="classifier disk_enospc")
    _require(disk_class, "gap_preserved", True, label="classifier disk_enospc")
    _require(disk_class, "raw_signature_preserved", True, label="classifier disk_enospc")

    disk = _mapping(disk_observation, label="disk observation")
    _require(disk, "pressure_performed", False, label="disk observation")
    _require(disk, "files_created_for_probe", False, label="disk observation")
    if int(disk.get("free_bytes", 0)) < I511B_MIN_FREE_BYTES_FOR_C or disk.get("safety_floor_met") is not True:
        raise ReconstructionValidationError("I5.11B disk safety floor is not met; C must not start.")

    if live_fault_workers:
        raise ReconstructionValidationError(
            f"I5.11B found live fault workers; refuse to freeze C plan: {list(live_fault_workers)!r}"
        )

    plans = _fault_plan()
    return {
        "schema_version": I511B_SCHEMA_VERSION,
        "phase": I511B_PHASE,
        "status": "failure_injection_runtime_and_evidence_preflight_frozen",
        "pass": True,
        "evidence_bindings": {
            "i5_11a_admission": a_bind,
            "post_i5_10d_delta_checkpoint": delta_bind,
            "i5_11b_preflight_receipt": receipt_bind,
        },
        "repository_anchors": deepcopy(dict(anchors)),
        "classifier_preflight": deepcopy(dict(classifier)),
        "disk_preflight": deepcopy(dict(disk)),
        "process_preflight": {
            "live_fault_workers": [],
            "no_fault_processes_running": True,
            "process_probe_is_read_only": True,
        },
        "fault_plan": plans,
        "campaign_contract": {
            "fault_scenarios": list(I511B_FAULT_SCENARIOS),
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
        "claim_boundaries": {
            "reconstruction_quality_comparison": False,
            "runtime_performance_ranking": False,
            "provider_ranking": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "enospc_classifier_gap_is_observation_not_fix": True,
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
        "preflight_receipt": deepcopy(receipt),
        "next": "I5.11C real bounded OOM / disk / crash / recovery campaign.",
    }
