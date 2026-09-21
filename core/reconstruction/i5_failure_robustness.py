"""I5.11A OOM / disk / crash / recovery inventory and experiment admission.

I5.11A is deliberately execution-free. It inventories the existing reconstruction
lifecycle containment, recovery, checkpoint, failure-classification, and external
process supervision contracts, then admits a bounded I5.11 campaign.

The admitted campaign never intentionally exhausts the real GPU and never fills
or pressures the user's real root filesystem. CUDA OOM and ENOSPC are injected
as synthetic fault signatures inside an isolated owned worker/workspace. The
existing G1 orphan-descendant crash path is retained as the crash control.

No fault is injected in A. No SfM/MVS/provider run is executed. No production
failure classifier, recovery engine, threshold, default, authority, Hybrid, or
Auto policy is changed here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError

I511A_SCHEMA_VERSION = 1
I511A_PHASE = "I5.11A"
I511A_RECEIPT_PHASE = "I5.11A_ADMISSION_RECEIPT"
I511A_I510D_FROZEN_SHA256 = "d43bcccb0f800fc5e6ff4108d25e2757e8ef5efca1fe14b53012027b8763f309"
I511A_POST_I510D_DELTA_FROZEN_SHA256 = "773f63c31d27d5746fe788f9a0463792a6d5061676fea6f44cb004f99e4a6247"
I511A_FAULT_SCENARIOS = ("cuda_oom", "disk_enospc", "crash_orphan_descendant")

I511A_REPOSITORY_ANCHORS: dict[str, str] = {
    "core/reconstruction/engine_hardening.py": "5e64126b2c3d11895d15449c6fda9d6f04ab770b891fb9cffa1927cbdaa238c0",
    "scripts/reconstruction_engine_fault_worker.py": "0ad2100dba991d917b10fc336f5c05a41df1f1af35923b654ef18b8290ef2a8d",
    "scripts/run_reconstruction_g1_real_hardening.py": "efbbc2089f420db5898598c1deb8cc92886592b722b8ee6ff0c195fe7f874c77",
    "core/runtime/failures.py": "ba05a440c51d4c5ef6816a586acd44504ddc9ead720243791c4bfe57db4229c4",
    "app/reconstruction_failure_recovery.py": "4da8cc6a3a1fff30e208c034ff0af99613e3ed3eb43e370a76199c01392e408a",
    "app/reconstruction_processing_job_controller.py": "b2a8775d7d3ddadfb430b37dd49a1ae2eb18748ec0ec178f0f0af764cf49af12",
    "app/offline_reconstruction_controller.py": "f08e5eba46cfc4b7d79703a386e139e0cade5913de4f5c3d4d43ed9e48b51150",
    "docs/phase_history/PHASE8_7G1_VALIDATION.md": "56754a7efeca8d81578a6ac1460a7a6a522e3758d4cb103b1dc310a91b5f3c5e",
    "docs/phase_history/PHASE8_7G3_VALIDATION.md": "4a30251505ab0bed476cb40e31db1def131e2d28a22fbd7c4792231e2fe2c518",
    "docs/phase_history/PHASE8_9E_VALIDATION.md": "08eba486039713b4f43dafe6635964c245cde904170e4d51177241695e6d9151",
    "docs/phase_history/I5_10D_MP_SFM_EVIDENCE_ONLY_QUALIFICATION_AND_CLOSURE.md": "85221c319436f72b952c1ba75c66df33f55c0ec493909eb7a51e52786e454ada",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.11A {label} must be an object.")
    return value


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = node.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.11A {label}.{key} contract drift: expected={expected!r} actual={actual!r}."
        )


def _valid_sha(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _binding(value: Mapping[str, Any], *, label: str, expected_sha256: str | None = None) -> dict[str, str]:
    node = _mapping(value, label=label)
    path = str(node.get("path", "")).strip()
    sha = str(node.get("sha256", "")).strip().lower()
    if not path or not _valid_sha(sha):
        raise ReconstructionValidationError(f"I5.11A {label} binding is malformed.")
    if expected_sha256 is not None and sha != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"I5.11A {label} SHA256 drift: expected={expected_sha256} actual={sha}."
        )
    return {"path": path, "sha256": sha}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_i5_11a_repository_anchors(repository_root: Path) -> dict[str, dict[str, Any]]:
    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"I5.11A repository root missing: {root}")
    verified: dict[str, dict[str, Any]] = {}
    for relative, expected in I511A_REPOSITORY_ANCHORS.items():
        path = root / relative
        if not path.is_file():
            raise ReconstructionValidationError(f"I5.11A repository anchor missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ReconstructionValidationError(
                f"I5.11A repository anchor drifted: {relative} expected={expected} actual={actual}"
            )
        verified[relative] = {"path": str(path), "sha256": actual, "verified": True}
    return verified


def repository_failure_robustness_inventory() -> dict[str, Any]:
    """Static inventory of inspected lifecycle/failure/recovery capabilities."""

    return {
        "inventory_is_static": True,
        "resource_pressure_performed": False,
        "whole_engine_supervisor": {
            "implementation": "core.reconstruction.engine_hardening",
            "process_group_boundary": True,
            "sigterm_then_sigkill_fallback": True,
            "failed_workspace_cleanup": True,
            "nonempty_preexisting_workspace_destroyed": False,
            "orphan_descendant_cleanup": True,
            "cancel_containment": True,
            "timeout_containment": True,
            "crash_containment": True,
            "safe_for_recovery_contract": True,
            "clean_recovery_evidence": True,
        },
        "historical_qualification": {
            "phase8_7g1_cancel_timeout_crash_cleanup": True,
            "phase8_7g2_atomic_checkpoint_resume": True,
            "phase8_7g3_repeated_interrupt_recovery": True,
            "phase8_8h3_terminal_resource_release": True,
            "phase8_9e_resource_recovery_protocol": True,
        },
        "runtime_failure_classification": {
            "cuda_oom_first_class": True,
            "cuda_oom_kind": "cuda_out_of_memory",
            "cuda_oom_requires_worker_restart": True,
            "worker_timeout_first_class": True,
            "worker_crash_first_class": True,
            "disk_enospc_first_class": False,
            "disk_enospc_gap_recorded": True,
        },
        "failure_recovery_presentation": {
            "fresh_retry_is_explicit": True,
            "retained_scene_or_checkpoint_blocks_ambiguous_retry": True,
            "session_identity_conflict_fails_closed": True,
            "automatic_live_recovery_remains_existing_runtime_owned": True,
        },
        "external_processing_supervision": {
            "process_group_termination": True,
            "kill_fallback": True,
            "bounded_protocol_buffer": True,
            "canonical_fallback_preserved_on_failure": True,
        },
    }


@dataclass(frozen=True, slots=True)
class I511AFailureAdmissionPolicy:
    fault_scenarios: tuple[str, ...] = I511A_FAULT_SCENARIOS
    fault_attempts_each: int = 1
    clean_recovery_after_each: bool = True
    recovery_attempts_each: int = 1
    actual_gpu_exhaustion: bool = False
    actual_root_filesystem_fill: bool = False
    actual_user_data_corruption: bool = False
    retry_until_pass: bool = False
    result_driven_retuning: bool = False
    production_default_modified: bool = False
    checkpoint_corruption: bool = False
    cancel_replay: bool = False
    timeout_replay: bool = False

    def __post_init__(self) -> None:
        if tuple(self.fault_scenarios) != I511A_FAULT_SCENARIOS:
            raise ReconstructionValidationError("I5.11A fault scenario set is frozen to OOM, ENOSPC, and crash.")
        if int(self.fault_attempts_each) != 1 or int(self.recovery_attempts_each) != 1:
            raise ReconstructionValidationError("I5.11A admits exactly one fault and one recovery attempt per scenario.")
        if self.clean_recovery_after_each is not True:
            raise ReconstructionValidationError("I5.11A requires a clean recovery attempt after each admitted fault.")
        if self.actual_gpu_exhaustion:
            raise ReconstructionValidationError("I5.11A forbids real GPU exhaustion; use bounded synthetic CUDA OOM injection.")
        if self.actual_root_filesystem_fill:
            raise ReconstructionValidationError("I5.11A forbids filling or pressuring the real root filesystem.")
        if self.actual_user_data_corruption:
            raise ReconstructionValidationError("I5.11A forbids corrupting real user/project/evidence data.")
        for name in (
            "retry_until_pass", "result_driven_retuning", "production_default_modified",
            "checkpoint_corruption", "cancel_replay", "timeout_replay",
        ):
            if getattr(self, name) is not False:
                raise ReconstructionValidationError(f"I5.11A requires {name}=False.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "fault_scenarios": list(self.fault_scenarios),
            "fault_attempts_each": self.fault_attempts_each,
            "clean_recovery_after_each": self.clean_recovery_after_each,
            "recovery_attempts_each": self.recovery_attempts_each,
            "actual_gpu_exhaustion": self.actual_gpu_exhaustion,
            "actual_root_filesystem_fill": self.actual_root_filesystem_fill,
            "actual_user_data_corruption": self.actual_user_data_corruption,
            "retry_until_pass": self.retry_until_pass,
            "result_driven_retuning": self.result_driven_retuning,
            "production_default_modified": self.production_default_modified,
            "checkpoint_corruption": self.checkpoint_corruption,
            "cancel_replay": self.cancel_replay,
            "timeout_replay": self.timeout_replay,
        }


def validate_i5_10d_for_i5_11a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.10D closure")
    _require(root, "schema_version", 1, label="I5.10D closure")
    _require(root, "phase", "I5.10D", label="I5.10D closure")
    _require(root, "status", "mp_sfm_evidence_qualified_and_closed", label="I5.10D closure")
    _require(root, "pass", True, label="I5.10D closure")
    decision = _mapping(root.get("decision"), label="I5.10D decision")
    _require(decision, "i5_10_complete", True, label="I5.10D decision")
    _require(decision, "i5_11_authorized", True, label="I5.10D decision")
    _require(decision, "i5_11_started", False, label="I5.10D decision")
    _require(decision, "i5_12_started", False, label="I5.10D decision")
    _require(decision, "i5_13_started", False, label="I5.10D decision")
    _require(decision, "i5_14_started", False, label="I5.10D decision")
    _require(decision, "hybrid_authorized", False, label="I5.10D decision")
    _require(decision, "auto_policy_implemented", False, label="I5.10D decision")
    _require(decision, "production_default_modified", False, label="I5.10D decision")
    scope = _mapping(root.get("scope"), label="I5.10D scope")
    _require(scope, "evidence_only_qualification_and_closure", True, label="I5.10D scope")
    _require(scope, "sfm_execution", False, label="I5.10D scope")
    _require(scope, "mvs_execution", False, label="I5.10D scope")
    return deepcopy(dict(root))


def validate_i5_11a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.11A admission receipt")
    _require(root, "schema_version", 1, label="I5.11A admission receipt")
    _require(root, "phase", I511A_RECEIPT_PHASE, label="I5.11A admission receipt")
    _require(root, "status", "pass", label="I5.11A admission receipt")
    _require(root, "pass", True, label="I5.11A admission receipt")
    scope = _mapping(root.get("scope"), label="I5.11A receipt scope")
    for key in ("fault_injection", "real_gpu_oom", "real_disk_fill", "sfm_execution", "mvs_execution", "provider_inference", "production_default_modified", "auto_policy_implemented", "hybrid_authorized"):
        _require(scope, key, False, label="I5.11A receipt scope")
    codes = _mapping(root.get("return_codes"), label="I5.11A receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        _require(codes, key, 0, label="I5.11A receipt return codes")
    return deepcopy(dict(root))


def build_i5_11a_failure_mode_admission(
    *,
    i5_10d_closure: Mapping[str, Any],
    i5_10d_binding: Mapping[str, Any],
    post_i5_10d_delta_binding: Mapping[str, Any],
    admission_receipt: Mapping[str, Any],
    admission_receipt_binding: Mapping[str, Any],
    repository_anchors: Mapping[str, Any],
    policy: I511AFailureAdmissionPolicy | None = None,
) -> dict[str, Any]:
    validate_i5_10d_for_i5_11a(i5_10d_closure)
    receipt = validate_i5_11a_receipt(admission_receipt)
    i510d_bind = _binding(i5_10d_binding, label="I5.10D closure", expected_sha256=I511A_I510D_FROZEN_SHA256)
    delta_bind = _binding(post_i5_10d_delta_binding, label="post-I5.10D delta checkpoint", expected_sha256=I511A_POST_I510D_DELTA_FROZEN_SHA256)
    receipt_bind = _binding(admission_receipt_binding, label="I5.11A admission receipt")
    anchor_map = _mapping(repository_anchors, label="repository anchors")
    if not anchor_map:
        raise ReconstructionValidationError("I5.11A repository anchors cannot be empty.")
    for name, value in anchor_map.items():
        node = _mapping(value, label=f"repository anchor {name}")
        _require(node, "verified", True, label=f"repository anchor {name}")
        if not _valid_sha(node.get("sha256")) or not str(node.get("path", "")).strip():
            raise ReconstructionValidationError(f"I5.11A repository anchor {name} is malformed.")

    frozen = policy or I511AFailureAdmissionPolicy()
    inventory = repository_failure_robustness_inventory()
    fault_matrix = {
        "cuda_oom": {
            "admitted": True,
            "injection": "synthetic_cuda_oom_signature_in_isolated_fault_worker",
            "signature": "simulated cuda out-of-memory",
            "expected_failure_kind": "cuda_out_of_memory",
            "generic_failure_classifier_has_first_class_oom": True,
            "consumes_real_gpu_memory": False,
            "changes_gpu_selection": False,
            "expected_supervisor_terminal_state": "crashed",
            "required_process_group_clean": True,
            "required_failed_workspace_removed": True,
            "clean_recovery_required": True,
        },
        "disk_enospc": {
            "admitted": True,
            "injection": "synthetic_errno_enospc_in_isolated_owned_workspace",
            "signature": "OSError(errno.ENOSPC, 'No space left on device')",
            "generic_failure_classifier_has_first_class_enospc": False,
            "raw_enospc_signature_must_be_preserved": True,
            "fills_real_filesystem": False,
            "writes_only_supervisor_owned_workspace": True,
            "expected_supervisor_terminal_state": "crashed",
            "required_process_group_clean": True,
            "required_failed_workspace_removed": True,
            "clean_recovery_required": True,
        },
        "crash_orphan_descendant": {
            "admitted": True,
            "injection": "existing_g1_parent_crash_with_live_descendant",
            "existing_g1_control": True,
            "expected_supervisor_terminal_state": "crashed",
            "required_process_group_clean": True,
            "required_failed_workspace_removed": True,
            "clean_recovery_required": True,
        },
    }
    return {
        "schema_version": I511A_SCHEMA_VERSION,
        "phase": I511A_PHASE,
        "status": "failure_mode_inventory_and_experiment_admitted",
        "pass": True,
        "repository_inventory": inventory,
        "admission_policy": frozen.as_dict(),
        "fault_matrix": fault_matrix,
        "outcome_semantics": {
            "admitted_injected_fault_is_expected_event": True,
            "fault_attempt_pass_requires_expected_signature_and_clean_containment": True,
            "clean_recovery_must_complete_after_each_fault": True,
            "failed_recovery_is_robustness_failure": True,
            "unexpected_infrastructure_failure_is_experiment_incomplete": True,
            "unexpected_source_or_worker_integrity_drift_is_experiment_incomplete": True,
            "fault_result_is_not_reconstruction_quality": True,
        },
        "i5_11b_preflight_requirements": {
            "bind_i5_11a_admission": True,
            "bind_i5_10d_closure": True,
            "bind_post_i5_10d_delta_checkpoint": True,
            "reverify_repository_anchors": True,
            "freeze_three_fault_mechanisms": True,
            "freeze_clean_recovery_after_each": True,
            "freeze_attempt_directory_schema": True,
            "freeze_expected_failure_signatures": True,
            "preflight_oom_classifier_with_synthetic_exception_only": True,
            "record_enospc_classifier_gap_without_production_change": True,
            "verify_disk_free_space_without_pressure": True,
            "verify_no_fault_processes_are_running": True,
            "execute_fault_injection": False,
            "real_gpu_oom": False,
            "real_disk_fill": False,
        },
        "i5_11c_campaign_contract": {
            "fault_scenarios": list(I511A_FAULT_SCENARIOS),
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
        "excluded_or_context_only": {
            "cancel": "historically_qualified_context_only",
            "timeout": "historically_qualified_context_only",
            "checkpoint_corruption": "out_of_scope_not_admitted",
            "real_cuda_memory_exhaustion": "forbidden_use_synthetic_signature",
            "real_root_filesystem_fill": "forbidden_use_synthetic_enospc",
            "user_or_evidence_data_corruption": "forbidden",
            "backend_quality_rerun": "out_of_scope",
        },
        "claim_boundaries": {
            "reconstruction_quality_comparison": False,
            "provider_ranking": False,
            "runtime_performance_ranking": False,
            "production_failure_classifier_change_authorized": False,
            "production_recovery_engine_change_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "hybrid_authorized": False,
            "auto_policy_implemented": False,
        },
        "evidence_bindings": {
            "i5_10d_closure": i510d_bind,
            "post_i5_10d_delta_checkpoint": delta_bind,
            "i5_11a_admission_receipt": receipt_bind,
        },
        "repository_anchors": deepcopy(dict(anchor_map)),
        "validation_receipt": receipt,
        "source_control": {
            "mode": "NO_GIT_GITEA",
            "post_i5_10d_delta_checkpoint_bound": True,
            "immutable_evidence_chain_required": True,
        },
        "scope": {
            "inventory_and_admission_only": True,
            "fault_injection": False,
            "real_gpu_oom": False,
            "real_disk_fill": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "provider_inference": False,
            "production_default_modified": False,
            "production_failure_classifier_modified": False,
            "recovery_engine_modified": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
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
            "hybrid_authorized": False,
            "auto_policy_implemented": False,
            "production_default_modified": False,
            "production_failure_classifier_change_authorized": False,
            "real_resource_exhaustion_authorized": False,
        },
        "next": "I5.11B controlled fault-injection runtime/evidence preflight.",
    }


__all__ = [
    "I511A_SCHEMA_VERSION", "I511A_PHASE", "I511A_RECEIPT_PHASE",
    "I511A_I510D_FROZEN_SHA256", "I511A_POST_I510D_DELTA_FROZEN_SHA256",
    "I511A_FAULT_SCENARIOS", "I511A_REPOSITORY_ANCHORS",
    "I511AFailureAdmissionPolicy", "repository_failure_robustness_inventory",
    "verify_i5_11a_repository_anchors", "validate_i5_10d_for_i5_11a",
    "validate_i5_11a_receipt", "build_i5_11a_failure_mode_admission",
]
