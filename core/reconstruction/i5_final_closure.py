"""I5.14B authoritative final I5 qualification and closure.

This module is evidence-only. It binds the frozen I5.14A closure-readiness
artifact, preserves its complete evidence synthesis and bounded limitations,
and marks I5 closed. It does not execute providers, SfM, MVS, scene/context
classification, hybrid reconstruction, or production Auto integration.

The post-closure immutable handoff (tiny I5.14B delta + full post-I5 source
checkpoint, both SHA256 sealed) is required by contract but is intentionally
created after the closure artifact so the artifact never claims future files
already exist.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError

I514B_SCHEMA_VERSION = 1
I514B_PHASE = "I5.14B"
I514B_RECEIPT_PHASE = "I5.14B_FINAL_I5_CLOSURE_RECEIPT"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_BOUNDED_LIMITATIONS = [
    "difficult_handheld_dense_robustness_degraded",
    "visual_stress_sparse_robustness_degraded",
    "forward_motion_sparse_sfm_rejected",
    "video_depth_anything_degraded",
    "remaining_single_image_mixed_terminal_quality",
    "mvs_runtime_performance_envelope_incomplete",
    "mvs_resource_envelope_incomplete",
    "matching_policy_candidate_rejected",
    "bounded_failure_recovery_not_universal",
    "hybrid_not_justified",
    "production_auto_deferred",
]

EXPECTED_SPARSE_ENVELOPE = {
    "difficult_handheld": "degraded",
    "low_overlap_weak_parallax": "pass",
    "visual_stress": "degraded",
    "forward_motion": "rejected",
    "dynamic_scene_proxy": "pass",
}

EXPECTED_DENSE_ENVELOPE = {
    "mvs": "pass",
    "rolling_depth": "pass",
    "video_depth_anything": "degraded",
    "remaining_single_image": "mixed_terminal_quality",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.14B {label} is missing or malformed.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I5.14B {label} contract drift: {key}={mapping.get(key)!r}, expected={expected!r}."
        )


def _binding(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    item = _mapping(value, label=label)
    path = str(item.get("path", "")).strip()
    digest = str(item.get("sha256", "")).strip().lower()
    if not path or not _SHA256_RE.fullmatch(digest):
        raise ReconstructionValidationError(f"I5.14B {label} binding is invalid.")
    return {"path": path, "sha256": digest}


def validate_i5_14b_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="closure receipt")
    _require(root, "schema_version", 1, label="closure receipt")
    _require(root, "phase", I514B_RECEIPT_PHASE, label="closure receipt")
    _require(root, "status", "pass", label="closure receipt")
    _require(root, "pass", True, label="closure receipt")

    scope = _mapping(root.get("scope"), label="receipt scope")
    for key in (
        "focused_i5_14b_tests",
        "full_i5_closure_regression",
        "evidence_only_final_closure",
        "import_probe",
        "runner_help_probe",
    ):
        _require(scope, key, True, label="receipt scope")
    for key in (
        "provider_execution",
        "sfm_execution",
        "mvs_execution",
        "scene_classifier_execution",
        "production_auto_integration",
        "production_default_modified",
        "provider_promoted",
    ):
        _require(scope, key, False, label="receipt scope")

    codes = _mapping(root.get("return_codes"), label="receipt return codes")
    for key in (
        "focused",
        "predecessor_unit",
        "predecessor_integration",
        "predecessor_stress",
        "import_probe",
        "runner_help_probe",
    ):
        _require(codes, key, 0, label="receipt return codes")
    return deepcopy(dict(root))


def validate_i5_14a_readiness(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.14A readiness")
    _require(root, "schema_version", 1, label="I5.14A readiness")
    _require(root, "phase", "I5.14A", label="I5.14A readiness")
    _require(root, "status", "final_i5_qualification_synthesized_closure_ready", label="I5.14A readiness")
    _require(root, "pass", True, label="I5.14A readiness")
    _require(root, "quality_status", "final_i5_closure_ready_with_bounded_limitations", label="I5.14A readiness")

    qualification = _mapping(root.get("qualification"), label="I5.14A qualification")
    for key in (
        "closure_ready",
        "all_required_phase_closures_bound",
        "all_required_phase_closures_passed_their_declared_contracts",
        "negative_results_preserved",
        "negative_results_are_not_closure_failures",
        "bounded_limitations_preserved",
        "production_auto_deferral_is_compatible_with_i5_closure",
    ):
        _require(qualification, key, True, label="I5.14A qualification")
    _require(qualification, "closure_artifact_count", 13, label="I5.14A qualification")

    synthesis = _mapping(root.get("evidence_synthesis"), label="I5.14A evidence synthesis")
    if dict(_mapping(synthesis.get("sparse_sfm"), label="sparse SfM envelope")) != EXPECTED_SPARSE_ENVELOPE:
        raise ReconstructionValidationError("I5.14B sparse SfM envelope drifted from I5.14A.")
    if dict(_mapping(synthesis.get("dense_provider"), label="dense provider envelope")) != EXPECTED_DENSE_ENVELOPE:
        raise ReconstructionValidationError("I5.14B dense provider envelope drifted from I5.14A.")

    matching = _mapping(synthesis.get("matching_policy"), label="matching policy synthesis")
    _require(matching, "sensitivity_observed", True, label="matching policy synthesis")
    _require(matching, "production_quality_recovery", False, label="matching policy synthesis")
    _require(matching, "candidate_quality", "rejected", label="matching policy synthesis")
    _require(matching, "production_default_change_authorized", False, label="matching policy synthesis")

    failure = _mapping(synthesis.get("failure_recovery"), label="failure recovery synthesis")
    _require(failure, "bounded_fault_recovery", "pass", label="failure recovery synthesis")
    _require(failure, "universal_failure_recovery", False, label="failure recovery synthesis")
    _require(failure, "real_resource_exhaustion_qualified", False, label="failure recovery synthesis")

    hybrid = _mapping(synthesis.get("hybrid"), label="hybrid synthesis")
    _require(hybrid, "quality_status", "hybrid_not_justified", label="hybrid synthesis")
    _require(hybrid, "execution_authorized", False, label="hybrid synthesis")

    auto = _mapping(synthesis.get("auto_policy"), label="Auto synthesis")
    _require(auto, "offline_shadow_qualified", True, label="Auto synthesis")
    _require(auto, "offline_replay_qualified", True, label="Auto synthesis")
    _require(auto, "production_auto_qualified", False, label="Auto synthesis")
    _require(auto, "production_auto_authorized", False, label="Auto synthesis")
    _require(auto, "integration_decision", "defer_production_auto", label="Auto synthesis")

    limitations = root.get("bounded_limitations")
    if not isinstance(limitations, list) or limitations != EXPECTED_BOUNDED_LIMITATIONS:
        raise ReconstructionValidationError(
            f"I5.14B bounded limitation set/order drifted: {limitations!r}."
        )

    claims = _mapping(root.get("claim_boundaries"), label="I5.14A claim boundaries")
    if not claims or any(value is not False for value in claims.values()):
        raise ReconstructionValidationError("I5.14B I5.14A claim boundaries were broadened.")

    scope = _mapping(root.get("scope"), label="I5.14A scope")
    _require(scope, "evidence_only_final_qualification_synthesis", True, label="I5.14A scope")
    for key in (
        "provider_execution",
        "sfm_execution",
        "mvs_execution",
        "scene_classifier_execution",
        "scenario_rematerialization",
        "hybrid_execution",
        "matching_policy_modified",
        "thresholds_modified",
        "production_auto_integration",
        "production_default_modified",
        "provider_promoted",
    ):
        _require(scope, key, False, label="I5.14A scope")

    decision = _mapping(root.get("decision"), label="I5.14A decision")
    _require(decision, "i5_1_through_i5_13_closed", True, label="I5.14A decision")
    _require(decision, "i5_14_started", True, label="I5.14A decision")
    _require(decision, "i5_14a_complete", True, label="I5.14A decision")
    _require(decision, "i5_14b_authorized", True, label="I5.14A decision")
    _require(decision, "i5_14b_started", False, label="I5.14A decision")
    _require(decision, "i5_complete", False, label="I5.14A decision")
    _require(decision, "production_auto_authorized", False, label="I5.14A decision")
    _require(decision, "production_default_modified", False, label="I5.14A decision")
    _require(decision, "provider_promoted", False, label="I5.14A decision")
    _require(decision, "provider_ranking_created", False, label="I5.14A decision")
    return deepcopy(dict(root))


def build_i5_14b_final_closure(
    *,
    readiness: Mapping[str, Any],
    readiness_binding: Mapping[str, Any],
    post_i5_14a_checkpoint_binding: Mapping[str, Any],
    closure_receipt: Mapping[str, Any],
    closure_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    ready = validate_i5_14a_readiness(readiness)
    ready_binding = _binding(readiness_binding, label="I5.14A readiness")
    checkpoint = _binding(post_i5_14a_checkpoint_binding, label="post-I5.14A checkpoint")
    receipt = validate_i5_14b_receipt(closure_receipt)
    receipt_binding = _binding(closure_receipt_binding, label="I5.14B closure receipt")
    counts = _mapping(receipt.get("counts"), label="closure receipt counts")

    phase_summary = {f"I5.{idx}": "closed" for idx in range(1, 15)}

    return {
        "schema_version": I514B_SCHEMA_VERSION,
        "phase": I514B_PHASE,
        "status": "final_i5_qualification_and_closure_complete",
        "pass": True,
        "quality_status": "final_i5_closed_with_bounded_limitations",
        "qualification": {
            "i5_complete": True,
            "i5_closed": True,
            "closure_with_bounded_limitations": True,
            "i5_14a_closure_ready": True,
            "all_required_phase_closures_bound": True,
            "all_required_phase_closures_contract_complete": True,
            "negative_results_preserved": True,
            "bounded_limitations_preserved": True,
            "offline_shadow_auto_qualified": True,
            "production_auto_deferral_preserved": True,
            "post_closure_immutable_handoff_required": True,
        },
        "phase_closure_summary": phase_summary,
        "evidence_synthesis": deepcopy(ready["evidence_synthesis"]),
        "bounded_limitations": deepcopy(ready["bounded_limitations"]),
        "architecture_invariants": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "native_metric_like_depth_establishes_physical_scale": False,
            "production_auto_authorized": False,
            "production_default_modified": False,
        },
        "evidence_bindings": {
            "i5_14a_closure_readiness": ready_binding,
            "post_i5_14a_source_checkpoint": checkpoint,
            "i5_14b_closure_receipt": receipt_binding,
        },
        "regression_summary": {
            "focused_i5_14b_tests": int(counts.get("focused_tests", 0)),
            "predecessor_tests": int(counts.get("predecessor_total_tests", 0)),
            "combined_exercised_tests": int(counts.get("combined_exercised_tests", 0)),
            "test_counts_are_not_quality_scores": True,
        },
        "immutable_handoff_contract": {
            "source_control_mode": "immutable_local_checkpoints",
            "git_or_gitea_required": False,
            "post_i5_14b_delta_required": True,
            "full_post_i5_source_checkpoint_required": True,
            "sha256_sidecars_required": True,
            "full_checkpoint_becomes_new_recovery_base": True,
            "pre_i5_delta_chain_may_be_archived_only_after_new_base_verified": True,
            "post_closure_handoff_checkpoint_created": False,
        },
        "scope": {
            "evidence_only_final_i5_closure": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scene_classifier_execution": False,
            "scenario_rematerialization": False,
            "hybrid_execution": False,
            "matching_policy_modified": False,
            "thresholds_modified": False,
            "production_auto_integration": False,
            "production_auto_activation": False,
            "production_default_modified": False,
            "provider_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
        },
        "claim_boundaries": {
            "universal_sfm_robustness": False,
            "universal_dense_provider_robustness": False,
            "universal_provider_winner": False,
            "global_provider_ranking": False,
            "global_scalar_score": False,
            "runtime_performance_rank": False,
            "cross_family_rank": False,
            "cross_gpu_equivalence": False,
            "production_auto_qualified": False,
            "production_auto_authorized": False,
            "certified_metric_scale": False,
        },
        "decision": {
            "i5_1_through_i5_13_closed": True,
            "i5_14_started": True,
            "i5_14a_complete": True,
            "i5_14b_complete": True,
            "i5_14_complete": True,
            "i5_complete": True,
            "i5_closed": True,
            "offline_auto_policy_qualified": True,
            "production_auto_authorized": False,
            "production_auto_integration_deferred": True,
            "production_default_modified": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "global_scalar_score_created": False,
            "final_immutable_handoff_required": True,
        },
        "next": (
            "Create and verify the post-I5.14B delta checkpoint plus a full post-I5 immutable source "
            "checkpoint with SHA256 sidecars; production Auto remains deferred."
        ),
    }


__all__ = [
    "I514B_SCHEMA_VERSION",
    "I514B_PHASE",
    "I514B_RECEIPT_PHASE",
    "EXPECTED_BOUNDED_LIMITATIONS",
    "EXPECTED_SPARSE_ENVELOPE",
    "EXPECTED_DENSE_ENVELOPE",
    "validate_i5_14a_readiness",
    "validate_i5_14b_receipt",
    "build_i5_14b_final_closure",
]
