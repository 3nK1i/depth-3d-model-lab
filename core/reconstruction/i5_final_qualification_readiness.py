"""I5.14A final I5 qualification synthesis and closure-readiness review.

This module is intentionally evidence-only.  It binds the authoritative I5
closure artifacts from I5.1C through I5.13D, preserves negative/degraded
results as bounded limitations, and decides whether I5.14B final closure may be
performed.  It does not execute providers, SfM, MVS, scene classification,
hybrid reconstruction, or production Auto integration.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError

I514A_SCHEMA_VERSION = 1
I514A_PHASE = "I5.14A"
I514A_RECEIPT_PHASE = "I5.14A_FINAL_QUALIFICATION_READINESS_RECEIPT"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

I514A_CLOSURE_SPECS: dict[str, dict[str, str]] = {
    "i5_1c": {"filename": "i5_1c_difficult_handheld_closure.json", "phase": "I5.1C", "status": "difficult_handheld_robustness_qualified", "quality": "degraded", "quality_path": "qualification"},
    "i5_2c": {"filename": "i5_2c_low_overlap_weak_parallax_closure.json", "phase": "I5.2C", "status": "low_overlap_weak_parallax_robustness_qualified", "quality": "pass", "quality_path": "qualification"},
    "i5_3c": {"filename": "i5_3c_blur_exposure_clutter_symmetry_closure.json", "phase": "I5.3C", "status": "blur_exposure_clutter_symmetry_robustness_qualified", "quality": "degraded", "quality_path": "qualification"},
    "i5_4c": {"filename": "i5_4c_forward_motion_kitti_closure.json", "phase": "I5.4C", "status": "forward_motion_kitti_robustness_qualified_and_closed", "quality": "rejected", "quality_path": "root"},
    "i5_5c": {"filename": "i5_5c_dynamic_scene_closure.json", "phase": "I5.5C", "status": "dynamic_scene_handling_robustness_qualified_and_closed", "quality": "pass", "quality_path": "root"},
    "i5_6c": {"filename": "i5_6c_mvs_robustness_closure.json", "phase": "I5.6C", "status": "mvs_robustness_qualified_and_closed", "quality": "pass", "quality_path": "root"},
    "i5_7c": {"filename": "i5_7c_rolling_depth_robustness_closure.json", "phase": "I5.7C", "status": "rolling_depth_robustness_qualified_and_closed", "quality": "pass", "quality_path": "root"},
    "i5_8c": {"filename": "i5_8c_video_depth_anything_robustness_closure.json", "phase": "I5.8C", "status": "video_depth_anything_robustness_qualified_and_closed", "quality": "degraded", "quality_path": "root"},
    "i5_9c": {"filename": "i5_9c_remaining_depth_provider_robustness_closure.json", "phase": "I5.9C", "status": "remaining_depth_provider_robustness_qualified_and_closed", "quality": "mixed_terminal_quality", "quality_path": "root"},
    "i5_10d": {"filename": "i5_10d_mp_sfm_qualification_and_closure.json", "phase": "I5.10D", "status": "mp_sfm_evidence_qualified_and_closed", "quality": "bounded_matching_policy_sensitivity_candidate_rejected", "quality_path": "root"},
    "i5_11d": {"filename": "i5_11d_failure_robustness_qualification_and_closure.json", "phase": "I5.11D", "status": "bounded_failure_recovery_evidence_qualified_and_closed", "quality": "bounded_fault_recovery_pass", "quality_path": "root"},
    "i5_12a": {"filename": "i5_12a_hybrid_evidence_justification_and_admission.json", "phase": "I5.12A", "status": "hybrid_evidence_justification_review_complete", "quality": "hybrid_not_justified", "quality_path": "root"},
    "i5_13d": {"filename": "i5_13d_auto_policy_production_integration_decision.json", "phase": "I5.13D", "status": "auto_policy_production_integration_decision_complete", "quality": "offline_auto_qualified_production_integration_deferred", "quality_path": "root"},
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.14A {label} is missing or malformed.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I5.14A {label} contract drift: {key}={mapping.get(key)!r}, expected={expected!r}."
        )


def _binding(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    item = _mapping(value, label=label)
    path = str(item.get("path", "")).strip()
    digest = str(item.get("sha256", "")).strip().lower()
    if not path or not _SHA256_RE.fullmatch(digest):
        raise ReconstructionValidationError(f"I5.14A {label} binding is invalid.")
    return {"path": path, "sha256": digest}


def validate_i5_14a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="readiness receipt")
    _require(root, "schema_version", 1, label="readiness receipt")
    _require(root, "phase", I514A_RECEIPT_PHASE, label="readiness receipt")
    _require(root, "status", "pass", label="readiness receipt")
    _require(root, "pass", True, label="readiness receipt")
    scope = _mapping(root.get("scope"), label="receipt scope")
    for key in ("focused_i5_14a_tests", "i5_closure_regression", "evidence_only_readiness_review", "import_probe", "runner_help_probe"):
        _require(scope, key, True, label="receipt scope")
    for key in ("provider_execution", "sfm_execution", "mvs_execution", "scene_classifier_execution", "production_auto_integration", "production_default_modified", "provider_promoted"):
        _require(scope, key, False, label="receipt scope")
    codes = _mapping(root.get("return_codes"), label="receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        _require(codes, key, 0, label="receipt return codes")
    return deepcopy(dict(root))


def _closure_quality(payload: Mapping[str, Any], spec: Mapping[str, str], *, label: str) -> str:
    if spec["quality_path"] == "qualification":
        node = _mapping(payload.get("qualification"), label=f"{label} qualification")
        return str(node.get("quality_status", ""))
    return str(payload.get("quality_status", ""))


def validate_i5_phase_closures(closures: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if set(closures) != set(I514A_CLOSURE_SPECS):
        missing = sorted(set(I514A_CLOSURE_SPECS) - set(closures))
        extra = sorted(set(closures) - set(I514A_CLOSURE_SPECS))
        raise ReconstructionValidationError(f"I5.14A closure set drifted: missing={missing}, extra={extra}.")
    validated: dict[str, dict[str, Any]] = {}
    for key, spec in I514A_CLOSURE_SPECS.items():
        root = _mapping(closures[key], label=key)
        _require(root, "phase", spec["phase"], label=key)
        _require(root, "status", spec["status"], label=key)
        _require(root, "pass", True, label=key)
        quality = _closure_quality(root, spec, label=key)
        if quality != spec["quality"]:
            raise ReconstructionValidationError(f"I5.14A {key} quality drift: {quality!r}, expected {spec['quality']!r}.")
        validated[key] = deepcopy(dict(root))

    q10 = _mapping(validated["i5_10d"].get("qualification"), label="I5.10D qualification")
    _require(q10, "matching_policy_sensitivity_observed", True, label="I5.10D qualification")
    _require(q10, "production_quality_recovery_observed", False, label="I5.10D qualification")

    q12 = _mapping(validated["i5_12a"].get("decision"), label="I5.12A decision")
    _require(q12, "hybrid_experiment_justified", False, label="I5.12A decision")
    _require(q12, "i5_12_complete", True, label="I5.12A decision")

    q13 = _mapping(validated["i5_13d"].get("qualification"), label="I5.13D qualification")
    _require(q13, "offline_policy_qualified", True, label="I5.13D qualification")
    _require(q13, "offline_replay_qualified", True, label="I5.13D qualification")
    _require(q13, "production_auto_qualified", False, label="I5.13D qualification")
    _require(q13, "production_auto_authorized", False, label="I5.13D qualification")
    _require(q13, "integration_decision", "defer_production_auto", label="I5.13D qualification")
    d13 = _mapping(validated["i5_13d"].get("decision"), label="I5.13D decision")
    _require(d13, "i5_13_complete", True, label="I5.13D decision")
    _require(d13, "i5_14_authorized", True, label="I5.13D decision")
    _require(d13, "production_auto_authorized", False, label="I5.13D decision")
    _require(d13, "production_default_modified", False, label="I5.13D decision")
    return validated


def build_i5_14a_final_qualification_readiness(
    *,
    closures: Mapping[str, Mapping[str, Any]],
    closure_bindings: Mapping[str, Mapping[str, Any]],
    post_i5_13d_checkpoint_binding: Mapping[str, Any],
    readiness_receipt: Mapping[str, Any],
    readiness_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_i5_phase_closures(closures)
    if set(closure_bindings) != set(I514A_CLOSURE_SPECS):
        raise ReconstructionValidationError("I5.14A closure binding set drifted.")
    bindings = {key: _binding(closure_bindings[key], label=f"{key} closure") for key in I514A_CLOSURE_SPECS}
    checkpoint = _binding(post_i5_13d_checkpoint_binding, label="post-I5.13D checkpoint")
    receipt = validate_i5_14a_receipt(readiness_receipt)
    receipt_binding = _binding(readiness_receipt_binding, label="I5.14A readiness receipt")

    sparse = {
        "difficult_handheld": "degraded",
        "low_overlap_weak_parallax": "pass",
        "visual_stress": "degraded",
        "forward_motion": "rejected",
        "dynamic_scene_proxy": "pass",
    }
    dense = {
        "mvs": "pass",
        "rolling_depth": "pass",
        "video_depth_anything": "degraded",
        "remaining_single_image": "mixed_terminal_quality",
    }
    limitations = [
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

    counts = _mapping(receipt.get("counts"), label="readiness receipt counts")
    return {
        "schema_version": I514A_SCHEMA_VERSION,
        "phase": I514A_PHASE,
        "status": "final_i5_qualification_synthesized_closure_ready",
        "pass": True,
        "quality_status": "final_i5_closure_ready_with_bounded_limitations",
        "qualification": {
            "closure_ready": True,
            "closure_artifact_count": len(I514A_CLOSURE_SPECS),
            "all_required_phase_closures_bound": True,
            "all_required_phase_closures_passed_their_declared_contracts": True,
            "negative_results_preserved": True,
            "negative_results_are_not_closure_failures": True,
            "bounded_limitations_preserved": True,
            "production_auto_deferral_is_compatible_with_i5_closure": True,
        },
        "evidence_synthesis": {
            "sparse_sfm": sparse,
            "dense_provider": dense,
            "matching_policy": {
                "sensitivity_observed": True,
                "production_quality_recovery": False,
                "candidate_quality": "rejected",
                "production_default_change_authorized": False,
            },
            "failure_recovery": {
                "bounded_fault_recovery": "pass",
                "universal_failure_recovery": False,
                "real_resource_exhaustion_qualified": False,
            },
            "hybrid": {
                "quality_status": "hybrid_not_justified",
                "execution_authorized": False,
            },
            "auto_policy": {
                "offline_shadow_qualified": True,
                "offline_replay_qualified": True,
                "production_auto_qualified": False,
                "production_auto_authorized": False,
                "integration_decision": "defer_production_auto",
            },
        },
        "bounded_limitations": limitations,
        "evidence_bindings": {
            "phase_closures": bindings,
            "post_i5_13d_source_checkpoint": checkpoint,
            "i5_14a_readiness_receipt": receipt_binding,
        },
        "regression_summary": {
            "focused_i5_14a_tests": int(counts.get("focused_tests", 0)),
            "predecessor_tests": int(counts.get("predecessor_total_tests", 0)),
            "combined_exercised_tests": int(counts.get("combined_exercised_tests", 0)),
            "test_counts_are_not_quality_scores": True,
        },
        "scope": {
            "evidence_only_final_qualification_synthesis": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scene_classifier_execution": False,
            "scenario_rematerialization": False,
            "hybrid_execution": False,
            "matching_policy_modified": False,
            "thresholds_modified": False,
            "production_auto_integration": False,
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
            "i5_14b_authorized": True,
            "i5_14b_started": False,
            "i5_complete": False,
            "production_auto_authorized": False,
            "production_default_modified": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
        },
        "next": "I5.14B final I5 closure and immutable handoff; no production Auto activation.",
    }


__all__ = [
    "I514A_SCHEMA_VERSION",
    "I514A_PHASE",
    "I514A_RECEIPT_PHASE",
    "I514A_CLOSURE_SPECS",
    "validate_i5_14a_receipt",
    "validate_i5_phase_closures",
    "build_i5_14a_final_qualification_readiness",
]
