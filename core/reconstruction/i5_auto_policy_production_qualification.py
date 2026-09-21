"""I5.13D evidence-only production Auto integration qualification / decision.

I5.13A admitted an evidence-bound conservative Auto design, I5.13B froze a
fully deterministic offline/shadow engine, and I5.13C replay-qualified that
engine against predeclared frozen evidence.  I5.13D is intentionally a decision
boundary, not an integration step: it determines whether those results are
sufficient to authorize production Auto in the current application.

The current production surface still lacks a qualified live-context -> frozen
Evidence-context mapping and production orchestration entry point.  Therefore a
passing I5.13D review closes I5.13 with the offline/shadow Auto policy qualified
while production Auto remains unqualified and unauthorized.  No provider, SfM,
MVS, scene classifier, or production Auto path is executed here.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError

I513D_SCHEMA_VERSION = 1
I513D_PHASE = "I5.13D"
I513D_RECEIPT_PHASE = "I5.13D_INTEGRATION_DECISION_RECEIPT"
I513D_POLICY_PROFILE = "evidence_bound_conservative_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.13D {label} is missing or malformed.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I5.13D {label} contract drift: {key}={mapping.get(key)!r}, expected={expected!r}."
        )


def _sha(value: Any, *, label: str) -> str:
    digest = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ReconstructionValidationError(f"I5.13D {label} is not a SHA-256 digest.")
    return digest


def _binding(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    item = _mapping(value, label=label)
    path = str(item.get("path", "")).strip()
    if not path:
        raise ReconstructionValidationError(f"I5.13D {label} path is missing.")
    return {"path": path, "sha256": _sha(item.get("sha256"), label=label)}


def validate_i5_13c_for_i5_13d(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.13C replay qualification")
    _require(root, "phase", "I5.13C", label="I5.13C replay qualification")
    _require(root, "status", "frozen_evidence_offline_auto_policy_replay_qualified", label="I5.13C replay qualification")
    _require(root, "pass", True, label="I5.13C replay qualification")
    _require(root, "quality_status", "offline_replay_pass", label="I5.13C replay qualification")
    _require(root, "policy_profile", I513D_POLICY_PROFILE, label="I5.13C replay qualification")

    replay_contract = _mapping(root.get("replay_contract"), label="I5.13C replay contract")
    for key in (
        "expected_decisions_declared_before_replay",
        "frozen_i5_13a_design_hash_verified",
        "frozen_i5_13b_engine_module_hash_verified",
        "frozen_evidence_only",
        "same_input_same_decision_required",
        "exact_reason_code_sequence_required",
        "no_result_driven_rule_tuning",
    ):
        _require(replay_contract, key, True, label="I5.13C replay contract")
    for key in ("provider_execution", "sfm_execution", "mvs_execution", "scenario_rematerialization"):
        _require(replay_contract, key, False, label="I5.13C replay contract")

    summary = _mapping(root.get("replay_summary"), label="I5.13C replay summary")
    _require(summary, "case_count", 10, label="I5.13C replay summary")
    _require(summary, "decision_class_counts", {"abstain": 8, "select": 2}, label="I5.13C replay summary")
    for key in (
        "all_expected_decisions_matched",
        "all_reason_codes_matched",
        "all_same_input_same_decision",
        "expected_decisions_declared_before_replay",
        "frozen_evidence_only",
        "unique_mvs_selection_verified",
        "unique_temporal_selection_verified",
        "temporal_zero_clean_abstention_verified",
        "single_image_ambiguity_abstention_verified",
        "forward_motion_hard_abstention_verified",
        "degraded_sparse_abstention_verified",
    ):
        _require(summary, key, True, label="I5.13C replay summary")
    for key in (
        "provider_execution",
        "sfm_execution",
        "mvs_execution",
        "scenario_rematerialization",
        "production_effect",
        "global_scalar_score_created",
        "provider_ranking_created",
        "result_driven_rule_tuning",
        "unknown_or_future_context_replayed",
        "engineer_override_counted_as_auto_replay",
    ):
        _require(summary, key, False, label="I5.13C replay summary")

    scope = _mapping(root.get("scope"), label="I5.13C scope")
    _require(scope, "offline_frozen_evidence_replay_completed", True, label="I5.13C scope")
    _require(scope, "offline_shadow_only", True, label="I5.13C scope")
    for key in (
        "auto_provider_selection_performed_in_production",
        "production_auto_integration",
        "provider_execution",
        "sfm_execution",
        "mvs_execution",
        "scene_classifier_implemented",
        "scenario_rematerialization",
        "production_default_modified",
        "provider_promoted",
    ):
        _require(scope, key, False, label="I5.13C scope")

    decision = _mapping(root.get("decision"), label="I5.13C decision")
    _require(decision, "i5_13a_complete", True, label="I5.13C decision")
    _require(decision, "i5_13b_complete", True, label="I5.13C decision")
    _require(decision, "i5_13c_complete", True, label="I5.13C decision")
    _require(decision, "i5_13d_authorized", True, label="I5.13C decision")
    _require(decision, "i5_13d_started", False, label="I5.13C decision")
    _require(decision, "offline_replay_qualified", True, label="I5.13C decision")
    _require(decision, "production_auto_authorized", False, label="I5.13C decision")
    _require(decision, "production_default_modified", False, label="I5.13C decision")
    return deepcopy(dict(root))


def inspect_i5_13d_production_integration_surface(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    architecture = root / "core/reconstruction/dense_provider_architecture.py"
    planning = root / "core/reconstruction/adaptive_production_planning.py"
    page = root / "app/widgets/reconstruction_page.py"
    engine = root / "core/reconstruction/i5_auto_policy_engine.py"
    for path in (architecture, planning, page, engine):
        if not path.is_file():
            raise ReconstructionValidationError(f"I5.13D production integration anchor missing: {path}")

    architecture_text = architecture.read_text(encoding="utf-8")
    planning_text = planning.read_text(encoding="utf-8")
    page_text = page.read_text(encoding="utf-8")
    engine_text = engine.read_text(encoding="utf-8")
    app_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in sorted((root / "app").rglob("*.py")))

    engine_ref_tokens = ("i5_auto_policy_engine", "evaluate_i5_13b_auto_policy", "I513BOfflineAutoContext")
    app_engine_ref = any(token in app_text for token in engine_ref_tokens)
    planning_auto_entrypoint = "DenseProviderSelectionMode.AUTO" in planning_text

    return {
        "auto_selection_mode_contract_exists": 'AUTO = "auto"' in architecture_text,
        "offline_engine_requires_explicit_frozen_context_tokens": (
            "I513BOfflineAutoContext" in engine_text and "explicit *frozen evidence context tokens*" in engine_text
        ),
        "production_scene_classifier_implemented": False,
        "live_context_to_evidence_token_mapping_qualified": False,
        "adaptive_production_planning_mode": (
            "explicit_provider_only"
            if "mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER" in planning_text and not planning_auto_entrypoint
            else "auto_or_mixed"
        ),
        "adaptive_planning_provider_selected_by_auto": '"provider_selected_by_auto": True' in planning_text,
        "production_ui_provider_selection_mode": (
            "inspection_only" if "Provider selection is inspection-only" in page_text else "unknown_or_active"
        ),
        "production_ui_provider_selection_activates_provider": (
            "Changing this selection never activates a provider." not in page_text
        ),
        "production_auto_engine_referenced_by_app": app_engine_ref,
        "production_auto_orchestration_entrypoint_implemented": planning_auto_entrypoint or app_engine_ref,
        "production_default_modified": False,
    }


def _validate_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    inv = deepcopy(dict(_mapping(value, label="production integration inventory")))
    expected_keys = {
        "auto_selection_mode_contract_exists",
        "offline_engine_requires_explicit_frozen_context_tokens",
        "production_scene_classifier_implemented",
        "live_context_to_evidence_token_mapping_qualified",
        "adaptive_production_planning_mode",
        "adaptive_planning_provider_selected_by_auto",
        "production_ui_provider_selection_mode",
        "production_ui_provider_selection_activates_provider",
        "production_auto_engine_referenced_by_app",
        "production_auto_orchestration_entrypoint_implemented",
        "production_default_modified",
    }
    if set(inv) != expected_keys:
        raise ReconstructionValidationError("I5.13D production integration inventory keys drifted.")
    if inv["auto_selection_mode_contract_exists"] is not True:
        raise ReconstructionValidationError("I5.13D requires the existing Auto selection-mode contract.")
    if inv["offline_engine_requires_explicit_frozen_context_tokens"] is not True:
        raise ReconstructionValidationError("I5.13D expected the frozen B engine to require explicit evidence-context tokens.")
    if inv["production_auto_orchestration_entrypoint_implemented"] is True:
        raise ReconstructionValidationError("I5.13D found an unauthorized production Auto entrypoint before the decision boundary.")
    if inv["production_default_modified"] is True:
        raise ReconstructionValidationError("I5.13D found an unauthorized production-default modification.")
    return inv


def qualify_i5_13d_production_integration(
    *,
    i5_13c_qualification: Mapping[str, Any],
    integration_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    c = validate_i5_13c_for_i5_13d(i5_13c_qualification)
    inv = _validate_inventory(integration_inventory)

    gaps: list[str] = []
    if not inv["live_context_to_evidence_token_mapping_qualified"]:
        gaps.append("live_context_to_evidence_token_mapping_not_qualified")
    if not inv["production_scene_classifier_implemented"]:
        gaps.append("production_scene_or_context_resolver_not_implemented")
    if inv["adaptive_production_planning_mode"] == "explicit_provider_only":
        gaps.append("adaptive_production_planning_explicit_provider_only")
    if inv["production_ui_provider_selection_mode"] == "inspection_only" and not inv["production_ui_provider_selection_activates_provider"]:
        gaps.append("production_ui_auto_activation_not_implemented")
    if not inv["production_auto_engine_referenced_by_app"]:
        gaps.append("production_auto_engine_not_wired_into_app")
    if not inv["production_auto_orchestration_entrypoint_implemented"]:
        gaps.append("production_auto_orchestration_entrypoint_missing")

    summary = _mapping(c.get("replay_summary"), label="I5.13C replay summary")
    prerequisites = {
        "qualified_live_context_resolver": True,
        "context_resolver_must_abstain_outside_frozen_evidence_envelope": True,
        "production_orchestration_binding_with_full_decision_provenance": True,
        "production_shadow_validation_before_live_activation": True,
        "no_result_driven_rule_tuning": True,
        "no_global_score_or_cross_family_rank": True,
        "pycolmap_camera_pose_authority_preserved": True,
    }
    return {
        "offline_policy_qualified": True,
        "offline_replay_qualified": True,
        "offline_replay_cases": int(summary["case_count"]),
        "offline_replay_selects": int(summary["decision_class_counts"]["select"]),
        "offline_replay_abstentions": int(summary["decision_class_counts"]["abstain"]),
        "production_auto_qualified": False,
        "production_auto_authorized": False,
        "decision": "defer_production_auto",
        "decision_reason": "offline_policy_determinism_is_qualified_but_live_context_resolution_and_production_orchestration_are_not",
        "blocking_gaps": gaps,
        "integration_inventory": inv,
        "required_future_prerequisites": prerequisites,
    }


def validate_i5_13d_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.13D decision receipt")
    _require(root, "phase", I513D_RECEIPT_PHASE, label="I5.13D decision receipt")
    _require(root, "status", "pass", label="I5.13D decision receipt")
    _require(root, "pass", True, label="I5.13D decision receipt")
    scope = _mapping(root.get("scope"), label="I5.13D receipt scope")
    for key in (
        "focused_i5_13d_tests",
        "i5_13c_regression",
        "production_integration_surface_regression",
        "evidence_only_decision",
        "import_probe",
        "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.13D receipt scope")
    for key in (
        "provider_execution",
        "sfm_execution",
        "mvs_execution",
        "scene_classifier_execution",
        "production_auto_integration",
        "production_default_modified",
        "provider_promoted",
    ):
        _require(scope, key, False, label="I5.13D receipt scope")
    codes = _mapping(root.get("return_codes"), label="I5.13D receipt return codes")
    for value in codes.values():
        if int(value) != 0:
            raise ReconstructionValidationError("I5.13D decision receipt contains a non-zero validation return code.")
    return deepcopy(dict(root))


def build_i5_13d_production_integration_decision(
    *,
    i5_13c_qualification: Mapping[str, Any],
    i5_13c_binding: Mapping[str, Any],
    post_i5_13c_checkpoint_binding: Mapping[str, Any],
    integration_inventory: Mapping[str, Any],
    integration_anchor_bindings: Mapping[str, Mapping[str, Any]],
    decision_receipt: Mapping[str, Any],
    decision_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    qualification = qualify_i5_13d_production_integration(
        i5_13c_qualification=i5_13c_qualification,
        integration_inventory=integration_inventory,
    )
    receipt = validate_i5_13d_receipt(decision_receipt)
    anchors = _mapping(integration_anchor_bindings, label="production integration anchor bindings")
    required_anchor_names = {
        "dense_provider_architecture",
        "adaptive_production_planning",
        "reconstruction_page",
        "i5_13b_engine_module",
    }
    if set(anchors) != required_anchor_names:
        raise ReconstructionValidationError("I5.13D integration anchor set drifted.")
    verified_anchors = {name: _binding(_mapping(value, label=name), label=name) for name, value in anchors.items()}
    counts = _mapping(receipt.get("counts"), label="I5.13D receipt counts")

    return {
        "schema_version": I513D_SCHEMA_VERSION,
        "phase": I513D_PHASE,
        "status": "auto_policy_production_integration_decision_complete",
        "pass": True,
        "quality_status": "offline_auto_qualified_production_integration_deferred",
        "policy_profile": I513D_POLICY_PROFILE,
        "qualification": {
            "offline_policy_qualified": qualification["offline_policy_qualified"],
            "offline_replay_qualified": qualification["offline_replay_qualified"],
            "offline_replay_cases": qualification["offline_replay_cases"],
            "offline_replay_selects": qualification["offline_replay_selects"],
            "offline_replay_abstentions": qualification["offline_replay_abstentions"],
            "production_auto_qualified": False,
            "production_auto_authorized": False,
            "integration_decision": qualification["decision"],
            "integration_decision_reason": qualification["decision_reason"],
        },
        "production_integration_inventory": qualification["integration_inventory"],
        "blocking_gaps": qualification["blocking_gaps"],
        "required_future_prerequisites": qualification["required_future_prerequisites"],
        "evidence_bindings": {
            "i5_13c_replay_qualification": _binding(i5_13c_binding, label="I5.13C replay qualification"),
            "post_i5_13c_source_checkpoint": _binding(post_i5_13c_checkpoint_binding, label="post-I5.13C source checkpoint"),
            "production_integration_anchors": verified_anchors,
            "i5_13d_decision_receipt": _binding(decision_receipt_binding, label="I5.13D decision receipt"),
        },
        "regression_summary": {
            "focused_i5_13d_tests": int(counts.get("focused_tests", 0)),
            "predecessor_tests": int(counts.get("predecessor_total_tests", 0)),
            "combined_exercised_tests": int(counts.get("combined_exercised_tests", 0)),
            "test_counts_are_not_policy_scores": True,
        },
        "scope": {
            "evidence_only_production_integration_decision": True,
            "offline_auto_policy_engine_qualified": True,
            "offline_replay_qualified": True,
            "production_auto_integration": False,
            "production_auto_activation": False,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scene_classifier_execution": False,
            "scenario_rematerialization": False,
            "matching_policy_modified": False,
            "hybrid_execution": False,
            "production_default_modified": False,
            "provider_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
        },
        "claim_boundaries": {
            "production_auto_qualified": False,
            "production_auto_authorized": False,
            "live_context_resolution_qualified": False,
            "universal_provider_winner": False,
            "global_provider_ranking": False,
            "global_scalar_score": False,
            "cross_family_rank": False,
            "runtime_performance_rank": False,
            "cross_gpu_equivalence": False,
            "universal_sfm_robustness": False,
            "universal_dense_provider_robustness": False,
        },
        "decision": {
            "i5_13_started": True,
            "i5_13a_complete": True,
            "i5_13b_complete": True,
            "i5_13c_complete": True,
            "i5_13d_complete": True,
            "i5_13_complete": True,
            "i5_14_authorized": True,
            "i5_14_started": False,
            "offline_auto_policy_qualified": True,
            "production_auto_authorized": False,
            "production_auto_integration_deferred": True,
            "production_default_modified": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "global_scalar_score_created": False,
            "hybrid_authorized": False,
            "matching_policy_change_authorized": False,
        },
        "next": "I5.14 final I5 qualification and closure; production Auto remains deferred until live context resolution and production orchestration are separately qualified.",
    }


__all__ = [
    "I513D_SCHEMA_VERSION",
    "I513D_PHASE",
    "I513D_RECEIPT_PHASE",
    "I513D_POLICY_PROFILE",
    "validate_i5_13c_for_i5_13d",
    "inspect_i5_13d_production_integration_surface",
    "qualify_i5_13d_production_integration",
    "validate_i5_13d_receipt",
    "build_i5_13d_production_integration_decision",
]
