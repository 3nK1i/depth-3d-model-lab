"""I5.13C frozen-evidence offline Auto-policy replay qualification.

This module does not execute reconstruction providers, SfM, MVS, or a live scene
classifier.  It replays the frozen I5.13B deterministic engine against an immutable,
predeclared matrix of evidence-context tokens and verifies exact decisions, provider
IDs, reason-code sequences, and byte-stable repeatability.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping

from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_auto_policy_engine import (
    I513B_ENGINE_PROFILE,
    I513B_PROVIDER_ID_BY_IMPLEMENTATION,
    I513BCompiledPolicy,
    I513BOfflineAutoContext,
    compile_i5_13b_policy,
    evaluate_i5_13b_auto_policy,
)

I513C_SCHEMA_VERSION = 1
I513C_PHASE = "I5.13C"
I513C_RECEIPT_PHASE = "I5.13C_REPLAY_RECEIPT"
I513C_REPLAY_PROFILE = I513B_ENGINE_PROFILE
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.13C {label} is missing or malformed.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I5.13C {label} contract drift: {key}={mapping.get(key)!r}, expected={expected!r}."
        )


def _sha(value: Any, *, label: str) -> str:
    digest = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ReconstructionValidationError(f"I5.13C {label} is not a SHA-256 digest.")
    return digest


def _binding(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    item = _mapping(value, label=label)
    path = str(item.get("path", "")).strip()
    if not path:
        raise ReconstructionValidationError(f"I5.13C {label} path is missing.")
    return {"path": path, "sha256": _sha(item.get("sha256"), label=label)}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def validate_i5_13b_for_i5_13c(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.13B engine freeze")
    _require(root, "phase", "I5.13B", label="I5.13B engine freeze")
    _require(root, "status", "deterministic_auto_policy_engine_frozen_offline_shadow", label="I5.13B engine freeze")
    _require(root, "pass", True, label="I5.13B engine freeze")
    _require(root, "quality_status", "offline_shadow_engine_pass", label="I5.13B engine freeze")
    _require(root, "policy_profile", I513C_REPLAY_PROFILE, label="I5.13B engine freeze")

    contract = _mapping(root.get("engine_contract"), label="I5.13B engine contract")
    _require(contract, "offline_shadow_only", True, label="I5.13B engine contract")
    _require(contract, "same_input_same_decision_required", True, label="I5.13B engine contract")
    _require(contract, "abstention_is_valid_terminal_decision", True, label="I5.13B engine contract")
    _require(contract, "auto_is_executable_provider", False, label="I5.13B engine contract")
    _require(contract, "global_scalar_score", False, label="I5.13B engine contract")
    _require(contract, "cross_family_rank", False, label="I5.13B engine contract")

    pre = _mapping(root.get("i5_13c_replay_preconditions"), label="I5.13C replay preconditions")
    for key in (
        "frozen_engine_module_hash_required",
        "frozen_i5_13a_design_hash_required",
        "expected_decisions_declared_before_replay",
        "frozen_evidence_only",
        "no_result_driven_rule_tuning",
    ):
        _require(pre, key, True, label="I5.13C replay preconditions")
    for key in ("provider_execution", "sfm_execution", "mvs_execution", "scenario_rematerialization"):
        _require(pre, key, False, label="I5.13C replay preconditions")

    decision = _mapping(root.get("decision"), label="I5.13B decision")
    _require(decision, "i5_13b_complete", True, label="I5.13B decision")
    _require(decision, "i5_13c_authorized", True, label="I5.13B decision")
    _require(decision, "i5_13c_started", False, label="I5.13B decision")
    _require(decision, "i5_13d_authorized", False, label="I5.13B decision")
    _require(decision, "production_auto_authorized", False, label="I5.13B decision")
    _require(decision, "production_default_modified", False, label="I5.13B decision")
    _require(decision, "provider_promoted", False, label="I5.13B decision")

    evidence = _mapping(root.get("evidence_bindings"), label="I5.13B evidence bindings")
    _binding(_mapping(evidence.get("i5_13a_admission"), label="I5.13A admission binding"), label="I5.13A admission binding")
    _binding(_mapping(evidence.get("i5_13b_engine_module"), label="I5.13B engine-module binding"), label="I5.13B engine-module binding")
    return deepcopy(dict(root))


def expected_i5_13c_replay_manifest() -> dict[str, dict[str, Any]]:
    """Immutable expectations declared before replay.

    The manifest contains only frozen evidence contexts compiled by I5.13B.  It does
    not add an unknown/future context and does not exercise engineer override as if it
    were an automatic Auto-policy decision.
    """
    select_reasons = ["auto_exact_unique_evidence_select", "offline_shadow_only"]
    ambiguous_reasons = [
        "auto_abstain",
        "multiple_clean_evidence_eligible_providers",
        "evidence_not_unique_even_after_availability",
    ]
    zero_clean_reasons = ["auto_abstain", "no_clean_evidence_eligible_provider"]
    degraded_sparse_reasons = ["auto_abstain", "sparse_context_degraded_auto_abstention"]
    manifest = {
        "mvs_unique_select": {
            "request_kind": "automatic",
            "sparse_context_id": "low_overlap_weak_parallax",
            "evidence_context_id": "i5_6_current_mvs_envelope",
            "provider_family": "mvs",
            "expected_decision_class": "select",
            "expected_selected_provider_id": "colmap-mvs-external",
            "expected_reason_codes": select_reasons,
        },
        "temporal_unique_select": {
            "request_kind": "automatic",
            "sparse_context_id": "dynamic_scene_proxy",
            "evidence_context_id": "i5_current_castle_temporal_envelope",
            "provider_family": "temporal_depth",
            "expected_decision_class": "select",
            "expected_selected_provider_id": "rolling-depth-temporal",
            "expected_reason_codes": select_reasons,
        },
        "temporal_motion_zero_clean_abstain": {
            "request_kind": "automatic",
            "sparse_context_id": "low_overlap_weak_parallax",
            "evidence_context_id": "g3_temporal_motion",
            "provider_family": "temporal_depth",
            "expected_decision_class": "abstain",
            "expected_selected_provider_id": "",
            "expected_reason_codes": zero_clean_reasons,
        },
        "temporal_static_zero_clean_abstain": {
            "request_kind": "automatic",
            "sparse_context_id": "low_overlap_weak_parallax",
            "evidence_context_id": "g3_temporal_static_sweep",
            "provider_family": "temporal_depth",
            "expected_decision_class": "abstain",
            "expected_selected_provider_id": "",
            "expected_reason_codes": zero_clean_reasons,
        },
        "single_image_complete_room_ambiguity_abstain": {
            "request_kind": "automatic",
            "sparse_context_id": "low_overlap_weak_parallax",
            "evidence_context_id": "g4_single_image_complete_room",
            "provider_family": "single_image_depth",
            "expected_decision_class": "abstain",
            "expected_selected_provider_id": "",
            "expected_reason_codes": ambiguous_reasons,
        },
        "single_image_low_texture_ambiguity_abstain": {
            "request_kind": "automatic",
            "sparse_context_id": "low_overlap_weak_parallax",
            "evidence_context_id": "g4_single_image_low_texture",
            "provider_family": "single_image_depth",
            "expected_decision_class": "abstain",
            "expected_selected_provider_id": "",
            "expected_reason_codes": ambiguous_reasons,
        },
        "single_image_weak_overlap_ambiguity_abstain": {
            "request_kind": "automatic",
            "sparse_context_id": "low_overlap_weak_parallax",
            "evidence_context_id": "g4_single_image_weak_overlap",
            "provider_family": "single_image_depth",
            "expected_decision_class": "abstain",
            "expected_selected_provider_id": "",
            "expected_reason_codes": ambiguous_reasons,
        },
        "forward_motion_hard_abstain": {
            "request_kind": "automatic",
            "sparse_context_id": "forward_motion",
            "evidence_context_id": "i5_6_current_mvs_envelope",
            "provider_family": "mvs",
            "expected_decision_class": "abstain",
            "expected_selected_provider_id": "",
            "expected_reason_codes": ["auto_abstain", "sparse_context_rejected_hard_abstention"],
        },
        "difficult_handheld_degraded_abstain": {
            "request_kind": "automatic",
            "sparse_context_id": "difficult_handheld",
            "evidence_context_id": "i5_6_current_mvs_envelope",
            "provider_family": "mvs",
            "expected_decision_class": "abstain",
            "expected_selected_provider_id": "",
            "expected_reason_codes": degraded_sparse_reasons,
        },
        "visual_stress_degraded_abstain": {
            "request_kind": "automatic",
            "sparse_context_id": "visual_stress",
            "evidence_context_id": "i5_6_current_mvs_envelope",
            "provider_family": "mvs",
            "expected_decision_class": "abstain",
            "expected_selected_provider_id": "",
            "expected_reason_codes": degraded_sparse_reasons,
        },
    }
    return deepcopy(manifest)


def _family(value: str) -> DenseProviderFamily:
    try:
        return DenseProviderFamily(value)
    except ValueError as exc:
        raise ReconstructionValidationError(f"I5.13C unknown provider family {value!r} in replay manifest.") from exc


def _available_provider_ids(family: DenseProviderFamily) -> tuple[str, ...]:
    ids = []
    for implementation, provider_id in I513B_PROVIDER_ID_BY_IMPLEMENTATION.items():
        if implementation is DenseProviderImplementation.COLMAP_MVS:
            impl_family = DenseProviderFamily.MVS
        elif implementation is DenseProviderImplementation.DA3:
            impl_family = DenseProviderFamily.MULTI_VIEW_AI_DEPTH
        elif implementation in {DenseProviderImplementation.ROLLING_DEPTH, DenseProviderImplementation.VIDEO_DEPTH_ANYTHING}:
            impl_family = DenseProviderFamily.TEMPORAL_DEPTH
        else:
            impl_family = DenseProviderFamily.SINGLE_IMAGE_DEPTH
        if impl_family is family:
            ids.append(provider_id)
    return tuple(ids)


def _request(case_name: str) -> DenseProviderSelectionRequest:
    return DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.AUTO,
        available_view_count=19,
        authoritative_sfm_available=True,
        authoritative_sfm_frame_id=f"i5_13c_replay_{case_name}",
        allow_fallback=False,
        engineer_override_provider_id="",
    )


def _context(case: Mapping[str, Any]) -> I513BOfflineAutoContext:
    family = _family(str(case["provider_family"]))
    return I513BOfflineAutoContext(
        sparse_context_id=str(case["sparse_context_id"]),
        evidence_context_id=str(case["evidence_context_id"]),
        provider_family=family,
        authoritative_sfm_production_usable=True,
        runtime_integrity_clean=True,
        available_provider_ids=_available_provider_ids(family),
    )


def _assert_b_engine_matches_admission(*, i5_13a_admission: Mapping[str, Any], i5_13b_freeze: Mapping[str, Any]) -> I513BCompiledPolicy:
    frozen = validate_i5_13b_for_i5_13c(i5_13b_freeze)
    policy = compile_i5_13b_policy(i5_13a_admission)
    compiled = _mapping(frozen.get("compiled_policy"), label="I5.13B compiled policy")
    if dict(policy.sparse_context_status) != dict(_mapping(compiled.get("sparse_context_status"), label="I5.13B sparse policy")):
        raise ReconstructionValidationError("I5.13C sparse policy no longer matches the frozen I5.13B engine.")
    frozen_bindings = _mapping(frozen.get("evidence_bindings"), label="I5.13B evidence bindings")
    a_binding = _binding(_mapping(frozen_bindings.get("i5_13a_admission"), label="I5.13A binding"), label="I5.13A binding")
    return policy


def replay_i5_13c_frozen_evidence(*, i5_13a_admission: Mapping[str, Any], i5_13b_freeze: Mapping[str, Any]) -> dict[str, Any]:
    policy = _assert_b_engine_matches_admission(i5_13a_admission=i5_13a_admission, i5_13b_freeze=i5_13b_freeze)
    manifest = expected_i5_13c_replay_manifest()
    cases: dict[str, dict[str, Any]] = {}
    class_counts = {"select": 0, "abstain": 0}
    all_expected = True
    all_reasons = True
    all_deterministic = True

    for name, expected in manifest.items():
        request = _request(name)
        context = _context(expected)
        repeated = [evaluate_i5_13b_auto_policy(policy, request, context).as_dict() for _ in range(3)]
        hashes = [_canonical_sha256(item) for item in repeated]
        deterministic = len(set(hashes)) == 1 and repeated[0] == repeated[1] == repeated[2]
        actual = repeated[0]
        actual_class = str(actual["decision_class"])
        selected = str(actual["decision"]["selected_provider_id"])
        reasons = list(actual["reason_codes"])
        decision_match = actual_class == expected["expected_decision_class"] and selected == expected["expected_selected_provider_id"]
        reason_match = reasons == list(expected["expected_reason_codes"])
        all_expected = all_expected and decision_match
        all_reasons = all_reasons and reason_match
        all_deterministic = all_deterministic and deterministic
        if actual_class not in class_counts:
            raise ReconstructionValidationError(f"I5.13C unexpected replay decision class {actual_class!r}.")
        class_counts[actual_class] += 1
        cases[name] = {
            "expected": deepcopy(expected),
            "actual_decision_class": actual_class,
            "actual_selected_provider_id": selected,
            "actual_reason_codes": reasons,
            "decision_match": decision_match,
            "reason_codes_match": reason_match,
            "same_input_same_decision": deterministic,
            "repeated_decision_sha256": hashes,
            "decision": actual["decision"],
            "evidence_eligible_provider_ids": actual["evidence_eligible_provider_ids"],
            "excluded_provider_reasons": actual["excluded_provider_reasons"],
            "matched_evidence_scenario_ids": actual["matched_evidence_scenario_ids"],
            "evidence_sha256_bindings": actual["evidence_sha256_bindings"],
            "production_effect": actual["production_effect"],
            "provider_execution": actual["provider_execution"],
            "sfm_execution": actual["sfm_execution"],
            "mvs_execution": actual["mvs_execution"],
        }

    if not all_expected:
        raise ReconstructionValidationError("I5.13C replay decision drifted from the predeclared expectation manifest.")
    if not all_reasons:
        raise ReconstructionValidationError("I5.13C replay reason codes drifted from the predeclared expectation manifest.")
    if not all_deterministic:
        raise ReconstructionValidationError("I5.13C replay was not deterministic for identical frozen inputs.")

    manifest_sha = _canonical_sha256(manifest)
    cases_sha = _canonical_sha256(cases)
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "cases": cases,
        "cases_sha256": cases_sha,
        "summary": {
            "case_count": len(cases),
            "decision_class_counts": class_counts,
            "all_expected_decisions_matched": all_expected,
            "all_reason_codes_matched": all_reasons,
            "all_same_input_same_decision": all_deterministic,
            "expected_decisions_declared_before_replay": True,
            "frozen_evidence_only": True,
            "unknown_or_future_context_replayed": False,
            "engineer_override_counted_as_auto_replay": False,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scenario_rematerialization": False,
            "production_effect": False,
            "global_scalar_score_created": False,
            "provider_ranking_created": False,
            "result_driven_rule_tuning": False,
        },
    }


def validate_i5_13c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.13C replay receipt")
    _require(root, "phase", I513C_RECEIPT_PHASE, label="I5.13C replay receipt")
    _require(root, "status", "pass", label="I5.13C replay receipt")
    _require(root, "pass", True, label="I5.13C replay receipt")
    scope = _mapping(root.get("scope"), label="I5.13C receipt scope")
    for key in (
        "focused_i5_13c_tests",
        "i5_13b_regression",
        "dense_selection_contract_regression",
        "deterministic_replay_stress",
        "import_probe",
        "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.13C receipt scope")
    for key in (
        "provider_execution",
        "sfm_execution",
        "mvs_execution",
        "scenario_rematerialization",
        "production_auto_integration",
        "production_default_modified",
        "provider_promoted",
    ):
        _require(scope, key, False, label="I5.13C receipt scope")
    codes = _mapping(root.get("return_codes"), label="I5.13C receipt return codes")
    for value in codes.values():
        if int(value) != 0:
            raise ReconstructionValidationError("I5.13C replay receipt contains a non-zero validation return code.")
    return deepcopy(dict(root))


def build_i5_13c_replay_qualification(
    *,
    i5_13a_admission: Mapping[str, Any],
    i5_13a_binding: Mapping[str, Any],
    i5_13b_freeze: Mapping[str, Any],
    i5_13b_binding: Mapping[str, Any],
    post_i5_13b_checkpoint_binding: Mapping[str, Any],
    replay_module_binding: Mapping[str, Any],
    engine_module_binding: Mapping[str, Any],
    replay_receipt: Mapping[str, Any],
    replay_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = validate_i5_13b_for_i5_13c(i5_13b_freeze)
    receipt = validate_i5_13c_receipt(replay_receipt)
    a_binding = _binding(i5_13a_binding, label="I5.13A admission")
    b_binding = _binding(i5_13b_binding, label="I5.13B engine freeze")
    checkpoint = _binding(post_i5_13b_checkpoint_binding, label="post-I5.13B source checkpoint")
    replay_module = _binding(replay_module_binding, label="I5.13C replay module")
    engine_module = _binding(engine_module_binding, label="I5.13B frozen engine module")
    receipt_binding = _binding(replay_receipt_binding, label="I5.13C replay receipt")

    frozen_bindings = _mapping(frozen.get("evidence_bindings"), label="I5.13B evidence bindings")
    frozen_a = _binding(_mapping(frozen_bindings.get("i5_13a_admission"), label="I5.13B A binding"), label="I5.13B A binding")
    frozen_engine = _binding(_mapping(frozen_bindings.get("i5_13b_engine_module"), label="I5.13B engine binding"), label="I5.13B engine binding")
    if a_binding["sha256"] != frozen_a["sha256"]:
        raise ReconstructionValidationError("I5.13C I5.13A design hash drifted from the I5.13B freeze.")
    if engine_module["sha256"] != frozen_engine["sha256"]:
        raise ReconstructionValidationError("I5.13C engine module hash drifted from the I5.13B freeze.")

    replay = replay_i5_13c_frozen_evidence(i5_13a_admission=i5_13a_admission, i5_13b_freeze=frozen)
    summary = replay["summary"]
    if summary["case_count"] != 10 or summary["decision_class_counts"] != {"select": 2, "abstain": 8}:
        raise ReconstructionValidationError("I5.13C replay-accounting contract drifted.")

    counts = _mapping(receipt.get("counts"), label="I5.13C receipt counts")
    return {
        "schema_version": I513C_SCHEMA_VERSION,
        "phase": I513C_PHASE,
        "status": "frozen_evidence_offline_auto_policy_replay_qualified",
        "pass": True,
        "quality_status": "offline_replay_pass",
        "policy_profile": I513C_REPLAY_PROFILE,
        "evidence_bindings": {
            "i5_13a_admission": a_binding,
            "i5_13b_engine_freeze": b_binding,
            "post_i5_13b_source_checkpoint": checkpoint,
            "i5_13b_engine_module": engine_module,
            "i5_13c_replay_module": replay_module,
            "i5_13c_replay_receipt": receipt_binding,
        },
        "replay_contract": {
            "expected_decisions_declared_before_replay": True,
            "expected_manifest_sha256": replay["manifest_sha256"],
            "frozen_i5_13a_design_hash_verified": True,
            "frozen_i5_13b_engine_module_hash_verified": True,
            "frozen_evidence_only": True,
            "same_input_same_decision_required": True,
            "exact_reason_code_sequence_required": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scenario_rematerialization": False,
            "no_result_driven_rule_tuning": True,
        },
        "expected_replay_manifest": replay["manifest"],
        "replay_cases": replay["cases"],
        "replay_summary": summary | {
            "replay_cases_sha256": replay["cases_sha256"],
            "unique_mvs_selection_verified": True,
            "unique_temporal_selection_verified": True,
            "temporal_zero_clean_abstention_verified": True,
            "single_image_ambiguity_abstention_verified": True,
            "forward_motion_hard_abstention_verified": True,
            "degraded_sparse_abstention_verified": True,
        },
        "regression_summary": {
            "focused_i5_13c_tests": int(counts.get("focused_tests", 0)),
            "predecessor_tests": int(counts.get("predecessor_total_tests", 0)),
            "combined_exercised_tests": int(counts.get("combined_exercised_tests", 0)),
            "test_counts_are_not_policy_scores": True,
        },
        "scope": {
            "auto_policy_engine_implemented": True,
            "offline_frozen_evidence_replay_completed": True,
            "offline_shadow_only": True,
            "auto_provider_selection_performed_in_production": False,
            "production_auto_integration": False,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scene_classifier_implemented": False,
            "scenario_rematerialization": False,
            "hybrid_execution": False,
            "matching_policy_modified": False,
            "provider_promoted": False,
            "production_default_modified": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
        },
        "claim_boundaries": {
            "production_auto_qualified": False,
            "production_auto_authorized": False,
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
            "i5_13d_authorized": True,
            "i5_13d_started": False,
            "i5_14_started": False,
            "auto_policy_engine_implemented": True,
            "offline_replay_qualified": True,
            "production_auto_authorized": False,
            "production_default_modified": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "global_scalar_score_created": False,
            "hybrid_authorized": False,
            "matching_policy_change_authorized": False,
        },
        "next": "I5.13D evidence-only production Auto integration qualification/decision (production Auto not yet authorized).",
    }


__all__ = [
    "I513C_SCHEMA_VERSION",
    "I513C_PHASE",
    "I513C_RECEIPT_PHASE",
    "I513C_REPLAY_PROFILE",
    "validate_i5_13b_for_i5_13c",
    "expected_i5_13c_replay_manifest",
    "replay_i5_13c_frozen_evidence",
    "validate_i5_13c_receipt",
    "build_i5_13c_replay_qualification",
]
