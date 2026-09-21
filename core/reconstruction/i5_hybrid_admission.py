"""I5.12A evidence-justification and admission review for optional hybrid reconstruction.

This step is evidence-only. It decides whether the frozen evidence actually
satisfies the historical I4.3A hybrid-admission hypothesis:

    MVS is cleaner while AI depth meaningfully recovers legitimate MVS holes.

I5.12A never runs MVS, DA3, SfM, fusion, provider inference, Auto policy, or a
hybrid candidate. A passing review can therefore end in either:

* ``hybrid_justified``: authorize I5.12B preflight for one bounded experiment; or
* ``hybrid_not_justified``: close I5.12 without experiment and authorize I5.13.

Global completeness alone is intentionally insufficient. The direct matched-view
``legitimate_hole_recovery`` dimension must prefer DA3 while the frozen cleanup
signals prefer COLMAP MVS.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError

I512A_SCHEMA_VERSION = 1
I512A_PHASE = "I5.12A"
I512A_RECEIPT_PHASE = "I5.12A_ADMISSION_RECEIPT"
I512A_G2_SCENARIO = "g2_weak_overlap_mvs_da3"
I512A_HYBRID_RULE = "mvs_cleaner_and_ai_depth_recovers_legitimate_mvs_holes"
I512A_CLEANLINESS_DIMENSIONS = (
    "false_geometry_and_floaters",
    "surface_noise_and_roughness",
    "false_bridging_or_thickening",
)
I512A_DIRECT_HOLE_DIMENSION = "legitimate_hole_recovery"
I512A_COMPLETENESS_DIMENSION = "global_completeness"


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.12A {label} must be an object.")
    return value


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = node.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.12A {label}.{key} contract drift: expected={expected!r} actual={actual!r}."
        )


def _valid_sha(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _binding(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    node = _mapping(value, label=label)
    path = str(node.get("path", "")).strip()
    sha = str(node.get("sha256", "")).strip().lower()
    if not path or not _valid_sha(sha):
        raise ReconstructionValidationError(f"I5.12A {label} binding is malformed.")
    return {"path": path, "sha256": sha}


def validate_i5_11d_for_i5_12a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.11D closure")
    _require(root, "schema_version", 1, label="I5.11D closure")
    _require(root, "phase", "I5.11D", label="I5.11D closure")
    _require(root, "status", "bounded_failure_recovery_evidence_qualified_and_closed", label="I5.11D closure")
    _require(root, "pass", True, label="I5.11D closure")
    decision = _mapping(root.get("decision"), label="I5.11D decision")
    _require(decision, "i5_11_complete", True, label="I5.11D decision")
    _require(decision, "i5_12_admission_review_authorized", True, label="I5.11D decision")
    _require(decision, "i5_12_started", False, label="I5.11D decision")
    _require(decision, "hybrid_authorized", False, label="I5.11D decision")
    _require(decision, "auto_policy_implemented", False, label="I5.11D decision")
    scope = _mapping(root.get("scope"), label="I5.11D scope")
    _require(scope, "hybrid_authorized", False, label="I5.11D scope")
    _require(scope, "auto_policy_implemented", False, label="I5.11D scope")
    if root.get("next") != "I5.12A hybrid experiment evidence-justification and admission review (hybrid execution not yet authorized).":
        raise ReconstructionValidationError("I5.12A predecessor next-boundary contract drifted.")
    return deepcopy(dict(root))


def validate_i5_6c_for_i5_12a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.6C MVS closure")
    _require(root, "phase", "I5.6C", label="I5.6C MVS closure")
    _require(root, "status", "mvs_robustness_qualified_and_closed", label="I5.6C MVS closure")
    _require(root, "pass", True, label="I5.6C MVS closure")
    qualification = _mapping(root.get("qualification"), label="I5.6C qualification")
    _require(qualification, "current_production_colmap_mvs_path_qualified_within_admitted_envelope", True, label="I5.6C qualification")
    _require(qualification, "universal_mvs_robustness_claimed", False, label="I5.6C qualification")
    _require(qualification, "provider_ranking_claimed", False, label="I5.6C qualification")
    authority = _mapping(root.get("authority"), label="I5.6C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.6C authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.6C authority")
    decision = _mapping(root.get("decision"), label="I5.6C decision")
    _require(decision, "i5_6_complete", True, label="I5.6C decision")
    _require(decision, "hybrid_authorized", False, label="I5.6C decision")
    _require(decision, "production_default_modified", False, label="I5.6C decision")
    return deepcopy(dict(root))


def validate_i5_2a_for_i5_12a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.2A admission")
    _require(root, "phase", "I5.2A", label="I5.2A admission")
    _require(root, "status", "low_overlap_weak_parallax_scenario_admitted", label="I5.2A admission")
    _require(root, "pass", True, label="I5.2A admission")
    bindings = _mapping(root.get("evidence_bindings"), label="I5.2A evidence bindings")
    g2 = _binding(_mapping(bindings.get("i4_3g2_final_closure"), label="I5.2A G2 binding"), label="I5.2A G2 binding")
    historical = _mapping(root.get("historical_baseline"), label="I5.2A historical baseline")
    _require(historical, "scenario_id", I512A_G2_SCENARIO, label="I5.2A historical baseline")
    _require(historical, "historical_review_scope", "authoritative_registered_views", label="I5.2A historical baseline")
    _require(historical, "historical_partial_registration_preserved", True, label="I5.2A historical baseline")
    decision = _mapping(root.get("decision"), label="I5.2A decision")
    _require(decision, "hybrid_authorized", False, label="I5.2A decision")
    return {"payload": deepcopy(dict(root)), "g2_binding": g2}


def validate_i4_3a_hybrid_rule(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ReconstructionValidationError("I5.12A I4.3A phase-history text is empty.")
    normalized = " ".join(text.split())
    required = (
        "A `hybrid` family is reserved",
        "MVS is cleaner while AI depth meaningfully recovers legitimate MVS holes",
        "No concrete hybrid provider is registered or authorized",
    )
    missing = [needle for needle in required if needle not in normalized]
    if missing:
        raise ReconstructionValidationError(f"I5.12A I4.3A hybrid-admission rule drifted: missing={missing}")
    return {
        "rule_id": I512A_HYBRID_RULE,
        "mvs_cleaner_required": True,
        "ai_legitimate_hole_recovery_required": True,
        "global_completeness_alone_sufficient": False,
        "direct_cross_provider_evidence_required": True,
    }


def _preference(prefs: Mapping[str, Any], dimension: str) -> str | None:
    value = prefs.get(dimension)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text not in {"colmap_mvs", "da3", "tie", "not_assessable"}:
        raise ReconstructionValidationError(
            f"I5.12A G2 preference {dimension} has unsupported value {text!r}."
        )
    return text


def evaluate_g2_hybrid_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I4.3G2 final closure")
    _require(root, "phase", "I4.3G2", label="I4.3G2 final closure")
    _require(root, "status", "formally_closed", label="I4.3G2 final closure")
    _require(root, "all_required_g2_scenarios_accounted", True, label="I4.3G2 final closure")
    _require(root, "all_required_manual_reviews_complete", True, label="I4.3G2 final closure")
    _require(root, "i4_3g2_closed", True, label="I4.3G2 final closure")
    synthesis = _mapping(root.get("synthesis"), label="I4.3G2 synthesis")
    _require(synthesis, "camera_pose_authority_owner", "pycolmap_sfm", label="I4.3G2 synthesis")
    for key in (
        "provider_selected", "provider_promoted", "universal_provider_winner_declared",
        "hybrid_provider_authorized", "metric_scale_claimed", "measurement_authoritative",
        "certified_metrology", "surface_mesh_generated",
    ):
        _require(synthesis, key, False, label="I4.3G2 synthesis")

    scenarios = _mapping(root.get("scenario_synthesis"), label="I4.3G2 scenario synthesis")
    weak = _mapping(scenarios.get(I512A_G2_SCENARIO), label="I4.3G2 weak-overlap scenario")
    _require(weak, "controlled_reveal_complete", True, label="I4.3G2 weak-overlap scenario")
    _require(weak, "manual_review_complete", True, label="I4.3G2 weak-overlap scenario")
    _require(weak, "global_provider_winner_declared", False, label="I4.3G2 weak-overlap scenario")
    _require(weak, "evidence_mode", "fresh_production_plus_matched_view", label="I4.3G2 weak-overlap scenario")
    if "partial_registration_preserved" in weak:
        _require(weak, "partial_registration_preserved", True, label="I4.3G2 weak-overlap scenario")
    if "full_scenario_claim_permitted" in weak:
        _require(weak, "full_scenario_claim_permitted", False, label="I4.3G2 weak-overlap scenario")
    _require(weak, "review_scope", "authoritative_registered_views", label="I4.3G2 weak-overlap scenario")

    prefs = _mapping(weak.get("provider_preferences"), label="I4.3G2 weak-overlap provider preferences")
    required_dims = set(I512A_CLEANLINESS_DIMENSIONS) | {I512A_DIRECT_HOLE_DIMENSION, I512A_COMPLETENESS_DIMENSION}
    missing = sorted(required_dims - set(prefs))
    if missing:
        raise ReconstructionValidationError(f"I5.12A G2 weak-overlap evidence lacks required review dimensions: {missing}")

    cleanliness = {dimension: _preference(prefs, dimension) for dimension in I512A_CLEANLINESS_DIMENSIONS}
    hole = _preference(prefs, I512A_DIRECT_HOLE_DIMENSION)
    completeness = _preference(prefs, I512A_COMPLETENESS_DIMENSION)
    mvs_cleaner = all(value == "colmap_mvs" for value in cleanliness.values())
    ai_legitimate_hole_recovery = hole == "da3"
    direct_condition_met = mvs_cleaner and ai_legitimate_hole_recovery

    return {
        "scenario_id": I512A_G2_SCENARIO,
        "review_scope": "authoritative_registered_views",
        "provider_preferences": {str(k): v for k, v in prefs.items()},
        "cleanliness_preferences": cleanliness,
        "legitimate_hole_recovery_preference": hole,
        "global_completeness_preference": completeness,
        "mvs_cleaner_observed": mvs_cleaner,
        "ai_legitimate_hole_recovery_observed": ai_legitimate_hole_recovery,
        "global_completeness_da3_observed": completeness == "da3",
        "global_completeness_alone_insufficient": True,
        "historical_hybrid_condition_met": direct_condition_met,
    }


def validate_i5_12a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.12A receipt")
    _require(root, "schema_version", 1, label="I5.12A receipt")
    _require(root, "phase", I512A_RECEIPT_PHASE, label="I5.12A receipt")
    _require(root, "status", "pass", label="I5.12A receipt")
    _require(root, "pass", True, label="I5.12A receipt")
    scope = _mapping(root.get("scope"), label="I5.12A receipt scope")
    for key in (
        "focused_i5_12a_tests", "predecessor_i5_11d_regression", "mvs_robustness_regression",
        "g2_visual_evidence_regression", "i5_2a_binding_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.12A receipt scope")
    for key in (
        "provider_execution", "sfm_execution", "mvs_execution", "da3_execution", "hybrid_execution",
        "auto_policy_implemented", "production_default_modified", "thresholds_modified",
    ):
        _require(scope, key, False, label="I5.12A receipt scope")
    counts = _mapping(root.get("counts"), label="I5.12A receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(value, int) and value >= 0 for value in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.12A receipt test counts must be non-negative integers.")
    if focused < 20 or predecessor < 20 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.12A receipt coverage is below the required admission-review floor.")
    rcs = _mapping(root.get("return_codes"), label="I5.12A receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.12A receipt {key} return code is not zero.")
    return deepcopy(dict(root))


def build_i5_12a_hybrid_admission_review(
    *,
    i5_11d_closure: Mapping[str, Any],
    i5_11d_binding: Mapping[str, Any],
    i5_6c_closure: Mapping[str, Any],
    i5_6c_binding: Mapping[str, Any],
    i5_2a_admission: Mapping[str, Any],
    i5_2a_binding: Mapping[str, Any],
    i4_3g2_closure: Mapping[str, Any],
    i4_3g2_binding: Mapping[str, Any],
    i4_3a_phase_history_text: str,
    i4_3a_binding: Mapping[str, Any],
    admission_receipt: Mapping[str, Any],
    admission_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    i511d = validate_i5_11d_for_i5_12a(i5_11d_closure)
    i56c = validate_i5_6c_for_i5_12a(i5_6c_closure)
    i52a_info = validate_i5_2a_for_i5_12a(i5_2a_admission)
    rule = validate_i4_3a_hybrid_rule(i4_3a_phase_history_text)
    evidence = evaluate_g2_hybrid_evidence(i4_3g2_closure)
    receipt = validate_i5_12a_receipt(admission_receipt)

    d_bind = _binding(i5_11d_binding, label="I5.11D closure")
    m_bind = _binding(i5_6c_binding, label="I5.6C closure")
    a_bind = _binding(i5_2a_binding, label="I5.2A admission")
    g2_bind = _binding(i4_3g2_binding, label="I4.3G2 closure")
    doc_bind = _binding(i4_3a_binding, label="I4.3A phase history")
    receipt_bind = _binding(admission_receipt_binding, label="I5.12A receipt")

    expected_g2 = i52a_info["g2_binding"]
    if g2_bind != expected_g2:
        raise ReconstructionValidationError(
            "I5.12A independently verified I4.3G2 closure does not match the closure bound by frozen I5.2A evidence."
        )

    # Latest roadmap guard: no hybrid or Auto work may already be active.
    latest_decision = _mapping(i511d.get("decision"), label="I5.11D decision")
    _require(latest_decision, "hybrid_authorized", False, label="I5.11D decision")
    _require(latest_decision, "auto_policy_implemented", False, label="I5.11D decision")
    mvs_qualification = _mapping(i56c.get("qualification"), label="I5.6C qualification")

    justified = bool(evidence["historical_hybrid_condition_met"])
    if justified:
        decision_status = "hybrid_justified"
        conclusion = "direct_g2_evidence_satisfies_mvs_cleaner_plus_legitimate_ai_hole_recovery_condition"
        next_step = "I5.12B hybrid integration/runtime/evidence preflight."
    else:
        decision_status = "hybrid_not_justified"
        conclusion = "direct_g2_evidence_does_not_satisfy_legitimate_ai_hole_recovery_condition"
        next_step = "I5.13 first evidence-driven Auto policy admission/design; I5.12 closes without hybrid execution."

    return {
        "schema_version": I512A_SCHEMA_VERSION,
        "phase": I512A_PHASE,
        "status": "hybrid_evidence_justification_review_complete",
        "pass": True,
        "quality_status": decision_status,
        "conclusion": conclusion,
        "historical_admission_rule": rule,
        "direct_cross_provider_evidence": evidence,
        "mvs_robustness_context": {
            "i5_6_quality_status": str(i56c.get("quality_status")),
            "current_production_colmap_mvs_path_qualified_within_admitted_envelope": bool(
                mvs_qualification.get("current_production_colmap_mvs_path_qualified_within_admitted_envelope")
            ),
            "universal_mvs_robustness_claimed": False,
            "runtime_performance_envelope_complete": bool(
                _mapping(i56c.get("limitations"), label="I5.6C limitations").get("runtime_performance_envelope_complete", False)
            ),
            "resource_envelope_complete": bool(
                _mapping(i56c.get("limitations"), label="I5.6C limitations").get("resource_envelope_complete", False)
            ),
        },
        "candidate_hypothesis": {
            "admitted": justified,
            "provider_pair": ["colmap_mvs", "da3"],
            "scenario_id": I512A_G2_SCENARIO,
            "baseline": "frozen_colmap_mvs_only_output",
            "candidate": "mvs_primary_with_strictly_gated_da3_legitimate_hole_supplement",
            "primary_changed_variable": "add_da3_geometry_only_at_predeclared_legitimate_mvs_holes",
            "camera_pose_authority_owner": "pycolmap_sfm",
            "mvs_remains_primary_geometry": True,
            "provider_native_outputs_must_remain_immutable": True,
            "provider_rerun_authorized_by_i5_12a": False,
            "sfm_rerun_authorized_by_i5_12a": False,
            "single_candidate_attempt_if_authorized": 1 if justified else 0,
            "global_completeness_only_is_not_admission_evidence": True,
        },
        "comparison_contract": {
            "production_native_baseline_primary": True,
            "matched_view_diagnostic_secondary": True,
            "allowed_dimensions": [
                "global_completeness", "legitimate_hole_recovery", "false_geometry_and_floaters",
                "surface_noise_and_roughness", "thin_structure_preservation", "edge_definition",
                "small_detail_preservation", "false_bridging_or_thickening",
            ],
            "global_scalar_score_forbidden": True,
            "universal_provider_winner_forbidden": True,
            "metric_scale_comparison_forbidden": True,
            "absolute_cross_run_pose_comparison_forbidden": True,
            "runtime_performance_ranking_forbidden": True,
        },
        "evidence_bindings": {
            "i5_11d_closure": d_bind,
            "i5_6c_mvs_closure": m_bind,
            "i5_2a_admission": a_bind,
            "i4_3g2_final_closure": g2_bind,
            "i4_3a_phase_history": doc_bind,
            "i5_12a_admission_receipt": receipt_bind,
        },
        "validation_receipt": receipt,
        "scope": {
            "evidence_only_admission_review": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "da3_execution": False,
            "hybrid_execution": False,
            "matched_view_rendering": False,
            "manual_review_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "i5_10_complete": True,
            "i5_11_complete": True,
            "i5_12_started": True,
            "i5_12a_complete": True,
            "hybrid_experiment_justified": justified,
            "i5_12b_authorized": justified,
            "i5_12b_started": False,
            "i5_12c_authorized": False,
            "i5_12_complete": not justified,
            "i5_13_authorized": not justified,
            "i5_13_started": False,
            "i5_14_started": False,
            "hybrid_authorized": False,
            "hybrid_execution_authorized": False,
            "provider_selected": False,
            "provider_promoted": False,
            "production_default_modified": False,
            "auto_policy_implemented": False,
        },
        "next": next_step,
    }


__all__ = [
    "I512A_SCHEMA_VERSION", "I512A_PHASE", "I512A_RECEIPT_PHASE", "I512A_G2_SCENARIO",
    "I512A_HYBRID_RULE", "I512A_CLEANLINESS_DIMENSIONS", "I512A_DIRECT_HOLE_DIMENSION",
    "validate_i5_11d_for_i5_12a", "validate_i5_6c_for_i5_12a", "validate_i5_2a_for_i5_12a",
    "validate_i4_3a_hybrid_rule", "evaluate_g2_hybrid_evidence", "validate_i5_12a_receipt",
    "build_i5_12a_hybrid_admission_review",
]
