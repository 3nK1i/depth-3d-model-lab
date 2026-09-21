"""I5.13A evidence synthesis and conservative Auto-policy admission/design.

This module is intentionally execution-free.  It consumes the frozen cross-provider
and I5 robustness closures and admits only the *design contract* for the first
evidence-driven Auto policy.  It does not select or execute a dense provider,
change PyCOLMAP authority, change a production default, implement Hybrid, alter
matching policy, create a provider ranking, or claim metric scale/metrology.

The admitted v1 policy is conservative and abstention-first:

* Auto is a selection policy, never an executable provider ID.
* PyCOLMAP SfM remains the sole camera/global-frame authority.
* Exact scenario evidence is required; cross-family/global ranking is forbidden.
* REJECTED evidence is ineligible for automatic primary selection.
* DEGRADED or conflicting evidence cannot become an automatic primary merely by
  counting PASS events elsewhere.
* Frozen historical rejections remain active and cannot be overwritten by a later
  different-scenario PASS.
* If evidence does not identify one defensible provider, Auto abstains with reason
  codes instead of inventing a tie-break.
* I5.13B is authorized only to implement the deterministic offline/shadow policy
  engine.  Production integration remains deferred to I5.13D.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
    DenseProviderSelectionMode,
    I43A_AUTO_POLICY_ID,
    I43A_AUTHORITATIVE_FRAME_OWNER,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i4_3_final_qualification import validate_i4_3g5b_closure

I513A_SCHEMA_VERSION = 1
I513A_PHASE = "I5.13A"
I513A_RECEIPT_PHASE = "I5.13A_ADMISSION_RECEIPT"
I513A_POLICY_PROFILE = "evidence_bound_conservative_v1"
I513A_DECISION_CLASSES = ("select", "abstain", "engineer_override")
I513A_PROVIDER_IMPLEMENTATIONS = tuple(item.value for item in DenseProviderImplementation)
I513A_PROVIDER_FAMILIES = tuple(item.value for item in DenseProviderFamily)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.13A {label} is missing or malformed.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"I5.13A {label} is missing or malformed.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I5.13A {label} contract drift: {key}={mapping.get(key)!r}, expected={expected!r}."
        )


def _binding(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    root = _mapping(value, label=f"{label} binding")
    path = str(root.get("path", "")).strip()
    sha = str(root.get("sha256", "")).strip().lower()
    if not path or not _SHA256_RE.fullmatch(sha):
        raise ReconstructionValidationError(f"I5.13A {label} binding is malformed.")
    return {"path": path, "sha256": sha}


def _decision_guards(payload: Mapping[str, Any], *, label: str) -> None:
    decision = _mapping(payload.get("decision"), label=f"{label} decision")
    for key in (
        "provider_selected",
        "provider_promoted",
        "provider_ranking_created",
        "universal_provider_winner_declared",
        "global_scalar_score_created",
        "auto_policy_implemented",
        "hybrid_authorized",
        "production_default_modified",
    ):
        if key in decision:
            _require(decision, key, False, label=f"{label} decision")


def _authority_guard(payload: Mapping[str, Any], *, label: str) -> None:
    authority = payload.get("authority")
    if not isinstance(authority, Mapping):
        return
    _require(authority, "camera_pose_authority_owner", I43A_AUTHORITATIVE_FRAME_OWNER, label=f"{label} authority")
    if "permanent_global_view_cap" in authority:
        _require(authority, "permanent_global_view_cap", None, label=f"{label} authority")
    for key in (
        "authoritative_sfm_modified",
        "provider_selected",
        "provider_promoted",
        "provider_ranking_created",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    ):
        if key in authority:
            _require(authority, key, False, label=f"{label} authority")


def _root(payload: Mapping[str, Any], *, phase: str, status: str, label: str) -> Mapping[str, Any]:
    root = _mapping(payload, label=label)
    _require(root, "phase", phase, label=label)
    _require(root, "status", status, label=label)
    _require(root, "pass", True, label=label)
    _decision_guards(root, label=label)
    _authority_guard(root, label=label)
    return root


def validate_i5_1c_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.1C", status="difficult_handheld_robustness_qualified", label="I5.1C closure")
    q = _mapping(root.get("qualification"), label="I5.1C qualification")
    _require(q, "quality_status", "degraded", label="I5.1C qualification")
    _require(q, "operating_envelope_classification", "outside_current_production_dense_robustness_envelope", label="I5.1C qualification")
    _require(q, "production_dense_gate_reached", False, label="I5.1C qualification")
    return {
        "quality_status": "degraded",
        "operating_envelope": q["operating_envelope_classification"],
        "primary_bottleneck": str(q.get("primary_robustness_bottleneck", "")),
        "auto_implication": "exact_difficult_handheld_evidence_requires_abstention_or_explicit_engineer_choice",
    }


def validate_i5_2c_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.2C", status="low_overlap_weak_parallax_robustness_qualified", label="I5.2C closure")
    q = _mapping(root.get("qualification"), label="I5.2C qualification")
    status = str(q.get("quality_status", ""))
    if status not in {"pass", "degraded", "rejected"}:
        raise ReconstructionValidationError("I5.13A I5.2C quality status is invalid.")
    _require(q, "low_overlap_authoritative_revalidation", True, label="I5.2C qualification")
    _require(q, "weak_parallax_authoritative_revalidation", True, label="I5.2C qualification")
    return {
        "quality_status": status,
        "operating_envelope": str(q.get("operating_envelope", "")),
        "partial_registration_preserved": bool(q.get("partial_registration_preserved")),
        "auto_implication": "scenario_specific_sparse_evidence_only_no_cross_scenario_generalization",
    }


def validate_i5_3c_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.3C", status="blur_exposure_clutter_symmetry_robustness_qualified", label="I5.3C closure")
    q = _mapping(root.get("qualification"), label="I5.3C qualification")
    _require(q, "quality_status", "degraded", label="I5.3C qualification")
    conclusions = _mapping(root.get("stress_conclusions"), label="I5.3C stress conclusions")
    required = {"blur", "underexposure", "overexposure", "clutter", "symmetry"}
    if not required.issubset(conclusions):
        raise ReconstructionValidationError("I5.13A I5.3C stress-conclusion coverage is incomplete.")
    return {
        "quality_status": "degraded",
        "operating_envelope": str(q.get("operating_envelope", "")),
        "blur": deepcopy(dict(_mapping(conclusions["blur"], label="I5.3C blur conclusion"))),
        "underexposure": deepcopy(dict(_mapping(conclusions["underexposure"], label="I5.3C underexposure conclusion"))),
        "overexposure": deepcopy(dict(_mapping(conclusions["overexposure"], label="I5.3C overexposure conclusion"))),
        "clutter": deepcopy(dict(_mapping(conclusions["clutter"], label="I5.3C clutter conclusion"))),
        "symmetry": deepcopy(dict(_mapping(conclusions["symmetry"], label="I5.3C symmetry conclusion"))),
        "auto_implication": "degraded_or_inconclusive_visual_stress_evidence_cannot_become_global_auto_preference",
    }


def validate_i5_4c_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.4C", status="forward_motion_kitti_robustness_qualified_and_closed", label="I5.4C closure")
    _require(root, "quality_status", "rejected", label="I5.4C closure")
    q = _mapping(root.get("qualification"), label="I5.4C qualification")
    _require(q, "quality_status", "rejected", label="I5.4C qualification")
    _require(q, "mapping_produced_sparse_model", False, label="I5.4C qualification")
    _require(q, "registered_images", 0, label="I5.4C qualification")
    return {
        "quality_status": "rejected",
        "operating_envelope": str(q.get("operating_envelope", "")),
        "primary_bottleneck": str(q.get("primary_bottleneck", "")),
        "hard_auto_abstention_required": True,
        "auto_must_not_mask_sparse_failure_with_dense_provider": True,
    }


def validate_i5_5c_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.5C", status="dynamic_scene_handling_robustness_qualified_and_closed", label="I5.5C closure")
    _require(root, "quality_status", "pass", label="I5.5C closure")
    q = _mapping(root.get("qualification"), label="I5.5C qualification")
    _require(q, "quality_status", "pass", label="I5.5C qualification")
    _require(q, "natural_dynamic_scene_ground_truth", False, label="I5.5C qualification")
    _require(q, "universal_dynamic_scene_robustness_claimed", False, label="I5.5C qualification")
    return {
        "quality_status": "pass",
        "operating_envelope": str(q.get("operating_envelope", "")),
        "natural_dynamic_scene_ground_truth": False,
        "auto_implication": "bounded_proxy_pass_does_not_authorize_universal_dynamic_scene_classification",
    }


def validate_i5_6c_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.6C", status="mvs_robustness_qualified_and_closed", label="I5.6C closure")
    _require(root, "quality_status", "pass", label="I5.6C closure")
    q = _mapping(root.get("qualification"), label="I5.6C qualification")
    _require(q, "current_production_colmap_mvs_path_qualified_within_admitted_envelope", True, label="I5.6C qualification")
    _require(q, "provider_ranking_claimed", False, label="I5.6C qualification")
    return {
        "provider": "colmap_mvs",
        "quality_status": "pass",
        "bounded_primary_eligibility": True,
        "universal_robustness": False,
        "runtime_performance_envelope_complete": bool(_mapping(root.get("limitations"), label="I5.6C limitations").get("runtime_performance_envelope_complete", False)),
        "resource_envelope_complete": bool(_mapping(root.get("limitations"), label="I5.6C limitations").get("resource_envelope_complete", False)),
    }


def validate_i5_7c_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.7C", status="rolling_depth_robustness_qualified_and_closed", label="I5.7C closure")
    _require(root, "quality_status", "pass", label="I5.7C closure")
    q = _mapping(root.get("qualification"), label="I5.7C qualification")
    _require(q, "current_frozen_foundation_rolling_depth_path_qualified_within_admitted_envelope", True, label="I5.7C qualification")
    _require(q, "historical_g3_motion_rejected_preserved", True, label="I5.7C qualification")
    _require(q, "historical_g3_static_sweep_degraded_preserved", True, label="I5.7C qualification")
    return {
        "provider": "rolling_depth",
        "current_castle_quality_status": "pass",
        "historical_static_sweep": "degraded",
        "historical_motion": "rejected",
        "historical_low_texture": "pass",
        "conflicting_or_mixed_scenario_evidence_preserved": True,
        "universal_robustness": False,
    }


def validate_i5_8c_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.8C", status="video_depth_anything_robustness_qualified_and_closed", label="I5.8C closure")
    _require(root, "quality_status", "degraded", label="I5.8C closure")
    q = _mapping(root.get("qualification"), label="I5.8C qualification")
    _require(q, "current_frozen_foundation_vda_path_execution_qualified_within_admitted_envelope", True, label="I5.8C qualification")
    _require(q, "current_frozen_foundation_vda_quality_clean_pass_claimed", False, label="I5.8C qualification")
    _require(q, "fusion_capacity_limitation_prevents_clean_quality_pass_claim", True, label="I5.8C qualification")
    return {
        "provider": "video_depth_anything",
        "quality_status": "degraded",
        "execution_qualified": True,
        "clean_auto_primary_eligible": False,
        "active_limitation": "fusion_capacity",
        "universal_robustness": False,
    }


def validate_i5_9c_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.9C", status="remaining_depth_provider_robustness_qualified_and_closed", label="I5.9C closure")
    _require(root, "quality_status", "mixed_terminal_quality", label="I5.9C closure")
    q = _mapping(root.get("qualification"), label="I5.9C qualification")
    _require(q, "combined_status_counts", {"pass": 35, "degraded": 1, "rejected": 0}, label="I5.9C qualification")
    providers = _mapping(root.get("provider_qualification"), label="I5.9C provider qualification")
    expected = {"metric3d_v2", "unidepth_v2", "moge_2", "depth_pro", "depth_anything_v2", "marigold"}
    if set(providers) != expected:
        raise ReconstructionValidationError("I5.13A I5.9C provider coverage drifted.")
    historical = _mapping(root.get("historical_g4_anchors"), label="I5.9C historical G4 anchors")
    limitations = _mapping(root.get("observed_limitations"), label="I5.9C limitations")
    _require(limitations, "historical_depth_anything_v2_low_texture_rejection_remains_active", True, label="I5.9C limitations")
    _require(limitations, "historical_marigold_low_texture_rejection_remains_active", True, label="I5.9C limitations")
    return {
        "current_castle_status_counts": deepcopy(dict(q["combined_status_counts"])),
        "current_castle_degraded_provider": str(q.get("current_castle_degraded_provider", "")),
        "current_castle_degraded_scenario": str(q.get("current_castle_degraded_scenario", "")),
        "historical_g4_anchors": deepcopy(dict(historical)),
        "historical_low_texture_rejections_remain_active": ["depth_anything_v2", "marigold"],
        "provider_qualification": deepcopy(dict(providers)),
        "auto_implication": "later_distinct_scenario_passes_do_not_overwrite_frozen_exact_scenario_rejections",
    }


def validate_i5_10d_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.10D", status="mp_sfm_evidence_qualified_and_closed", label="I5.10D closure")
    q = _mapping(root.get("qualification"), label="I5.10D qualification")
    _require(q, "conclusion", "matching_policy_sensitivity_with_partial_sparse_recovery_but_no_production_quality_recovery", label="I5.10D qualification")
    candidate = _mapping(root.get("candidate_quality"), label="I5.10D candidate quality")
    _require(candidate, "production_quality_status", "rejected", label="I5.10D candidate quality")
    return {
        "matching_policy_sensitivity_observed": True,
        "production_quality_recovery_observed": False,
        "candidate_quality_status": "rejected",
        "auto_matching_policy_change_authorized": False,
    }


def validate_i5_11d_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.11D", status="bounded_failure_recovery_evidence_qualified_and_closed", label="I5.11D closure")
    _require(root, "quality_status", "bounded_fault_recovery_pass", label="I5.11D closure")
    q = _mapping(root.get("qualification"), label="I5.11D qualification")
    for key in (
        "bounded_fault_recovery_pass",
        "all_expected_fault_signatures_observed",
        "all_process_groups_clean",
        "all_recoveries_completed",
        "all_repository_integrity_preserved",
    ):
        _require(q, key, True, label="I5.11D qualification")
    _require(q, "runtime_integrity_failure", False, label="I5.11D qualification")
    return {
        "bounded_fault_recovery_pass": True,
        "runtime_integrity_failure": False,
        "system_integrity_gate_supported": True,
        "universal_failure_recovery_claim": False,
    }


def validate_i5_12a_for_i5_13a(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(payload, phase="I5.12A", status="hybrid_evidence_justification_review_complete", label="I5.12A closure")
    _require(root, "quality_status", "hybrid_not_justified", label="I5.12A closure")
    decision = _mapping(root.get("decision"), label="I5.12A decision")
    _require(decision, "i5_12_complete", True, label="I5.12A decision")
    _require(decision, "i5_13_authorized", True, label="I5.12A decision")
    _require(decision, "i5_13_started", False, label="I5.12A decision")
    _require(decision, "hybrid_authorized", False, label="I5.12A decision")
    _require(decision, "hybrid_execution_authorized", False, label="I5.12A decision")
    return {
        "quality_status": "hybrid_not_justified",
        "hybrid_family_auto_eligible": False,
        "hybrid_execution_authorized": False,
        "i5_13_authorized": True,
    }


def validate_i5_13a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.13A receipt")
    _require(root, "schema_version", 1, label="I5.13A receipt")
    _require(root, "phase", I513A_RECEIPT_PHASE, label="I5.13A receipt")
    _require(root, "status", "pass", label="I5.13A receipt")
    _require(root, "pass", True, label="I5.13A receipt")
    scope = _mapping(root.get("scope"), label="I5.13A receipt scope")
    for key in (
        "focused_i5_13a_tests",
        "dense_architecture_regression",
        "cross_provider_closure_regression",
        "i5_10_to_i5_12_regression",
        "import_probe",
        "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.13A receipt scope")
    for key in (
        "provider_execution",
        "sfm_execution",
        "mvs_execution",
        "auto_policy_engine_implemented",
        "auto_provider_selection_performed",
        "production_auto_integration",
        "production_default_modified",
        "hybrid_execution",
    ):
        _require(scope, key, False, label="I5.13A receipt scope")
    counts = _mapping(root.get("counts"), label="I5.13A receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.13A receipt test counts are invalid.")
    if focused < 20 or predecessor < 20 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.13A receipt coverage is below the admission floor.")
    rcs = _mapping(root.get("return_codes"), label="I5.13A receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.13A receipt {key} return code is not zero.")
    return deepcopy(dict(root))


def _architecture_contract() -> dict[str, Any]:
    if DenseProviderSelectionMode.AUTO.value != I43A_AUTO_POLICY_ID:
        raise ReconstructionValidationError("I5.13A canonical Auto selection-mode contract drifted.")
    if I43A_AUTO_POLICY_ID in I513A_PROVIDER_IMPLEMENTATIONS:
        raise ReconstructionValidationError("I5.13A Auto must never be an executable provider implementation.")
    return {
        "auto_policy_id": I43A_AUTO_POLICY_ID,
        "selection_mode": DenseProviderSelectionMode.AUTO.value,
        "auto_is_selection_policy_only": True,
        "auto_is_executable_provider": False,
        "provider_families": list(I513A_PROVIDER_FAMILIES),
        "provider_implementations": list(I513A_PROVIDER_IMPLEMENTATIONS),
        "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
        "engineer_override_remains_separate_from_automatic_selection": True,
        "fallback_must_be_explicit": True,
        "unselected_auto_decision_with_reason_codes_is_valid": True,
    }


def build_i5_13a_auto_policy_admission(
    *,
    i4_3g5b_closure: Mapping[str, Any],
    i4_3g5b_binding: Mapping[str, Any],
    i5_1c_closure: Mapping[str, Any],
    i5_1c_binding: Mapping[str, Any],
    i5_2c_closure: Mapping[str, Any],
    i5_2c_binding: Mapping[str, Any],
    i5_3c_closure: Mapping[str, Any],
    i5_3c_binding: Mapping[str, Any],
    i5_4c_closure: Mapping[str, Any],
    i5_4c_binding: Mapping[str, Any],
    i5_5c_closure: Mapping[str, Any],
    i5_5c_binding: Mapping[str, Any],
    i5_6c_closure: Mapping[str, Any],
    i5_6c_binding: Mapping[str, Any],
    i5_7c_closure: Mapping[str, Any],
    i5_7c_binding: Mapping[str, Any],
    i5_8c_closure: Mapping[str, Any],
    i5_8c_binding: Mapping[str, Any],
    i5_9c_closure: Mapping[str, Any],
    i5_9c_binding: Mapping[str, Any],
    i5_10d_closure: Mapping[str, Any],
    i5_10d_binding: Mapping[str, Any],
    i5_11d_closure: Mapping[str, Any],
    i5_11d_binding: Mapping[str, Any],
    i5_12a_closure: Mapping[str, Any],
    i5_12a_binding: Mapping[str, Any],
    post_i5_12a_checkpoint_binding: Mapping[str, Any],
    admission_receipt: Mapping[str, Any],
    admission_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    architecture = _architecture_contract()
    g5 = validate_i4_3g5b_closure(i4_3g5b_closure)
    s1 = validate_i5_1c_for_i5_13a(i5_1c_closure)
    s2 = validate_i5_2c_for_i5_13a(i5_2c_closure)
    s3 = validate_i5_3c_for_i5_13a(i5_3c_closure)
    s4 = validate_i5_4c_for_i5_13a(i5_4c_closure)
    s5 = validate_i5_5c_for_i5_13a(i5_5c_closure)
    p6 = validate_i5_6c_for_i5_13a(i5_6c_closure)
    p7 = validate_i5_7c_for_i5_13a(i5_7c_closure)
    p8 = validate_i5_8c_for_i5_13a(i5_8c_closure)
    p9 = validate_i5_9c_for_i5_13a(i5_9c_closure)
    m10 = validate_i5_10d_for_i5_13a(i5_10d_closure)
    r11 = validate_i5_11d_for_i5_13a(i5_11d_closure)
    h12 = validate_i5_12a_for_i5_13a(i5_12a_closure)
    receipt = validate_i5_13a_receipt(admission_receipt)

    bindings = {
        "i4_3g5b_cross_provider_closure": _binding(i4_3g5b_binding, label="I4.3G5B closure"),
        "i5_1c_difficult_handheld": _binding(i5_1c_binding, label="I5.1C closure"),
        "i5_2c_low_overlap_weak_parallax": _binding(i5_2c_binding, label="I5.2C closure"),
        "i5_3c_visual_stress": _binding(i5_3c_binding, label="I5.3C closure"),
        "i5_4c_forward_motion": _binding(i5_4c_binding, label="I5.4C closure"),
        "i5_5c_dynamic_scene": _binding(i5_5c_binding, label="I5.5C closure"),
        "i5_6c_mvs": _binding(i5_6c_binding, label="I5.6C closure"),
        "i5_7c_rolling_depth": _binding(i5_7c_binding, label="I5.7C closure"),
        "i5_8c_video_depth_anything": _binding(i5_8c_binding, label="I5.8C closure"),
        "i5_9c_single_image_depth": _binding(i5_9c_binding, label="I5.9C closure"),
        "i5_10d_mp_sfm": _binding(i5_10d_binding, label="I5.10D closure"),
        "i5_11d_failure_recovery": _binding(i5_11d_binding, label="I5.11D closure"),
        "i5_12a_hybrid_review": _binding(i5_12a_binding, label="I5.12A closure"),
        "post_i5_12a_source_checkpoint": _binding(post_i5_12a_checkpoint_binding, label="post-I5.12A source checkpoint"),
        "i5_13a_admission_receipt": _binding(admission_receipt_binding, label="I5.13A receipt"),
    }

    terminal_events = list(g5["terminal_quality_events"])
    if len(terminal_events) != 5:
        raise ReconstructionValidationError("I5.13A requires the five frozen I4.3G terminal quality events.")

    return {
        "schema_version": I513A_SCHEMA_VERSION,
        "phase": I513A_PHASE,
        "status": "auto_policy_evidence_synthesis_and_design_admitted",
        "pass": True,
        "quality_status": "auto_policy_design_admitted",
        "policy_profile": I513A_POLICY_PROFILE,
        "evidence_bindings": bindings,
        "architecture_contract": architecture,
        "evidence_coverage": {
            "cross_provider_family_count": int(g5["coverage"]["qualification_family_count"]),
            "cross_provider_scenario_count": int(g5["coverage"]["scenario_count"]),
            "provider_implementation_count": int(g5["coverage"]["provider_implementation_count"]),
            "i5_sparse_sfm_robustness_phases": ["I5.1C", "I5.2C", "I5.3C", "I5.4C", "I5.5C"],
            "i5_dense_provider_robustness_phases": ["I5.6C", "I5.7C", "I5.8C", "I5.9C"],
            "matching_policy_phase": "I5.10D",
            "failure_recovery_phase": "I5.11D",
            "hybrid_review_phase": "I5.12A",
            "all_required_evidence_classes_bound": True,
            "global_provider_ranking_created": False,
        },
        "evidence_synthesis": {
            "sparse_sfm_operating_envelope": {
                "difficult_handheld": s1,
                "low_overlap_weak_parallax": s2,
                "visual_stress": s3,
                "forward_motion": s4,
                "dynamic_scene_proxy": s5,
            },
            "provider_operating_envelope": {
                "cross_provider_terminal_quality_events": deepcopy(terminal_events),
                "mvs": p6,
                "rolling_depth": p7,
                "video_depth_anything": p8,
                "single_image_depth": p9,
                "cross_family_direct_ranking_valid": False,
                "terminal_event_counts_are_not_scores": True,
            },
            "matching_policy_context": m10,
            "failure_recovery_context": r11,
            "hybrid_context": h12,
        },
        "auto_policy_v1_design": {
            "policy_id": I43A_AUTO_POLICY_ID,
            "profile": I513A_POLICY_PROFILE,
            "decision_classes": list(I513A_DECISION_CLASSES),
            "primary_principle": "select_only_on_exact_unique_evidence_support_else_abstain",
            "hard_gates": {
                "authoritative_pycolmap_sfm_required": True,
                "authoritative_sfm_must_be_production_usable": True,
                "runtime_integrity_must_be_clean": True,
                "provider_must_be_available": True,
                "evidence_context_must_be_recognized": True,
                "hybrid_family_forbidden": True,
            },
            "eligibility_rules": {
                "automatic_primary_requires_frozen_pass_evidence": True,
                "rejected_evidence_is_ineligible": True,
                "degraded_evidence_is_not_automatic_primary": True,
                "historical_exact_scenario_rejection_remains_active": True,
                "later_distinct_scenario_pass_does_not_overwrite_rejection": True,
                "conflicting_or_mixed_evidence_blocks_automatic_primary": True,
                "unknown_or_unqualified_scenario_abstains": True,
                "sparse_sfm_failure_cannot_be_masked_by_dense_provider_selection": True,
            },
            "ambiguity_rules": {
                "multiple_eligible_without_direct_scenario_preference_abstain": True,
                "zero_eligible_abstain": True,
                "no_global_scalar_score": True,
                "no_provider_win_count": True,
                "no_cross_family_rank": True,
                "no_runtime_performance_tiebreak": True,
                "no_gpu_model_performance_ranking": True,
            },
            "fallback_rules": {
                "fallback_requires_request_allow_fallback": True,
                "fallback_must_be_evidence_eligible_for_same_context": True,
                "fallback_must_be_explicit_in_decision": True,
                "retry_until_pass_forbidden": True,
                "implicit_global_fallback_chain_forbidden": True,
            },
            "engineer_override_rules": {
                "override_is_not_automatic_selection": True,
                "override_must_be_explicit_and_provenanced": True,
                "override_does_not_rewrite_evidence": True,
            },
            "forbidden_policy_actions": {
                "change_sfm_matching_strategy": True,
                "select_hybrid_family": True,
                "claim_metric_scale": True,
                "claim_measurement_authority": True,
                "claim_certified_metrology": True,
                "promote_provider": True,
                "change_production_default": True,
                "relax_quality_thresholds": True,
                "invent_runtime_or_resource_rank": True,
            },
            "explainability_contract": {
                "selected_or_abstained_provider_decision_required": True,
                "reason_codes_required": True,
                "considered_provider_ids_required": True,
                "excluded_provider_reasons_required": True,
                "matched_evidence_scenario_ids_required": True,
                "evidence_sha256_bindings_required": True,
                "global_confidence_scalar_forbidden": True,
            },
            "production_context_boundary": {
                "i5_13b_engine_is_offline_shadow_only": True,
                "i5_13b_may_consume_frozen_evidence_context_tokens": True,
                "production_scene_classifier_not_authorized_in_i5_13a": True,
                "live_provider_execution_not_authorized_in_i5_13a": True,
                "production_auto_integration_deferred_to_i5_13d": True,
            },
        },
        "offline_replay_contract": {
            "i5_13c_must_replay_frozen_evidence_only": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scenario_rematerialization": False,
            "expected_decisions_must_be_declared_before_replay": True,
            "policy_output_deterministic": True,
            "same_input_same_decision_required": True,
            "abstention_is_valid_terminal_decision": True,
            "no_result_driven_rule_tuning": True,
        },
        "scope": {
            "evidence_only_admission_and_design": True,
            "auto_policy_engine_implemented": False,
            "auto_provider_selection_performed": False,
            "production_auto_integration": False,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "matched_view_rendering": False,
            "manual_review_rerun": False,
            "hybrid_execution": False,
            "matching_policy_modified": False,
            "provider_promoted": False,
            "production_default_modified": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "claim_boundaries": {
            "universal_provider_winner": False,
            "global_provider_ranking": False,
            "global_scalar_score": False,
            "cross_family_rank": False,
            "runtime_performance_rank": False,
            "cross_gpu_equivalence": False,
            "universal_sfm_robustness": False,
            "universal_dense_provider_robustness": False,
            "production_auto_qualified": False,
        },
        "regression_summary": {
            "focused_i5_13a_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "test_counts_are_not_policy_scores": True,
        },
        "decision": {
            "i5_10_complete": True,
            "i5_11_complete": True,
            "i5_12_complete": True,
            "i5_13_started": True,
            "i5_13a_complete": True,
            "i5_13b_authorized": True,
            "i5_13b_started": False,
            "i5_13c_authorized": False,
            "i5_13c_started": False,
            "i5_13d_authorized": False,
            "i5_13d_started": False,
            "i5_14_started": False,
            "auto_policy_design_admitted": True,
            "auto_policy_engine_implemented": False,
            "production_auto_authorized": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "global_scalar_score_created": False,
            "hybrid_authorized": False,
            "matching_policy_change_authorized": False,
            "production_default_modified": False,
        },
        "next": "I5.13B deterministic evidence-bound Auto policy engine (offline/shadow only; no production integration).",
    }


__all__ = [
    "I513A_SCHEMA_VERSION",
    "I513A_PHASE",
    "I513A_RECEIPT_PHASE",
    "I513A_POLICY_PROFILE",
    "I513A_DECISION_CLASSES",
    "build_i5_13a_auto_policy_admission",
    "validate_i5_13a_receipt",
    "validate_i5_1c_for_i5_13a",
    "validate_i5_2c_for_i5_13a",
    "validate_i5_3c_for_i5_13a",
    "validate_i5_4c_for_i5_13a",
    "validate_i5_5c_for_i5_13a",
    "validate_i5_6c_for_i5_13a",
    "validate_i5_7c_for_i5_13a",
    "validate_i5_8c_for_i5_13a",
    "validate_i5_9c_for_i5_13a",
    "validate_i5_10d_for_i5_13a",
    "validate_i5_11d_for_i5_13a",
    "validate_i5_12a_for_i5_13a",
]
