"""I4.3H2 formal Production Dense-Provider Architecture closure.

H2 is deliberately evidence-only. It consumes the frozen H1 qualification,
requires independent hash re-verification of H1 and the evidence chain H1
bound, confirms that the current in-repository provider architecture still
matches the H1 audit, and only then formally closes I4.3 and authorizes I4.4.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError

I43H2_SCHEMA_VERSION = 1
I43H2_PHASE = "I4.3H2"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_EXPECTED_IMPLEMENTATIONS = (
    "colmap_mvs",
    "da3",
    "rolling_depth",
    "video_depth_anything",
    "metric3d_v2",
    "unidepth_v2",
    "moge_2",
    "depth_pro",
    "depth_anything_v2",
    "marigold",
)

_EXPECTED_TERMINAL_QUALITY_EVENTS = {
    ("temporal_depth", "g3_temporal_static_sweep", "rolling_depth", "degraded"),
    ("temporal_depth", "g3_temporal_motion", "rolling_depth", "rejected"),
    ("temporal_depth", "g3_temporal_motion", "video_depth_anything", "degraded"),
    ("single_image_depth", "g4_single_image_low_texture", "depth_anything_v2", "rejected"),
    ("single_image_depth", "g4_single_image_low_texture", "marigold", "rejected"),
}

_DECISION_FALSE = (
    "provider_selected",
    "provider_promoted",
    "provider_ranking_created",
    "universal_provider_winner_declared",
    "global_scalar_score_created",
    "auto_policy_implemented",
    "hybrid_authorized",
)

_AUTHORITY_FALSE = (
    "authoritative_sfm_modified",
    "provider_selected",
    "provider_promoted",
    "auto_policy_implemented",
    "hybrid_authorized",
    "metric_scale_claimed",
    "measurement_authoritative",
    "certified_metrology",
    "surface_mesh_generated",
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3H2 {label} must be an object.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"I4.3H2 {label} must be an array.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I4.3H2 {label}.{key} contract drift: expected={expected!r} actual={mapping.get(key)!r}."
        )


def _validate_sha_binding(value: Any, *, label: str) -> dict[str, str]:
    binding = _mapping(value, label=label)
    path = str(binding.get("path", "")).strip()
    sha = str(binding.get("sha256", "")).strip().lower()
    if not path:
        raise ReconstructionValidationError(f"I4.3H2 {label} path is empty.")
    if not _SHA256_RE.fullmatch(sha):
        raise ReconstructionValidationError(f"I4.3H2 {label} SHA256 is malformed.")
    return {"path": path, "sha256": sha}


def _terminal_events(value: Any) -> tuple[tuple[str, str, str, str], ...]:
    events: list[tuple[str, str, str, str]] = []
    for raw in _sequence(value, label="terminal_quality_events"):
        item = _mapping(raw, label="terminal quality event")
        events.append(
            (
                str(item.get("family_id", "")),
                str(item.get("scenario_id", "")),
                str(item.get("provider_implementation", "")),
                str(item.get("production_status", "")),
            )
        )
    return tuple(events)


@dataclass(frozen=True, slots=True)
class H2EvidenceBinding:
    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        path = str(self.path).strip()
        sha = str(self.sha256).strip().lower()
        if not path or not _SHA256_RE.fullmatch(sha):
            raise ReconstructionValidationError("I4.3H2 H1 evidence binding is malformed.")
        return {"path": path, "sha256": sha}


def validate_i4_3h1_qualification_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the frozen H1 artifact proves the complete qualification gate."""

    if (
        payload.get("phase") != "I4.3H1"
        or payload.get("status") != "qualification_complete"
        or payload.get("pass") is not True
    ):
        raise ReconstructionValidationError("I4.3H2 requires a passing frozen I4.3H1 qualification audit.")

    scope = _mapping(payload.get("scope"), label="H1 scope")
    _require(scope, "evidence_and_repository_contract_audit", True, label="H1 scope")
    for key in (
        "provider_execution_started",
        "provider_runtime_probe_started",
        "sfm_rerun",
        "matched_view_rerun",
        "manual_review_rerun",
        "formal_i4_3_closure_performed",
        "i4_4_work_started",
    ):
        _require(scope, key, False, label="H1 scope")

    evidence_bindings = _mapping(payload.get("evidence_bindings"), label="H1 evidence_bindings")
    if set(evidence_bindings) != {"i4_3g5b_closure", "i4_3h1_regression_receipt"}:
        raise ReconstructionValidationError("I4.3H2 H1 immediate evidence-binding set drifted.")
    normalized_h1_bindings = {
        key: _validate_sha_binding(value, label=f"H1 evidence binding {key}")
        for key, value in evidence_bindings.items()
    }

    predecessor_bindings_raw = _mapping(
        payload.get("g5b_predecessor_bindings"), label="H1 G5B predecessor bindings"
    )
    if len(predecessor_bindings_raw) != 5:
        raise ReconstructionValidationError("I4.3H2 requires all five G5B predecessor bindings from H1.")
    normalized_predecessors = {
        key: _validate_sha_binding(value, label=f"G5B predecessor binding {key}")
        for key, value in predecessor_bindings_raw.items()
    }

    chain = _mapping(payload.get("closure_chain_audit"), label="H1 closure_chain_audit")
    for key in (
        "i4_3g_formally_closed",
        "g5a_deterministic_rebuild_verified_by_g5b",
        "g5b_sha256_verified",
        "g5b_predecessor_sha256_bindings_reverified",
    ):
        _require(chain, key, True, label="H1 closure_chain_audit")
    _require(chain, "qualification_family_count", 3, label="H1 closure_chain_audit")
    _require(chain, "scenario_count", 11, label="H1 closure_chain_audit")
    _require(chain, "provider_implementation_count", 10, label="H1 closure_chain_audit")

    architecture = _mapping(payload.get("architecture_audit"), label="H1 architecture_audit")
    _require(architecture, "provider_family_count", 5, label="H1 architecture_audit")
    _require(architecture, "executable_provider_family_count", 4, label="H1 architecture_audit")
    _require(architecture, "hybrid_family_reserved", True, label="H1 architecture_audit")
    _require(architecture, "hybrid_implementation_authorized", False, label="H1 architecture_audit")
    _require(architecture, "provider_implementation_count", 10, label="H1 architecture_audit")
    if tuple(architecture.get("integrated_provider_implementations", ())) != _EXPECTED_IMPLEMENTATIONS:
        raise ReconstructionValidationError("I4.3H2 H1 integrated-provider inventory drifted.")
    for key in (
        "all_provider_contract_modules_importable",
        "all_provider_implementation_contracts_valid",
        "all_provider_adaptive_planning_contracts_valid",
        "progressive_status_required",
    ):
        _require(architecture, key, True, label="H1 architecture_audit")
    _require(
        architecture,
        "adaptive_planning_coefficients_are_empirical_runtime_claims",
        False,
        label="H1 architecture_audit",
    )
    _require(architecture, "permanent_global_view_cap", None, label="H1 architecture_audit")
    inventory = _mapping(architecture.get("provider_inventory"), label="H1 provider_inventory")
    if set(inventory) != set(_EXPECTED_IMPLEMENTATIONS):
        raise ReconstructionValidationError("I4.3H2 H1 provider inventory coverage drifted.")
    for implementation, raw in inventory.items():
        item = _mapping(raw, label=f"H1 provider inventory {implementation}")
        _require(item, "module_contract_imported", True, label=f"H1 provider inventory {implementation}")
        _require(item, "backend_runtime_probe_started", False, label=f"H1 provider inventory {implementation}")
        _require(
            item,
            "scenario_qualification_evidence_closed",
            True,
            label=f"H1 provider inventory {implementation}",
        )
        _require(item, "globally_promoted", False, label=f"H1 provider inventory {implementation}")
        adaptive = _mapping(item.get("adaptive_planning"), label=f"H1 adaptive planning {implementation}")
        _require(
            adaptive,
            "relative_cost_is_empirical_runtime_claim",
            False,
            label=f"H1 adaptive planning {implementation}",
        )

    qualification = _mapping(payload.get("qualification_audit"), label="H1 qualification_audit")
    _require(qualification, "qualification_family_count", 3, label="H1 qualification_audit")
    _require(qualification, "scenario_count", 11, label="H1 qualification_audit")
    _require(qualification, "provider_implementation_count", 10, label="H1 qualification_audit")
    _require(qualification, "production_native_evidence_primary", True, label="H1 qualification_audit")
    _require(qualification, "matched_view_diagnostic_secondary", True, label="H1 qualification_audit")
    _require(
        qualification,
        "scenario_specific_operating_envelopes_preserved",
        True,
        label="H1 qualification_audit",
    )
    _require(qualification, "terminal_quality_event_count", 5, label="H1 qualification_audit")
    _require(qualification, "quality_event_counts_are_not_provider_scores", True, label="H1 qualification_audit")
    _require(qualification, "direct_cross_family_provider_ranking_valid", False, label="H1 qualification_audit")
    _require(
        qualification,
        "all_ten_integrated_providers_have_closed_family_qualification_evidence",
        True,
        label="H1 qualification_audit",
    )
    _require(qualification, "global_provider_promotion_performed", False, label="H1 qualification_audit")
    terminal_events = _terminal_events(qualification.get("terminal_quality_events"))
    if set(terminal_events) != _EXPECTED_TERMINAL_QUALITY_EVENTS or len(terminal_events) != 5:
        raise ReconstructionValidationError("I4.3H2 H1 terminal-quality event set drifted.")

    provenance = _mapping(payload.get("provenance_audit"), label="H1 provenance_audit")
    for key in (
        "frozen_g_closure_bound",
        "immediate_g_predecessor_artifacts_rehashed",
        "family_provenance_conclusions_preserved",
        "provider_contract_integration_stages_preserved",
    ):
        _require(provenance, key, True, label="H1 provenance_audit")
    _require(provenance, "provider_specific_semantics_collapsed", False, label="H1 provenance_audit")
    _require(provenance, "raw_nested_artifact_tree_rehashed_by_h1", False, label="H1 provenance_audit")

    regression = _mapping(payload.get("regression_audit"), label="H1 regression_audit")
    _require(regression, "receipt_bound", True, label="H1 regression_audit")
    _require(regression, "predecessor_unit_tests", 444, label="H1 regression_audit")
    _require(regression, "predecessor_integration_tests", 86, label="H1 regression_audit")
    _require(regression, "predecessor_stress_tests", 138, label="H1 regression_audit")
    _require(regression, "predecessor_total_tests", 668, label="H1 regression_audit")
    _require(regression, "all_required_suites_passed", True, label="H1 regression_audit")
    focused = regression.get("focused_h1_tests")
    if isinstance(focused, bool) or not isinstance(focused, int) or focused < 15:
        raise ReconstructionValidationError("I4.3H2 H1 focused-test count is unexpectedly small.")

    experimental = _mapping(payload.get("experimental_surfaces"), label="H1 experimental_surfaces")
    _require(experimental, "auto_selection_contract_scaffold_exists", True, label="H1 experimental_surfaces")
    _require(experimental, "auto_policy_implemented", False, label="H1 experimental_surfaces")
    _require(experimental, "hybrid_family_contract_reserved", True, label="H1 experimental_surfaces")
    _require(experimental, "hybrid_authorized", False, label="H1 experimental_surfaces")
    _require(
        experimental,
        "first_evidence_driven_auto_policy_deferred_to_i5",
        True,
        label="H1 experimental_surfaces",
    )

    decision = _mapping(payload.get("decision"), label="H1 decision")
    _require(decision, "i4_3h1_qualification_complete", True, label="H1 decision")
    _require(decision, "i4_3h2_authorized", True, label="H1 decision")
    _require(decision, "i4_3_closed", False, label="H1 decision")
    _require(decision, "i4_4_authorized", False, label="H1 decision")
    for key in _DECISION_FALSE:
        _require(decision, key, False, label="H1 decision")

    authority = _mapping(payload.get("authority"), label="H1 authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="H1 authority")
    _require(authority, "permanent_global_view_cap", None, label="H1 authority")
    for key in _AUTHORITY_FALSE:
        _require(authority, key, False, label="H1 authority")

    if payload.get("next") != "I4.3H2 formal I4.3 closure.":
        raise ReconstructionValidationError("I4.3H2 H1 next-boundary contract drifted.")

    return {
        "evidence_bindings": normalized_h1_bindings,
        "g5b_predecessor_bindings": normalized_predecessors,
        "architecture_audit": deepcopy(dict(architecture)),
        "terminal_quality_events": deepcopy(list(qualification["terminal_quality_events"])),
        "focused_h1_tests": focused,
    }


def build_i4_3h2_final_closure(
    *,
    i4_3h1_audit: Mapping[str, Any],
    evidence_binding: H2EvidenceBinding,
    h1_bound_artifacts_reverified: bool,
    g5b_predecessor_artifacts_reverified: bool,
    current_architecture_matches_h1: bool,
) -> dict[str, Any]:
    """Build the formal H2 closure only after every independent gate passes."""

    validated = validate_i4_3h1_qualification_audit(i4_3h1_audit)
    if h1_bound_artifacts_reverified is not True:
        raise ReconstructionValidationError("I4.3H2 requires independent re-verification of H1-bound artifacts.")
    if g5b_predecessor_artifacts_reverified is not True:
        raise ReconstructionValidationError("I4.3H2 requires independent re-verification of all G5B predecessors.")
    if current_architecture_matches_h1 is not True:
        raise ReconstructionValidationError("I4.3H2 current provider architecture no longer matches frozen H1.")

    h1_binding = evidence_binding.as_dict()
    return {
        "schema_version": I43H2_SCHEMA_VERSION,
        "phase": I43H2_PHASE,
        "status": "formally_closed",
        "pass": True,
        "scope": {
            "evidence_only": True,
            "provider_execution_started": False,
            "provider_runtime_probe_started": False,
            "sfm_rerun": False,
            "matched_view_rerun": False,
            "manual_review_rerun": False,
            "formal_i4_3_closure_performed": True,
            "i4_4_work_started": False,
        },
        "evidence_binding": {"i4_3h1_audit": h1_binding},
        "h1_bound_evidence": deepcopy(validated["evidence_bindings"]),
        "g5b_predecessor_bindings": deepcopy(validated["g5b_predecessor_bindings"]),
        "closure_chain": {
            "h1_sha256_verified": True,
            "h1_qualification_complete": True,
            "h2_authorized_by_h1": True,
            "h1_bound_g5b_sha256_reverified": True,
            "h1_bound_regression_receipt_sha256_reverified": True,
            "g5b_predecessor_sha256_bindings_reverified": True,
            "i4_3g_formally_closed": True,
            "current_repository_architecture_matches_h1": True,
            "full_i4_3_regression_receipt_preserved": True,
        },
        "coverage": {
            "provider_family_count": 5,
            "executable_provider_family_count": 4,
            "provider_implementation_count": 10,
            "qualification_family_count": 3,
            "qualification_scenario_count": 11,
            "terminal_quality_event_count": 5,
        },
        "qualification_closure": {
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "scenario_specific_operating_envelopes_preserved": True,
            "terminal_quality_events": deepcopy(validated["terminal_quality_events"]),
            "quality_event_counts_are_not_provider_scores": True,
            "direct_cross_family_provider_ranking_valid": False,
            "universal_provider_conclusion_drawn": False,
        },
        "provenance_closure": {
            "h1_audit_sha256_bound": True,
            "h1_immediate_evidence_rehashed": True,
            "g5b_predecessor_evidence_rehashed": True,
            "family_provenance_conclusions_preserved": True,
            "provider_specific_semantics_preserved": True,
            "raw_nested_artifact_tree_recursively_rehashed_by_h2": False,
        },
        "regression_summary": {
            "predecessor_total_tests": 668,
            "focused_h1_tests": validated["focused_h1_tests"],
            "all_required_suites_passed": True,
            "h2_does_not_reinterpret_test_counts_as_runtime_quality": True,
        },
        "decision": {
            "i4_3h1_qualification_complete": True,
            "i4_3h2_formal_closure_complete": True,
            "i4_3_production_dense_provider_architecture_frozen": True,
            "i4_3_closed": True,
            "i4_4_authorized": True,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "handoff_to_i4_4": {
            "i4_4_authorized": True,
            "i4_4_work_started": False,
            "first_boundary": "I4.4A focused regression",
            "evidence_driven_auto_policy_deferred_to_i5": True,
            "hybrid_policy_deferred_unless_later_evidence_authorizes_it": True,
        },
        "next": "I4.4A focused regression.",
    }


__all__ = [
    "I43H2_SCHEMA_VERSION",
    "I43H2_PHASE",
    "H2EvidenceBinding",
    "validate_i4_3h1_qualification_audit",
    "build_i4_3h2_final_closure",
]
