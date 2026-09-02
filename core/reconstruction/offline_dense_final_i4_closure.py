"""I4.4D final I4 closure and offline-dense-foundation freeze.

I4.4D is the final evidence-only closure boundary for I4. It consumes the
completed I4.4C provenance/authority audit plus a narrow closure regression
receipt, independently re-verifies I4.4C's immediate content-addressed
predecessor bindings, re-audits the current repository authority contract, and
formally freezes the offline dense foundation before I5.

It deliberately does not execute providers, rerun PyCOLMAP/SfM, regenerate
matched views, repeat manual review, create provider rankings/scores, implement
Auto, authorize Hybrid, claim dense-provider metric scale, create measurement
or certified-metrology authority, generate a surface mesh, or start I5 work.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_provenance_authority_audit import (
    audit_current_offline_dense_authority_contracts,
)

I44D_SCHEMA_VERSION = 1
I44D_PHASE = "I4.4D"
I44D_RECEIPT_PHASE = "I4.4D_FINAL_CLOSURE_RECEIPT"
I44D_MIN_PREDECESSOR_TESTS = 90
I44D_MIN_PREDECESSOR_TEST_FILES = 9

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_TERMINAL_QUALITY_EVENTS = {
    ("temporal_depth", "g3_temporal_static_sweep", "rolling_depth", "degraded"),
    ("temporal_depth", "g3_temporal_motion", "rolling_depth", "rejected"),
    ("temporal_depth", "g3_temporal_motion", "video_depth_anything", "degraded"),
    ("single_image_depth", "g4_single_image_low_texture", "depth_anything_v2", "rejected"),
    ("single_image_depth", "g4_single_image_low_texture", "marigold", "rejected"),
}
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
_DECISION_FALSE = (
    "provider_selected",
    "provider_promoted",
    "provider_ranking_created",
    "universal_provider_winner_declared",
    "global_scalar_score_created",
    "auto_policy_implemented",
    "hybrid_authorized",
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.4D {label} must be an object.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"I4.4D {label} must be an array.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I4.4D {label}.{key} contract drift: expected={expected!r} actual={mapping.get(key)!r}."
        )


def _non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReconstructionValidationError(f"I4.4D {label} must be a non-negative integer.")
    return value


def _terminal_events(value: Any) -> set[tuple[str, str, str, str]]:
    events: set[tuple[str, str, str, str]] = set()
    for raw in _sequence(value, label="terminal_quality_events"):
        item = _mapping(raw, label="terminal quality event")
        events.add((
            str(item.get("family_id", "")),
            str(item.get("scenario_id", "")),
            str(item.get("provider_implementation", "")),
            str(item.get("production_status", "")),
        ))
    return events


@dataclass(frozen=True, slots=True)
class FinalI4EvidenceBinding:
    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        path = str(self.path).strip()
        sha = str(self.sha256).strip().lower()
        if not path or not _SHA256_RE.fullmatch(sha):
            raise ReconstructionValidationError("I4.4D evidence binding is malformed.")
        return {"path": path, "sha256": sha}


def _validate_authority(authority: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=label)
    _require(authority, "permanent_global_view_cap", None, label=label)
    drifted = [key for key in _AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.4D {label} forbidden authority drift: {drifted}")
    return deepcopy(dict(authority))


def validate_i4_4c_result_for_i4_4d(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate I4.4C as the sole immediate authorization boundary for D."""
    if payload.get("phase") != "I4.4C" or payload.get("status") != "provenance_authority_audit_complete" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.4D requires the passing frozen I4.4C provenance/authority audit.")

    scope = _mapping(payload.get("scope"), label="I4.4C scope")
    _require(scope, "evidence_and_repository_contract_audit", True, label="I4.4C scope")
    for key in (
        "provider_execution_started",
        "provider_runtime_probe_started",
        "sfm_rerun",
        "matched_view_rerun",
        "manual_review_rerun",
        "new_provider_comparison_started",
        "final_i4_closure_started",
    ):
        _require(scope, key, False, label="I4.4C scope")

    bindings = _mapping(payload.get("evidence_bindings"), label="I4.4C evidence_bindings")
    expected_bindings = {
        "i4_3h2_closure",
        "i4_4a_result",
        "i4_4b_result",
        "i4_4c_audit_receipt",
    }
    if set(bindings) != expected_bindings:
        raise ReconstructionValidationError("I4.4D I4.4C evidence-binding inventory drifted.")

    provenance = _mapping(payload.get("provenance_chain_audit"), label="I4.4C provenance_chain_audit")
    for key in (
        "h2_h1_chain_reverified",
        "h1_bound_artifacts_reverified",
        "g5b_predecessor_artifacts_reverified",
        "i4_4a_chain_reverified",
        "i4_4b_chain_reverified",
        "current_repository_authority_matches_frozen_h1",
        "formal_i4_3_evidence_graph_rehashed",
        "i4_4_predecessor_chain_rehashed",
        "family_provenance_conclusions_preserved",
        "provider_specific_semantics_preserved",
        "no_stronger_raw_artifact_claim_created",
    ):
        _require(provenance, key, True, label="I4.4C provenance_chain_audit")
    _require(
        provenance,
        "raw_nested_provider_artifact_tree_recursively_rehashed_by_i4_4c",
        False,
        label="I4.4C provenance_chain_audit",
    )

    authority_audit = _mapping(payload.get("authority_audit"), label="I4.4C authority_audit")
    _require(authority_audit, "camera_pose_authority_owner", "pycolmap_sfm", label="I4.4C authority_audit")
    _require(authority_audit, "default_reconstruction_scale_status", "arbitrary", label="I4.4C authority_audit")
    _require(authority_audit, "permanent_global_view_cap", None, label="I4.4C authority_audit")
    for key in (
        "authoritative_sfm_modified",
        "provider_native_pose_becomes_canonical",
        "dense_provider_metric_scale_claimed",
        "dense_provider_measurement_authoritative",
        "dense_provider_certified_metrology",
        "dense_provider_surface_mesh_authority",
        "provider_execution_started",
    ):
        _require(authority_audit, key, False, label="I4.4C authority_audit")
    architecture = _mapping(authority_audit.get("provider_architecture"), label="I4.4C provider_architecture")
    _require(architecture, "provider_family_count", 5, label="I4.4C provider_architecture")
    _require(architecture, "executable_provider_family_count", 4, label="I4.4C provider_architecture")
    _require(architecture, "provider_implementation_count", 10, label="I4.4C provider_architecture")
    _require(architecture, "permanent_global_view_cap", None, label="I4.4C provider_architecture")

    qualification = _mapping(payload.get("qualification_guard"), label="I4.4C qualification_guard")
    _require(qualification, "terminal_quality_event_count", 5, label="I4.4C qualification_guard")
    for key in (
        "production_native_evidence_primary",
        "matched_view_diagnostic_secondary",
        "quality_event_counts_are_not_provider_scores",
    ):
        _require(qualification, key, True, label="I4.4C qualification_guard")
    for key in ("provider_ranking_created", "universal_provider_winner_declared", "global_scalar_score_created"):
        _require(qualification, key, False, label="I4.4C qualification_guard")
    events = _terminal_events(qualification.get("terminal_quality_events"))
    if events != _EXPECTED_TERMINAL_QUALITY_EVENTS or len(events) != 5:
        raise ReconstructionValidationError("I4.4D I4.4C terminal-quality event set drifted.")

    decision = _mapping(payload.get("decision"), label="I4.4C decision")
    for key in ("i4_3_closed", "i4_4a_complete", "i4_4b_complete", "i4_4c_complete", "i4_4d_authorized"):
        _require(decision, key, True, label="I4.4C decision")
    for key in ("i4_4d_started",) + _DECISION_FALSE:
        _require(decision, key, False, label="I4.4C decision")

    authority = _validate_authority(_mapping(payload.get("authority"), label="I4.4C authority"), label="I4.4C authority")
    if payload.get("next") != "I4.4D final I4 closure + freeze.":
        raise ReconstructionValidationError("I4.4D I4.4C next-boundary contract drifted.")
    return {
        "evidence_bindings": deepcopy(dict(bindings)),
        "provenance": deepcopy(dict(provenance)),
        "authority_audit": deepcopy(dict(authority_audit)),
        "authority": authority,
        "terminal_quality_events": events,
        "regression_summary": deepcopy(dict(_mapping(payload.get("regression_summary"), label="I4.4C regression_summary"))),
    }


def validate_i4_4d_closure_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != I44D_RECEIPT_PHASE or payload.get("status") != "pass" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.4D requires a passing final-closure regression receipt.")
    scope = _mapping(payload.get("scope"), label="closure receipt scope")
    for key in (
        "focused_i4_4d_tests",
        "predecessor_i4_4_unit_regression",
        "predecessor_i4_4_integration_regression",
        "predecessor_i4_4_stress_regression",
        "import_probe",
        "runner_help_probe",
    ):
        _require(scope, key, True, label="closure receipt scope")
    for key in (
        "provider_execution_started",
        "provider_runtime_probe_started",
        "sfm_rerun",
        "matched_view_rerun",
        "manual_review_rerun",
        "new_provider_comparison_started",
        "i5_work_started",
    ):
        _require(scope, key, False, label="closure receipt scope")

    counts = _mapping(payload.get("counts"), label="closure receipt counts")
    focused = _non_negative_int(counts.get("focused_tests"), label="focused_tests")
    file_count = _non_negative_int(counts.get("predecessor_test_files"), label="predecessor_test_files")
    unit = _non_negative_int(counts.get("predecessor_unit_tests"), label="predecessor_unit_tests")
    integration = _non_negative_int(counts.get("predecessor_integration_tests"), label="predecessor_integration_tests")
    stress = _non_negative_int(counts.get("predecessor_stress_tests"), label="predecessor_stress_tests")
    total = _non_negative_int(counts.get("predecessor_total_tests"), label="predecessor_total_tests")
    combined = _non_negative_int(counts.get("combined_exercised_tests"), label="combined_exercised_tests")
    if focused < 20:
        raise ReconstructionValidationError("I4.4D focused-test count is unexpectedly small.")
    if file_count < I44D_MIN_PREDECESSOR_TEST_FILES or total < I44D_MIN_PREDECESSOR_TESTS:
        raise ReconstructionValidationError("I4.4D predecessor I4.4 regression is too narrow.")
    if total != unit + integration + stress or combined != focused + total:
        raise ReconstructionValidationError("I4.4D closure regression counts are inconsistent.")

    codes = _mapping(payload.get("return_codes"), label="closure receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        _require(codes, key, 0, label="closure receipt return_codes")
    return {"counts": deepcopy(dict(counts))}


def build_i4_4d_final_i4_closure(
    *,
    i4_4c_result: Mapping[str, Any],
    i4_4c_binding: FinalI4EvidenceBinding,
    closure_receipt: Mapping[str, Any],
    closure_receipt_binding: FinalI4EvidenceBinding,
    i4_4c_bound_artifacts_reverified: bool,
    current_repository_authority_matches_i4_4c: bool,
) -> dict[str, Any]:
    validated_c = validate_i4_4c_result_for_i4_4d(i4_4c_result)
    validated_receipt = validate_i4_4d_closure_receipt(closure_receipt)
    if i4_4c_bound_artifacts_reverified is not True:
        raise ReconstructionValidationError("I4.4D requires independent re-verification of all I4.4C immediate bindings.")
    if current_repository_authority_matches_i4_4c is not True:
        raise ReconstructionValidationError("I4.4D current repository authority no longer matches frozen I4.4C.")

    current_authority = audit_current_offline_dense_authority_contracts()
    if current_authority != validated_c["authority_audit"]:
        raise ReconstructionValidationError("I4.4D current offline-dense authority contract drifted from I4.4C.")

    authority = deepcopy(validated_c["authority"])
    events = set(validated_c["terminal_quality_events"])
    return {
        "schema_version": I44D_SCHEMA_VERSION,
        "phase": I44D_PHASE,
        "status": "formally_closed",
        "pass": True,
        "scope": {
            "evidence_only": True,
            "formal_i4_closure_performed": True,
            "offline_dense_foundation_freeze_performed": True,
            "provider_execution_started": False,
            "provider_runtime_probe_started": False,
            "sfm_rerun": False,
            "matched_view_rerun": False,
            "manual_review_rerun": False,
            "new_provider_comparison_started": False,
            "i5_work_started": False,
        },
        "evidence_binding": {
            "i4_4c_provenance_authority_audit": i4_4c_binding.as_dict(),
            "i4_4d_closure_receipt": closure_receipt_binding.as_dict(),
        },
        "i4_4c_bound_evidence": deepcopy(validated_c["evidence_bindings"]),
        "closure_chain": {
            "i4_4c_sha256_verified": True,
            "i4_4c_complete": True,
            "i4_4d_authorized_by_i4_4c": True,
            "i4_4c_bound_artifacts_sha256_reverified": True,
            "formal_i4_3_evidence_graph_preserved": True,
            "i4_4a_complete": True,
            "i4_4b_complete": True,
            "i4_4c_complete_before_final_closure": True,
            "current_repository_authority_matches_i4_4c": True,
            "final_i4_closure_receipt_preserved": True,
        },
        "coverage": {
            "provider_family_count": 5,
            "executable_provider_family_count": 4,
            "provider_implementation_count": 10,
            "qualification_family_count": 3,
            "qualification_scenario_count": 11,
            "terminal_quality_event_count": 5,
            "i4_4_boundary_count": 4,
            "i4_4_completed_boundary_count": 4,
        },
        "qualification_closure": {
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "scenario_specific_operating_envelopes_preserved": True,
            "terminal_quality_events": [
                {
                    "family_id": family,
                    "scenario_id": scenario_id,
                    "provider_implementation": provider,
                    "production_status": status,
                }
                for family, scenario_id, provider, status in sorted(events)
            ],
            "quality_event_counts_are_not_provider_scores": True,
            "direct_cross_family_provider_ranking_valid": False,
            "universal_provider_conclusion_drawn": False,
        },
        "provenance_closure": {
            "i4_4c_audit_sha256_bound": True,
            "i4_4c_immediate_evidence_rehashed": True,
            "formal_i4_3_provenance_conclusions_preserved": True,
            "family_provenance_conclusions_preserved": True,
            "provider_specific_semantics_preserved": True,
            "raw_nested_provider_artifact_tree_recursively_rehashed_by_i4_4d": False,
            "no_stronger_raw_artifact_claim_created": True,
        },
        "regression_summary": {
            "focused_i4_4d_tests": validated_receipt["counts"]["focused_tests"],
            "predecessor_i4_4_tests": validated_receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": validated_receipt["counts"]["combined_exercised_tests"],
            "i4_4c_affected_tests_preserved": validated_c["regression_summary"].get("affected_total_tests"),
            "all_required_suites_passed": True,
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority_audit": current_authority,
        "authority": authority,
        "decision": {
            "i4_3_closed": True,
            "i4_4a_complete": True,
            "i4_4b_complete": True,
            "i4_4c_complete": True,
            "i4_4d_final_closure_complete": True,
            "i4_4_offline_dense_foundation_frozen": True,
            "i4_closed": True,
            "i5_authorized": True,
            "i5_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "handoff_to_i5": {
            "i5_authorized": True,
            "i5_work_started": False,
            "first_boundary": "I5.1 difficult handheld captures",
            "evidence_driven_auto_policy_deferred_to_i5_13": True,
            "hybrid_experiment_deferred_to_i5_12_if_justified": True,
            "offline_dense_foundation_authority_locked": True,
            "dense_provider_metric_scale_not_established_by_i4": True,
        },
        "next": "I5.1 difficult handheld captures.",
    }


__all__ = [
    "I44D_SCHEMA_VERSION",
    "I44D_PHASE",
    "I44D_RECEIPT_PHASE",
    "FinalI4EvidenceBinding",
    "validate_i4_4c_result_for_i4_4d",
    "validate_i4_4d_closure_receipt",
    "build_i4_4d_final_i4_closure",
]
