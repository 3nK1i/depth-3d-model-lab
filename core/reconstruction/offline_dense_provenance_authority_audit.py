"""I4.4C provenance / authority audit for the frozen offline dense foundation.

I4.4C is an additive, evidence-only audit. It verifies that the formal I4.3
closure and the completed I4.4A/I4.4B results still form one coherent
content-addressed provenance chain and that the current repository preserves
all frozen camera/scale/provider authority boundaries.

This module deliberately does not execute providers, rerun PyCOLMAP/SfM,
regenerate matched views, repeat manual review, implement Auto, authorize
Hybrid, claim dense-provider metric scale, create measurement/certified-
metrology authority, generate a surface mesh, or perform I4.4D closure.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from core.reconstruction.dense_provider_architecture import dense_provider_family_contracts
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i4_3_final_qualification import audit_i4_3_provider_architecture
from core.reconstruction.offline_dense_cross_provider_visual_verification import (
    validate_i4_4a_result_for_i4_4b,
)
from core.reconstruction.offline_dense_foundation_regression import validate_i4_3h2_for_i4_4a
from core.reconstruction.sfm_contracts import SfmScaleStatus

I44C_SCHEMA_VERSION = 1
I44C_PHASE = "I4.4C"
I44C_RECEIPT_PHASE = "I4.4C_PROVENANCE_AUTHORITY_AUDIT_RECEIPT"
I44C_MIN_AFFECTED_TESTS = 180
I44C_MIN_AFFECTED_TEST_FILES = 20

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
        raise ReconstructionValidationError(f"I4.4C {label} must be an object.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"I4.4C {label} must be an array.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I4.4C {label}.{key} contract drift: expected={expected!r} actual={mapping.get(key)!r}."
        )


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReconstructionValidationError(f"I4.4C {label} must be a non-negative integer.")
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
class ProvenanceAuthorityEvidenceBinding:
    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        path = str(self.path).strip()
        sha = str(self.sha256).strip().lower()
        if not path or not _SHA256_RE.fullmatch(sha):
            raise ReconstructionValidationError("I4.4C evidence binding is malformed.")
        return {"path": path, "sha256": sha}


def _validate_authority(authority: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=label)
    _require(authority, "permanent_global_view_cap", None, label=label)
    drifted = [key for key in _AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.4C {label} forbidden authority drift: {drifted}")
    return deepcopy(dict(authority))


def validate_i4_4b_result_for_i4_4c(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate I4.4B as the immediate predecessor authorization for I4.4C."""
    if payload.get("phase") != "I4.4B" or payload.get("status") != "visual_verification_complete" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.4C requires the passing frozen I4.4B visual-verification result.")

    scope = _mapping(payload.get("scope"), label="I4.4B scope")
    for key in ("evidence_only", "cross_provider_visual_verification_only"):
        _require(scope, key, True, label="I4.4B scope")
    for key in (
        "provider_execution_started",
        "provider_runtime_probe_started",
        "sfm_rerun",
        "matched_view_rerun",
        "manual_review_rerun",
        "provenance_authority_final_audit_started",
        "final_i4_closure_started",
    ):
        _require(scope, key, False, label="I4.4B scope")

    coverage = _mapping(payload.get("coverage"), label="I4.4B coverage")
    expected_coverage = {
        "qualification_family_count": 3,
        "scenario_count": 11,
        "provider_implementation_count": 10,
        "fresh_visual_review_scenario_count": 10,
        "historical_visual_review_scenario_count": 1,
        "review_dimension_total": 21,
        "all_visual_review_families_accounted": True,
    }
    for key, expected in expected_coverage.items():
        _require(coverage, key, expected, label="I4.4B coverage")

    visual = _mapping(payload.get("cross_family_visual_verification"), label="I4.4B cross_family_visual_verification")
    for key in (
        "production_native_evidence_primary",
        "matched_view_diagnostic_secondary",
        "all_visual_reviews_complete",
        "historical_evidence_preserved",
        "quality_event_counts_are_not_provider_scores",
    ):
        _require(visual, key, True, label="I4.4B cross_family_visual_verification")
    for key in (
        "manual_preferences_aggregated_into_score",
        "direct_cross_family_provider_ranking_valid",
        "universal_provider_winner_declared",
    ):
        _require(visual, key, False, label="I4.4B cross_family_visual_verification")
    events = _terminal_events(visual.get("terminal_quality_events"))
    if events != _EXPECTED_TERMINAL_QUALITY_EVENTS or len(events) != 5:
        raise ReconstructionValidationError("I4.4C I4.4B terminal quality-event set drifted.")

    decision = _mapping(payload.get("decision"), label="I4.4B decision")
    for key in ("i4_3_closed", "i4_4a_complete", "i4_4b_complete", "i4_4c_authorized"):
        _require(decision, key, True, label="I4.4B decision")
    for key in ("i4_4c_started", "i4_4d_started") + _DECISION_FALSE:
        _require(decision, key, False, label="I4.4B decision")

    authority = _validate_authority(_mapping(payload.get("authority"), label="I4.4B authority"), label="I4.4B authority")
    if payload.get("next") != "I4.4C provenance / authority audit.":
        raise ReconstructionValidationError("I4.4C I4.4B next-boundary contract drifted.")
    return {
        "coverage": deepcopy(dict(coverage)),
        "authority": authority,
        "terminal_quality_events": events,
    }


def validate_i4_4c_audit_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != I44C_RECEIPT_PHASE or payload.get("status") != "pass" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.4C requires a passing provenance/authority audit receipt.")
    scope = _mapping(payload.get("scope"), label="audit receipt scope")
    for key in (
        "focused_i4_4c_tests",
        "affected_provenance_unit_regression",
        "affected_provenance_integration_regression",
        "affected_provenance_stress_regression",
        "import_probe",
        "runner_help_probe",
    ):
        _require(scope, key, True, label="audit receipt scope")
    for key in (
        "provider_execution_started",
        "provider_runtime_probe_started",
        "sfm_rerun",
        "matched_view_rerun",
        "manual_review_rerun",
        "final_i4_closure_started",
    ):
        _require(scope, key, False, label="audit receipt scope")

    counts = _mapping(payload.get("counts"), label="audit receipt counts")
    focused = _positive_int(counts.get("focused_tests"), label="focused_tests")
    file_count = _positive_int(counts.get("affected_test_files"), label="affected_test_files")
    unit = _positive_int(counts.get("affected_unit_tests"), label="affected_unit_tests")
    integration = _positive_int(counts.get("affected_integration_tests"), label="affected_integration_tests")
    stress = _positive_int(counts.get("affected_stress_tests"), label="affected_stress_tests")
    total = _positive_int(counts.get("affected_total_tests"), label="affected_total_tests")
    if focused < 20:
        raise ReconstructionValidationError("I4.4C focused-test count is unexpectedly small.")
    if file_count < I44C_MIN_AFFECTED_TEST_FILES or total < I44C_MIN_AFFECTED_TESTS:
        raise ReconstructionValidationError("I4.4C affected provenance/authority regression is too narrow.")
    if total != unit + integration + stress:
        raise ReconstructionValidationError("I4.4C affected regression counts do not sum to affected_total_tests.")

    codes = _mapping(payload.get("return_codes"), label="audit receipt return_codes")
    for key in ("focused", "affected_unit", "affected_integration", "affected_stress", "import_probe", "runner_help_probe"):
        _require(codes, key, 0, label="audit receipt return_codes")
    return {"counts": deepcopy(dict(counts))}


def audit_current_offline_dense_authority_contracts() -> dict[str, Any]:
    """Re-audit repository-native provider/SfM authority without backend execution."""
    architecture = audit_i4_3_provider_architecture()
    if architecture["provider_family_count"] != 5:
        raise ReconstructionValidationError("I4.4C provider-family inventory drifted.")
    if architecture["executable_provider_family_count"] != 4:
        raise ReconstructionValidationError("I4.4C executable-provider-family inventory drifted.")
    if architecture["provider_implementation_count"] != 10:
        raise ReconstructionValidationError("I4.4C provider implementation inventory drifted.")
    if architecture["permanent_global_view_cap"] is not None:
        raise ReconstructionValidationError("I4.4C detected a permanent global dense-view cap.")

    families = dense_provider_family_contracts()
    for contract in families:
        if not contract.requires_authoritative_sfm or not contract.preserves_authoritative_sfm_frame:
            raise ReconstructionValidationError(f"I4.4C {contract.family.value} no longer preserves authoritative SfM.")
        if any((
            contract.establishes_metric_scale,
            contract.establishes_camera_pose_authority,
            contract.establishes_measurement_authority,
            contract.establishes_certified_metrology,
            contract.generates_surface_mesh,
        )):
            raise ReconstructionValidationError(f"I4.4C {contract.family.value} acquired forbidden authority.")

    return {
        "provider_architecture": deepcopy(architecture),
        "camera_pose_authority_owner": "pycolmap_sfm",
        "authoritative_sfm_modified": False,
        "provider_native_pose_becomes_canonical": False,
        "default_reconstruction_scale_status": SfmScaleStatus.ARBITRARY.value,
        "dense_provider_metric_scale_claimed": False,
        "dense_provider_measurement_authoritative": False,
        "dense_provider_certified_metrology": False,
        "dense_provider_surface_mesh_authority": False,
        "permanent_global_view_cap": None,
        "provider_execution_started": False,
    }


def build_i4_4c_provenance_authority_audit(
    *,
    i4_3h2_closure: Mapping[str, Any],
    i4_3h2_binding: ProvenanceAuthorityEvidenceBinding,
    i4_4a_result: Mapping[str, Any],
    i4_4a_binding: ProvenanceAuthorityEvidenceBinding,
    i4_4b_result: Mapping[str, Any],
    i4_4b_binding: ProvenanceAuthorityEvidenceBinding,
    audit_receipt: Mapping[str, Any],
    audit_receipt_binding: ProvenanceAuthorityEvidenceBinding,
    h2_h1_chain_reverified: bool,
    h1_bound_artifacts_reverified: bool,
    g5b_predecessor_artifacts_reverified: bool,
    i4_4a_chain_reverified: bool,
    i4_4b_chain_reverified: bool,
    current_repository_authority_matches_frozen_h1: bool,
) -> dict[str, Any]:
    validated_h2 = validate_i4_3h2_for_i4_4a(i4_3h2_closure)
    validated_i44a = validate_i4_4a_result_for_i4_4b(i4_4a_result)
    validated_i44b = validate_i4_4b_result_for_i4_4c(i4_4b_result)
    validated_receipt = validate_i4_4c_audit_receipt(audit_receipt)

    gates = {
        "h2_h1_chain_reverified": h2_h1_chain_reverified,
        "h1_bound_artifacts_reverified": h1_bound_artifacts_reverified,
        "g5b_predecessor_artifacts_reverified": g5b_predecessor_artifacts_reverified,
        "i4_4a_chain_reverified": i4_4a_chain_reverified,
        "i4_4b_chain_reverified": i4_4b_chain_reverified,
        "current_repository_authority_matches_frozen_h1": current_repository_authority_matches_frozen_h1,
    }
    failed = [key for key, value in gates.items() if value is not True]
    if failed:
        raise ReconstructionValidationError(f"I4.4C provenance/authority gates are incomplete: {failed}")

    current_authority = audit_current_offline_dense_authority_contracts()
    h2_authority = deepcopy(validated_h2["authority"])
    if h2_authority != validated_i44a["authority"] or h2_authority != validated_i44b["authority"]:
        raise ReconstructionValidationError("I4.4C H2/I4.4A/I4.4B authority records disagree.")
    if current_authority["camera_pose_authority_owner"] != h2_authority["camera_pose_authority_owner"]:
        raise ReconstructionValidationError("I4.4C current camera-pose authority no longer matches frozen H2.")
    if current_authority["permanent_global_view_cap"] != h2_authority["permanent_global_view_cap"]:
        raise ReconstructionValidationError("I4.4C current view-cap authority no longer matches frozen H2.")

    h2_events = set(_terminal_events(i4_3h2_closure["qualification_closure"]["terminal_quality_events"]))
    i44a_events = set(validated_i44a["terminal_quality_events"])
    i44b_events = set(validated_i44b["terminal_quality_events"])
    if h2_events != _EXPECTED_TERMINAL_QUALITY_EVENTS or h2_events != i44a_events or h2_events != i44b_events:
        raise ReconstructionValidationError("I4.4C terminal quality-event provenance drifted across H2/A/B.")

    return {
        "schema_version": I44C_SCHEMA_VERSION,
        "phase": I44C_PHASE,
        "status": "provenance_authority_audit_complete",
        "pass": True,
        "scope": {
            "evidence_and_repository_contract_audit": True,
            "provider_execution_started": False,
            "provider_runtime_probe_started": False,
            "sfm_rerun": False,
            "matched_view_rerun": False,
            "manual_review_rerun": False,
            "new_provider_comparison_started": False,
            "final_i4_closure_started": False,
        },
        "evidence_bindings": {
            "i4_3h2_closure": i4_3h2_binding.as_dict(),
            "i4_4a_result": i4_4a_binding.as_dict(),
            "i4_4b_result": i4_4b_binding.as_dict(),
            "i4_4c_audit_receipt": audit_receipt_binding.as_dict(),
        },
        "provenance_chain_audit": {
            **gates,
            "formal_i4_3_evidence_graph_rehashed": True,
            "i4_4_predecessor_chain_rehashed": True,
            "family_provenance_conclusions_preserved": True,
            "provider_specific_semantics_preserved": True,
            "raw_nested_provider_artifact_tree_recursively_rehashed_by_i4_4c": False,
            "no_stronger_raw_artifact_claim_created": True,
        },
        "authority_audit": current_authority,
        "qualification_guard": {
            "terminal_quality_event_count": 5,
            "terminal_quality_events": [
                {
                    "family_id": family,
                    "scenario_id": scenario_id,
                    "provider_implementation": provider,
                    "production_status": status,
                }
                for family, scenario_id, provider, status in sorted(h2_events)
            ],
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "quality_event_counts_are_not_provider_scores": True,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
        },
        "regression_summary": deepcopy(validated_receipt["counts"]),
        "authority": h2_authority,
        "decision": {
            "i4_3_closed": True,
            "i4_4a_complete": True,
            "i4_4b_complete": True,
            "i4_4c_complete": True,
            "i4_4d_authorized": True,
            "i4_4d_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "next": "I4.4D final I4 closure + freeze.",
    }


__all__ = [
    "I44C_SCHEMA_VERSION",
    "I44C_PHASE",
    "I44C_RECEIPT_PHASE",
    "ProvenanceAuthorityEvidenceBinding",
    "validate_i4_4b_result_for_i4_4c",
    "validate_i4_4c_audit_receipt",
    "audit_current_offline_dense_authority_contracts",
    "build_i4_4c_provenance_authority_audit",
]
