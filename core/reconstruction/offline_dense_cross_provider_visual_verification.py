"""I4.4B cross-provider visual verification for the frozen offline dense foundation.

This boundary is evidence-only.  It consumes the formally closed I4.3 visual
qualification closures together with the frozen I4.3H2 closure and the
completed I4.4A focused regression result.  It verifies that visual-review
coverage across the three executable provider families remains complete,
scenario-specific, and bounded by the same authority/provenance locks.

It deliberately does not rerun providers, rerun PyCOLMAP, regenerate matched
views, repeat manual review, create a cross-family score/ranking, select or
promote a provider, implement Auto, authorize Hybrid, claim metric scale,
create measurement or metrology authority, or perform I4.4C/I4.4D.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from core.reconstruction.cross_provider_qualification import qualification_scenario_matrix
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_foundation_regression import validate_i4_3h2_for_i4_4a
from core.reconstruction.single_image_provider_matched_view_review import (
    I43G4C_PROVIDER_IMPLEMENTATIONS,
    I43G4C_REVIEW_DIMENSIONS,
)
from core.reconstruction.temporal_provider_matched_view_review import (
    I43G3C_PROVIDER_IMPLEMENTATIONS,
    I43G3C_REVIEW_DIMENSIONS,
)

I44B_SCHEMA_VERSION = 1
I44B_PHASE = "I4.4B"
I44B_RECEIPT_PHASE = "I4.4B_VISUAL_VERIFICATION_RECEIPT"
I44B_MIN_AFFECTED_TESTS = 150
I44B_MIN_AFFECTED_TEST_FILES = 18

_ALLOWED_TERMINAL_STATUSES = ("pass", "degraded", "rejected")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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
        raise ReconstructionValidationError(f"I4.4B {label} must be an object.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"I4.4B {label} must be an array.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I4.4B {label}.{key} contract drift: expected={expected!r} actual={mapping.get(key)!r}."
        )


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReconstructionValidationError(f"I4.4B {label} must be a non-negative integer.")
    return value


def _scenario_ids(substep: str) -> tuple[str, ...]:
    return tuple(item.scenario_id for item in qualification_scenario_matrix() if item.substep == substep)


def _scenario_dimensions(scenario_id: str) -> tuple[str, ...]:
    matches = [item for item in qualification_scenario_matrix() if item.scenario_id == scenario_id]
    if len(matches) != 1:
        raise ReconstructionValidationError(f"I4.4B could not resolve scenario dimensions for {scenario_id}.")
    return tuple(matches[0].manual_review_dimensions)


def _provider_list(substep: str) -> tuple[str, ...]:
    seen: list[str] = []
    for item in qualification_scenario_matrix():
        if item.substep != substep:
            continue
        for provider in item.provider_implementations:
            value = provider.value
            if value not in seen:
                seen.append(value)
    return tuple(seen)


@dataclass(frozen=True, slots=True)
class VisualVerificationEvidenceBinding:
    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        path = str(self.path).strip()
        sha = str(self.sha256).strip().lower()
        if not path or not _SHA256_RE.fullmatch(sha):
            raise ReconstructionValidationError("I4.4B evidence binding is malformed.")
        return {"path": path, "sha256": sha}


def _validate_authority(authority: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=label)
    _require(authority, "permanent_global_view_cap", None, label=label)
    drifted = [key for key in _AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.4B {label} forbidden authority drift: {drifted}")
    return deepcopy(dict(authority))


def _terminal_events(value: Any) -> set[tuple[str, str, str, str]]:
    result: set[tuple[str, str, str, str]] = set()
    for raw in _sequence(value, label="terminal_quality_events"):
        item = _mapping(raw, label="terminal quality event")
        result.add((
            str(item.get("family_id", "")),
            str(item.get("scenario_id", "")),
            str(item.get("provider_implementation", "")),
            str(item.get("production_status", "")),
        ))
    return result


def validate_i4_4a_result_for_i4_4b(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != "I4.4A" or payload.get("status") != "focused_regression_passed" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.4B requires the passing frozen I4.4A focused regression result.")

    scope = _mapping(payload.get("scope"), label="I4.4A scope")
    for key in (
        "focused_regression_only",
        "provider_execution_started",
        "provider_runtime_probe_started",
        "sfm_rerun",
        "matched_view_rerun",
        "manual_visual_verification_started",
        "provenance_authority_final_audit_started",
        "final_i4_closure_started",
    ):
        expected = True if key == "focused_regression_only" else False
        _require(scope, key, expected, label="I4.4A scope")

    decision = _mapping(payload.get("decision"), label="I4.4A decision")
    for key in ("i4_3_closed", "i4_4a_complete", "i4_4b_authorized"):
        _require(decision, key, True, label="I4.4A decision")
    for key in ("i4_4b_started", "i4_4c_started", "i4_4d_started", "auto_policy_implemented", "hybrid_authorized"):
        _require(decision, key, False, label="I4.4A decision")

    authority = _validate_authority(_mapping(payload.get("authority"), label="I4.4A authority"), label="I4.4A authority")
    quality = _mapping(payload.get("qualification_guard"), label="I4.4A qualification_guard")
    _require(quality, "terminal_quality_event_count", 5, label="I4.4A qualification_guard")
    _require(quality, "quality_event_counts_are_not_provider_scores", True, label="I4.4A qualification_guard")
    for key in ("provider_ranking_created", "universal_provider_winner_declared"):
        _require(quality, key, False, label="I4.4A qualification_guard")
    events = _terminal_events(quality.get("terminal_quality_events"))
    if len(events) != 5:
        raise ReconstructionValidationError("I4.4B I4.4A terminal quality-event coverage drifted.")
    return {"authority": authority, "terminal_quality_events": events}


def validate_i4_4b_visual_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != I44B_RECEIPT_PHASE or payload.get("status") != "pass" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.4B requires a passing visual-verification receipt.")
    scope = _mapping(payload.get("scope"), label="receipt scope")
    for key in (
        "focused_i4_4b_tests",
        "affected_visual_unit_regression",
        "affected_visual_integration_regression",
        "affected_visual_stress_regression",
        "import_probe",
        "runner_help_probe",
    ):
        _require(scope, key, True, label="receipt scope")
    for key in ("provider_execution_started", "sfm_rerun", "visual_review_rerun", "manual_review_rerun"):
        _require(scope, key, False, label="receipt scope")

    counts = _mapping(payload.get("counts"), label="receipt counts")
    focused = _positive_int(counts.get("focused_tests"), label="focused_tests")
    file_count = _positive_int(counts.get("affected_test_files"), label="affected_test_files")
    unit = _positive_int(counts.get("affected_unit_tests"), label="affected_unit_tests")
    integration = _positive_int(counts.get("affected_integration_tests"), label="affected_integration_tests")
    stress = _positive_int(counts.get("affected_stress_tests"), label="affected_stress_tests")
    total = _positive_int(counts.get("affected_total_tests"), label="affected_total_tests")
    if focused < 20:
        raise ReconstructionValidationError("I4.4B focused-test count is unexpectedly small.")
    if file_count < I44B_MIN_AFFECTED_TEST_FILES or total < I44B_MIN_AFFECTED_TESTS:
        raise ReconstructionValidationError("I4.4B affected visual regression is too narrow.")
    if total != unit + integration + stress:
        raise ReconstructionValidationError("I4.4B affected regression counts do not sum to affected_total_tests.")

    codes = _mapping(payload.get("return_codes"), label="receipt return_codes")
    for key in ("focused", "affected_unit", "affected_integration", "affected_stress", "import_probe", "runner_help_probe"):
        _require(codes, key, 0, label="receipt return_codes")
    return {"counts": deepcopy(dict(counts))}


def validate_g2_visual_closure(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != "I4.3G2" or payload.get("status") != "formally_closed":
        raise ReconstructionValidationError("I4.4B requires the formally closed I4.3G2 visual family closure.")
    _require(payload, "scenario_count", 5, label="G2 closure")
    _require(payload, "all_required_g2_scenarios_accounted", True, label="G2 closure")
    _require(payload, "all_required_manual_reviews_complete", True, label="G2 closure")
    _require(payload, "i4_3g2_closed", True, label="G2 closure")
    _require(payload, "i4_3g3_authorized", True, label="G2 closure")
    if "fresh_scenario_count" in payload:
        _require(payload, "fresh_scenario_count", 4, label="G2 closure")
    if "historical_scenario_count" in payload:
        _require(payload, "historical_scenario_count", 1, label="G2 closure")
    if "g2c_manual_qualification_complete" in payload:
        _require(payload, "g2c_manual_qualification_complete", True, label="G2 closure")

    synthesis = _mapping(payload.get("synthesis"), label="G2 synthesis")
    _require(synthesis, "camera_pose_authority_owner", "pycolmap_sfm", label="G2 synthesis")
    for key in (
        "auto_provider_selection_implemented",
        "hybrid_provider_authorized",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "provider_promoted",
        "provider_selected",
        "surface_mesh_generated",
        "universal_provider_winner_declared",
    ):
        _require(synthesis, key, False, label="G2 synthesis")

    scenario_ids = _scenario_ids("I4.3G2")
    scenarios = _mapping(payload.get("scenario_synthesis"), label="G2 scenario_synthesis")
    if set(scenarios) != set(scenario_ids):
        raise ReconstructionValidationError("I4.4B G2 scenario coverage drifted.")

    historical = "g2_house_mvs_da3_historical"
    fresh = [sid for sid in scenario_ids if sid != historical]
    dims = _scenario_dimensions(fresh[0])
    for scenario_id in fresh:
        scenario = _mapping(scenarios.get(scenario_id), label=f"G2 {scenario_id}")
        _require(scenario, "manual_review_complete", True, label=f"G2 {scenario_id}")
        _require(scenario, "global_provider_winner_declared", False, label=f"G2 {scenario_id}")
        _require(scenario, "evidence_mode", "fresh_production_plus_matched_view", label=f"G2 {scenario_id}")
        _require(scenario, "controlled_reveal_complete", True, label=f"G2 {scenario_id}")
        prefs = _mapping(scenario.get("provider_preferences"), label=f"G2 {scenario_id} provider_preferences")
        if set(prefs) != set(dims):
            raise ReconstructionValidationError(f"I4.4B G2 {scenario_id} review dimensions drifted.")
    house = _mapping(scenarios.get(historical), label="G2 historical house")
    _require(house, "manual_review_complete", True, label="G2 historical house")
    _require(house, "global_provider_winner_declared", False, label="G2 historical house")
    _require(house, "evidence_mode", "historical_read_only", label="G2 historical house")
    _require(house, "provider_reexecution_performed", False, label="G2 historical house")

    return {
        "family_id": "mvs_da3",
        "provider_implementations": list(_provider_list("I4.3G2")),
        "scenario_ids": list(scenario_ids),
        "review_dimensions": list(dims),
        "manual_review_dimension_count": len(dims),
        "fresh_visual_review_scenarios": len(fresh),
        "historical_visual_review_scenarios": 1,
        "all_manual_reviews_complete": True,
        "historical_read_only_scenarios": [historical],
        "review_scopes_preserved": True,
    }


def _validate_status_map(value: Any, *, label: str, providers: tuple[str, ...]) -> dict[str, str]:
    statuses = _mapping(value, label=label)
    if set(statuses) != set(providers):
        raise ReconstructionValidationError(f"I4.4B {label} provider coverage drifted.")
    invalid = {provider: status for provider, status in statuses.items() if str(status) not in _ALLOWED_TERMINAL_STATUSES}
    if invalid:
        raise ReconstructionValidationError(f"I4.4B {label} contains invalid statuses: {invalid}")
    return {str(provider): str(status) for provider, status in statuses.items()}


def validate_g3_visual_closure(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != "I4.3G3C" or payload.get("status") != "formally_closed" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.4B requires the passing/formally-closed I4.3G3C closure.")
    _require(payload, "scenario_count", 3, label="G3 closure")
    _require(payload, "all_required_temporal_scenarios_accounted", True, label="G3 closure")
    _require(payload, "all_required_manual_reviews_complete", True, label="G3 closure")
    _require(payload, "i4_3g3_closed", True, label="G3 closure")
    _require(payload, "i4_3g4_authorized", True, label="G3 closure")

    scenarios = _mapping(payload.get("scenario_synthesis"), label="G3 scenario_synthesis")
    scenario_ids = _scenario_ids("I4.3G3")
    if set(scenarios) != set(scenario_ids):
        raise ReconstructionValidationError("I4.4B G3 scenario coverage drifted.")
    events: set[tuple[str, str, str, str]] = set()
    dims = tuple(I43G3C_REVIEW_DIMENSIONS)
    providers = tuple(I43G3C_PROVIDER_IMPLEMENTATIONS)
    for scenario_id in scenario_ids:
        scenario = _mapping(scenarios.get(scenario_id), label=f"G3 {scenario_id}")
        statuses = _validate_status_map(scenario.get("production_statuses"), label=f"G3 {scenario_id} production_statuses", providers=providers)
        _require(scenario, "manual_review_complete", True, label=f"G3 {scenario_id}")
        _require(scenario, "matched_view_diagnostic_secondary", True, label=f"G3 {scenario_id}")
        _require(scenario, "production_rejection_overridden", False, label=f"G3 {scenario_id}")
        _require(scenario, "global_provider_winner_declared", False, label=f"G3 {scenario_id}")
        prefs = _mapping(scenario.get("provider_preferences"), label=f"G3 {scenario_id} provider_preferences")
        if set(prefs) != set(dims):
            raise ReconstructionValidationError(f"I4.4B G3 {scenario_id} review dimensions drifted.")
        for provider, status in statuses.items():
            if status in {"degraded", "rejected"}:
                events.add(("temporal_depth", scenario_id, provider, status))

    decision = _mapping(payload.get("decision"), label="G3 decision")
    for key in ("production_native_evidence_primary", "matched_view_diagnostic_secondary", "scenario_specific_verdicts_preserved", "quality_failures_preserved"):
        _require(decision, key, True, label="G3 decision")
    for key in ("provider_selected", "provider_promoted", "universal_provider_winner_declared", "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized"):
        _require(decision, key, False, label="G3 decision")
    authority = _validate_authority(_mapping(payload.get("authority"), label="G3 authority"), label="G3 authority")
    return {
        "family_id": "temporal_depth",
        "provider_implementations": list(providers),
        "scenario_ids": list(scenario_ids),
        "review_dimensions": list(dims),
        "manual_review_dimension_count": len(dims),
        "fresh_visual_review_scenarios": len(scenario_ids),
        "historical_visual_review_scenarios": 0,
        "all_manual_reviews_complete": True,
        "matched_view_diagnostic_secondary": True,
        "terminal_quality_events": [
            {
                "family_id": family,
                "scenario_id": scenario_id,
                "provider_implementation": provider,
                "production_status": status,
            }
            for family, scenario_id, provider, status in sorted(events)
        ],
        "authority": authority,
    }


def validate_g4_visual_closure(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != "I4.3G4C" or payload.get("status") != "formally_closed" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I4.4B requires the passing/formally-closed I4.3G4C closure.")
    _require(payload, "scenario_count", 3, label="G4 closure")
    _require(payload, "all_required_single_image_scenarios_accounted", True, label="G4 closure")
    _require(payload, "all_required_manual_reviews_complete", True, label="G4 closure")
    _require(payload, "all_six_provider_outcomes_accounted_per_scenario", True, label="G4 closure")
    _require(payload, "i4_3g4_closed", True, label="G4 closure")
    _require(payload, "i4_3g5_authorized", True, label="G4 closure")

    scenarios = _mapping(payload.get("scenario_synthesis"), label="G4 scenario_synthesis")
    scenario_ids = _scenario_ids("I4.3G4")
    if set(scenarios) != set(scenario_ids):
        raise ReconstructionValidationError("I4.4B G4 scenario coverage drifted.")
    events: set[tuple[str, str, str, str]] = set()
    dims = tuple(I43G4C_REVIEW_DIMENSIONS)
    providers = tuple(I43G4C_PROVIDER_IMPLEMENTATIONS)
    for scenario_id in scenario_ids:
        scenario = _mapping(scenarios.get(scenario_id), label=f"G4 {scenario_id}")
        statuses = _validate_status_map(scenario.get("production_statuses"), label=f"G4 {scenario_id} production_statuses", providers=providers)
        _require(scenario, "manual_review_complete", True, label=f"G4 {scenario_id}")
        _require(scenario, "matched_view_diagnostic_secondary", True, label=f"G4 {scenario_id}")
        _require(scenario, "production_rejection_overridden", False, label=f"G4 {scenario_id}")
        _require(scenario, "provider_ranking_created", False, label=f"G4 {scenario_id}")
        _require(scenario, "global_provider_winner_declared", False, label=f"G4 {scenario_id}")
        prefs = _mapping(scenario.get("provider_preferences"), label=f"G4 {scenario_id} provider_preferences")
        not_assessable = _mapping(scenario.get("provider_not_assessable"), label=f"G4 {scenario_id} provider_not_assessable")
        if set(prefs) != set(dims) or set(not_assessable) != set(dims):
            raise ReconstructionValidationError(f"I4.4B G4 {scenario_id} review dimensions drifted.")
        for provider, status in statuses.items():
            if status in {"degraded", "rejected"}:
                events.add(("single_image_depth", scenario_id, provider, status))

    decision = _mapping(payload.get("decision"), label="G4 decision")
    for key in ("production_native_evidence_primary", "matched_view_diagnostic_secondary", "scenario_specific_verdicts_preserved", "quality_rejections_preserved"):
        _require(decision, key, True, label="G4 decision")
    for key in (
        "provider_selected",
        "provider_promoted",
        "provider_ranking_created",
        "universal_provider_winner_declared",
        "global_scalar_score_created",
        "auto_policy_implemented",
        "hybrid_authorized",
    ):
        _require(decision, key, False, label="G4 decision")
    authority = _validate_authority(_mapping(payload.get("authority"), label="G4 authority"), label="G4 authority")
    return {
        "family_id": "single_image_depth",
        "provider_implementations": list(providers),
        "scenario_ids": list(scenario_ids),
        "review_dimensions": list(dims),
        "manual_review_dimension_count": len(dims),
        "fresh_visual_review_scenarios": len(scenario_ids),
        "historical_visual_review_scenarios": 0,
        "all_manual_reviews_complete": True,
        "matched_view_diagnostic_secondary": True,
        "terminal_quality_events": [
            {
                "family_id": family,
                "scenario_id": scenario_id,
                "provider_implementation": provider,
                "production_status": status,
            }
            for family, scenario_id, provider, status in sorted(events)
        ],
        "authority": authority,
    }


def build_i4_4b_cross_provider_visual_verification(
    *,
    i4_3h2_closure: Mapping[str, Any],
    i4_3h2_binding: VisualVerificationEvidenceBinding,
    i4_4a_result: Mapping[str, Any],
    i4_4a_binding: VisualVerificationEvidenceBinding,
    i4_3g2_closure: Mapping[str, Any],
    i4_3g2_binding: VisualVerificationEvidenceBinding,
    i4_3g3c_closure: Mapping[str, Any],
    i4_3g3c_binding: VisualVerificationEvidenceBinding,
    i4_3g4c_closure: Mapping[str, Any],
    i4_3g4c_binding: VisualVerificationEvidenceBinding,
    visual_receipt: Mapping[str, Any],
    visual_receipt_binding: VisualVerificationEvidenceBinding,
) -> dict[str, Any]:
    validated_h2 = validate_i4_3h2_for_i4_4a(i4_3h2_closure)
    validated_i44a = validate_i4_4a_result_for_i4_4b(i4_4a_result)
    validated_receipt = validate_i4_4b_visual_receipt(visual_receipt)
    g2 = validate_g2_visual_closure(i4_3g2_closure)
    g3 = validate_g3_visual_closure(i4_3g3c_closure)
    g4 = validate_g4_visual_closure(i4_3g4c_closure)

    h2_events = _terminal_events(validated_h2["terminal_quality_events"])
    i44a_events = set(validated_i44a["terminal_quality_events"])  # type: ignore[arg-type]
    visual_events = _terminal_events(g3["terminal_quality_events"] + g4["terminal_quality_events"])
    if visual_events != h2_events or visual_events != i44a_events:
        raise ReconstructionValidationError("I4.4B terminal quality-event set drifted from frozen H2/I4.4A evidence.")

    coverage = {
        "qualification_family_count": 3,
        "scenario_count": 11,
        "provider_implementation_count": 10,
        "fresh_visual_review_scenario_count": 10,
        "historical_visual_review_scenario_count": 1,
        "review_dimension_total": g2["manual_review_dimension_count"] + g3["manual_review_dimension_count"] + g4["manual_review_dimension_count"],
        "all_visual_review_families_accounted": True,
    }

    authority = deepcopy(validated_i44a["authority"])
    return {
        "schema_version": I44B_SCHEMA_VERSION,
        "phase": I44B_PHASE,
        "status": "visual_verification_complete",
        "pass": True,
        "scope": {
            "evidence_only": True,
            "cross_provider_visual_verification_only": True,
            "provider_execution_started": False,
            "provider_runtime_probe_started": False,
            "sfm_rerun": False,
            "matched_view_rerun": False,
            "manual_review_rerun": False,
            "provenance_authority_final_audit_started": False,
            "final_i4_closure_started": False,
        },
        "evidence_bindings": {
            "i4_3h2_closure": i4_3h2_binding.as_dict(),
            "i4_4a_result": i4_4a_binding.as_dict(),
            "i4_3g2_closure": i4_3g2_binding.as_dict(),
            "i4_3g3c_closure": i4_3g3c_binding.as_dict(),
            "i4_3g4c_closure": i4_3g4c_binding.as_dict(),
            "i4_4b_visual_receipt": visual_receipt_binding.as_dict(),
        },
        "coverage": coverage,
        "family_visual_verification": {
            "mvs_da3": deepcopy(g2),
            "temporal_depth": deepcopy(g3),
            "single_image_depth": deepcopy(g4),
        },
        "cross_family_visual_verification": {
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "all_visual_reviews_complete": True,
            "historical_evidence_preserved": True,
            "manual_preferences_aggregated_into_score": False,
            "quality_event_counts_are_not_provider_scores": True,
            "direct_cross_family_provider_ranking_valid": False,
            "universal_provider_winner_declared": False,
            "terminal_quality_events": [
                {
                    "family_id": family,
                    "scenario_id": scenario_id,
                    "provider_implementation": provider,
                    "production_status": status,
                }
                for family, scenario_id, provider, status in sorted(h2_events)
            ],
        },
        "regression_summary": deepcopy(validated_receipt["counts"]),
        "authority": authority,
        "decision": {
            "i4_3_closed": True,
            "i4_4a_complete": True,
            "i4_4b_complete": True,
            "i4_4c_authorized": True,
            "i4_4c_started": False,
            "i4_4d_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "next": "I4.4C provenance / authority audit.",
    }


__all__ = [
    "I44B_SCHEMA_VERSION",
    "I44B_PHASE",
    "I44B_RECEIPT_PHASE",
    "VisualVerificationEvidenceBinding",
    "validate_i4_4a_result_for_i4_4b",
    "validate_i4_4b_visual_receipt",
    "validate_g2_visual_closure",
    "validate_g3_visual_closure",
    "validate_g4_visual_closure",
    "build_i4_4b_cross_provider_visual_verification",
]
