"""I4.3G5A cross-provider evidence synthesis.

G5A is evidence-only. It consumes the frozen G1 methodology plus the formally
closed G2, G3 and G4 qualification families, verifies their authority and
scenario-specific decision contracts, and writes one cross-family operating-
envelope synthesis for G5B.

It deliberately does not execute a provider, rerun PyCOLMAP, create a global
score/ranking, select/promote a provider, implement Auto, authorize Hybrid,
claim metric scale, create measurement/metrology authority, or close I4.3G.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.reconstruction.cross_provider_qualification import qualification_scenario_matrix
from core.reconstruction.errors import ReconstructionValidationError

I43G5A_SCHEMA_VERSION = 1
I43G5A_PHASE = "I4.3G5A"
I43G5A_FAMILY_IDS = ("mvs_da3", "temporal_depth", "single_image_depth")
_ALLOWED_TERMINAL_STATUSES = ("pass", "degraded", "rejected")

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
        raise ReconstructionValidationError(f"I4.3G5A {label} is missing or malformed.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"I4.3G5A {label} is missing or malformed.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I4.3G5A {label} contract drift: {key}={mapping.get(key)!r}, expected={expected!r}."
        )


def _require_authority(authority: Mapping[str, Any], *, label: str) -> None:
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm":
        raise ReconstructionValidationError(f"I4.3G5A {label} changed PyCOLMAP camera authority.")
    if authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError(f"I4.3G5A {label} introduced a permanent global view cap.")
    drifted = [key for key in _AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G5A {label} forbidden authority drift: {drifted}")


def _frozen_scenario_specs() -> tuple[dict[str, Any], ...]:
    return tuple(item.as_dict() for item in qualification_scenario_matrix())


def validate_g1_framework(g1: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    if g1.get("phase") != "I4.3G1" or g1.get("status") != "pass" or g1.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G5A requires the passing frozen I4.3G1 framework.")

    coverage = _mapping(g1.get("coverage"), label="G1 coverage")
    expected_coverage = {
        "scenario_count": 11,
        "provider_implementation_count": 10,
        "g2_scenarios": 5,
        "g3_scenarios": 3,
        "g4_scenarios": 3,
    }
    for key, value in expected_coverage.items():
        _require(coverage, key, value, label="G1 coverage")

    expected_specs = _frozen_scenario_specs()
    expected_by_id = {item["scenario_id"]: item for item in expected_specs}
    matrix = _sequence(g1.get("scenario_matrix"), label="G1 scenario_matrix")
    actual_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in matrix:
        item = _mapping(raw, label="G1 scenario entry")
        scenario_id = str(item.get("scenario_id", ""))
        if not scenario_id or scenario_id in actual_by_id:
            raise ReconstructionValidationError("I4.3G5A G1 scenario IDs are missing or duplicated.")
        actual_by_id[scenario_id] = item
    if set(actual_by_id) != set(expected_by_id):
        raise ReconstructionValidationError("I4.3G5A G1 scenario coverage drifted from the frozen repository matrix.")

    family_scenarios: dict[str, list[str]] = {"I4.3G2": [], "I4.3G3": [], "I4.3G4": []}
    for scenario_id, expected in expected_by_id.items():
        actual = actual_by_id[scenario_id]
        if actual.get("substep") != expected["substep"]:
            raise ReconstructionValidationError(f"I4.3G5A G1 substep drift: {scenario_id}.")
        if tuple(actual.get("provider_implementations", ())) != tuple(expected["provider_implementations"]):
            raise ReconstructionValidationError(f"I4.3G5A G1 provider matrix drift: {scenario_id}.")
        if tuple(actual.get("comparison_modes", ())) != tuple(expected["comparison_modes"]):
            raise ReconstructionValidationError(f"I4.3G5A G1 comparison-mode drift: {scenario_id}.")
        if actual.get("required") is not True or actual.get("manual_review_required") is not True:
            raise ReconstructionValidationError(f"I4.3G5A G1 required/manual-review contract drift: {scenario_id}.")
        family_scenarios[expected["substep"]].append(scenario_id)

    rules = _mapping(g1.get("decision_rules"), label="G1 decision_rules")
    for key in (
        "production_native_plan_is_primary_operating_envelope",
        "matched_view_diagnostic_is_secondary_causal_diagnostic",
        "scenario_specific_verdicts_required",
        "global_scalar_score_forbidden",
    ):
        _require(rules, key, True, label="G1 decision_rules")
    for key in (
        "auto_policy_implemented",
        "hybrid_experiment_authorized",
        "provider_promotion_performed",
        "universal_provider_winner_assumed",
    ):
        _require(rules, key, False, label="G1 decision_rules")

    _require_authority(_mapping(g1.get("authority"), label="G1 authority"), label="G1 authority")
    return {key: tuple(value) for key, value in family_scenarios.items()}


def _validate_g2_closure(g2: Mapping[str, Any]) -> Mapping[str, Any]:
    if g2.get("phase") != "I4.3G2" or g2.get("status") != "formally_closed":
        raise ReconstructionValidationError("I4.3G5A requires the formally closed I4.3G2 closure.")
    _require(g2, "i4_3g2_closed", True, label="G2 closure")
    _require(g2, "i4_3g3_authorized", True, label="G2 closure")
    synthesis = _mapping(g2.get("synthesis"), label="G2 synthesis")
    if synthesis.get("camera_pose_authority_owner") != "pycolmap_sfm":
        raise ReconstructionValidationError("I4.3G5A G2 synthesis changed PyCOLMAP camera authority.")
    required_false = (
        "auto_provider_selection_implemented",
        "hybrid_provider_authorized",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "provider_promoted",
        "provider_selected",
        "surface_mesh_generated",
        "universal_provider_winner_declared",
    )
    drifted = [key for key in required_false if synthesis.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G5A G2 synthesis authority drift: {drifted}")
    return synthesis


def _validate_family_closure(
    closure: Mapping[str, Any],
    *,
    phase: str,
    scenario_ids: tuple[str, ...],
    providers: tuple[str, ...],
    closed_key: str,
    authorized_key: str,
) -> Mapping[str, Any]:
    if closure.get("phase") != phase or closure.get("status") != "formally_closed" or closure.get("pass") is not True:
        raise ReconstructionValidationError(f"I4.3G5A requires the passing/formally-closed {phase} closure.")
    _require(closure, "scenario_count", len(scenario_ids), label=f"{phase} closure")
    _require(closure, closed_key, True, label=f"{phase} closure")
    _require(closure, authorized_key, True, label=f"{phase} closure")

    synthesis = _mapping(closure.get("scenario_synthesis"), label=f"{phase} scenario_synthesis")
    if set(synthesis) != set(scenario_ids):
        raise ReconstructionValidationError(f"I4.3G5A {phase} scenario synthesis coverage drifted.")
    for scenario_id in scenario_ids:
        scenario = _mapping(synthesis[scenario_id], label=f"{phase} {scenario_id}")
        statuses = _mapping(scenario.get("production_statuses"), label=f"{phase} {scenario_id} production_statuses")
        if set(statuses) != set(providers):
            raise ReconstructionValidationError(f"I4.3G5A {phase} {scenario_id} provider coverage drifted.")
        invalid = {provider: status for provider, status in statuses.items() if status not in _ALLOWED_TERMINAL_STATUSES}
        if invalid:
            raise ReconstructionValidationError(
                f"I4.3G5A {phase} {scenario_id} contains invalid terminal statuses: {invalid}"
            )
        if scenario.get("manual_review_complete") is not True:
            raise ReconstructionValidationError(f"I4.3G5A {phase} {scenario_id} manual review is incomplete.")
        if scenario.get("matched_view_diagnostic_secondary") is not True:
            raise ReconstructionValidationError(f"I4.3G5A {phase} {scenario_id} diagnostic precedence drifted.")
        if scenario.get("production_rejection_overridden") is not False:
            raise ReconstructionValidationError(f"I4.3G5A {phase} {scenario_id} overrides a production rejection.")

    decision = _mapping(closure.get("decision"), label=f"{phase} decision")
    for key in (
        "production_native_evidence_primary",
        "matched_view_diagnostic_secondary",
        "scenario_specific_verdicts_preserved",
    ):
        _require(decision, key, True, label=f"{phase} decision")
    for key in (
        "provider_selected",
        "provider_promoted",
        "universal_provider_winner_declared",
        "global_scalar_score_created",
        "auto_policy_implemented",
        "hybrid_authorized",
    ):
        _require(decision, key, False, label=f"{phase} decision")
    if "provider_ranking_created" in decision:
        _require(decision, "provider_ranking_created", False, label=f"{phase} decision")

    _require_authority(_mapping(closure.get("authority"), label=f"{phase} authority"), label=f"{phase} authority")
    return synthesis


def _terminal_outcome_counts(
    scenario_synthesis: Mapping[str, Any],
    providers: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    counts = {
        provider: {"pass": 0, "degraded": 0, "rejected": 0}
        for provider in providers
    }
    for scenario in scenario_synthesis.values():
        statuses = _mapping(_mapping(scenario, label="scenario synthesis").get("production_statuses"), label="production_statuses")
        for provider in providers:
            counts[provider][str(statuses[provider])] += 1
    return counts


def _quality_events(
    *,
    family_id: str,
    scenario_synthesis: Mapping[str, Any],
) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for scenario_id, raw in scenario_synthesis.items():
        scenario = _mapping(raw, label=f"{family_id} {scenario_id}")
        statuses = _mapping(scenario.get("production_statuses"), label=f"{family_id} {scenario_id} statuses")
        for provider, status in statuses.items():
            if status in {"degraded", "rejected"}:
                events.append({
                    "family_id": family_id,
                    "scenario_id": str(scenario_id),
                    "provider_implementation": str(provider),
                    "production_status": str(status),
                })
    return events


def _optional_g2_scenario_synthesis(g2: Mapping[str, Any]) -> tuple[str, Mapping[str, Any] | None]:
    candidates: tuple[tuple[str, Any], ...] = (
        ("scenario_synthesis", g2.get("scenario_synthesis")),
        ("scenarios", g2.get("scenarios")),
        ("synthesis.scenario_synthesis", _mapping(g2.get("synthesis"), label="G2 synthesis").get("scenario_synthesis")),
        ("synthesis.scenarios", _mapping(g2.get("synthesis"), label="G2 synthesis").get("scenarios")),
    )
    for source, value in candidates:
        if isinstance(value, Mapping):
            return source, value
    return "formal_closure_synthesis_only", None


@dataclass(frozen=True, slots=True)
class G5AEvidenceBinding:
    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": str(self.sha256).lower()}


def build_i4_3g5a_cross_provider_synthesis(
    *,
    i4_3g1_snapshot: Mapping[str, Any],
    i4_3g2_closure: Mapping[str, Any],
    i4_3g3c_closure: Mapping[str, Any],
    i4_3g4c_closure: Mapping[str, Any],
    evidence_bindings: Mapping[str, G5AEvidenceBinding] | None = None,
) -> dict[str, Any]:
    """Build the evidence-only G5A cross-family operating-envelope synthesis."""

    family_scenarios = validate_g1_framework(i4_3g1_snapshot)
    g2_synthesis = _validate_g2_closure(i4_3g2_closure)

    g3_ids = family_scenarios["I4.3G3"]
    g4_ids = family_scenarios["I4.3G4"]
    g3_providers = ("rolling_depth", "video_depth_anything")
    g4_providers = (
        "metric3d_v2",
        "unidepth_v2",
        "moge_2",
        "depth_pro",
        "depth_anything_v2",
        "marigold",
    )
    g3_synthesis = _validate_family_closure(
        i4_3g3c_closure,
        phase="I4.3G3C",
        scenario_ids=g3_ids,
        providers=g3_providers,
        closed_key="i4_3g3_closed",
        authorized_key="i4_3g4_authorized",
    )
    g4_synthesis = _validate_family_closure(
        i4_3g4c_closure,
        phase="I4.3G4C",
        scenario_ids=g4_ids,
        providers=g4_providers,
        closed_key="i4_3g4_closed",
        authorized_key="i4_3g5_authorized",
    )

    g2_detail_source, g2_detail = _optional_g2_scenario_synthesis(i4_3g2_closure)
    g2_ids = family_scenarios["I4.3G2"]
    if g2_detail is not None and set(g2_detail) != set(g2_ids):
        missing = sorted(set(g2_ids) - set(g2_detail))
        extra = sorted(set(g2_detail) - set(g2_ids))
        raise ReconstructionValidationError(
            f"I4.3G5A G2 scenario detail coverage drifted: missing={missing} extra={extra}"
        )

    quality_events = _quality_events(family_id="temporal_depth", scenario_synthesis=g3_synthesis)
    quality_events.extend(_quality_events(family_id="single_image_depth", scenario_synthesis=g4_synthesis))

    bindings = evidence_bindings or {}
    expected_binding_names = {"i4_3g1_snapshot", "i4_3g2_closure", "i4_3g3c_closure", "i4_3g4c_closure"}
    if bindings and set(bindings) != expected_binding_names:
        raise ReconstructionValidationError("I4.3G5A evidence binding set is incomplete or contains unexpected entries.")

    return {
        "schema_version": I43G5A_SCHEMA_VERSION,
        "phase": I43G5A_PHASE,
        "status": "synthesis_complete",
        "pass": True,
        "scope": {
            "evidence_only": True,
            "provider_execution_started": False,
            "sfm_rerun": False,
            "matched_view_rerun": False,
            "manual_review_rerun": False,
            "formal_i4_3g_closure_performed": False,
        },
        "evidence_bindings": {
            key: value.as_dict()
            for key, value in bindings.items()
        },
        "coverage": {
            "qualification_family_count": 3,
            "scenario_count": 11,
            "provider_implementation_count": 10,
            "g2_scenarios": len(g2_ids),
            "g3_scenarios": len(g3_ids),
            "g4_scenarios": len(g4_ids),
            "all_g1_scenarios_accounted_by_formally_closed_families": True,
        },
        "family_synthesis": {
            "mvs_da3": {
                "family_role": "static_multiview_dense_reconstruction",
                "provider_implementations": ["colmap_mvs", "da3"],
                "scenario_ids": list(g2_ids),
                "formal_closure_phase": "I4.3G2",
                "formal_closure_valid": True,
                "frozen_closure_synthesis": deepcopy(dict(g2_synthesis)),
                "scenario_detail_source": g2_detail_source,
                "scenario_detail": deepcopy(dict(g2_detail)) if g2_detail is not None else None,
                "historical_house_evidence_read_only": True,
                "direct_rank_comparison_with_other_families_valid": False,
            },
            "temporal_depth": {
                "family_role": "ordered_temporal_depth_evidence",
                "provider_implementations": list(g3_providers),
                "scenario_ids": list(g3_ids),
                "formal_closure_phase": "I4.3G3C",
                "formal_closure_valid": True,
                "scenario_synthesis": deepcopy(dict(g3_synthesis)),
                "terminal_outcome_counts_not_score": _terminal_outcome_counts(g3_synthesis, g3_providers),
                "direct_rank_comparison_with_other_families_valid": False,
            },
            "single_image_depth": {
                "family_role": "single_image_depth_aligned_to_authoritative_sfm",
                "provider_implementations": list(g4_providers),
                "scenario_ids": list(g4_ids),
                "formal_closure_phase": "I4.3G4C",
                "formal_closure_valid": True,
                "scenario_synthesis": deepcopy(dict(g4_synthesis)),
                "terminal_outcome_counts_not_score": _terminal_outcome_counts(g4_synthesis, g4_providers),
                "direct_rank_comparison_with_other_families_valid": False,
            },
        },
        "cross_family_synthesis": {
            "family_roles_preserved": True,
            "scenario_specific_operating_envelopes_preserved": True,
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "manual_preferences_aggregated_into_score": False,
            "direct_cross_family_provider_ranking_valid": False,
            "terminal_quality_events": quality_events,
            "quality_event_counts_are_not_provider_scores": True,
            "no_universal_provider_conclusion_drawn": True,
        },
        "decision": {
            "g5a_cross_provider_synthesis_complete": True,
            "i4_3g5b_authorized": True,
            "i4_3g_closed": False,
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "scenario_specific_verdicts_preserved": True,
            "quality_rejections_preserved": True,
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
        "next": "I4.3G5B formal I4.3G closure.",
    }


__all__ = [
    "I43G5A_SCHEMA_VERSION",
    "I43G5A_PHASE",
    "I43G5A_FAMILY_IDS",
    "G5AEvidenceBinding",
    "validate_g1_framework",
    "build_i4_3g5a_cross_provider_synthesis",
]
