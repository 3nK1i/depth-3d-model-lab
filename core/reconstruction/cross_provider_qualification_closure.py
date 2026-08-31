"""I4.3G5B formal cross-provider qualification closure.

G5B is evidence-only. It consumes the frozen I4.3G5A synthesis after the
runner has re-verified the full G1 -> G2/G3/G4 -> G5A evidence chain, validates
that the synthesis still preserves all qualification/authority invariants, and
formally closes I4.3G.

It does not execute providers, rerun PyCOLMAP, rerender matched views, repeat
manual review, create a global score/ranking, select/promote a provider,
implement Auto, authorize Hybrid, claim metric scale, create measurement or
metrology authority, generate a surface mesh, or perform I4.3H work.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from core.reconstruction.cross_provider_qualification import qualification_scenario_matrix
from core.reconstruction.errors import ReconstructionValidationError

I43G5B_SCHEMA_VERSION = 1
I43G5B_PHASE = "I4.3G5B"
I43G5B_EXPECTED_G5A_PHASE = "I4.3G5A"

_EXPECTED_FAMILIES = ("mvs_da3", "temporal_depth", "single_image_depth")
_EXPECTED_PROVIDER_IMPLEMENTATIONS = (
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
_EXPECTED_TERMINAL_QUALITY_EVENTS = (
    ("temporal_depth", "g3_temporal_static_sweep", "rolling_depth", "degraded"),
    ("temporal_depth", "g3_temporal_motion", "rolling_depth", "rejected"),
    ("temporal_depth", "g3_temporal_motion", "video_depth_anything", "degraded"),
    ("single_image_depth", "g4_single_image_low_texture", "depth_anything_v2", "rejected"),
    ("single_image_depth", "g4_single_image_low_texture", "marigold", "rejected"),
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
_DECISION_FALSE = (
    "provider_selected",
    "provider_promoted",
    "provider_ranking_created",
    "universal_provider_winner_declared",
    "global_scalar_score_created",
    "auto_policy_implemented",
    "hybrid_authorized",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3G5B {label} is missing or malformed.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"I4.3G5B {label} is missing or malformed.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I4.3G5B {label} contract drift: {key}={mapping.get(key)!r}, expected={expected!r}."
        )


def _scenario_ids_by_substep() -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {"I4.3G2": [], "I4.3G3": [], "I4.3G4": []}
    for item in qualification_scenario_matrix():
        grouped[item.substep].append(item.scenario_id)
    return {key: tuple(value) for key, value in grouped.items()}


def _validate_sha_binding(value: Any, *, label: str) -> dict[str, str]:
    binding = _mapping(value, label=label)
    path = str(binding.get("path", "")).strip()
    sha = str(binding.get("sha256", "")).strip().lower()
    if not path:
        raise ReconstructionValidationError(f"I4.3G5B {label} path is empty.")
    if not _SHA256_RE.fullmatch(sha):
        raise ReconstructionValidationError(f"I4.3G5B {label} SHA256 is malformed.")
    return {"path": path, "sha256": sha}


def _terminal_quality_events(value: Any) -> tuple[tuple[str, str, str, str], ...]:
    events = []
    for raw in _sequence(value, label="G5A terminal_quality_events"):
        item = _mapping(raw, label="G5A terminal quality event")
        events.append((
            str(item.get("family_id", "")),
            str(item.get("scenario_id", "")),
            str(item.get("provider_implementation", "")),
            str(item.get("production_status", "")),
        ))
    return tuple(events)


def validate_i4_3g5a_synthesis(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen G5A synthesis as the sole semantic input to G5B."""

    if (
        payload.get("phase") != I43G5B_EXPECTED_G5A_PHASE
        or payload.get("status") != "synthesis_complete"
        or payload.get("pass") is not True
    ):
        raise ReconstructionValidationError("I4.3G5B requires the passing frozen I4.3G5A synthesis.")

    scope = _mapping(payload.get("scope"), label="G5A scope")
    _require(scope, "evidence_only", True, label="G5A scope")
    for key in (
        "provider_execution_started",
        "sfm_rerun",
        "matched_view_rerun",
        "manual_review_rerun",
        "formal_i4_3g_closure_performed",
    ):
        _require(scope, key, False, label="G5A scope")

    coverage = _mapping(payload.get("coverage"), label="G5A coverage")
    expected_coverage = {
        "qualification_family_count": 3,
        "scenario_count": 11,
        "provider_implementation_count": 10,
        "g2_scenarios": 5,
        "g3_scenarios": 3,
        "g4_scenarios": 3,
        "all_g1_scenarios_accounted_by_formally_closed_families": True,
    }
    for key, expected in expected_coverage.items():
        _require(coverage, key, expected, label="G5A coverage")

    expected_ids = _scenario_ids_by_substep()
    families = _mapping(payload.get("family_synthesis"), label="G5A family_synthesis")
    if tuple(families) != _EXPECTED_FAMILIES and set(families) != set(_EXPECTED_FAMILIES):
        raise ReconstructionValidationError("I4.3G5B G5A provider-family coverage drifted.")

    expected_family_contracts = {
        "mvs_da3": {
            "role": "static_multiview_dense_reconstruction",
            "providers": ("colmap_mvs", "da3"),
            "scenarios": expected_ids["I4.3G2"],
            "phase": "I4.3G2",
        },
        "temporal_depth": {
            "role": "ordered_temporal_depth_evidence",
            "providers": ("rolling_depth", "video_depth_anything"),
            "scenarios": expected_ids["I4.3G3"],
            "phase": "I4.3G3C",
        },
        "single_image_depth": {
            "role": "single_image_depth_aligned_to_authoritative_sfm",
            "providers": (
                "metric3d_v2", "unidepth_v2", "moge_2", "depth_pro", "depth_anything_v2", "marigold"
            ),
            "scenarios": expected_ids["I4.3G4"],
            "phase": "I4.3G4C",
        },
    }
    seen_providers: list[str] = []
    for family_id, expected in expected_family_contracts.items():
        family = _mapping(families.get(family_id), label=f"G5A family {family_id}")
        _require(family, "family_role", expected["role"], label=f"G5A family {family_id}")
        if tuple(family.get("provider_implementations", ())) != expected["providers"]:
            raise ReconstructionValidationError(f"I4.3G5B G5A {family_id} provider coverage drifted.")
        if tuple(family.get("scenario_ids", ())) != expected["scenarios"]:
            raise ReconstructionValidationError(f"I4.3G5B G5A {family_id} scenario coverage drifted.")
        _require(family, "formal_closure_phase", expected["phase"], label=f"G5A family {family_id}")
        _require(family, "formal_closure_valid", True, label=f"G5A family {family_id}")
        _require(family, "direct_rank_comparison_with_other_families_valid", False, label=f"G5A family {family_id}")
        seen_providers.extend(str(item) for item in family.get("provider_implementations", ()))
    if tuple(seen_providers) != _EXPECTED_PROVIDER_IMPLEMENTATIONS:
        raise ReconstructionValidationError("I4.3G5B G5A 10-provider implementation coverage drifted.")

    cross = _mapping(payload.get("cross_family_synthesis"), label="G5A cross_family_synthesis")
    for key in (
        "family_roles_preserved",
        "scenario_specific_operating_envelopes_preserved",
        "production_native_evidence_primary",
        "matched_view_diagnostic_secondary",
        "quality_event_counts_are_not_provider_scores",
        "no_universal_provider_conclusion_drawn",
    ):
        _require(cross, key, True, label="G5A cross_family_synthesis")
    for key in ("manual_preferences_aggregated_into_score", "direct_cross_family_provider_ranking_valid"):
        _require(cross, key, False, label="G5A cross_family_synthesis")
    events = _terminal_quality_events(cross.get("terminal_quality_events"))
    if set(events) != set(_EXPECTED_TERMINAL_QUALITY_EVENTS) or len(events) != len(_EXPECTED_TERMINAL_QUALITY_EVENTS):
        raise ReconstructionValidationError(
            f"I4.3G5B G5A terminal quality-event set drifted: {events!r}"
        )

    decision = _mapping(payload.get("decision"), label="G5A decision")
    for key in (
        "g5a_cross_provider_synthesis_complete",
        "i4_3g5b_authorized",
        "production_native_evidence_primary",
        "matched_view_diagnostic_secondary",
        "scenario_specific_verdicts_preserved",
        "quality_rejections_preserved",
    ):
        _require(decision, key, True, label="G5A decision")
    _require(decision, "i4_3g_closed", False, label="G5A decision")
    for key in _DECISION_FALSE:
        _require(decision, key, False, label="G5A decision")

    authority = _mapping(payload.get("authority"), label="G5A authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="G5A authority")
    _require(authority, "permanent_global_view_cap", None, label="G5A authority")
    for key in _AUTHORITY_FALSE:
        _require(authority, key, False, label="G5A authority")

    bindings = _mapping(payload.get("evidence_bindings"), label="G5A evidence_bindings")
    expected_binding_names = (
        "i4_3g1_snapshot",
        "i4_3g2_closure",
        "i4_3g3c_closure",
        "i4_3g4c_closure",
    )
    if set(bindings) != set(expected_binding_names):
        raise ReconstructionValidationError("I4.3G5B G5A predecessor evidence-binding set drifted.")
    normalized_bindings = {
        key: _validate_sha_binding(bindings[key], label=f"G5A binding {key}")
        for key in expected_binding_names
    }

    return {
        "coverage": deepcopy(dict(coverage)),
        "family_synthesis": deepcopy(dict(families)),
        "cross_family_synthesis": deepcopy(dict(cross)),
        "predecessor_bindings": normalized_bindings,
    }


@dataclass(frozen=True, slots=True)
class G5BClosureBinding:
    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        path = str(self.path).strip()
        sha = str(self.sha256).strip().lower()
        if not path or not _SHA256_RE.fullmatch(sha):
            raise ReconstructionValidationError("I4.3G5B closure evidence binding is malformed.")
        return {"path": path, "sha256": sha}


def build_i4_3g5b_final_closure(
    *,
    i4_3g5a_synthesis: Mapping[str, Any],
    evidence_bindings: Mapping[str, G5BClosureBinding],
    chain_rebuild_verified: bool,
) -> dict[str, Any]:
    """Build the formal I4.3G closure after the G5A chain is verified."""

    validated = validate_i4_3g5a_synthesis(i4_3g5a_synthesis)
    if chain_rebuild_verified is not True:
        raise ReconstructionValidationError("I4.3G5B requires a verified deterministic G5A chain rebuild.")
    bindings = evidence_bindings
    expected_binding_names = {
        "i4_3g1_snapshot",
        "i4_3g2_closure",
        "i4_3g3c_closure",
        "i4_3g4c_closure",
        "i4_3g5a_synthesis",
    }
    if set(bindings) != expected_binding_names:
        raise ReconstructionValidationError("I4.3G5B final evidence-binding set is incomplete or unexpected.")

    cross = validated["cross_family_synthesis"]
    family_synthesis = validated["family_synthesis"]
    terminal_quality_events = deepcopy(list(cross["terminal_quality_events"]))

    return {
        "schema_version": I43G5B_SCHEMA_VERSION,
        "phase": I43G5B_PHASE,
        "status": "formally_closed",
        "pass": True,
        "scope": {
            "evidence_only": True,
            "provider_execution_started": False,
            "sfm_rerun": False,
            "matched_view_rerun": False,
            "manual_review_rerun": False,
            "formal_i4_3g_closure_performed": True,
            "i4_3h_work_started": False,
        },
        "evidence_bindings": {
            key: value.as_dict()
            for key, value in bindings.items()
        },
        "closure_chain": {
            "g1_framework_bound": True,
            "g2_formally_closed": True,
            "g3_formally_closed": True,
            "g4_formally_closed": True,
            "g5a_synthesis_complete": True,
            "g5a_deterministically_rebuilt_from_predecessors": True,
            "all_predecessor_sha256_bindings_verified": True,
        },
        "coverage": deepcopy(validated["coverage"]),
        "operating_envelope_summary": {
            "family_roles": {
                family_id: {
                    "family_role": family["family_role"],
                    "provider_implementations": deepcopy(list(family["provider_implementations"])),
                    "scenario_ids": deepcopy(list(family["scenario_ids"])),
                    "formal_closure_phase": family["formal_closure_phase"],
                    "formal_closure_valid": True,
                    "direct_rank_comparison_with_other_families_valid": False,
                }
                for family_id, family in family_synthesis.items()
            },
            "scenario_specific_operating_envelopes_preserved": True,
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "terminal_quality_events": terminal_quality_events,
            "terminal_quality_event_count": len(terminal_quality_events),
            "quality_event_counts_are_not_provider_scores": True,
            "direct_cross_family_provider_ranking_valid": False,
        },
        "decision": {
            "g5a_cross_provider_synthesis_complete": True,
            "i4_3g_closed": True,
            "i4_3h_authorized": True,
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
        "next": "I4.3H provider architecture closure.",
    }


__all__ = [
    "I43G5B_SCHEMA_VERSION",
    "I43G5B_PHASE",
    "G5BClosureBinding",
    "validate_i4_3g5a_synthesis",
    "build_i4_3g5b_final_closure",
]
