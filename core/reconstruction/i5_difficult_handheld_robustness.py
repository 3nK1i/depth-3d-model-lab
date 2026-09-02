"""I5.1A difficult-handheld historical baseline admission.

I5.1A starts the difficult-handheld robustness phase without replaying providers
or SfM. Historical Horse standalone summary JSONs are not required: the frozen,
committed I4 phase-history documents are the provenance source for the historical
baseline claims. The current difficult-handheld replay remains I5.1B.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError

I51A_SCHEMA_VERSION = 2
I51A_PHASE = "I5.1A"
I51A_RECEIPT_PHASE = "I5.1A_BASELINE_RECEIPT"
I51A_MIN_FOCUSED_TESTS = 25
I51A_MIN_AFFECTED_TESTS = 30
I51A_MIN_COMBINED_TESTS = 55

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
_DECISION_FALSE = (
    "provider_selected",
    "provider_promoted",
    "provider_ranking_created",
    "universal_provider_winner_declared",
    "global_scalar_score_created",
    "auto_policy_implemented",
    "hybrid_authorized",
)


@dataclass(frozen=True, slots=True)
class I51EvidenceBinding:
    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        path = str(self.path).strip()
        sha = str(self.sha256).strip().lower()
        if not path or not _SHA256_RE.fullmatch(sha):
            raise ReconstructionValidationError("I5.1A evidence binding is malformed.")
        return {"path": path, "sha256": sha}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.1A {label} must be an object.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = mapping.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.1A {label}.{key} contract drift: expected={expected!r} actual={actual!r}."
        )


def _non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReconstructionValidationError(f"I5.1A {label} must be a non-negative integer.")
    return value


def _contains(text: str, needle: str, *, label: str) -> None:
    normalized_text = " ".join(text.split())
    normalized_needle = " ".join(needle.split())
    if normalized_needle not in normalized_text:
        raise ReconstructionValidationError(f"I5.1A committed phase-history drift: missing {label}.")


def validate_i4_4d_for_i5_1(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate final I4 closure as the authorization boundary for I5.1."""
    if payload.get("phase") != "I4.4D" or payload.get("status") != "formally_closed" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.1A requires the passing frozen I4.4D final I4 closure.")

    decision = _mapping(payload.get("decision"), label="I4.4D decision")
    for key in (
        "i4_3_closed", "i4_4a_complete", "i4_4b_complete", "i4_4c_complete",
        "i4_4d_final_closure_complete", "i4_4_offline_dense_foundation_frozen",
        "i4_closed", "i5_authorized",
    ):
        _require(decision, key, True, label="I4.4D decision")
    _require(decision, "i5_started", False, label="I4.4D decision")
    for key in _DECISION_FALSE:
        _require(decision, key, False, label="I4.4D decision")

    handoff = _mapping(payload.get("handoff_to_i5"), label="I4.4D handoff_to_i5")
    _require(handoff, "i5_authorized", True, label="I4.4D handoff_to_i5")
    _require(handoff, "i5_work_started", False, label="I4.4D handoff_to_i5")
    _require(handoff, "first_boundary", "I5.1 difficult handheld captures", label="I4.4D handoff_to_i5")
    _require(handoff, "evidence_driven_auto_policy_deferred_to_i5_13", True, label="I4.4D handoff_to_i5")
    _require(handoff, "hybrid_experiment_deferred_to_i5_12_if_justified", True, label="I4.4D handoff_to_i5")
    _require(handoff, "offline_dense_foundation_authority_locked", True, label="I4.4D handoff_to_i5")
    _require(handoff, "dense_provider_metric_scale_not_established_by_i4", True, label="I4.4D handoff_to_i5")

    authority = _mapping(payload.get("authority"), label="I4.4D authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I4.4D authority")
    _require(authority, "permanent_global_view_cap", None, label="I4.4D authority")
    drift = [key for key in _AUTHORITY_FALSE if authority.get(key) is not False]
    if drift:
        raise ReconstructionValidationError(f"I5.1A I4.4D forbidden authority drift: {drift}")

    audit = _mapping(payload.get("authority_audit"), label="I4.4D authority_audit")
    _require(audit, "camera_pose_authority_owner", "pycolmap_sfm", label="I4.4D authority_audit")
    _require(audit, "default_reconstruction_scale_status", "arbitrary", label="I4.4D authority_audit")
    _require(audit, "permanent_global_view_cap", None, label="I4.4D authority_audit")

    if payload.get("next") != "I5.1 difficult handheld captures.":
        raise ReconstructionValidationError("I5.1A I4.4D next-boundary contract drifted.")

    return {
        "decision": deepcopy(dict(decision)),
        "handoff": deepcopy(dict(handoff)),
        "authority": deepcopy(dict(authority)),
        "authority_audit": deepcopy(dict(audit)),
    }


def validate_horse_phase_history(
    *,
    i4_1_4_doc: str,
    i4_1_5_doc: str,
    i4_1_6b_doc: str,
    i4_1_7_doc: str,
) -> dict[str, Any]:
    """Recover only claims explicitly preserved in committed I4 phase history."""
    for label, text in (
        ("I4.1.4", i4_1_4_doc),
        ("I4.1.5", i4_1_5_doc),
        ("I4.1.6B", i4_1_6b_doc),
        ("I4.1.7", i4_1_7_doc),
    ):
        if not isinstance(text, str) or not text.strip():
            raise ReconstructionValidationError(f"I5.1A {label} phase history must be non-empty text.")

    _contains(i4_1_4_doc, "On the Horse orbit this improved DA3/SfM alignment to 36/40, but the dense-consistency path still collapsed to two views.", label="I4.1.4 Horse outcome")
    _contains(i4_1_4_doc, "Threshold relaxation is never authorized by this milestone.", label="I4.1.4 threshold guardrail")

    _contains(i4_1_5_doc, "I4.1.4 established a mixed failure: the Horse capture regime is difficult **and** local temporal density materially affects dense-consistency success.", label="I4.1.5 mixed-failure conclusion")
    _contains(i4_1_5_doc, "The Horse sequence remains an I5 robustness case", label="I4.1.5 I5 preservation")

    _contains(i4_1_6b_doc, "`uniform_localized_dense_pool`", label="I4.1.6B uniform plan")
    _contains(i4_1_6b_doc, "`global_anchors_plus_local_neighbors`", label="I4.1.6B local-neighbor plan")
    _contains(i4_1_6b_doc, ">=10 connected", label="I4.1.6B 10-view production gate")
    _contains(i4_1_6b_doc, "Dense threshold relaxation is never authorized by this step.", label="I4.1.6B threshold guardrail")

    _contains(i4_1_7_doc, "the difficult Horse I4.1.6B result, which proved that decoupled sampling materially improves real dense consistency but did not reach the 10-view production gate.", label="I4.1.7 difficult-Horse conclusion")
    _contains(i4_1_7_doc, "No threshold relaxation is allowed.", label="I4.1.7 threshold guardrail")

    return {
        "historical_capture_regime_difficult": True,
        "historical_local_temporal_density_materially_affects_dense_consistency": True,
        "historical_full_span_da3_sfm_alignment": "36/40",
        "historical_dense_path_collapsed_to_views": 2,
        "historical_decoupled_sampling_materially_improved_real_dense_consistency": True,
        "historical_dense_gate_reached": False,
        "historical_dense_gate_required_connected_views": 10,
        "historical_candidate_plans": [
            "global_anchors_plus_local_neighbors",
            "uniform_localized_dense_pool",
        ],
        "historical_horse_preserved_for_i5": True,
        "threshold_relaxation_authorized": False,
        "evidence_level": "committed_phase_history",
    }


def validate_i5_1a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != I51A_RECEIPT_PHASE or payload.get("status") != "pass" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.1A requires a passing baseline regression receipt.")

    scope = _mapping(payload.get("scope"), label="baseline receipt scope")
    for key in (
        "focused_i5_1a_tests", "affected_i4_horse_regression", "i4_4d_handoff_regression",
        "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="baseline receipt scope")
    for key in (
        "provider_execution_started", "sfm_rerun", "dense_pair_rerun",
        "current_difficult_handheld_replay_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="baseline receipt scope")

    counts = _mapping(payload.get("counts"), label="baseline receipt counts")
    focused = _non_negative_int(counts.get("focused_tests"), label="focused_tests")
    affected = _non_negative_int(counts.get("affected_total_tests"), label="affected_total_tests")
    combined = _non_negative_int(counts.get("combined_exercised_tests"), label="combined_exercised_tests")
    if focused < I51A_MIN_FOCUSED_TESTS or affected < I51A_MIN_AFFECTED_TESTS or combined < I51A_MIN_COMBINED_TESTS:
        raise ReconstructionValidationError("I5.1A regression receipt test coverage is below the required baseline.")
    if combined != focused + affected:
        raise ReconstructionValidationError("I5.1A regression receipt combined count is inconsistent.")

    return_codes = _mapping(payload.get("return_codes"), label="baseline receipt return_codes")
    for key in ("focused", "affected_unit", "affected_integration", "affected_stress", "import_probe", "runner_help_probe"):
        if return_codes.get(key) != 0:
            raise ReconstructionValidationError(f"I5.1A regression receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def build_i5_1a_difficult_handheld_baseline(
    *,
    i4_4d_result: Mapping[str, Any],
    i4_4d_binding: I51EvidenceBinding,
    i4_1_4_doc: str,
    i4_1_4_binding: I51EvidenceBinding,
    i4_1_5_doc: str,
    i4_1_5_binding: I51EvidenceBinding,
    i4_1_6b_doc: str,
    i4_1_6b_binding: I51EvidenceBinding,
    i4_1_7_doc: str,
    i4_1_7_binding: I51EvidenceBinding,
    baseline_receipt: Mapping[str, Any],
    baseline_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    frozen = validate_i4_4d_for_i5_1(i4_4d_result)
    historical = validate_horse_phase_history(
        i4_1_4_doc=i4_1_4_doc,
        i4_1_5_doc=i4_1_5_doc,
        i4_1_6b_doc=i4_1_6b_doc,
        i4_1_7_doc=i4_1_7_doc,
    )
    receipt = validate_i5_1a_receipt(baseline_receipt)

    return {
        "schema_version": I51A_SCHEMA_VERSION,
        "phase": I51A_PHASE,
        "status": "historical_baseline_admitted",
        "pass": True,
        "scope": {
            "historical_evidence_admission_only": True,
            "difficult_handheld_robustness_phase_started": True,
            "provider_execution_started": False,
            "provider_runtime_probe_started": False,
            "sfm_rerun": False,
            "dense_pair_rerun": False,
            "fusion_rerun": False,
            "filtering_rerun": False,
            "persistence_rerun": False,
            "current_difficult_handheld_replay_started": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
        },
        "evidence_bindings": {
            "i4_4d_final_i4_closure": i4_4d_binding.as_dict(),
            "i4_1_4_phase_history": i4_1_4_binding.as_dict(),
            "i4_1_5_phase_history": i4_1_5_binding.as_dict(),
            "i4_1_6b_phase_history": i4_1_6b_binding.as_dict(),
            "i4_1_7_phase_history": i4_1_7_binding.as_dict(),
            "i5_1a_baseline_receipt": baseline_receipt_binding.as_dict(),
        },
        "scenario": {
            "scenario_id": "i5_1_horse_difficult_handheld_baseline",
            "capture_class": "difficult_handheld_static_world_capture",
            "source_role": "historical_robustness_target",
            "fresh_source_identity_required_for_i5_1b": True,
        },
        "historical_baseline": {
            **historical,
            "historical_outcome": "degraded_robustness_case",
            "historical_failure_is_preserved_evidence": True,
            "historical_failure_reclassified_as_pass": False,
            "historical_runtime_summary_json_reconstructed": False,
            "historical_counts_are_not_provider_scores": True,
        },
        "provenance": {
            "i4_4d_final_closure_sha256_bound": True,
            "committed_horse_phase_history_sha256_bound": True,
            "historical_horse_standalone_summary_json_bound": False,
            "synthetic_historical_summary_json_created": False,
            "historical_raw_video_content_rehashed_by_i5_1a": False,
            "historical_nested_native_artifact_tree_recursively_rehashed_by_i5_1a": False,
            "no_stronger_historical_source_identity_claim_created": True,
            "current_replay_must_establish_fresh_source_identity": True,
        },
        "regression_summary": {
            "focused_i5_1a_tests": receipt["counts"]["focused_tests"],
            "affected_tests": receipt["counts"]["affected_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "all_required_suites_passed": True,
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority_audit": frozen["authority_audit"],
        "authority": frozen["authority"],
        "decision": {
            "i4_closed": True,
            "i5_authorized": True,
            "i5_1_started": True,
            "i5_1a_complete": True,
            "i5_1_current_replay_authorized": True,
            "i5_1_complete": False,
            "i5_2_authorized": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "roadmap_guard": {
            "i5_2_low_overlap_not_started": True,
            "i5_3_blur_exposure_clutter_symmetry_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_evaluation_not_started": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.1B current frozen-foundation difficult-handheld replay.",
    }

# ---------------------------------------------------------------------------
# I5.1B — current frozen-foundation difficult-handheld replay
# ---------------------------------------------------------------------------

I51B_SCHEMA_VERSION = 1
I51B_PHASE = "I5.1B"

_I51B_DECISION_FALSE = (
    "provider_selected",
    "provider_promoted",
    "provider_ranking_created",
    "universal_provider_winner_declared",
    "global_scalar_score_created",
    "auto_policy_implemented",
    "hybrid_authorized",
    "dense_consistency_threshold_relaxation_authorized",
    "metric_scale_claimed",
    "measurement_authoritative",
    "certified_metrology",
    "surface_mesh_generated",
)


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReconstructionValidationError(f"I5.1B {label} must be a positive integer.")
    return int(value)


def _positive_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReconstructionValidationError(f"I5.1B {label} must be a positive finite number.")
    result = float(value)
    if result <= 0.0 or result != result or result in (float("inf"), float("-inf")):
        raise ReconstructionValidationError(f"I5.1B {label} must be a positive finite number.")
    return result


def validate_i5_1a_for_replay(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen I5.1A authorization boundary for fresh replay."""
    if payload.get("phase") != "I5.1A" or payload.get("status") != "historical_baseline_admitted" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.1B requires a passing frozen I5.1A historical baseline.")

    scenario = _mapping(payload.get("scenario"), label="I5.1A scenario")
    _require(scenario, "fresh_source_identity_required_for_i5_1b", True, label="I5.1A scenario")
    if str(scenario.get("scenario_id", "")) != "i5_1_horse_difficult_handheld_baseline":
        raise ReconstructionValidationError("I5.1B predecessor scenario identity drifted.")

    historical = _mapping(payload.get("historical_baseline"), label="I5.1A historical_baseline")
    _require(historical, "historical_failure_is_preserved_evidence", True, label="I5.1A historical_baseline")
    _require(historical, "historical_failure_reclassified_as_pass", False, label="I5.1A historical_baseline")
    _require(historical, "historical_dense_gate_reached", False, label="I5.1A historical_baseline")
    if _non_negative_int(historical.get("historical_dense_path_collapsed_to_views"), label="historical_dense_path_collapsed_to_views") != 2:
        raise ReconstructionValidationError("I5.1B historical Horse two-view baseline drifted.")
    if _positive_int(historical.get("historical_dense_gate_required_connected_views"), label="historical_dense_gate_required_connected_views") != 10:
        raise ReconstructionValidationError("I5.1B historical Horse 10-view gate drifted.")

    provenance = _mapping(payload.get("provenance"), label="I5.1A provenance")
    _require(provenance, "historical_raw_video_content_rehashed_by_i5_1a", False, label="I5.1A provenance")
    _require(provenance, "no_stronger_historical_source_identity_claim_created", True, label="I5.1A provenance")
    _require(provenance, "current_replay_must_establish_fresh_source_identity", True, label="I5.1A provenance")

    authority = _mapping(payload.get("authority"), label="I5.1A authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.1A authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.1A authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented",
        "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative",
        "certified_metrology", "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.1A authority")

    authority_audit = _mapping(payload.get("authority_audit"), label="I5.1A authority_audit")
    _require(authority_audit, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.1A authority_audit")
    _require(authority_audit, "default_reconstruction_scale_status", "arbitrary", label="I5.1A authority_audit")
    _require(authority_audit, "permanent_global_view_cap", None, label="I5.1A authority_audit")

    decision = _mapping(payload.get("decision"), label="I5.1A decision")
    for key in ("i4_closed", "i5_authorized", "i5_1_started", "i5_1a_complete", "i5_1_current_replay_authorized"):
        _require(decision, key, True, label="I5.1A decision")
    _require(decision, "i5_1_complete", False, label="I5.1A decision")
    _require(decision, "i5_2_authorized", False, label="I5.1A decision")
    for key in _I51B_DECISION_FALSE:
        _require(decision, key, False, label="I5.1A decision")

    if payload.get("next") != "I5.1B current frozen-foundation difficult-handheld replay.":
        raise ReconstructionValidationError("I5.1B predecessor next-boundary contract drifted.")

    return {
        "scenario": deepcopy(dict(scenario)),
        "historical_baseline": deepcopy(dict(historical)),
        "provenance": deepcopy(dict(provenance)),
        "authority": deepcopy(dict(authority)),
        "authority_audit": deepcopy(dict(authority_audit)),
        "decision": deepcopy(dict(decision)),
    }


def validate_i5_1b_source_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(payload, label="source_identity")
    path = str(source.get("path", "")).strip()
    sha = str(source.get("sha256", "")).strip().lower()
    if not path:
        raise ReconstructionValidationError("I5.1B fresh source path is required.")
    if not _SHA256_RE.fullmatch(sha):
        raise ReconstructionValidationError("I5.1B fresh source SHA256 is malformed.")
    result = {
        "path": path,
        "sha256": sha,
        "size_bytes": _positive_int(source.get("size_bytes"), label="source size_bytes"),
        "total_frames": _positive_int(source.get("total_frames"), label="source total_frames"),
        "fps": _positive_float(source.get("fps"), label="source fps"),
        "duration_seconds": _positive_float(source.get("duration_seconds"), label="source duration_seconds"),
        "width": _positive_int(source.get("width"), label="source width"),
        "height": _positive_int(source.get("height"), label="source height"),
    }
    return result


def _i51b_i1_selected_views(payload: Mapping[str, Any]) -> int:
    selected = payload.get("selected_path")
    if not isinstance(selected, Mapping):
        return 0
    value = selected.get("view_count", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReconstructionValidationError("I5.1B I1 selected_path.view_count is invalid.")
    return int(value)


def validate_i5_1b_i1(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(payload, label="I1 summary")
    if summary.get("phase") != "I1":
        raise ReconstructionValidationError("I5.1B requires an I1 replay summary.")
    status = str(summary.get("status", ""))
    if status not in {"pass", "degraded", "failed"}:
        raise ReconstructionValidationError("I5.1B I1 replay status is invalid.")
    authority = summary.get("authority")
    if isinstance(authority, Mapping):
        for key in ("metric_scale_claimed", "measurement_authoritative", "surface_mesh_generated"):
            _require(authority, key, False, label="I1 authority")
    if status == "failed" and not str(summary.get("failed_stage", "")).strip():
        raise ReconstructionValidationError("I5.1B failed I1 summary must name failed_stage.")
    _i51b_i1_selected_views(summary)
    return deepcopy(dict(summary))


def validate_i5_1b_localization(payload: Mapping[str, Any], *, source_sha256: str) -> dict[str, Any]:
    summary = _mapping(payload, label="I4.1.6A localization summary")
    if summary.get("phase") != "I4.1.6A" or summary.get("status") not in {"localization_ready", "localization_degraded"}:
        raise ReconstructionValidationError("I5.1B localization summary contract drifted.")
    source = _mapping(summary.get("source"), label="I4.1.6A source")
    if str(source.get("source_sha256", "")).lower() != str(source_sha256).lower():
        raise ReconstructionValidationError("I5.1B localization source SHA256 does not match the fresh replay source.")
    decision = _mapping(summary.get("decision"), label="I4.1.6A decision")
    for key in ("single_authoritative_sfm_frame", "global_full_span_sampling_preserved", "extra_dense_candidates_are_observations_not_second_map", "extra_dense_candidates_require_pose_localization"):
        _require(decision, key, True, label="I4.1.6A decision")
    for key in (
        "production_sampling_integration_authorized", "production_default_changed",
        "dense_consistency_threshold_relaxation_authorized", "metric_scale_claimed",
        "measurement_authoritative", "surface_mesh_generated",
    ):
        _require(decision, key, False, label="I4.1.6A decision")
    plans = summary.get("plans")
    if not isinstance(plans, list) or not plans:
        raise ReconstructionValidationError("I5.1B localization summary requires plan evidence.")
    for plan in plans:
        p = _mapping(plan, label="I4.1.6A plan")
        _positive_int(p.get("planned_candidates"), label="localization planned_candidates")
        extra = _non_negative_int(p.get("extra_candidates"), label="localization extra_candidates")
        accepted = _non_negative_int(p.get("accepted_extra_candidates"), label="localization accepted_extra_candidates")
        if accepted > extra:
            raise ReconstructionValidationError("I5.1B localization accepted extras exceed extra candidates.")
        fraction = float(p.get("extra_localization_fraction", -1.0))
        if not 0.0 <= fraction <= 1.0:
            raise ReconstructionValidationError("I5.1B localization fraction must be in [0,1].")
    return deepcopy(dict(summary))


def validate_i5_1b_dense_ab(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(payload, label="I4.1.6B dense A/B summary")
    if summary.get("phase") != "I4.1.6B" or summary.get("status") != "real_dense_ab_ready":
        raise ReconstructionValidationError("I5.1B dense A/B summary contract drifted.")
    decision = _mapping(summary.get("decision"), label="I4.1.6B decision")
    for key in ("single_authoritative_sfm_frame", "global_full_span_sampling_preserved", "extra_dense_candidates_are_observations_not_second_map", "extra_dense_candidates_pose_localized"):
        _require(decision, key, True, label="I4.1.6B decision")
    for key in (
        "production_sampling_integration_authorized", "production_default_changed",
        "metric_scale_claimed", "measurement_authoritative", "surface_mesh_generated",
    ):
        _require(decision, key, False, label="I4.1.6B decision")
    thresholds = _mapping(summary.get("thresholds"), label="I4.1.6B thresholds")
    _require(thresholds, "threshold_relaxation", False, label="I4.1.6B thresholds")
    if _positive_int(thresholds.get("max_bridge_gap"), label="dense max_bridge_gap") > 6:
        raise ReconstructionValidationError("I5.1B dense max_bridge_gap exceeds the qualified range.")
    plans = summary.get("plans")
    if not isinstance(plans, list) or not plans:
        raise ReconstructionValidationError("I5.1B dense A/B summary requires plan evidence.")
    for plan in plans:
        p = _mapping(plan, label="I4.1.6B plan")
        _positive_int(p.get("planned_candidates"), label="dense planned_candidates")
        accepted = _non_negative_int(p.get("alignment_accepted"), label="dense alignment_accepted")
        if accepted > int(p.get("planned_candidates")):
            raise ReconstructionValidationError("I5.1B dense alignment accepted count exceeds candidates.")
        selected = _non_negative_int(p.get("selected_qualified_views"), label="dense selected_qualified_views")
        if p.get("dense_gate_pass") is True and selected < 10:
            raise ReconstructionValidationError("I5.1B dense gate cannot pass below ten selected qualified views.")
    return deepcopy(dict(summary))


def _validate_i5_1b_replay_parameters(payload: Mapping[str, Any]) -> dict[str, Any]:
    params = _mapping(payload, label="replay_parameters")
    profile = str(params.get("profile", ""))
    matching = str(params.get("matching", ""))
    if profile not in {"efficient", "balanced", "high_quality"}:
        raise ReconstructionValidationError("I5.1B replay profile is invalid.")
    if matching != "exhaustive":
        raise ReconstructionValidationError("I5.1B controlled Horse replay requires exhaustive matching; strategy evaluation belongs to I5.10.")
    result = {
        "profile": profile,
        "matching": matching,
        "max_keyframes": _positive_int(params.get("max_keyframes"), label="max_keyframes"),
        "max_dense_views": _positive_int(params.get("max_dense_views"), label="max_dense_views"),
        "dense_candidate_cap": _positive_int(params.get("dense_candidate_cap"), label="dense_candidate_cap"),
        "video_target_fps": _positive_float(params.get("video_target_fps"), label="video_target_fps"),
        "local_target_fps": _positive_float(params.get("local_target_fps"), label="local_target_fps"),
        "process_res": _positive_int(params.get("process_res"), label="process_res"),
        "controlled_experiment_bounds_only": True,
        "permanent_global_view_cap": None,
    }
    if result["max_keyframes"] < 10 or result["max_dense_views"] < 10:
        raise ReconstructionValidationError("I5.1B replay requires at least ten global keyframes/dense views.")
    if result["dense_candidate_cap"] < result["max_keyframes"]:
        raise ReconstructionValidationError("I5.1B dense candidate cap must cover the global replay bound.")
    return result


def _fresh_best_selected(dense_ab: Mapping[str, Any] | None, global_selected: int) -> int:
    best = int(global_selected)
    if dense_ab is None:
        return best
    plans = dense_ab.get("plans", [])
    if isinstance(plans, list):
        for plan in plans:
            if isinstance(plan, Mapping):
                value = plan.get("selected_qualified_views", 0)
                if isinstance(value, int) and not isinstance(value, bool):
                    best = max(best, int(value))
    return best


def build_i5_1b_difficult_handheld_replay(
    *,
    i5_1a_result: Mapping[str, Any],
    i5_1a_sha256: str,
    source_identity: Mapping[str, Any],
    i1_summary: Mapping[str, Any],
    localization_summary: Mapping[str, Any] | None,
    dense_ab_summary: Mapping[str, Any] | None,
    stage_return_codes: Mapping[str, Any],
    replay_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    predecessor = validate_i5_1a_for_replay(i5_1a_result)
    predecessor_sha = str(i5_1a_sha256).strip().lower()
    if not _SHA256_RE.fullmatch(predecessor_sha):
        raise ReconstructionValidationError("I5.1B I5.1A SHA256 binding is malformed.")
    source = validate_i5_1b_source_identity(source_identity)
    i1 = validate_i5_1b_i1(i1_summary)
    localization = None if localization_summary is None else validate_i5_1b_localization(
        localization_summary, source_sha256=source["sha256"]
    )
    dense = None if dense_ab_summary is None else validate_i5_1b_dense_ab(dense_ab_summary)
    params = _validate_i5_1b_replay_parameters(replay_parameters)

    return_codes = _mapping(stage_return_codes, label="stage_return_codes")
    i1_rc = return_codes.get("i1")
    loc_rc = return_codes.get("localization")
    dense_rc = return_codes.get("dense_ab")
    if isinstance(i1_rc, bool) or not isinstance(i1_rc, int):
        raise ReconstructionValidationError("I5.1B I1 return code must be an integer.")
    if i1_rc not in {0, 2}:
        raise ReconstructionValidationError("I5.1B I1 return code is outside the qualified replay contract {0,2}.")
    if localization is None and loc_rc is not None:
        raise ReconstructionValidationError("I5.1B localization return code exists without localization evidence.")
    if localization is not None and (isinstance(loc_rc, bool) or not isinstance(loc_rc, int)):
        raise ReconstructionValidationError("I5.1B localization return code must be an integer when localization ran.")
    if localization is not None and loc_rc not in {0, 3}:
        raise ReconstructionValidationError("I5.1B localization return code is outside the qualified replay contract {0,3}.")
    if dense is None and dense_rc is not None:
        raise ReconstructionValidationError("I5.1B dense return code exists without dense A/B evidence.")
    if dense is not None and (isinstance(dense_rc, bool) or not isinstance(dense_rc, int)):
        raise ReconstructionValidationError("I5.1B dense return code must be an integer when dense A/B ran.")
    if dense is not None and dense_rc != 0:
        raise ReconstructionValidationError("I5.1B dense A/B evidence requires a zero runner return code.")
    if dense is not None:
        dense_source = dense.get("source")
        if not isinstance(dense_source, Mapping) or str(dense_source.get("video", "")) != source["path"]:
            raise ReconstructionValidationError("I5.1B dense A/B source path does not match the fresh replay source.")

    i1_status = str(i1.get("status"))
    global_selected = _i51b_i1_selected_views(i1)
    dense_gate = False
    recommended_plan = None
    if dense is not None:
        plans = dense.get("plans", [])
        dense_gate = any(bool(item.get("dense_gate_pass")) for item in plans if isinstance(item, Mapping))
        decision = _mapping(dense.get("decision"), label="I4.1.6B decision")
        recommended_plan = str(decision.get("recommended_plan", "")) or None

    if i1_status == "failed" and str(i1.get("failed_stage")) == "sfm":
        quality_status = "rejected"
        failed_stage = "sfm"
    elif global_selected >= 10 or dense_gate:
        quality_status = "pass"
        failed_stage = None
    else:
        quality_status = "degraded"
        failed_stage = str(i1.get("failed_stage", "")) or (
            "decoupled_localization" if localization is not None and localization.get("status") == "localization_degraded" else "dense_qualification"
        )

    best_selected = _fresh_best_selected(dense, global_selected)
    historical = predecessor["historical_baseline"]
    historical_selected = int(historical["historical_dense_path_collapsed_to_views"])

    execution = {
        "current_difficult_handheld_replay_started": True,
        "global_i1_executed": True,
        "sfm_rerun": True,
        "global_da3_dense_executed": bool(i1_status != "failed" or str(i1.get("failed_stage")) != "sfm"),
        "localization_executed": localization is not None,
        "dense_ab_executed": dense is not None,
        "da3_extra_candidates_executed": bool(dense is not None),
        "dense_pair_rerun": bool(dense is not None),
        "fusion_may_execute_inside_i1": True,
        "production_default_modified": False,
        "stage_return_codes": {
            "i1": int(i1_rc),
            "localization": None if loc_rc is None else int(loc_rc),
            "dense_ab": None if dense_rc is None else int(dense_rc),
        },
    }

    return {
        "schema_version": I51B_SCHEMA_VERSION,
        "phase": I51B_PHASE,
        "status": "replay_evidence_complete",
        "pass": True,
        "scope": {
            "fresh_difficult_handheld_replay_only": True,
            "provider_specific_robustness_campaign": False,
            "mp_sfm_strategy_evaluation": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
            "i5_1_final_closure": False,
        },
        "source_identity": source,
        "provenance": {
            "i5_1a_sha256_bound": True,
            "i5_1a_sha256": predecessor_sha,
            "fresh_source_sha256_bound": True,
            "fresh_source_sha256": source["sha256"],
            "historical_raw_video_sha256_available": False,
            "fresh_vs_historical_content_hash_equality_claimed": False,
            "historical_runtime_summary_json_reconstructed": False,
            "current_stage_summaries_preserved": True,
        },
        "replay_parameters": params,
        "quality": {
            "status": quality_status,
            "failed_stage": failed_stage,
            "fresh_global_selected_qualified_views": int(global_selected),
            "fresh_best_selected_qualified_views": int(best_selected),
            "fresh_dense_gate_reached": bool(global_selected >= 10 or dense_gate),
            "recommended_decoupled_plan": recommended_plan,
            "quality_status_is_legitimate_operating_envelope_evidence": True,
            "quality_status_is_not_provider_ranking": True,
        },
        "historical_comparison": {
            "historical_outcome": historical["historical_outcome"],
            "historical_selected_qualified_views": historical_selected,
            "historical_dense_gate_reached": False,
            "historical_dense_gate_required_connected_views": 10,
            "fresh_global_selected_qualified_views": int(global_selected),
            "fresh_best_selected_qualified_views": int(best_selected),
            "selected_view_gain_over_historical": int(best_selected - historical_selected),
            "fresh_dense_gate_reached": bool(global_selected >= 10 or dense_gate),
            "historical_raw_video_sha256_available": False,
            "fresh_vs_historical_content_hash_equality_claimed": False,
            "comparison_is_scenario_continuity_not_content_hash_equivalence": True,
            "counts_are_not_provider_scores": True,
        },
        "stage_evidence": {
            "i1_summary": deepcopy(dict(i1)),
            "localization_summary": None if localization is None else deepcopy(dict(localization)),
            "dense_ab_summary": None if dense is None else deepcopy(dict(dense)),
        },
        "execution": execution,
        "authority_audit": deepcopy(predecessor["authority_audit"]),
        "authority": deepcopy(predecessor["authority"]),
        "decision": {
            "i4_closed": True,
            "i5_authorized": True,
            "i5_1_started": True,
            "i5_1a_complete": True,
            "i5_1b_complete": True,
            "i5_1c_authorized": True,
            "i5_1_complete": False,
            "i5_2_authorized": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "roadmap_guard": {
            "i5_2_low_overlap_not_started": True,
            "i5_3_blur_exposure_clutter_symmetry_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_evaluation_not_started": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.1C difficult-handheld robustness qualification and closure.",
    }

# ---------------------------------------------------------------------------
# I5.1C — difficult-handheld robustness qualification and closure
# ---------------------------------------------------------------------------

I51C_SCHEMA_VERSION = 1
I51C_PHASE = "I5.1C"
I51C_RECEIPT_PHASE = "I5.1C_QUALIFICATION_RECEIPT"
I51C_MIN_FOCUSED_TESTS = 15
I51C_MIN_PREDECESSOR_TESTS = 60
I51C_MIN_COMBINED_TESTS = 80


def _i51c_require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = mapping.get(key)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.1C {label}.{key} contract drift: expected={expected!r} actual={actual!r}."
        )


def validate_i5_1b_for_i5_1c(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact evidence-complete Horse replay being closed by I5.1C."""
    if payload.get("phase") != I51B_PHASE or payload.get("status") != "replay_evidence_complete" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.1C requires the passing evidence-complete I5.1B replay record.")

    scope = _mapping(payload.get("scope"), label="I5.1B scope")
    _i51c_require(scope, "fresh_difficult_handheld_replay_only", True, label="I5.1B scope")
    _i51c_require(scope, "i5_1_final_closure", False, label="I5.1B scope")
    for key in ("provider_specific_robustness_campaign", "mp_sfm_strategy_evaluation", "auto_policy_work_started", "hybrid_experiment_started"):
        _i51c_require(scope, key, False, label="I5.1B scope")

    provenance = _mapping(payload.get("provenance"), label="I5.1B provenance")
    _i51c_require(provenance, "i5_1a_sha256_bound", True, label="I5.1B provenance")
    predecessor_sha = str(provenance.get("i5_1a_sha256", "")).strip().lower()
    if not _SHA256_RE.fullmatch(predecessor_sha):
        raise ReconstructionValidationError("I5.1C I5.1B predecessor SHA256 binding is malformed.")
    _i51c_require(provenance, "fresh_source_sha256_bound", True, label="I5.1B provenance")
    _i51c_require(provenance, "historical_raw_video_sha256_available", False, label="I5.1B provenance")
    _i51c_require(provenance, "fresh_vs_historical_content_hash_equality_claimed", False, label="I5.1B provenance")
    _i51c_require(provenance, "historical_runtime_summary_json_reconstructed", False, label="I5.1B provenance")
    _i51c_require(provenance, "current_stage_summaries_preserved", True, label="I5.1B provenance")

    source = validate_i5_1b_source_identity(_mapping(payload.get("source_identity"), label="I5.1B source_identity"))
    if source["sha256"] != str(provenance.get("fresh_source_sha256", "")).strip().lower():
        raise ReconstructionValidationError("I5.1C fresh source identity does not match the I5.1B provenance binding.")

    quality = _mapping(payload.get("quality"), label="I5.1B quality")
    _i51c_require(quality, "status", "degraded", label="I5.1B quality")
    _i51c_require(quality, "failed_stage", "dense_pair_qualification", label="I5.1B quality")
    _i51c_require(quality, "fresh_global_selected_qualified_views", 2, label="I5.1B quality")
    _i51c_require(quality, "fresh_best_selected_qualified_views", 7, label="I5.1B quality")
    _i51c_require(quality, "fresh_dense_gate_reached", False, label="I5.1B quality")
    _i51c_require(quality, "recommended_decoupled_plan", "uniform_localized_dense_pool", label="I5.1B quality")
    _i51c_require(quality, "quality_status_is_legitimate_operating_envelope_evidence", True, label="I5.1B quality")
    _i51c_require(quality, "quality_status_is_not_provider_ranking", True, label="I5.1B quality")

    comparison = _mapping(payload.get("historical_comparison"), label="I5.1B historical_comparison")
    _i51c_require(comparison, "historical_selected_qualified_views", 2, label="I5.1B historical_comparison")
    _i51c_require(comparison, "historical_dense_gate_reached", False, label="I5.1B historical_comparison")
    _i51c_require(comparison, "historical_dense_gate_required_connected_views", 10, label="I5.1B historical_comparison")
    _i51c_require(comparison, "fresh_global_selected_qualified_views", 2, label="I5.1B historical_comparison")
    _i51c_require(comparison, "fresh_best_selected_qualified_views", 7, label="I5.1B historical_comparison")
    _i51c_require(comparison, "selected_view_gain_over_historical", 5, label="I5.1B historical_comparison")
    _i51c_require(comparison, "fresh_dense_gate_reached", False, label="I5.1B historical_comparison")
    _i51c_require(comparison, "historical_raw_video_sha256_available", False, label="I5.1B historical_comparison")
    _i51c_require(comparison, "fresh_vs_historical_content_hash_equality_claimed", False, label="I5.1B historical_comparison")
    _i51c_require(comparison, "comparison_is_scenario_continuity_not_content_hash_equivalence", True, label="I5.1B historical_comparison")
    _i51c_require(comparison, "counts_are_not_provider_scores", True, label="I5.1B historical_comparison")

    execution = _mapping(payload.get("execution"), label="I5.1B execution")
    for key in ("current_difficult_handheld_replay_started", "global_i1_executed", "sfm_rerun", "global_da3_dense_executed", "localization_executed", "dense_ab_executed", "da3_extra_candidates_executed", "dense_pair_rerun"):
        _i51c_require(execution, key, True, label="I5.1B execution")
    _i51c_require(execution, "production_default_modified", False, label="I5.1B execution")
    stage_return_codes = _mapping(execution.get("stage_return_codes"), label="I5.1B stage_return_codes")
    if stage_return_codes.get("i1") not in {0, 2} or stage_return_codes.get("localization") not in {0, 3} or stage_return_codes.get("dense_ab") != 0:
        raise ReconstructionValidationError("I5.1C I5.1B stage return-code contract drifted.")

    stage_evidence = _mapping(payload.get("stage_evidence"), label="I5.1B stage_evidence")
    i1 = validate_i5_1b_i1(_mapping(stage_evidence.get("i1_summary"), label="I5.1B I1 stage summary"))
    if i1.get("status") != "degraded" or i1.get("failed_stage") != "dense_pair_qualification" or _i51b_i1_selected_views(i1) != 2:
        raise ReconstructionValidationError("I5.1C I5.1B global dense-degradation boundary drifted.")
    localization = validate_i5_1b_localization(
        _mapping(stage_evidence.get("localization_summary"), label="I5.1B localization stage summary"),
        source_sha256=source["sha256"],
    )
    if localization.get("status") != "localization_ready":
        raise ReconstructionValidationError("I5.1C requires the completed I5.1B localization stage.")
    dense = validate_i5_1b_dense_ab(_mapping(stage_evidence.get("dense_ab_summary"), label="I5.1B dense A/B stage summary"))
    dense_decision = _mapping(dense.get("decision"), label="I5.1B dense A/B decision")
    _i51c_require(dense_decision, "recommended_plan", "uniform_localized_dense_pool", label="I5.1B dense A/B decision")
    _i51c_require(dense_decision, "real_dense_ab_validation_pass", False, label="I5.1B dense A/B decision")
    _i51c_require(dense_decision, "i4_1_6c_production_binding_authorized", False, label="I5.1B dense A/B decision")

    plans = {str(item.get("plan_name")): item for item in dense.get("plans", []) if isinstance(item, Mapping)}
    if set(plans) != {"uniform_localized_dense_pool", "global_anchors_plus_local_neighbors"}:
        raise ReconstructionValidationError("I5.1C dense A/B plan inventory drifted.")
    uniform = _mapping(plans["uniform_localized_dense_pool"], label="uniform dense plan")
    anchors = _mapping(plans["global_anchors_plus_local_neighbors"], label="anchor-neighbor dense plan")
    for plan, aligned, planned, selected in ((uniform, 93, 102, 7), (anchors, 109, 118, 6)):
        _i51c_require(plan, "alignment_accepted", aligned, label="I5.1B dense plan")
        _i51c_require(plan, "planned_candidates", planned, label="I5.1B dense plan")
        _i51c_require(plan, "selected_qualified_views", selected, label="I5.1B dense plan")
        _i51c_require(plan, "dense_gate_pass", False, label="I5.1B dense plan")

    decision = _mapping(payload.get("decision"), label="I5.1B decision")
    for key in ("i4_closed", "i5_authorized", "i5_1_started", "i5_1a_complete", "i5_1b_complete", "i5_1c_authorized"):
        _i51c_require(decision, key, True, label="I5.1B decision")
    _i51c_require(decision, "i5_1_complete", False, label="I5.1B decision")
    _i51c_require(decision, "i5_2_authorized", False, label="I5.1B decision")
    for key in _I51B_DECISION_FALSE:
        _i51c_require(decision, key, False, label="I5.1B decision")

    authority = _mapping(payload.get("authority"), label="I5.1B authority")
    _i51c_require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.1B authority")
    _i51c_require(authority, "permanent_global_view_cap", None, label="I5.1B authority")
    for key in _AUTHORITY_FALSE:
        _i51c_require(authority, key, False, label="I5.1B authority")
    authority_audit = _mapping(payload.get("authority_audit"), label="I5.1B authority_audit")
    _i51c_require(authority_audit, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.1B authority_audit")
    _i51c_require(authority_audit, "default_reconstruction_scale_status", "arbitrary", label="I5.1B authority_audit")
    _i51c_require(authority_audit, "permanent_global_view_cap", None, label="I5.1B authority_audit")

    if payload.get("next") != "I5.1C difficult-handheld robustness qualification and closure.":
        raise ReconstructionValidationError("I5.1C predecessor next-boundary contract drifted.")

    return {
        "provenance": deepcopy(dict(provenance)),
        "source_identity": source,
        "quality": deepcopy(dict(quality)),
        "historical_comparison": deepcopy(dict(comparison)),
        "execution": deepcopy(dict(execution)),
        "i1_summary": deepcopy(dict(i1)),
        "localization_summary": deepcopy(dict(localization)),
        "dense_ab_summary": deepcopy(dict(dense)),
        "authority": deepcopy(dict(authority)),
        "authority_audit": deepcopy(dict(authority_audit)),
        "decision": deepcopy(dict(decision)),
    }


def validate_i5_1b_oom_diagnostic(text: str) -> dict[str, Any]:
    """Admit only the explicitly preserved pre-correction host-OOM facts."""
    if not isinstance(text, str) or not text.strip():
        raise ReconstructionValidationError("I5.1C failed-attempt diagnostic must be non-empty text.")
    required = (
        "i5_1b_difficult_handheld_replay_20260902_133628",
        "I4.1.6A Decoupled Dense-Candidate Localization: LOCALIZATION_READY",
        "I4.1.6B aligning/projecting extra 147/176",
        "oom-kill",
        "task=python",
        "Out of memory: Killed process",
        "anon-rss:10942216kB",
    )
    missing = [needle for needle in required if needle not in text]
    if missing:
        raise ReconstructionValidationError(f"I5.1C failed-attempt diagnostic is missing required OOM evidence: {missing}")
    return {
        "pre_correction_host_oom_observed": True,
        "failed_attempt_directory_timestamp": "20260902_133628",
        "i4_1_6a_completed_before_oom": True,
        "failure_stage": "i4_1_6b_dense_alignment_projection",
        "last_observed_candidate": "147/176",
        "killed_process": "python",
        "killed_python_anon_rss_kib": 10942216,
        "failed_attempt_preserved": True,
        "failed_attempt_reclassified_as_quality_result": False,
        "event_is_supporting_robustness_evidence": True,
    }


def validate_i5_1c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != I51C_RECEIPT_PHASE or payload.get("status") != "pass" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.1C requires a passing qualification regression receipt.")
    scope = _mapping(payload.get("scope"), label="I5.1C receipt scope")
    for key in ("focused_i5_1c_tests", "predecessor_i5_1_regression", "oom_correction_regression", "import_probe", "runner_help_probe"):
        _i51c_require(scope, key, True, label="I5.1C receipt scope")
    for key in ("provider_execution_started", "sfm_rerun", "dense_pair_rerun", "quality_replay_rerun", "thresholds_modified", "auto_policy_implemented", "hybrid_authorized"):
        _i51c_require(scope, key, False, label="I5.1C receipt scope")

    counts = _mapping(payload.get("counts"), label="I5.1C receipt counts")
    focused = _non_negative_int(counts.get("focused_tests"), label="I5.1C focused_tests")
    predecessor = _non_negative_int(counts.get("predecessor_total_tests"), label="I5.1C predecessor_total_tests")
    combined = _non_negative_int(counts.get("combined_exercised_tests"), label="I5.1C combined_exercised_tests")
    if focused < I51C_MIN_FOCUSED_TESTS or predecessor < I51C_MIN_PREDECESSOR_TESTS or combined < I51C_MIN_COMBINED_TESTS:
        raise ReconstructionValidationError("I5.1C regression receipt test coverage is below the required closure floor.")
    if combined != focused + predecessor:
        raise ReconstructionValidationError("I5.1C regression receipt combined count is inconsistent.")

    return_codes = _mapping(payload.get("return_codes"), label="I5.1C receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if return_codes.get(key) != 0:
            raise ReconstructionValidationError(f"I5.1C regression receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def build_i5_1c_difficult_handheld_closure(
    *,
    i5_1a_result: Mapping[str, Any],
    i5_1a_binding: I51EvidenceBinding,
    i5_1b_result: Mapping[str, Any],
    i5_1b_binding: I51EvidenceBinding,
    oom_diagnostic: str,
    oom_diagnostic_binding: I51EvidenceBinding,
    qualification_receipt: Mapping[str, Any],
    qualification_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    baseline = validate_i5_1a_for_replay(i5_1a_result)
    replay = validate_i5_1b_for_i5_1c(i5_1b_result)
    oom = validate_i5_1b_oom_diagnostic(oom_diagnostic)
    receipt = validate_i5_1c_receipt(qualification_receipt)

    a_binding = i5_1a_binding.as_dict()
    b_binding = i5_1b_binding.as_dict()
    oom_binding = oom_diagnostic_binding.as_dict()
    receipt_binding = qualification_receipt_binding.as_dict()
    if replay["provenance"]["i5_1a_sha256"] != a_binding["sha256"]:
        raise ReconstructionValidationError("I5.1C I5.1B predecessor SHA does not match the independently verified I5.1A artifact.")

    if replay["authority"] != baseline["authority"] or replay["authority_audit"] != baseline["authority_audit"]:
        raise ReconstructionValidationError("I5.1C authority drifted between I5.1A and I5.1B.")

    quality = replay["quality"]
    comparison = replay["historical_comparison"]
    dense = replay["dense_ab_summary"]
    plans = {str(item.get("plan_name")): item for item in dense.get("plans", []) if isinstance(item, Mapping)}
    uniform = plans["uniform_localized_dense_pool"]
    anchors = plans["global_anchors_plus_local_neighbors"]

    return {
        "schema_version": I51C_SCHEMA_VERSION,
        "phase": I51C_PHASE,
        "status": "difficult_handheld_robustness_qualified",
        "pass": True,
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "provider_execution_started": False,
            "sfm_rerun": False,
            "dense_pair_rerun": False,
            "quality_replay_rerun": False,
            "manual_review_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "provider_specific_robustness_campaign_started": False,
            "mp_sfm_strategy_evaluation_started": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
        },
        "evidence_bindings": {
            "i5_1a_baseline": a_binding,
            "i5_1b_replay": b_binding,
            "i5_1b_failed_attempt_diagnostic": oom_binding,
            "i5_1c_qualification_receipt": receipt_binding,
        },
        "provenance": {
            "i5_1a_reverified": True,
            "i5_1b_reverified": True,
            "i5_1b_i5_1a_binding_reverified": True,
            "fresh_source_identity_preserved": True,
            "fresh_source_sha256": replay["source_identity"]["sha256"],
            "historical_raw_video_sha256_available": False,
            "fresh_vs_historical_content_hash_equality_claimed": False,
            "comparison_is_scenario_continuity_not_content_hash_equivalence": True,
            "failed_attempt_diagnostic_sha256_bound": True,
            "corrected_replay_is_quality_qualification_source": True,
            "failed_attempt_is_supporting_resilience_evidence_only": True,
            "no_synthetic_historical_runtime_json_created": True,
        },
        "qualification": {
            "scenario_id": baseline["scenario"]["scenario_id"],
            "capture_class": baseline["scenario"]["capture_class"],
            "quality_status": quality["status"],
            "historical_selected_qualified_views": comparison["historical_selected_qualified_views"],
            "fresh_global_selected_qualified_views": quality["fresh_global_selected_qualified_views"],
            "fresh_best_selected_qualified_views": quality["fresh_best_selected_qualified_views"],
            "selected_view_gain_over_historical": comparison["selected_view_gain_over_historical"],
            "required_connected_qualified_views": comparison["historical_dense_gate_required_connected_views"],
            "production_dense_gate_reached": quality["fresh_dense_gate_reached"],
            "material_improvement_over_historical_baseline": comparison["selected_view_gain_over_historical"] > 0,
            "improvement_is_insufficient_for_production_gate": True,
            "operating_envelope_classification": "outside_current_production_dense_robustness_envelope",
            "primary_robustness_bottleneck": "local_dense_consistency_connectivity",
            "global_sfm_completed_before_dense_degradation": True,
            "decoupled_localization_completed": True,
            "recommended_scenario_experiment_plan": quality["recommended_decoupled_plan"],
            "recommended_plan_is_scenario_experiment_result_not_production_default": True,
            "counts_are_not_provider_scores": True,
            "quality_status_is_not_provider_ranking": True,
            "threshold_relaxation_authorized": False,
            "retry_until_quality_pass_performed": False,
        },
        "dense_ab_observation": {
            "global_baseline_selected": 2,
            "global_baseline_pair_pass_fraction": dense["baseline_global_only"]["pair_pass_fraction"],
            "uniform_localized_dense_pool": {
                "aligned": uniform["alignment_accepted"],
                "planned": uniform["planned_candidates"],
                "pair_pass_fraction": uniform["pair_pass_fraction"],
                "selected": uniform["selected_qualified_views"],
                "dense_gate": uniform["dense_gate_pass"],
            },
            "global_anchors_plus_local_neighbors": {
                "aligned": anchors["alignment_accepted"],
                "planned": anchors["planned_candidates"],
                "pair_pass_fraction": anchors["pair_pass_fraction"],
                "selected": anchors["selected_qualified_views"],
                "dense_gate": anchors["dense_gate_pass"],
            },
        },
        "robustness_event": {
            **oom,
            "root_cause_classification": "unbounded_bulk_dense_record_memory_lifetime",
            "semantics_preserving_memory_lifetime_correction_applied": True,
            "corrected_replay_completed": True,
            "corrected_replay_quality_status": quality["status"],
            "correction_changed_reconstruction_inputs": False,
            "correction_changed_candidate_plans": False,
            "correction_changed_quality_thresholds": False,
            "correction_changed_camera_authority": False,
            "correction_changed_production_defaults": False,
            "i5_11_oom_recovery_closed_early": False,
            "oom_event_reserved_for_later_i5_11_resilience_context": True,
        },
        "regression_summary": {
            "focused_i5_1c_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "all_required_suites_passed": True,
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority_audit": deepcopy(replay["authority_audit"]),
        "authority": deepcopy(replay["authority"]),
        "decision": {
            "i4_closed": True,
            "i5_authorized": True,
            "i5_1_started": True,
            "i5_1a_complete": True,
            "i5_1b_complete": True,
            "i5_1c_complete": True,
            "i5_1_complete": True,
            "i5_1_quality_status": "degraded",
            "i5_2_authorized": True,
            "i5_2_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "roadmap_guard": {
            "i5_2_low_overlap_authorized_not_started": True,
            "i5_3_blur_exposure_clutter_symmetry_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_evaluation_not_started": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.2 low overlap / weak parallax.",
    }
