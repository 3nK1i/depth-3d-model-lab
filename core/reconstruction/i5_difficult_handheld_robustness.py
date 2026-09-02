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
