"""I5.7C RollingDepth robustness qualification and closure.

I5.7C is evidence-only. It consumes the frozen I5.7A admission and the exact
I5.7B real replay, including the preserved castle_control recovery provenance,
to close I5.7 with a bounded scientific operating envelope.

No RollingDepth inference, SfM/MVS rerun, provider comparison, threshold tuning,
Auto, Hybrid, MP-SfM, metric-scale promotion, or production-default mutation is
performed here.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_rolling_depth_robustness import (
    I57A_FROZEN_G3_ROLLING_DEPTH_STATUSES,
)
from core.reconstruction.i5_rolling_depth_robustness_replay import (
    I57B_PHASE,
    I57B_SCHEMA_VERSION,
    I57B_SCENARIO_IDS,
    I57B_RECOVERY_MISSING_PERFORMANCE_FIELDS,
    I57B_RECOVERY_SCENARIO_ID,
    I57BReplayPolicy,
    validate_i5_7a_for_i5_7b,
)

I57C_SCHEMA_VERSION = 1
I57C_PHASE = "I5.7C"
I57C_RECEIPT_PHASE = "I5.7C_QUALIFICATION_RECEIPT"
I57C_I57A_FROZEN_SHA256 = "040ee6ce45f32e9f3f51087ce5f5b8764ced9a33e978d8024d04867efa5fae61"
I57C_I57B_FROZEN_SHA256 = "516264b62866bf0b02ce80006d7ba72b15cef23576d065435f792b391d5f29df"


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{label} must be an object.")
    return value


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if node.get(key) != expected:
        raise ReconstructionValidationError(
            f"{label}.{key} must be {expected!r}; found {node.get(key)!r}."
        )


def _valid_sha(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _validate_binding(
    binding: I51EvidenceBinding, *, label: str, expected_sha256: str | None = None,
) -> None:
    if not str(binding.path).strip() or not _valid_sha(binding.sha256):
        raise ReconstructionValidationError(f"{label} evidence binding is malformed.")
    if expected_sha256 is not None and binding.sha256.lower() != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"{label} SHA256 drifted: expected={expected_sha256} actual={binding.sha256}"
        )


def _validate_mapping_binding(node: Any, *, label: str) -> dict[str, Any]:
    binding = deepcopy(dict(_mapping(node, label=label)))
    if not str(binding.get("path", "")).strip() or not _valid_sha(binding.get("sha256")):
        raise ReconstructionValidationError(f"{label} is malformed.")
    return binding


def _nonnegative_number_or_none(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReconstructionValidationError(f"{label} must be a non-negative number or null.")
    number = float(value)
    if number < 0.0 or number != number or number in (float("inf"), float("-inf")):
        raise ReconstructionValidationError(f"{label} must be finite and non-negative.")
    return number


def _validate_recovery_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    recovery = deepcopy(dict(payload))
    _require(recovery, "scenario_id", I57B_RECOVERY_SCENARIO_ID, label="I5.7B recovery provenance")
    _require(recovery, "original_wrapper_failure_after_provider_execution", True, label="I5.7B recovery provenance")
    _require(recovery, "provider_execution_rerun", False, label="I5.7B recovery provenance")
    _require(recovery, "retry_until_pass", False, label="I5.7B recovery provenance")
    _require(recovery, "thresholds_modified", False, label="I5.7B recovery provenance")
    _require(recovery, "production_default_modified", False, label="I5.7B recovery provenance")
    _require(recovery, "sfm_rerun", False, label="I5.7B recovery provenance")
    _require(recovery, "mvs_rerun", False, label="I5.7B recovery provenance")
    _require(recovery, "manifest_verified", True, label="I5.7B recovery provenance")
    _require(recovery, "console_log_verified", True, label="I5.7B recovery provenance")
    _require(recovery, "current_frozen_sfm_revalidated", True, label="I5.7B recovery provenance")
    if tuple(recovery.get("missing_performance_fields", ())) != I57B_RECOVERY_MISSING_PERFORMANCE_FIELDS:
        raise ReconstructionValidationError("I5.7C recovered control missing-performance contract drifted.")
    _validate_mapping_binding(recovery.get("original_manifest"), label="I5.7B original recovery manifest")
    _validate_mapping_binding(recovery.get("original_console_log"), label="I5.7B original recovery console log")
    if not str(recovery.get("original_attempt_path", "")).strip():
        raise ReconstructionValidationError("I5.7C recovery provenance lacks original attempt path.")
    if not str(recovery.get("original_exception_message", "")).strip():
        raise ReconstructionValidationError("I5.7C recovery provenance lacks original wrapper exception.")
    return recovery


def validate_i5_7b_for_i5_7c(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact evidence-complete I5.7B replay being closed by I5.7C."""

    _require(payload, "schema_version", I57B_SCHEMA_VERSION, label="I5.7B replay")
    _require(payload, "phase", I57B_PHASE, label="I5.7B replay")
    _require(payload, "status", "current_frozen_foundation_rolling_depth_robustness_replayed", label="I5.7B replay")
    _require(payload, "pass", True, label="I5.7B replay")
    _require(payload, "quality_status", "pass", label="I5.7B replay")

    scope = _mapping(payload.get("scope"), label="I5.7B scope")
    for key in (
        "real_production_rolling_depth_replay", "six_admitted_castle_scenarios_attempted",
        "sequential_execution", "full_registered_temporal_sequence_per_scenario",
        "recovery_used", "five_remaining_scenarios_executed_in_recovery_attempt",
    ):
        _require(scope, key, True, label="I5.7B scope")
    _require(scope, "castle_control_provider_rerun", False, label="I5.7B scope")
    for key in (
        "adaptive_view_reduction", "sfm_rerun", "bundle_adjustment_rerun", "source_rematerialized",
        "mvs_rerun", "da3_rerun", "video_depth_anything_rerun", "provider_comparison_started",
        "thresholds_modified", "production_default_modified", "result_driven_retuning",
        "retry_until_pass", "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.7B scope")

    bindings = _mapping(payload.get("evidence_bindings"), label="I5.7B evidence bindings")
    a_binding = _validate_mapping_binding(bindings.get("i5_7a_admission"), label="I5.7B I5.7A binding")
    if a_binding["sha256"].lower() != I57C_I57A_FROZEN_SHA256:
        raise ReconstructionValidationError("I5.7B no longer binds the exact frozen I5.7A artifact.")
    _validate_mapping_binding(bindings.get("i5_7b_replay_receipt"), label="I5.7B replay receipt binding")
    _validate_mapping_binding(bindings.get("nested_i5_6a_castle_sfm_admission"), label="I5.7B nested I5.6A binding")

    preflight = _mapping(payload.get("runtime_preflight"), label="I5.7B runtime preflight")
    _require(preflight, "rolling_depth_runtime_ready", True, label="I5.7B runtime preflight")
    _require(preflight, "runtime_lock_valid", True, label="I5.7B runtime preflight")
    _require(preflight, "checkpoint_ready", True, label="I5.7B runtime preflight")
    _require(preflight, "cuda_available", True, label="I5.7B runtime preflight")
    _require(preflight, "device", "cuda:0", label="I5.7B runtime preflight")
    _require(preflight, "precision", "fp16", label="I5.7B runtime preflight")
    _require(preflight, "production_config_unchanged", True, label="I5.7B runtime preflight")

    policy = _mapping(payload.get("execution_policy"), label="I5.7B execution policy")
    frozen_policy = I57BReplayPolicy().as_dict()
    if dict(policy) != frozen_policy:
        raise ReconstructionValidationError("I5.7C I5.7B execution policy drifted from the frozen production policy.")

    records = _mapping(payload.get("scenario_records"), label="I5.7B scenario records")
    if len(records) != len(I57B_SCENARIO_IDS) or set(records) != set(I57B_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.7C requires exactly the six admitted I5.7B scenario records.")

    for sid in I57B_SCENARIO_IDS:
        record = _mapping(records.get(sid), label=f"I5.7B {sid} record")
        _require(record, "scenario_id", sid, label=f"I5.7B {sid} record")
        _require(record, "runtime_status", "pass", label=f"I5.7B {sid} record")
        _require(record, "provider_execution_terminal", True, label=f"I5.7B {sid} record")
        source = record.get("source_registered_views")
        processed = record.get("processed_views")
        if not isinstance(source, int) or source < 2 or processed != source:
            raise ReconstructionValidationError(f"I5.7C {sid} did not preserve complete registered-sequence processing.")
        metrics = _mapping(record.get("native_metrics"), label=f"I5.7B {sid} native metrics")
        if metrics.get("processed_views") != source:
            raise ReconstructionValidationError(f"I5.7C {sid} native metric view count drifted.")
        _require(metrics, "alignment_accepted", True, label=f"I5.7B {sid} native metrics")
        if int(metrics.get("retained_correspondences", 0)) <= 0 or int(metrics.get("contributing_views", 0)) <= 0:
            raise ReconstructionValidationError(f"I5.7C {sid} lacks accepted alignment support.")
        if int(metrics.get("accepted_fusion_views", 0)) <= 0 or int(metrics.get("canonical_point_count", 0)) <= 0:
            raise ReconstructionValidationError(f"I5.7C {sid} lacks accepted canonical fusion evidence.")
        authority = _mapping(record.get("authority"), label=f"I5.7B {sid} authority")
        _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.7B {sid} authority")
        _require(authority, "authoritative_sfm_modified", False, label=f"I5.7B {sid} authority")
        _require(authority, "single_authoritative_sfm_frame_preserved", True, label=f"I5.7B {sid} authority")
        _require(authority, "sfm_rerun", False, label=f"I5.7B {sid} authority")
        _require(authority, "bundle_adjustment_rerun", False, label=f"I5.7B {sid} authority")
        _require(authority, "source_rematerialized", False, label=f"I5.7B {sid} authority")
        _require(authority, "per_frame_alignment_performed", False, label=f"I5.7B {sid} authority")
        _require(authority, "permanent_global_view_cap", None, label=f"I5.7B {sid} authority")
        for key in (
            "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
            "surface_mesh_generated", "provider_promotion_performed",
        ):
            _require(authority, key, False, label=f"I5.7B {sid} authority")

    control = _mapping(records[I57B_RECOVERY_SCENARIO_ID], label="I5.7B recovered control")
    control_recovery = _validate_recovery_provenance(
        _mapping(control.get("recovery"), label="I5.7B recovered control provenance")
    )
    root_recovery = _validate_recovery_provenance(
        _mapping(payload.get("recovery_provenance"), label="I5.7B root recovery provenance")
    )
    if root_recovery != control_recovery:
        raise ReconstructionValidationError("I5.7C root/control recovery provenance drifted.")
    control_metrics = _mapping(control.get("native_metrics"), label="I5.7B control native metrics")
    for key in I57B_RECOVERY_MISSING_PERFORMANCE_FIELDS:
        if control_metrics.get(key) is not None:
            raise ReconstructionValidationError(f"I5.7C recovered control {key} must remain explicitly missing.")
    for sid in I57B_SCENARIO_IDS:
        if sid != I57B_RECOVERY_SCENARIO_ID and records[sid].get("recovery") is not None:
            raise ReconstructionValidationError("Only castle_control may carry I5.7B recovery provenance.")

    historical = _mapping(payload.get("historical_temporal_anchors_preserved"), label="I5.7B historical anchors")
    for sid, expected in I57A_FROZEN_G3_ROLLING_DEPTH_STATUSES.items():
        node = _mapping(historical.get(sid), label=f"I5.7B historical {sid}")
        _require(node, "rolling_depth_production_status", expected, label=f"I5.7B historical {sid}")
        _require(node, "historical_result_overwritten", False, label=f"I5.7B historical {sid}")
        _require(node, "rerun_authorized_by_i5_7a", False, label=f"I5.7B historical {sid}")

    exclusions = _mapping(payload.get("excluded_or_deferred_scenarios_preserved"), label="I5.7B exclusions")
    expected_exclusions = {
        "castle_blur": "BLOCKED_UPSTREAM",
        "castle_underexposure": "BLOCKED_UPSTREAM",
        "castle_clutter": "NOT_FORMALLY_ADMITTED",
        "kitti_forward_motion": "BLOCKED_UPSTREAM",
        "difficult_handheld_horse": "DEFERRED_NOT_ADMITTED",
    }
    for sid, expected_status in expected_exclusions.items():
        node = _mapping(exclusions.get(sid), label=f"I5.7B exclusion {sid}")
        _require(node, "status", expected_status, label=f"I5.7B exclusion {sid}")
        _require(node, "rolling_depth_failure", False, label=f"I5.7B exclusion {sid}")

    observations = _mapping(payload.get("observations"), label="I5.7B observations")
    _require(observations, "classification_threshold_used", None, label="I5.7B observations")
    for key in (
        "global_scalar_score_created", "provider_ranking_created", "universal_provider_winner_declared",
        "cross_run_absolute_geometry_comparison", "cross_run_scale_shift_comparison",
        "ground_truth_dense_depth_available", "ground_truth_surface_accuracy_available", "metric_scale_available",
    ):
        _require(observations, key, False, label="I5.7B observations")
    _require(observations, "resource_metrics_are_observations_not_qualification_thresholds", True, label="I5.7B observations")
    _require(observations, "quality_interpretation_deferred_to_i5_7c", True, label="I5.7B observations")

    authority = _mapping(payload.get("authority"), label="I5.7B authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.7B authority")
    _require(authority, "authoritative_sfm_modified", False, label="I5.7B authority")
    _require(authority, "single_authoritative_sfm_frame_preserved", True, label="I5.7B authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.7B authority")
    for key in (
        "rolling_depth_pose_authority", "provider_selected", "provider_promoted", "provider_ranking_created",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.7B authority")

    decision = _mapping(payload.get("decision"), label="I5.7B decision")
    _require(decision, "i5_7_started", True, label="I5.7B decision")
    _require(decision, "i5_7a_complete", True, label="I5.7B decision")
    _require(decision, "i5_7b_complete", True, label="I5.7B decision")
    _require(decision, "i5_7c_authorized", True, label="I5.7B decision")
    _require(decision, "i5_7c_started", False, label="I5.7B decision")
    _require(decision, "i5_8_started", False, label="I5.7B decision")
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized", "mp_sfm_authorized",
        "threshold_relaxation_authorized", "production_default_modified",
    ):
        _require(decision, key, False, label="I5.7B decision")

    roadmap = _mapping(payload.get("roadmap_guard"), label="I5.7B roadmap guard")
    _require(roadmap, "i5_7c_evidence_only_qualification_authorized", True, label="I5.7B roadmap guard")
    _require(roadmap, "i5_8_not_started", True, label="I5.7B roadmap guard")
    _require(roadmap, "i5_10_mp_sfm_deferred", True, label="I5.7B roadmap guard")
    _require(roadmap, "i5_12_hybrid_deferred_if_justified", True, label="I5.7B roadmap guard")
    _require(roadmap, "i5_13_auto_deferred", True, label="I5.7B roadmap guard")
    _require(payload, "next", "I5.7C RollingDepth robustness qualification and closure.", label="I5.7B replay")
    return deepcopy(dict(payload))


def validate_i5_7c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I57C_SCHEMA_VERSION, label="I5.7C receipt")
    _require(payload, "phase", I57C_RECEIPT_PHASE, label="I5.7C receipt")
    _require(payload, "status", "pass", label="I5.7C receipt")
    _require(payload, "pass", True, label="I5.7C receipt")
    scope = _mapping(payload.get("scope"), label="I5.7C receipt scope")
    for key in (
        "focused_i5_7c_tests", "predecessor_i5_7b_regression", "predecessor_i5_7a_regression",
        "historical_g3_preservation_regression", "recovery_provenance_regression",
        "authority_guard_regression", "roadmap_guard_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.7C receipt scope")
    for key in (
        "rolling_depth_inference", "sfm_rerun", "mvs_rerun", "provider_comparison_started",
        "thresholds_modified", "production_default_modified", "retry_until_pass", "mp_sfm_started",
        "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.7C receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.7C receipt counts")
    focused = int(counts.get("focused_tests", 0))
    predecessor = int(counts.get("predecessor_total_tests", 0))
    combined = int(counts.get("combined_exercised_tests", 0))
    if focused <= 0 or predecessor <= 0 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.7C receipt test counts are inconsistent.")
    rcs = _mapping(payload.get("return_codes"), label="I5.7C receipt return codes")
    for key in (
        "focused", "predecessor_unit", "predecessor_integration", "predecessor_stress",
        "import_probe", "runner_help_probe",
    ):
        if int(rcs.get(key, -1)) != 0:
            raise ReconstructionValidationError(f"I5.7C receipt return code is non-zero: {key}")
    return deepcopy(dict(payload))


def build_i5_7c_rolling_depth_robustness_closure(
    *,
    i5_7a_result: Mapping[str, Any],
    i5_7a_binding: I51EvidenceBinding,
    i5_7b_result: Mapping[str, Any],
    i5_7b_binding: I51EvidenceBinding,
    qualification_receipt: Mapping[str, Any],
    qualification_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    admission = validate_i5_7a_for_i5_7b(i5_7a_result)
    replay = validate_i5_7b_for_i5_7c(i5_7b_result)
    receipt = validate_i5_7c_receipt(qualification_receipt)
    _validate_binding(i5_7a_binding, label="I5.7C I5.7A", expected_sha256=I57C_I57A_FROZEN_SHA256)
    _validate_binding(i5_7b_binding, label="I5.7C I5.7B", expected_sha256=I57C_I57B_FROZEN_SHA256)
    _validate_binding(qualification_receipt_binding, label="I5.7C receipt")

    replay_a = _mapping(replay["evidence_bindings"]["i5_7a_admission"], label="I5.7B I5.7A binding")
    if str(replay_a.get("sha256", "")).lower() != i5_7a_binding.sha256.lower():
        raise ReconstructionValidationError("I5.7C I5.7B predecessor SHA does not match the independently verified I5.7A artifact.")
    if dict(_mapping(replay.get("execution_policy"), label="I5.7B execution policy")) != I57BReplayPolicy().as_dict():
        raise ReconstructionValidationError("I5.7C frozen RollingDepth execution policy drifted.")

    records = _mapping(replay.get("scenario_records"), label="I5.7B scenario records")
    passed = [sid for sid in I57B_SCENARIO_IDS if records[sid].get("runtime_status") == "pass"]
    recovery = deepcopy(dict(_mapping(replay.get("recovery_provenance"), label="I5.7B recovery provenance")))
    historical = deepcopy(dict(_mapping(replay.get("historical_temporal_anchors_preserved"), label="I5.7B historical anchors")))
    exclusions = deepcopy(dict(_mapping(replay.get("excluded_or_deferred_scenarios_preserved"), label="I5.7B exclusions")))

    control_metrics = _mapping(records[I57B_RECOVERY_SCENARIO_ID].get("native_metrics"), label="I5.7B control native metrics")
    recovered_control_performance_missing = all(
        control_metrics.get(key) is None for key in I57B_RECOVERY_MISSING_PERFORMANCE_FIELDS
    )

    return {
        "schema_version": I57C_SCHEMA_VERSION,
        "phase": I57C_PHASE,
        "status": "rolling_depth_robustness_qualified_and_closed",
        "pass": True,
        "quality_status": "pass",
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "i5_7a_admission_reused": True,
            "i5_7b_real_replay_reused": True,
            "recovery_evidence_reused": True,
            "rolling_depth_inference": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "mvs_rerun": False,
            "source_rematerialized": False,
            "provider_comparison_started": False,
            "da3_rerun": False,
            "video_depth_anything_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "evidence_bindings": {
            "i5_7a_admission": i5_7a_binding.as_dict(),
            "i5_7b_real_replay": i5_7b_binding.as_dict(),
            "i5_7c_qualification_receipt": qualification_receipt_binding.as_dict(),
        },
        "provenance": {
            "i5_7b_quality_status_preserved": True,
            "i5_7b_scenario_runtime_outcomes_preserved": True,
            "i5_7b_recovery_provenance_preserved": True,
            "historical_g3_temporal_results_preserved": True,
            "upstream_exclusions_preserved": True,
            "historical_result_overwritten": False,
            "castle_control_provider_execution_rerun": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
        },
        "qualification": {
            "quality_status": "pass",
            "operating_envelope": "qualified_on_six_admitted_current_castle_temporal_scenarios_with_historical_temporal_limitations_preserved",
            "admitted_current_replay_scenario_count": len(I57B_SCENARIO_IDS),
            "runtime_pass_scenario_count": len(passed),
            "runtime_pass_scenarios": passed,
            "all_six_admitted_current_replay_scenarios_pass": len(passed) == len(I57B_SCENARIO_IDS),
            "current_frozen_foundation_rolling_depth_path_qualified_within_admitted_envelope": True,
            "historical_g3_static_sweep_degraded_preserved": historical["g3_temporal_static_sweep"]["rolling_depth_production_status"] == "degraded",
            "historical_g3_motion_rejected_preserved": historical["g3_temporal_motion"]["rolling_depth_production_status"] == "rejected",
            "historical_g3_low_texture_pass_preserved": historical["g3_temporal_low_texture"]["rolling_depth_production_status"] == "pass",
            "historical_temporal_evidence_prevents_universal_robustness_claim": True,
            "runtime_performance_qualification_claimed": False,
            "resource_robustness_qualification_claimed": False,
            "long_sequence_resource_qualification_claimed": False,
            "provider_specific_cancellation_qualification_claimed": False,
            "rtx_4090_qualification_claimed": False,
            "rtx_5090_qualification_claimed": False,
            "cross_gpu_equivalence_qualification_claimed": False,
            "ground_truth_dense_depth_available": False,
            "ground_truth_surface_accuracy_available": False,
            "ground_truth_surface_accuracy_claimed": False,
            "classification_threshold_used": None,
            "global_scalar_score_created": False,
            "universal_rolling_depth_robustness_claimed": False,
            "universal_provider_winner_claimed": False,
            "provider_ranking_claimed": False,
            "metric_scale_claimed": False,
        },
        "recovery_provenance": recovery,
        "historical_temporal_anchors": historical,
        "excluded_or_deferred_scenarios": exclusions,
        "limitations": {
            "recovered_control_wall_time_unavailable": recovered_control_performance_missing and control_metrics.get("wall_time_s") is None,
            "recovered_control_resource_metrics_unavailable": recovered_control_performance_missing and control_metrics.get("peak_vram_mib") is None and control_metrics.get("peak_system_ram_used_mib") is None,
            "runtime_performance_envelope_complete": False,
            "resource_envelope_complete": False,
            "rtx_2060_current_machine_observations_do_not_establish_cross_gpu_equivalence": True,
            "rtx_4090_not_qualified": True,
            "rtx_5090_not_qualified": True,
            "long_sequence_resource_envelope_not_qualified": True,
            "provider_specific_cancellation_not_qualified": True,
            "ground_truth_dense_depth_unavailable": True,
            "ground_truth_surface_accuracy_unavailable": True,
            "upstream_excluded_scenarios_are_not_reclassified_as_rolling_depth_failures": True,
            "historical_g3_static_sweep_degraded_remains_active_evidence": True,
            "historical_g3_motion_rejection_remains_active_evidence": True,
            "castle_suite_is_not_universal_rolling_depth_ground_truth": True,
            "arbitrary_sfm_gauge_preserved": True,
        },
        "regression_summary": {
            "focused_i5_7c_tests": receipt["counts"].get("focused_tests"),
            "predecessor_tests": receipt["counts"].get("predecessor_total_tests"),
            "combined_exercised_tests": receipt["counts"].get("combined_exercised_tests"),
            "all_required_suites_passed": True,
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "rolling_depth_pose_authority": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_7_complete": True,
            "i5_8_authorized": True,
            "i5_8_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "mp_sfm_authorized": False,
            "threshold_relaxation_authorized": False,
            "production_default_modified": False,
        },
        "roadmap_guard": {
            "i5_8_video_depth_anything_authorized_not_started": True,
            "i5_9_remaining_depth_provider_robustness_not_started": True,
            "i5_10_mp_sfm_deferred": True,
            "i5_11_oom_disk_crash_recovery_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_auto_deferred": True,
            "i5_14_final_i5_closure_deferred": True,
        },
        "next": "I5.8 Video Depth Anything robustness.",
    }


__all__ = [
    "I57C_SCHEMA_VERSION", "I57C_PHASE", "I57C_RECEIPT_PHASE",
    "I57C_I57A_FROZEN_SHA256", "I57C_I57B_FROZEN_SHA256",
    "validate_i5_7b_for_i5_7c", "validate_i5_7c_receipt",
    "build_i5_7c_rolling_depth_robustness_closure",
]
