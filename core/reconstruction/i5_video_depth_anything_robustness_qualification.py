"""I5.8C Video Depth Anything robustness qualification and closure.

I5.8C is evidence-only. It consumes the frozen I5.8A admission and the exact
I5.8B real replay. The current replay is execution-complete but quality-DEGRADED:
all admitted registered frames aligned successfully while every scenario reached
the existing production fusion capacity. That limitation is preserved rather
than tuned away or relabeled as a clean quality pass.

No Video Depth Anything inference, SfM/MVS rerun, provider comparison, capacity
increase, threshold tuning, Auto, Hybrid, MP-SfM, metric-scale promotion, or
production-default mutation is performed here.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_video_depth_anything_robustness import (
    I58A_FROZEN_G3_VDA_STATUSES,
)
from core.reconstruction.i5_video_depth_anything_robustness_replay import (
    I58B_PHASE,
    I58B_SCHEMA_VERSION,
    I58B_SCENARIO_IDS,
    I58BReplayPolicy,
    validate_i5_8a_for_i5_8b,
)

I58C_SCHEMA_VERSION = 1
I58C_PHASE = "I5.8C"
I58C_RECEIPT_PHASE = "I5.8C_QUALIFICATION_RECEIPT"
I58C_I58A_FROZEN_SHA256 = "1d635cdcefe02e1ee3b66dfaa8af4df49fb952e58cbf4446b50c2d562369d90a"
I58C_I58B_FROZEN_SHA256 = "24eab2c95869ea96792c6599eeb94ce2801d5920d4a37b9a015a5428e568bb63"
I58C_VDA_CHECKPOINT_SHA256 = "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33"
I58C_VDA_SOURCE_REVISION = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
I58C_RESOLVED_PROFILE = "efficient"
I58C_RESOLVED_ENCODER = "vits"
I58C_RESOLVED_DEVICE = "cuda:0"
I58C_RESOLVED_PRECISION = "fp16"
I58C_PRODUCTION_MAX_VOXELS = 1_500_000


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


def _positive_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReconstructionValidationError(f"{label} must be a positive finite number.")
    number = float(value)
    if number <= 0.0 or number != number or number in (float("inf"), float("-inf")):
        raise ReconstructionValidationError(f"{label} must be a positive finite number.")
    return number


def validate_i5_8b_for_i5_8c(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact evidence-complete, degraded-quality I5.8B replay."""

    _require(payload, "schema_version", I58B_SCHEMA_VERSION, label="I5.8B replay")
    _require(payload, "phase", I58B_PHASE, label="I5.8B replay")
    _require(
        payload,
        "status",
        "current_frozen_foundation_video_depth_anything_robustness_replayed",
        label="I5.8B replay",
    )
    _require(payload, "pass", True, label="I5.8B replay")
    _require(payload, "quality_status", "degraded", label="I5.8B replay")

    scope = _mapping(payload.get("scope"), label="I5.8B scope")
    for key in (
        "real_production_video_depth_anything_replay",
        "six_admitted_castle_scenarios_attempted",
        "sequential_execution",
        "full_registered_temporal_sequence_per_scenario",
        "hardware_resolved_profile_used",
    ):
        _require(scope, key, True, label="I5.8B scope")
    for key in (
        "adaptive_view_reduction", "sfm_rerun", "bundle_adjustment_rerun",
        "source_rematerialized", "mvs_rerun", "rolling_depth_rerun",
        "provider_comparison_started", "thresholds_modified",
        "production_default_modified", "result_driven_retuning",
        "retry_until_pass", "mp_sfm_started", "auto_policy_implemented",
        "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.8B scope")

    bindings = _mapping(payload.get("evidence_bindings"), label="I5.8B evidence bindings")
    a_binding = _validate_mapping_binding(bindings.get("i5_8a_admission"), label="I5.8B I5.8A binding")
    if a_binding["sha256"].lower() != I58C_I58A_FROZEN_SHA256:
        raise ReconstructionValidationError("I5.8B no longer binds the exact frozen I5.8A artifact.")
    _validate_mapping_binding(bindings.get("i5_8b_replay_receipt"), label="I5.8B replay receipt binding")
    _validate_mapping_binding(bindings.get("nested_i5_6a_castle_sfm_admission"), label="I5.8B nested I5.6A binding")

    preflight = _mapping(payload.get("runtime_preflight"), label="I5.8B runtime preflight")
    _require(preflight, "video_depth_anything_runtime_ready", True, label="I5.8B runtime preflight")
    _require(preflight, "resolved_profile", I58C_RESOLVED_PROFILE, label="I5.8B runtime preflight")
    _require(preflight, "resolved_encoder", I58C_RESOLVED_ENCODER, label="I5.8B runtime preflight")
    _require(preflight, "device", I58C_RESOLVED_DEVICE, label="I5.8B runtime preflight")
    _require(preflight, "precision", I58C_RESOLVED_PRECISION, label="I5.8B runtime preflight")
    _require(preflight, "checkpoint_sha256", I58C_VDA_CHECKPOINT_SHA256, label="I5.8B runtime preflight")
    _require(preflight, "source_revision", I58C_VDA_SOURCE_REVISION, label="I5.8B runtime preflight")
    _require(preflight, "production_config_unchanged", True, label="I5.8B runtime preflight")
    production_config = _mapping(preflight.get("production_config"), label="I5.8B production config")
    _require(production_config, "max_voxels", I58C_PRODUCTION_MAX_VOXELS, label="I5.8B production config")
    runtime_config = _mapping(preflight.get("runtime_config"), label="I5.8B runtime config")
    _require(runtime_config, "profile", I58C_RESOLVED_PROFILE, label="I5.8B runtime config")
    _require(runtime_config, "hardware_tier", I58C_RESOLVED_PROFILE, label="I5.8B runtime config")
    _require(runtime_config, "encoder", I58C_RESOLVED_ENCODER, label="I5.8B runtime config")
    _require(runtime_config, "device", I58C_RESOLVED_DEVICE, label="I5.8B runtime config")
    _require(runtime_config, "precision", I58C_RESOLVED_PRECISION, label="I5.8B runtime config")
    if "metric_video_depth_anything_vits.pth" not in str(runtime_config.get("checkpoint", "")):
        raise ReconstructionValidationError("I5.8C expected the frozen efficient/ViT-S VDA checkpoint.")

    policy = _mapping(payload.get("execution_policy"), label="I5.8B execution policy")
    if dict(policy) != I58BReplayPolicy().as_dict():
        raise ReconstructionValidationError("I5.8C I5.8B execution policy drifted from the frozen replay policy.")

    records = _mapping(payload.get("scenario_records"), label="I5.8B scenario records")
    if len(records) != len(I58B_SCENARIO_IDS) or set(records) != set(I58B_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.8C requires exactly the six admitted I5.8B scenario records.")

    for sid in I58B_SCENARIO_IDS:
        record = _mapping(records.get(sid), label=f"I5.8B {sid} record")
        _require(record, "scenario_id", sid, label=f"I5.8B {sid} record")
        _require(record, "runtime_status", "degraded", label=f"I5.8B {sid} record")
        _require(record, "provider_execution_terminal", True, label=f"I5.8B {sid} record")
        source = record.get("source_registered_views")
        processed = record.get("processed_views")
        if not isinstance(source, int) or source < 2 or processed != source:
            raise ReconstructionValidationError(f"I5.8C {sid} did not preserve complete registered-sequence processing.")

        metrics = _mapping(record.get("native_metrics"), label=f"I5.8B {sid} native metrics")
        if int(metrics.get("processed_views", 0)) != source:
            raise ReconstructionValidationError(f"I5.8C {sid} native metric view count drifted.")
        if int(metrics.get("alignment_accepted_views", -1)) != source:
            raise ReconstructionValidationError(f"I5.8C {sid} no longer has full registered-frame alignment acceptance.")
        if int(metrics.get("alignment_rejected_views", -1)) != 0:
            raise ReconstructionValidationError(f"I5.8C {sid} contains an alignment rejection not present in the frozen replay conclusion.")
        if float(metrics.get("alignment_accepted_fraction", -1.0)) != 1.0:
            raise ReconstructionValidationError(f"I5.8C {sid} alignment accepted fraction drifted from 1.0.")
        drops = int(metrics.get("fusion_capacity_drops", 0))
        if drops <= 0:
            raise ReconstructionValidationError(f"I5.8C {sid} must preserve the observed fusion-capacity degradation.")
        if int(metrics.get("canonical_point_count", 0)) != I58C_PRODUCTION_MAX_VOXELS:
            raise ReconstructionValidationError(f"I5.8C {sid} canonical point count no longer matches the frozen production fusion capacity.")
        for key in ("wall_time_s", "peak_vram_mib", "peak_system_ram_used_mib", "frame_latency_mean_ms"):
            _positive_number(metrics.get(key), label=f"I5.8B {sid} {key}")

        execution = _mapping(record.get("execution_result"), label=f"I5.8B {sid} execution result")
        _require(execution, "status", "degraded", label=f"I5.8B {sid} execution result")
        reasons = execution.get("degradation_reasons")
        expected_reason = f"fusion_capacity_drops:{drops}"
        if not isinstance(reasons, list) or reasons != [expected_reason]:
            raise ReconstructionValidationError(
                f"I5.8C {sid} degradation must remain solely the frozen fusion-capacity reason {expected_reason!r}."
            )
        _require(execution, "failure_reason", "", label=f"I5.8B {sid} execution result")

        authority = _mapping(record.get("authority"), label=f"I5.8B {sid} authority")
        _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.8B {sid} authority")
        _require(authority, "authoritative_sfm_modified", False, label=f"I5.8B {sid} authority")
        _require(authority, "single_authoritative_sfm_frame_preserved", True, label=f"I5.8B {sid} authority")
        _require(authority, "sfm_rerun", False, label=f"I5.8B {sid} authority")
        _require(authority, "bundle_adjustment_rerun", False, label=f"I5.8B {sid} authority")
        _require(authority, "source_rematerialized", False, label=f"I5.8B {sid} authority")
        _require(authority, "per_frame_alignment_performed", True, label=f"I5.8B {sid} authority")
        _require(authority, "sequence_global_alignment_performed", False, label=f"I5.8B {sid} authority")
        _require(authority, "native_estimated_intrinsics_are_canonical_authority", False, label=f"I5.8B {sid} authority")
        _require(authority, "native_metric_depth_establishes_physical_metric_scale", False, label=f"I5.8B {sid} authority")
        _require(authority, "permanent_global_view_cap", None, label=f"I5.8B {sid} authority")
        for key in (
            "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
            "surface_mesh_generated", "provider_promotion_performed",
        ):
            _require(authority, key, False, label=f"I5.8B {sid} authority")

    historical = _mapping(payload.get("historical_temporal_anchors_preserved"), label="I5.8B historical anchors")
    for sid, expected in I58A_FROZEN_G3_VDA_STATUSES.items():
        node = _mapping(historical.get(sid), label=f"I5.8B historical {sid}")
        _require(node, "video_depth_anything_production_status", expected, label=f"I5.8B historical {sid}")
        _require(node, "historical_result_overwritten", False, label=f"I5.8B historical {sid}")

    observations = _mapping(payload.get("observations"), label="I5.8B observations")
    _require(observations, "universal_video_depth_anything_robustness_claimed", False, label="I5.8B observations")
    _require(observations, "historical_motion_degraded_can_be_overwritten_by_current_pass", False, label="I5.8B observations")
    _require(observations, "quality_interpretation_deferred_to_i5_8c", True, label="I5.8B observations")

    authority = _mapping(payload.get("authority"), label="I5.8B authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.8B authority")
    _require(authority, "authoritative_sfm_modified", False, label="I5.8B authority")
    _require(authority, "single_authoritative_sfm_frame_preserved", True, label="I5.8B authority")
    _require(authority, "video_depth_anything_pose_authority", False, label="I5.8B authority")
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.8B authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.8B authority")

    decision = _mapping(payload.get("decision"), label="I5.8B decision")
    _require(decision, "i5_7_complete", True, label="I5.8B decision")
    _require(decision, "i5_8_started", True, label="I5.8B decision")
    _require(decision, "i5_8a_complete", True, label="I5.8B decision")
    _require(decision, "i5_8b_complete", True, label="I5.8B decision")
    _require(decision, "i5_8c_authorized", True, label="I5.8B decision")
    _require(decision, "i5_8c_started", False, label="I5.8B decision")
    _require(decision, "i5_9_started", False, label="I5.8B decision")
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created",
        "auto_policy_implemented", "hybrid_authorized", "mp_sfm_authorized",
        "threshold_relaxation_authorized", "production_default_modified",
    ):
        _require(decision, key, False, label="I5.8B decision")
    return deepcopy(dict(payload))


def validate_i5_8c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.8C receipt")
    _require(payload, "phase", I58C_RECEIPT_PHASE, label="I5.8C receipt")
    _require(payload, "status", "pass", label="I5.8C receipt")
    _require(payload, "pass", True, label="I5.8C receipt")
    scope = _mapping(payload.get("scope"), label="I5.8C receipt scope")
    for key in (
        "focused_i5_8c_tests", "predecessor_i5_8b_regression", "predecessor_i5_8a_regression",
        "historical_g3_preservation_regression", "fusion_capacity_provenance_regression",
        "authority_guard_regression", "roadmap_guard_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.8C receipt scope")
    for key in (
        "video_depth_anything_inference", "sfm_rerun", "mvs_rerun", "provider_comparison_started",
        "thresholds_modified", "production_default_modified", "retry_until_pass", "mp_sfm_started",
        "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.8C receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.8C receipt counts")
    focused = int(counts.get("focused_tests", 0))
    predecessor = int(counts.get("predecessor_total_tests", 0))
    combined = int(counts.get("combined_exercised_tests", 0))
    if focused <= 0 or predecessor <= 0 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.8C receipt test counts are inconsistent.")
    rcs = _mapping(payload.get("return_codes"), label="I5.8C receipt return codes")
    for key in (
        "focused", "predecessor_unit", "predecessor_integration", "predecessor_stress",
        "import_probe", "runner_help_probe",
    ):
        if int(rcs.get(key, -1)) != 0:
            raise ReconstructionValidationError(f"I5.8C receipt return code is non-zero: {key}")
    return deepcopy(dict(payload))


def build_i5_8c_video_depth_anything_robustness_closure(
    *,
    i5_8a_result: Mapping[str, Any],
    i5_8a_binding: I51EvidenceBinding,
    i5_8b_result: Mapping[str, Any],
    i5_8b_binding: I51EvidenceBinding,
    qualification_receipt: Mapping[str, Any],
    qualification_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    admission = validate_i5_8a_for_i5_8b(i5_8a_result)
    replay = validate_i5_8b_for_i5_8c(i5_8b_result)
    receipt = validate_i5_8c_receipt(qualification_receipt)
    _validate_binding(i5_8a_binding, label="I5.8C I5.8A", expected_sha256=I58C_I58A_FROZEN_SHA256)
    _validate_binding(i5_8b_binding, label="I5.8C I5.8B", expected_sha256=I58C_I58B_FROZEN_SHA256)
    _validate_binding(qualification_receipt_binding, label="I5.8C receipt")

    replay_a = _mapping(replay["evidence_bindings"]["i5_8a_admission"], label="I5.8B I5.8A binding")
    if str(replay_a.get("sha256", "")).lower() != i5_8a_binding.sha256.lower():
        raise ReconstructionValidationError("I5.8C I5.8B predecessor SHA does not match the independently verified I5.8A artifact.")
    if dict(_mapping(replay.get("execution_policy"), label="I5.8B execution policy")) != I58BReplayPolicy().as_dict():
        raise ReconstructionValidationError("I5.8C frozen VDA replay policy drifted.")

    records = _mapping(replay.get("scenario_records"), label="I5.8B scenario records")
    degraded = [sid for sid in I58B_SCENARIO_IDS if records[sid].get("runtime_status") == "degraded"]
    capacity = {
        sid: int(_mapping(records[sid].get("native_metrics"), label=f"I5.8B {sid} native metrics")["fusion_capacity_drops"])
        for sid in I58B_SCENARIO_IDS
    }
    historical = deepcopy(dict(_mapping(replay.get("historical_temporal_anchors_preserved"), label="I5.8B historical anchors")))
    exclusions = deepcopy(dict(_mapping(replay.get("excluded_or_deferred_scenarios_preserved"), label="I5.8B exclusions")))

    current_metrics_complete = all(
        _mapping(records[sid].get("native_metrics"), label=f"I5.8B {sid} metrics").get(key) is not None
        for sid in I58B_SCENARIO_IDS
        for key in ("wall_time_s", "peak_vram_mib", "peak_system_ram_used_mib", "frame_latency_mean_ms")
    )

    return {
        "schema_version": I58C_SCHEMA_VERSION,
        "phase": I58C_PHASE,
        "status": "video_depth_anything_robustness_qualified_and_closed",
        "pass": True,
        "quality_status": "degraded",
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "i5_8a_admission_reused": True,
            "i5_8b_real_replay_reused": True,
            "video_depth_anything_inference": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "mvs_rerun": False,
            "source_rematerialized": False,
            "provider_comparison_started": False,
            "rolling_depth_rerun": False,
            "thresholds_modified": False,
            "fusion_capacity_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "evidence_bindings": {
            "i5_8a_admission": i5_8a_binding.as_dict(),
            "i5_8b_real_replay": i5_8b_binding.as_dict(),
            "i5_8c_qualification_receipt": qualification_receipt_binding.as_dict(),
        },
        "provenance": {
            "i5_8b_quality_status_preserved": True,
            "i5_8b_scenario_runtime_outcomes_preserved": True,
            "historical_g3_temporal_results_preserved": True,
            "upstream_exclusions_preserved": True,
            "historical_result_overwritten": False,
            "fusion_capacity_degradation_preserved": True,
            "fusion_capacity_increased_to_obtain_pass": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
        },
        "qualification": {
            "quality_status": "degraded",
            "operating_envelope": "execution_qualified_but_quality_degraded_on_six_admitted_current_castle_temporal_scenarios_due_to_existing_fusion_capacity_with_historical_motion_degradation_preserved",
            "admitted_current_replay_scenario_count": len(I58B_SCENARIO_IDS),
            "current_replay_terminal_scenario_count": len(degraded),
            "current_replay_degraded_scenario_count": len(degraded),
            "current_replay_degraded_scenarios": degraded,
            "all_six_current_scenarios_terminal": len(degraded) == len(I58B_SCENARIO_IDS),
            "all_current_registered_frames_alignment_accepted": True,
            "all_six_current_scenarios_fusion_capacity_degraded": all(value > 0 for value in capacity.values()),
            "current_frozen_foundation_vda_path_execution_qualified_within_admitted_envelope": True,
            "current_frozen_foundation_vda_quality_clean_pass_claimed": False,
            "historical_g3_static_sweep_pass_preserved": historical["g3_temporal_static_sweep"]["video_depth_anything_production_status"] == "pass",
            "historical_g3_motion_degraded_preserved": historical["g3_temporal_motion"]["video_depth_anything_production_status"] == "degraded",
            "historical_g3_low_texture_pass_preserved": historical["g3_temporal_low_texture"]["video_depth_anything_production_status"] == "pass",
            "historical_temporal_evidence_prevents_universal_robustness_claim": True,
            "fusion_capacity_limitation_prevents_clean_quality_pass_claim": True,
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
            "universal_video_depth_anything_robustness_claimed": False,
            "universal_provider_winner_claimed": False,
            "provider_ranking_claimed": False,
            "metric_scale_claimed": False,
        },
        "runtime_identity": {
            "hardware_observation": "NVIDIA GeForce RTX 2060",
            "resolved_profile": I58C_RESOLVED_PROFILE,
            "resolved_encoder": I58C_RESOLVED_ENCODER,
            "device": I58C_RESOLVED_DEVICE,
            "precision": I58C_RESOLVED_PRECISION,
            "checkpoint_sha256": I58C_VDA_CHECKPOINT_SHA256,
            "source_revision": I58C_VDA_SOURCE_REVISION,
            "production_max_voxels": I58C_PRODUCTION_MAX_VOXELS,
        },
        "fusion_capacity_observations": capacity,
        "historical_temporal_anchors": historical,
        "excluded_or_deferred_scenarios": exclusions,
        "limitations": {
            "fusion_capacity_degradation_remains_active": True,
            "production_fusion_capacity_increased": False,
            "all_current_alignment_acceptance_does_not_remove_fusion_capacity_limitation": True,
            "current_six_scenario_performance_observations_complete": current_metrics_complete,
            "runtime_performance_envelope_complete": False,
            "resource_envelope_complete": False,
            "rtx_2060_current_machine_observations_do_not_establish_cross_gpu_equivalence": True,
            "rtx_4090_not_qualified": True,
            "rtx_5090_not_qualified": True,
            "long_sequence_resource_envelope_not_qualified": True,
            "provider_specific_cancellation_not_qualified": True,
            "ground_truth_dense_depth_unavailable": True,
            "ground_truth_surface_accuracy_unavailable": True,
            "upstream_excluded_scenarios_are_not_reclassified_as_vda_failures": True,
            "historical_g3_motion_degradation_remains_active_evidence": True,
            "castle_suite_is_not_universal_vda_ground_truth": True,
            "arbitrary_sfm_gauge_preserved": True,
        },
        "regression_summary": {
            "focused_i5_8c_tests": receipt["counts"].get("focused_tests"),
            "predecessor_tests": receipt["counts"].get("predecessor_total_tests"),
            "combined_exercised_tests": receipt["counts"].get("combined_exercised_tests"),
            "all_required_suites_passed": True,
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "video_depth_anything_pose_authority": False,
            "native_estimated_intrinsics_are_canonical_authority": False,
            "native_metric_depth_establishes_physical_metric_scale": False,
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
            "i5_8_complete": True,
            "i5_9_authorized": True,
            "i5_9_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "mp_sfm_authorized": False,
            "threshold_relaxation_authorized": False,
            "fusion_capacity_increase_authorized": False,
            "production_default_modified": False,
        },
        "roadmap_guard": {
            "i5_9_remaining_depth_provider_robustness_authorized_not_started": True,
            "i5_10_mp_sfm_deferred": True,
            "i5_11_oom_disk_crash_recovery_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_auto_deferred": True,
            "i5_14_final_i5_closure_deferred": True,
        },
        "next": "I5.9 Remaining depth-provider robustness.",
    }


__all__ = [
    "I58C_SCHEMA_VERSION", "I58C_PHASE", "I58C_RECEIPT_PHASE",
    "I58C_I58A_FROZEN_SHA256", "I58C_I58B_FROZEN_SHA256",
    "I58C_VDA_CHECKPOINT_SHA256", "I58C_VDA_SOURCE_REVISION",
    "validate_i5_8b_for_i5_8c", "validate_i5_8c_receipt",
    "build_i5_8c_video_depth_anything_robustness_closure",
]
