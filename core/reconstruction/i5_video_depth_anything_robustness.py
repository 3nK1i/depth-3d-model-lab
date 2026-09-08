"""I5.8A Video Depth Anything robustness admission and envelope design.

I5.8A is execution-free. It consumes the formally closed I5.7 artifact, the
exact I5.6A Castle/PyCOLMAP admission matrix, and the frozen I4.3G3C temporal
qualification evidence to define a provider-specific real replay envelope for
Video Depth Anything (VDA).

Unlike RollingDepth, the existing VDA production path is a stateful streaming
cache that emits one native depth map per ordered registered SfM view and then
aligns each registered frame independently into the authoritative PyCOLMAP SfM
gauge. I5.8A freezes that existing behavior rather than forcing VDA into the
RollingDepth sequence-global alignment model.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_mvs_robustness import I56A_SCENARIO_IDS, validate_i5_6a_for_i5_6b
from core.reconstruction.single_image_provider_qualification import I43G3C_FROZEN_CLOSURE_SHA256
from core.reconstruction.video_depth_anything_dense_provider import (
    I43D1_PROVIDER_ID,
    VideoDepthAnythingProductionConfig,
)

I58A_SCHEMA_VERSION = 1
I58A_PHASE = "I5.8A"
I58A_RECEIPT_PHASE = "I5.8A_ADMISSION_RECEIPT"
I58A_PROVIDER_IMPLEMENTATION = "video_depth_anything"
I58A_PROVIDER_ID = I43D1_PROVIDER_ID
I58A_I57C_FROZEN_SHA256 = "b76beeb82db0a4477fdea9212ac3f693725d9b57c8c3fafdf307eae4438a50fc"
I58A_I56A_FROZEN_SHA256 = "506fb778d82ddc46350cbe4922ebe8a2ba43e0df4fa7818687a5768ebe0afc7a"
I58A_G3C_FROZEN_SHA256 = I43G3C_FROZEN_CLOSURE_SHA256
I58A_SCENARIO_IDS = I56A_SCENARIO_IDS
I58A_G3_SCENARIOS = (
    "g3_temporal_static_sweep",
    "g3_temporal_motion",
    "g3_temporal_low_texture",
)
I58A_FROZEN_G3_VDA_STATUSES = {
    "g3_temporal_static_sweep": "pass",
    "g3_temporal_motion": "degraded",
    "g3_temporal_low_texture": "pass",
}

I58A_RUNTIME_VARIANTS = {
    "efficient": {
        "encoder": "vits",
        "checkpoint": "checkpoints/video-depth-anything/metric_video_depth_anything_vits.pth",
        "model_label": "Metric Video Depth Anything Small",
        "input_size": 392,
        "max_resolution": 960,
    },
    "balanced": {
        "encoder": "vitb",
        "checkpoint": "checkpoints/video-depth-anything/metric_video_depth_anything_vitb.pth",
        "model_label": "Metric Video Depth Anything Base",
        "input_size": 448,
        "max_resolution": 1280,
    },
    "ultra": {
        "encoder": "vitl",
        "checkpoint": "checkpoints/video-depth-anything/metric_video_depth_anything_vitl.pth",
        "model_label": "Metric Video Depth Anything Large",
        "input_size": 518,
        "max_resolution": 1920,
    },
}

_SCENARIO_INTERPRETATION = {
    "castle_control": "streaming_cache_registered_sequence_control",
    "castle_low_overlap": "registered_keyframe_spacing_and_cache_continuity_stress_not_mvs_overlap_equivalence",
    "castle_weak_parallax": "narrow_baseline_streaming_temporal_stability_stress",
    "castle_overexposure": "photometric_overexposure_streaming_temporal_stress",
    "castle_symmetry": "registered_subset_repetitive_structure_streaming_temporal_stress",
    "castle_dynamic_proxy": "dynamic_motion_boundary_streaming_cache_stress",
}


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


def _validate_binding(binding: I51EvidenceBinding, *, label: str, expected_sha256: str | None = None) -> None:
    if not str(binding.path).strip() or not _valid_sha(binding.sha256):
        raise ReconstructionValidationError(f"{label} evidence binding is malformed.")
    if expected_sha256 is not None and binding.sha256.lower() != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"{label} SHA256 drifted from the frozen predecessor: expected={expected_sha256} actual={binding.sha256}"
        )


@dataclass(frozen=True, slots=True)
class I58AAdmissionPolicy:
    scenario_ids: tuple[str, ...] = I58A_SCENARIO_IDS
    provider_id: str = I58A_PROVIDER_ID
    provider_implementation: str = I58A_PROVIDER_IMPLEMENTATION
    temporal_mode: str = "official_streaming_cache"
    stateful_temporal_sequence: bool = True
    source_order_required: bool = True
    stable_pixel_grid_required: bool = True
    worker_model_loads_once_per_sequence: bool = True
    warmup_reset_before_user_sequence: bool = True
    monotonic_worker_frame_ids: bool = True
    canonical_alignment: str = "per_registered_frame_affine_to_authoritative_sfm"
    per_frame_alignment_performed: bool = True
    sequence_global_alignment_performed: bool = False
    runtime_profile_resolution: str = "hardware_resolved_compute_profile_auto"
    checkpoint_must_match_resolved_profile: bool = True
    sequential_scenario_execution: bool = True
    permanent_global_view_cap: int | None = None
    result_driven_retuning: bool = False
    provider_comparison: bool = False

    def __post_init__(self) -> None:
        if tuple(self.scenario_ids) != I58A_SCENARIO_IDS:
            raise ReconstructionValidationError("I5.8A scenario matrix is frozen to the six admitted Castle cases.")
        if self.provider_id != I58A_PROVIDER_ID or self.provider_implementation != I58A_PROVIDER_IMPLEMENTATION:
            raise ReconstructionValidationError("I5.8A must bind the existing Video Depth Anything production provider.")
        if self.temporal_mode != "official_streaming_cache":
            raise ReconstructionValidationError("I5.8A must preserve the existing VDA official streaming-cache mode.")
        if not all((
            self.stateful_temporal_sequence,
            self.source_order_required,
            self.stable_pixel_grid_required,
            self.worker_model_loads_once_per_sequence,
            self.warmup_reset_before_user_sequence,
            self.monotonic_worker_frame_ids,
        )):
            raise ReconstructionValidationError("I5.8A must preserve VDA stateful ordered streaming semantics.")
        if self.canonical_alignment != "per_registered_frame_affine_to_authoritative_sfm":
            raise ReconstructionValidationError("I5.8A must preserve VDA per-registered-frame SfM alignment.")
        if self.per_frame_alignment_performed is not True or self.sequence_global_alignment_performed is not False:
            raise ReconstructionValidationError("I5.8A cannot replace VDA per-frame alignment with RollingDepth-style global alignment.")
        if self.runtime_profile_resolution != "hardware_resolved_compute_profile_auto" or self.checkpoint_must_match_resolved_profile is not True:
            raise ReconstructionValidationError("I5.8A must preserve hardware-resolved production profile/checkpoint binding.")
        if self.sequential_scenario_execution is not True:
            raise ReconstructionValidationError("I5.8A formal replay is sequential by design.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I5.8A cannot introduce a permanent global view cap.")
        if self.result_driven_retuning is not False or self.provider_comparison is not False:
            raise ReconstructionValidationError("I5.8A forbids result-driven retuning and provider comparison.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_ids": list(self.scenario_ids),
            "provider_id": self.provider_id,
            "provider_implementation": self.provider_implementation,
            "production_adapter": "VideoDepthAnythingProductionProviderAdapter",
            "temporal_mode": self.temporal_mode,
            "stateful_temporal_sequence": self.stateful_temporal_sequence,
            "source_order_required": self.source_order_required,
            "stable_pixel_grid_required": self.stable_pixel_grid_required,
            "worker_model_loads_once_per_sequence": self.worker_model_loads_once_per_sequence,
            "warmup_reset_before_user_sequence": self.warmup_reset_before_user_sequence,
            "monotonic_worker_frame_ids": self.monotonic_worker_frame_ids,
            "canonical_alignment": self.canonical_alignment,
            "per_frame_alignment_performed": self.per_frame_alignment_performed,
            "sequence_global_alignment_performed": self.sequence_global_alignment_performed,
            "runtime_profile_resolution": self.runtime_profile_resolution,
            "checkpoint_must_match_resolved_profile": self.checkpoint_must_match_resolved_profile,
            "runtime_variants": deepcopy(I58A_RUNTIME_VARIANTS),
            "production_config": VideoDepthAnythingProductionConfig().as_dict(),
            "sequential_scenario_execution": self.sequential_scenario_execution,
            "permanent_global_view_cap": self.permanent_global_view_cap,
            "result_driven_retuning": self.result_driven_retuning,
            "provider_comparison": self.provider_comparison,
        }


def validate_i5_7c_for_i5_8(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.7C closure")
    _require(payload, "phase", "I5.7C", label="I5.7C closure")
    _require(payload, "status", "rolling_depth_robustness_qualified_and_closed", label="I5.7C closure")
    _require(payload, "pass", True, label="I5.7C closure")
    _require(payload, "quality_status", "pass", label="I5.7C closure")
    scope = _mapping(payload.get("scope"), label="I5.7C scope")
    _require(scope, "evidence_only_qualification_and_closure", True, label="I5.7C scope")
    for key in (
        "rolling_depth_inference", "sfm_rerun", "bundle_adjustment_rerun", "mvs_rerun",
        "provider_comparison_started", "video_depth_anything_rerun", "thresholds_modified",
        "production_default_modified", "result_driven_retuning", "retry_until_pass",
        "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.7C scope")
    qualification = _mapping(payload.get("qualification"), label="I5.7C qualification")
    _require(qualification, "universal_rolling_depth_robustness_claimed", False, label="I5.7C qualification")
    _require(qualification, "provider_ranking_claimed", False, label="I5.7C qualification")
    _require(qualification, "metric_scale_claimed", False, label="I5.7C qualification")
    authority = _mapping(payload.get("authority"), label="I5.7C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.7C authority")
    _require(authority, "single_authoritative_sfm_frame_preserved", True, label="I5.7C authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.7C authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "provider_ranking_created",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.7C authority")
    decision = _mapping(payload.get("decision"), label="I5.7C decision")
    _require(decision, "i5_7_complete", True, label="I5.7C decision")
    _require(decision, "i5_8_authorized", True, label="I5.7C decision")
    _require(decision, "i5_8_started", False, label="I5.7C decision")
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized", "mp_sfm_authorized",
        "threshold_relaxation_authorized", "production_default_modified",
    ):
        _require(decision, key, False, label="I5.7C decision")
    roadmap = _mapping(payload.get("roadmap_guard"), label="I5.7C roadmap guard")
    _require(roadmap, "i5_8_video_depth_anything_authorized_not_started", True, label="I5.7C roadmap guard")
    _require(payload, "next", "I5.8 Video Depth Anything robustness.", label="I5.7C closure")
    return deepcopy(dict(payload))


def validate_i4_3g3c_for_i5_8(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I4.3G3C closure")
    _require(payload, "phase", "I4.3G3C", label="I4.3G3C closure")
    _require(payload, "status", "formally_closed", label="I4.3G3C closure")
    _require(payload, "pass", True, label="I4.3G3C closure")
    _require(payload, "scenario_count", 3, label="I4.3G3C closure")
    _require(payload, "all_required_temporal_scenarios_accounted", True, label="I4.3G3C closure")
    _require(payload, "all_required_manual_reviews_complete", True, label="I4.3G3C closure")
    synthesis = _mapping(payload.get("scenario_synthesis"), label="I4.3G3C synthesis")
    if set(synthesis) != set(I58A_G3_SCENARIOS):
        raise ReconstructionValidationError("I5.8A requires exactly the three frozen I4.3G3C temporal scenarios.")
    for sid, expected in I58A_FROZEN_G3_VDA_STATUSES.items():
        node = _mapping(synthesis.get(sid), label=f"I4.3G3C {sid}")
        statuses = _mapping(node.get("production_statuses"), label=f"I4.3G3C {sid} production statuses")
        _require(statuses, "video_depth_anything", expected, label=f"I4.3G3C {sid} production statuses")
        _require(node, "manual_review_complete", True, label=f"I4.3G3C {sid}")
        _require(node, "matched_view_diagnostic_secondary", True, label=f"I4.3G3C {sid}")
        _require(node, "production_rejection_overridden", False, label=f"I4.3G3C {sid}")
    decision = _mapping(payload.get("decision"), label="I4.3G3C decision")
    _require(decision, "production_native_evidence_primary", True, label="I4.3G3C decision")
    _require(decision, "scenario_specific_verdicts_preserved", True, label="I4.3G3C decision")
    _require(decision, "quality_failures_preserved", True, label="I4.3G3C decision")
    for key in (
        "provider_selected", "provider_promoted", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(decision, key, False, label="I4.3G3C decision")
    authority = _mapping(payload.get("authority"), label="I4.3G3C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I4.3G3C authority")
    _require(authority, "permanent_global_view_cap", None, label="I4.3G3C authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented",
        "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I4.3G3C authority")
    return deepcopy(dict(payload))


def validate_i5_8a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I58A_SCHEMA_VERSION, label="I5.8A receipt")
    _require(payload, "phase", I58A_RECEIPT_PHASE, label="I5.8A receipt")
    _require(payload, "status", "pass", label="I5.8A receipt")
    _require(payload, "pass", True, label="I5.8A receipt")
    scope = _mapping(payload.get("scope"), label="I5.8A receipt scope")
    for key in (
        "focused_i5_8a_tests", "vda_provider_regression", "temporal_g3_regression",
        "i5_6a_source_binding_regression", "i5_7c_authorization_regression",
        "authority_guard_regression", "roadmap_guard_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.8A receipt scope")
    for key in (
        "video_depth_anything_inference", "sfm_rerun", "mvs_rerun", "provider_comparison_started",
        "thresholds_modified", "production_default_modified", "result_driven_retuning", "retry_until_pass",
        "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.8A receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.8A receipt counts")
    focused = int(counts.get("focused_tests", 0))
    predecessor = int(counts.get("predecessor_total_tests", 0))
    combined = int(counts.get("combined_exercised_tests", 0))
    if focused <= 0 or predecessor <= 0 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.8A receipt test counts are inconsistent.")
    rcs = _mapping(payload.get("return_codes"), label="I5.8A receipt return codes")
    for key in (
        "focused", "predecessor_unit", "predecessor_integration", "predecessor_stress",
        "import_probe", "runner_help_probe",
    ):
        if int(rcs.get(key, -1)) != 0:
            raise ReconstructionValidationError(f"I5.8A receipt return code is non-zero: {key}")
    return deepcopy(dict(payload))


def build_i5_8a_video_depth_anything_robustness_admission(
    *,
    i5_7c_closure: Mapping[str, Any],
    i5_7c_binding: I51EvidenceBinding,
    i5_6a_admission: Mapping[str, Any],
    i5_6a_binding: I51EvidenceBinding,
    i4_3g3c_closure: Mapping[str, Any],
    i4_3g3c_binding: I51EvidenceBinding,
    admission_receipt: Mapping[str, Any],
    admission_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    validate_i5_7c_for_i5_8(i5_7c_closure)
    i56a = validate_i5_6a_for_i5_6b(i5_6a_admission)
    g3c = validate_i4_3g3c_for_i5_8(i4_3g3c_closure)
    receipt = validate_i5_8a_receipt(admission_receipt)
    _validate_binding(i5_7c_binding, label="I5.8A I5.7C", expected_sha256=I58A_I57C_FROZEN_SHA256)
    _validate_binding(i5_6a_binding, label="I5.8A I5.6A", expected_sha256=I58A_I56A_FROZEN_SHA256)
    _validate_binding(i4_3g3c_binding, label="I5.8A I4.3G3C", expected_sha256=I58A_G3C_FROZEN_SHA256)
    _validate_binding(admission_receipt_binding, label="I5.8A admission receipt")

    policy = I58AAdmissionPolicy().as_dict()
    source_matrix = _mapping(i56a.get("scenario_matrix"), label="I5.6A scenario matrix")
    scenario_matrix: dict[str, Any] = {}
    for sid in I58A_SCENARIO_IDS:
        src = _mapping(source_matrix.get(sid), label=f"I5.6A {sid}")
        registered = int(src.get("registered_images", src.get("registered_image_count", 0)))
        input_images = int(src.get("input_images", 0))
        if registered < 2:
            raise ReconstructionValidationError(f"I5.8A {sid} has fewer than two authoritative registered views.")
        scenario_matrix[sid] = {
            "scenario_id": sid,
            "admitted_for_i5_8b": True,
            "video_depth_anything_execution_required": True,
            "scientific_interpretation": _SCENARIO_INTERPRETATION[sid],
            "source_registered_scope_only": True,
            "input_images": input_images,
            "registered_images": registered,
            "registration_ratio": float(src.get("registration_ratio", registered / float(max(1, input_images)))),
            "registered_images_sha256": src.get("registered_images_sha256"),
            "registered_image_names": deepcopy(src.get("registered_image_names", [])),
            "selected_sparse_sha256": src.get("selected_sparse_sha256"),
            "workspace_root": src.get("workspace_root"),
            "report_binding": deepcopy(src.get("report_binding")),
            "camera_pose_authority_owner": "pycolmap_sfm",
            "gauge": "arbitrary",
            "sfm_rehydrated_without_rerun": True,
            "ordered_registered_keyframe_stream": True,
            "original_full_video_continuity_claimed": False,
            "vda_streaming_cache_state_preserved_across_registered_views": True,
            "per_registered_frame_sfm_alignment_expected": True,
            "sequence_global_alignment_expected": False,
            "mvs_specific_metrics_are_video_depth_anything_criteria": False,
        }

    historical: dict[str, Any] = {}
    synthesis = _mapping(g3c.get("scenario_synthesis"), label="I4.3G3C synthesis")
    for sid in I58A_G3_SCENARIOS:
        statuses = _mapping(_mapping(synthesis[sid], label=sid).get("production_statuses"), label=f"{sid} statuses")
        historical[sid] = {
            "video_depth_anything_production_status": statuses["video_depth_anything"],
            "historical_result_overwritten": False,
            "rerun_authorized_by_i5_8a": False,
        }

    return {
        "schema_version": I58A_SCHEMA_VERSION,
        "phase": I58A_PHASE,
        "status": "video_depth_anything_robustness_scenarios_admitted",
        "pass": True,
        "scope": {
            "scenario_admission_only": True,
            "i5_7_closed_predecessor_reused": True,
            "i5_6a_castle_sfm_matrix_reused": True,
            "historical_g3_temporal_closure_reused": True,
            "video_depth_anything_inference": False,
            "rolling_depth_inference": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "mvs_rerun": False,
            "provider_comparison_started": False,
            "source_rematerialized": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "evidence_bindings": {
            "i5_7c_rolling_depth_closure": i5_7c_binding.as_dict(),
            "i5_6a_castle_sfm_admission": i5_6a_binding.as_dict(),
            "i4_3g3c_temporal_closure": i4_3g3c_binding.as_dict(),
            "i5_8a_admission_receipt": admission_receipt_binding.as_dict(),
        },
        "provider_contract": {
            **policy,
            "family": "temporal_depth",
            "execution": "external_process",
            "native_output": "depth_evidence",
            "sequence_scope": "all_registered_authoritative_sfm_views_in_source_order",
            "worker_session_id_fixed_per_sequence": True,
            "model_warmup_is_not_first_user_frame": True,
            "native_metric_depth_preserved_as_evidence": True,
            "native_estimated_intrinsics_preserved_as_evidence": True,
            "native_metric_depth_promoted_to_canonical_metric_scale": False,
            "native_estimated_intrinsics_consumed_for_canonical_projection": False,
            "canonical_projection_pose_source": "pycolmap_sfm",
            "provider_promotion_performed": False,
        },
        "scenario_matrix": scenario_matrix,
        "historical_temporal_anchors": historical,
        "excluded_or_deferred_scenarios": {
            "castle_blur": {
                "status": "BLOCKED_UPSTREAM",
                "reason": "authoritative_pycolmap_sfm_unavailable_after_blur_stress",
                "video_depth_anything_failure": False,
            },
            "castle_underexposure": {
                "status": "BLOCKED_UPSTREAM",
                "reason": "authoritative_pycolmap_sfm_unavailable_after_pre_sfm_ingestion_rejection",
                "video_depth_anything_failure": False,
            },
            "castle_clutter": {
                "status": "NOT_FORMALLY_ADMITTED",
                "reason": "clutter_semantic_stress_not_revalidated_in_frozen_predecessor",
                "video_depth_anything_failure": False,
            },
            "kitti_forward_motion": {
                "status": "BLOCKED_UPSTREAM",
                "reason": "incremental_mapping_produced_no_authoritative_sparse_model_or_registered_cameras",
                "video_depth_anything_failure": False,
                "mp_sfm_not_pulled_forward": True,
            },
            "difficult_handheld_horse": {
                "status": "DEFERRED_NOT_ADMITTED",
                "reason": "scientifically_applicable_but_exact_temporal_source_and_authoritative_sfm_binding_not_present_in_i5_6a_admission_matrix",
                "video_depth_anything_failure": False,
                "provenance_not_invented": True,
            },
        },
        "i5_8b_evidence_contract": {
            "new_timestamped_attempt_directory_per_real_attempt": True,
            "failed_interrupted_blocked_rejected_attempts_preserved": True,
            "runtime_preflight_required": True,
            "hardware_resolved_profile_recorded": True,
            "resolved_checkpoint_must_match_probed_checkpoint": True,
            "source_revision_and_checkpoint_identity_required": True,
            "source_and_authoritative_sfm_integrity_revalidation_required": True,
            "ordered_registered_sequence_identity_required": True,
            "stable_pixel_grid_and_single_camera_model_required": True,
            "warmup_reset_before_user_sequence_required": True,
            "single_worker_model_load_per_sequence_required": True,
            "native_vda_depth_stack_required": True,
            "native_frame_latency_records_required": True,
            "per_registered_frame_alignment_evidence_required": True,
            "sequence_global_alignment_forbidden": True,
            "canonical_fusion_evidence_required_when_minimum_alignment_support_is_met": True,
            "wall_time_observation_required": True,
            "peak_vram_observation_required_when_available": True,
            "peak_host_ram_observation_required_when_available": True,
            "missing_resource_or_performance_metrics_must_remain_missing": True,
            "pass_degraded_rejected_are_valid_terminal_scientific_outcomes": True,
            "rejected_requires_explicit_provider_quality_failure_not_runtime_or_integrity_failure": True,
            "runtime_or_integrity_failure_must_fail_attempt_nonzero": True,
            "retry_until_pass": False,
            "result_driven_retuning": False,
            "global_scalar_score_created": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
        },
        "known_qualification_gaps": {
            "historical_g3_motion_degraded_remains_active_evidence": True,
            "runtime_performance_envelope_complete": False,
            "resource_envelope_complete": False,
            "rtx_4090_qualified": False,
            "rtx_5090_qualified": False,
            "cross_gpu_equivalence_qualified": False,
            "long_sequence_resource_envelope_qualified": False,
            "provider_specific_cancellation_qualified": False,
            "ground_truth_dense_depth_available": False,
            "ground_truth_surface_accuracy_available": False,
            "metric_scale_available": False,
            "certified_metrology_available": False,
            "registered_keyframe_stream_is_not_claimed_to_equal_full_original_video_temporal_continuity": True,
        },
        "regression_summary": {
            "focused_i5_8a_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "video_depth_anything_pose_authority": False,
            "native_estimated_intrinsics_are_canonical_authority": False,
            "native_metric_depth_establishes_physical_metric_scale": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_7_complete": True,
            "i5_8_started": True,
            "i5_8a_complete": True,
            "i5_8b_authorized": True,
            "i5_8b_started": False,
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
            "production_default_modified": False,
        },
        "roadmap_guard": {
            "i5_8b_real_vda_replay_authorized_not_started": True,
            "i5_9_remaining_depth_provider_robustness_not_started": True,
            "i5_10_mp_sfm_deferred": True,
            "i5_11_oom_disk_crash_recovery_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_auto_deferred": True,
            "i5_14_final_i5_closure_deferred": True,
        },
        "next": "I5.8B real frozen-foundation Video Depth Anything robustness replay.",
    }


__all__ = [
    "I58A_SCHEMA_VERSION", "I58A_PHASE", "I58A_RECEIPT_PHASE",
    "I58A_PROVIDER_IMPLEMENTATION", "I58A_PROVIDER_ID", "I58A_I57C_FROZEN_SHA256",
    "I58A_I56A_FROZEN_SHA256", "I58A_G3C_FROZEN_SHA256", "I58A_SCENARIO_IDS",
    "I58A_G3_SCENARIOS", "I58A_FROZEN_G3_VDA_STATUSES", "I58A_RUNTIME_VARIANTS",
    "I58AAdmissionPolicy", "validate_i5_7c_for_i5_8", "validate_i4_3g3c_for_i5_8",
    "validate_i5_8a_receipt", "build_i5_8a_video_depth_anything_robustness_admission",
]
