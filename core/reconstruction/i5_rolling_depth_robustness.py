"""I5.7A RollingDepth robustness admission and envelope design.

I5.7A is execution-free. It binds the frozen I5.6 closure, the exact I5.6A
Castle/SfM admission matrix, and the formally closed I4.3G3C temporal evidence
to define the provider-specific RollingDepth robustness replay envelope for
I5.7B.

No RollingDepth inference, SfM/MVS rerun, provider comparison, threshold tuning,
Auto, Hybrid, MP-SfM, metric-scale promotion, or production-default mutation is
performed here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_mvs_robustness import I56A_SCENARIO_IDS, validate_i5_6a_for_i5_6b
from core.reconstruction.rolling_depth_dense_provider import (
    I43D2_CANONICAL_DEPTH_POLARITY,
    I43D2_NATIVE_DEPTH_POLARITY,
    I43D2_NATIVE_DEPTH_SEMANTICS,
    I43D2_NATIVE_TO_CANONICAL_MULTIPLIER,
    I43D2_PROVIDER_ID,
    RollingDepthProductionConfig,
)
from core.reconstruction.single_image_provider_qualification import I43G3C_FROZEN_CLOSURE_SHA256

I57A_SCHEMA_VERSION = 1
I57A_PHASE = "I5.7A"
I57A_RECEIPT_PHASE = "I5.7A_ADMISSION_RECEIPT"
I57A_PROVIDER_IMPLEMENTATION = "rolling_depth"
I57A_PROVIDER_ID = I43D2_PROVIDER_ID
I57A_I56A_FROZEN_SHA256 = "506fb778d82ddc46350cbe4922ebe8a2ba43e0df4fa7818687a5768ebe0afc7a"
I57A_I56C_FROZEN_SHA256 = "448414d95e4e9c87b77ca5b3a5b94f6b2b22bb595d1ec81df8f2ec48c1424ccb"
I57A_SCENARIO_IDS = I56A_SCENARIO_IDS
I57A_G3_SCENARIOS = (
    "g3_temporal_static_sweep",
    "g3_temporal_motion",
    "g3_temporal_low_texture",
)
I57A_FROZEN_G3_ROLLING_DEPTH_STATUSES = {
    "g3_temporal_static_sweep": "degraded",
    "g3_temporal_motion": "rejected",
    "g3_temporal_low_texture": "pass",
}

_SCENARIO_INTERPRETATION = {
    "castle_control": "temporal_sequence_control",
    "castle_low_overlap": "temporal_support_spacing_stress_not_mvs_overlap_equivalence",
    "castle_weak_parallax": "narrow_baseline_temporal_geometry_stress",
    "castle_overexposure": "photometric_overexposure_temporal_stress",
    "castle_symmetry": "registered_subset_symmetry_temporal_stress",
    "castle_dynamic_proxy": "dynamic_motion_boundary_temporal_stress",
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
class I57AAdmissionPolicy:
    scenario_ids: tuple[str, ...] = I57A_SCENARIO_IDS
    provider_id: str = I57A_PROVIDER_ID
    provider_implementation: str = I57A_PROVIDER_IMPLEMENTATION
    stateful_temporal_sequence: bool = True
    source_order_required: bool = True
    sequence_alignment: str = "one_global_affine_to_authoritative_sfm"
    per_frame_alignment: bool = False
    default_device: str = "cuda:0"
    default_precision: str = "fp16"
    sequential_execution: bool = True
    permanent_global_view_cap: int | None = None
    result_driven_retuning: bool = False
    provider_comparison: bool = False

    def __post_init__(self) -> None:
        if tuple(self.scenario_ids) != I57A_SCENARIO_IDS:
            raise ReconstructionValidationError("I5.7A scenario matrix is frozen to the six admitted Castle cases.")
        if self.provider_id != I57A_PROVIDER_ID or self.provider_implementation != I57A_PROVIDER_IMPLEMENTATION:
            raise ReconstructionValidationError("I5.7A must bind the existing RollingDepth production provider.")
        if self.stateful_temporal_sequence is not True or self.source_order_required is not True:
            raise ReconstructionValidationError("I5.7A must preserve stateful ordered temporal-sequence semantics.")
        if self.sequence_alignment != "one_global_affine_to_authoritative_sfm" or self.per_frame_alignment is not False:
            raise ReconstructionValidationError("I5.7A must preserve one sequence-global RollingDepth alignment and forbid per-frame fitting.")
        if self.default_device != "cuda:0" or self.default_precision != "fp16":
            raise ReconstructionValidationError("I5.7A cannot silently change the frozen production device/precision defaults.")
        if self.sequential_execution is not True:
            raise ReconstructionValidationError("I5.7A formal replay is sequential by design.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I5.7A cannot introduce a permanent global view cap.")
        if self.result_driven_retuning is not False or self.provider_comparison is not False:
            raise ReconstructionValidationError("I5.7A forbids result-driven retuning and provider comparison.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_ids": list(self.scenario_ids),
            "provider_id": self.provider_id,
            "provider_implementation": self.provider_implementation,
            "production_adapter": "RollingDepthProductionProviderAdapter",
            "stateful_temporal_sequence": self.stateful_temporal_sequence,
            "source_order_required": self.source_order_required,
            "sequence_alignment": self.sequence_alignment,
            "per_frame_alignment": self.per_frame_alignment,
            "default_device": self.default_device,
            "default_precision": self.default_precision,
            "production_config": RollingDepthProductionConfig().as_dict(),
            "sequential_execution": self.sequential_execution,
            "permanent_global_view_cap": self.permanent_global_view_cap,
            "result_driven_retuning": self.result_driven_retuning,
            "provider_comparison": self.provider_comparison,
        }


def validate_i5_6c_for_i5_7(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.6C closure")
    _require(payload, "phase", "I5.6C", label="I5.6C closure")
    _require(payload, "status", "mvs_robustness_qualified_and_closed", label="I5.6C closure")
    _require(payload, "pass", True, label="I5.6C closure")
    scope = _mapping(payload.get("scope"), label="I5.6C scope")
    _require(scope, "evidence_only_qualification_and_closure", True, label="I5.6C scope")
    for key in (
        "sfm_rerun", "bundle_adjustment_rerun", "mvs_rerun", "patch_match_rerun",
        "stereo_fusion_rerun", "provider_comparison_started", "da3_rerun",
        "thresholds_modified", "production_default_modified", "result_driven_retuning",
        "retry_until_pass", "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.6C scope")
    authority = _mapping(payload.get("authority"), label="I5.6C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.6C authority")
    _require(authority, "authoritative_sfm_modified", False, label="I5.6C authority")
    _require(authority, "single_authoritative_sfm_frame_preserved", True, label="I5.6C authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.6C authority")
    for key in (
        "provider_promoted", "provider_ranking_created", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.6C authority")
    decision = _mapping(payload.get("decision"), label="I5.6C decision")
    _require(decision, "i5_6_complete", True, label="I5.6C decision")
    _require(decision, "i5_7_authorized", True, label="I5.6C decision")
    _require(decision, "i5_7_started", False, label="I5.6C decision")
    for key in (
        "provider_promoted", "provider_ranking_created", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized",
        "mp_sfm_authorized", "threshold_relaxation_authorized", "production_default_modified",
    ):
        _require(decision, key, False, label="I5.6C decision")
    roadmap = _mapping(payload.get("roadmap_guard"), label="I5.6C roadmap guard")
    _require(roadmap, "i5_7_rollingdepth_authorized", True, label="I5.6C roadmap guard")
    _require(roadmap, "i5_8_video_depth_anything_not_started", True, label="I5.6C roadmap guard")
    _require(roadmap, "i5_10_mp_sfm_deferred", True, label="I5.6C roadmap guard")
    _require(roadmap, "i5_12_hybrid_deferred_if_justified", True, label="I5.6C roadmap guard")
    _require(roadmap, "i5_13_auto_deferred", True, label="I5.6C roadmap guard")
    return deepcopy(dict(payload))


def validate_i4_3g3c_for_i5_7(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I4.3G3C closure")
    _require(payload, "phase", "I4.3G3C", label="I4.3G3C closure")
    _require(payload, "status", "formally_closed", label="I4.3G3C closure")
    _require(payload, "pass", True, label="I4.3G3C closure")
    _require(payload, "scenario_count", 3, label="I4.3G3C closure")
    _require(payload, "all_required_temporal_scenarios_accounted", True, label="I4.3G3C closure")
    _require(payload, "all_required_manual_reviews_complete", True, label="I4.3G3C closure")
    synthesis = _mapping(payload.get("scenario_synthesis"), label="I4.3G3C synthesis")
    if set(synthesis) != set(I57A_G3_SCENARIOS):
        raise ReconstructionValidationError("I5.7A requires exactly the three frozen I4.3G3C temporal scenarios.")
    for sid, expected in I57A_FROZEN_G3_ROLLING_DEPTH_STATUSES.items():
        node = _mapping(synthesis.get(sid), label=f"I4.3G3C {sid}")
        statuses = _mapping(node.get("production_statuses"), label=f"I4.3G3C {sid} production statuses")
        _require(statuses, "rolling_depth", expected, label=f"I4.3G3C {sid} production statuses")
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
        "authoritative_sfm_modified", "provider_selected", "provider_promoted",
        "auto_policy_implemented", "hybrid_authorized", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I4.3G3C authority")
    return deepcopy(dict(payload))


def validate_i5_7a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I57A_SCHEMA_VERSION, label="I5.7A receipt")
    _require(payload, "phase", I57A_RECEIPT_PHASE, label="I5.7A receipt")
    _require(payload, "status", "pass", label="I5.7A receipt")
    _require(payload, "pass", True, label="I5.7A receipt")
    scope = _mapping(payload.get("scope"), label="I5.7A receipt scope")
    for key in (
        "focused_i5_7a_tests", "rolling_depth_provider_regression", "temporal_g3_regression",
        "i5_6_predecessor_regression", "authority_guard_regression", "roadmap_guard_regression",
        "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.7A receipt scope")
    for key in (
        "rolling_depth_inference", "sfm_rerun", "mvs_rerun", "provider_comparison_started",
        "thresholds_modified", "production_default_modified", "result_driven_retuning",
        "retry_until_pass", "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.7A receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.7A receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(v, int) and v >= 0 for v in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.7A receipt test counts must be non-negative integers.")
    if focused < 10 or predecessor < 20 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.7A receipt coverage is below the admission floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.7A receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.7A receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def build_i5_7a_rolling_depth_robustness_admission(
    *,
    i5_6a_admission: Mapping[str, Any],
    i5_6a_binding: I51EvidenceBinding,
    i5_6c_closure: Mapping[str, Any],
    i5_6c_binding: I51EvidenceBinding,
    i4_3g3c_closure: Mapping[str, Any],
    i4_3g3c_binding: I51EvidenceBinding,
    admission_receipt: Mapping[str, Any],
    admission_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    admission = validate_i5_6a_for_i5_6b(i5_6a_admission)
    closure = validate_i5_6c_for_i5_7(i5_6c_closure)
    g3 = validate_i4_3g3c_for_i5_7(i4_3g3c_closure)
    receipt = validate_i5_7a_receipt(admission_receipt)
    _validate_binding(i5_6a_binding, label="I5.7A I5.6A", expected_sha256=I57A_I56A_FROZEN_SHA256)
    _validate_binding(i5_6c_binding, label="I5.7A I5.6C", expected_sha256=I57A_I56C_FROZEN_SHA256)
    _validate_binding(i4_3g3c_binding, label="I5.7A I4.3G3C", expected_sha256=I43G3C_FROZEN_CLOSURE_SHA256)
    _validate_binding(admission_receipt_binding, label="I5.7A receipt")

    matrix = _mapping(admission.get("scenario_matrix"), label="I5.6A scenario matrix")
    scenario_matrix: dict[str, Any] = {}
    for sid in I57A_SCENARIO_IDS:
        source = _mapping(matrix.get(sid), label=f"I5.6A {sid}")
        registered = int(source.get("registered_images", 0))
        input_images = int(source.get("input_images", 0))
        if registered < 2 or input_images < registered:
            raise ReconstructionValidationError(f"I5.7A {sid} lacks reusable authoritative temporal SfM coverage.")
        scenario_matrix[sid] = {
            "scenario_id": sid,
            "admitted_for_i5_7b": True,
            "rolling_depth_execution_required": True,
            "scientific_interpretation": _SCENARIO_INTERPRETATION[sid],
            "input_images": input_images,
            "registered_images": registered,
            "quality_status": source.get("quality_status"),
            "source_registered_scope_only": sid == "castle_symmetry",
            "same_authoritative_pycolmap_sfm_reused": True,
            "source_rematerialization_authorized": False,
            "sfm_rerun_authorized": False,
            "provider_comparison_authorized": False,
            "mvs_specific_metrics_are_rolling_depth_criteria": False,
        }

    g3_synthesis = _mapping(g3.get("scenario_synthesis"), label="I4.3G3C synthesis")
    historical = {
        sid: {
            "scenario_id": sid,
            "rolling_depth_production_status": _mapping(g3_synthesis[sid].get("production_statuses"), label=f"{sid} statuses")["rolling_depth"],
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "historical_result_overwritten": False,
            "rerun_authorized_by_i5_7a": False,
        }
        for sid in I57A_G3_SCENARIOS
    }

    policy = I57AAdmissionPolicy().as_dict()
    return {
        "schema_version": I57A_SCHEMA_VERSION,
        "phase": I57A_PHASE,
        "status": "rolling_depth_robustness_admission_complete",
        "pass": True,
        "scope": {
            "admission_and_envelope_design_only": True,
            "rolling_depth_inference": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "mvs_rerun": False,
            "da3_rerun": False,
            "video_depth_anything_rerun": False,
            "provider_comparison_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "evidence_bindings": {
            "i5_6a_castle_sfm_admission": i5_6a_binding.as_dict(),
            "i5_6c_mvs_closure": i5_6c_binding.as_dict(),
            "i4_3g3c_temporal_closure": i4_3g3c_binding.as_dict(),
            "i5_7a_admission_receipt": admission_receipt_binding.as_dict(),
        },
        "provider_contract": {
            "provider_id": I57A_PROVIDER_ID,
            "implementation": I57A_PROVIDER_IMPLEMENTATION,
            "family": "temporal_depth",
            "native_depth_semantics": I43D2_NATIVE_DEPTH_SEMANTICS,
            "native_depth_polarity": I43D2_NATIVE_DEPTH_POLARITY,
            "canonical_depth_polarity": I43D2_CANONICAL_DEPTH_POLARITY,
            "native_to_canonical_multiplier": I43D2_NATIVE_TO_CANONICAL_MULTIPLIER,
            "stateful_sequence": True,
            "ordered_registered_sequence_required": True,
            "sequence_alignment": "global_affine_to_authoritative_sfm",
            "per_frame_alignment": False,
            "sequence_affine_establishes_metric_scale": False,
            "runtime_isolated_external_process": True,
            "persistent_multi_job_model_server": False,
            "provider_specific_cancellation_contract_qualified": False,
            "runtime_preflight_required_for_i5_7b": True,
        },
        "execution_policy": policy,
        "scenario_matrix": scenario_matrix,
        "historical_temporal_anchors": historical,
        "excluded_or_deferred_scenarios": {
            "castle_blur": {
                "status": "BLOCKED_UPSTREAM",
                "reason": "authoritative_pycolmap_sfm_unavailable_after_pre_sfm_ingestion_rejection",
                "rolling_depth_failure": False,
            },
            "castle_underexposure": {
                "status": "BLOCKED_UPSTREAM",
                "reason": "authoritative_pycolmap_sfm_unavailable_after_pre_sfm_ingestion_rejection",
                "rolling_depth_failure": False,
            },
            "castle_clutter": {
                "status": "NOT_FORMALLY_ADMITTED",
                "reason": "clutter_semantic_stress_not_revalidated_in_frozen_predecessor",
                "rolling_depth_failure": False,
            },
            "kitti_forward_motion": {
                "status": "BLOCKED_UPSTREAM",
                "reason": "incremental_mapping_produced_no_authoritative_sparse_model_or_registered_cameras",
                "rolling_depth_failure": False,
                "mp_sfm_not_pulled_forward": True,
            },
            "difficult_handheld_horse": {
                "status": "DEFERRED_NOT_ADMITTED",
                "reason": "scientifically_applicable_but_exact_temporal_source_and_authoritative_sfm_binding_not_present_in_i5_6a_admission_matrix",
                "rolling_depth_failure": False,
                "provenance_not_invented": True,
            },
        },
        "i5_7b_evidence_contract": {
            "new_timestamped_attempt_directory_per_real_attempt": True,
            "failed_interrupted_blocked_rejected_attempts_preserved": True,
            "runtime_preflight_and_pinned_source_checkpoint_identity_required": True,
            "source_and_authoritative_sfm_integrity_revalidation_required": True,
            "ordered_temporal_input_identity_required": True,
            "native_provider_evidence_required": True,
            "global_sequence_alignment_evidence_required": True,
            "canonical_fusion_evidence_required_when_alignment_accepted": True,
            "wall_time_observation_required": True,
            "peak_vram_observation_required_when_available": True,
            "peak_host_ram_observation_required_when_available": True,
            "missing_resource_or_performance_metrics_must_remain_missing": True,
            "pass_degraded_rejected_are_all_valid_terminal_scientific_outcomes": True,
            "global_scalar_score_created": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
        },
        "known_qualification_gaps": {
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
        },
        "regression_summary": {
            "focused_i5_7a_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "rolling_depth_pose_authority": False,
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
            "i5_6_complete": True,
            "i5_7_started": True,
            "i5_7a_complete": True,
            "i5_7b_authorized": True,
            "i5_7b_started": False,
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
            "i5_7b_real_rolling_depth_replay_authorized_not_started": True,
            "i5_8_video_depth_anything_not_started": True,
            "i5_10_mp_sfm_deferred": True,
            "i5_11_oom_disk_crash_recovery_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_auto_deferred": True,
        },
        "next": "I5.7B real frozen-foundation RollingDepth robustness replay.",
    }


__all__ = [
    "I57A_SCHEMA_VERSION", "I57A_PHASE", "I57A_RECEIPT_PHASE", "I57A_PROVIDER_IMPLEMENTATION",
    "I57A_PROVIDER_ID", "I57A_I56A_FROZEN_SHA256", "I57A_I56C_FROZEN_SHA256",
    "I57A_SCENARIO_IDS", "I57A_G3_SCENARIOS", "I57A_FROZEN_G3_ROLLING_DEPTH_STATUSES",
    "I57AAdmissionPolicy", "validate_i5_6c_for_i5_7", "validate_i4_3g3c_for_i5_7",
    "validate_i5_7a_receipt", "build_i5_7a_rolling_depth_robustness_admission",
]
