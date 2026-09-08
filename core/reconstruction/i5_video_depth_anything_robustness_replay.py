"""I5.8B real frozen-foundation Video Depth Anything robustness replay contract.

I5.8B executes the six I5.8A-admitted Castle scenarios through the existing
I4.3D1 production Video Depth Anything adapter.  It does not change the provider,
SfM, MVS, thresholds, runtime profile resolver, or production defaults.  The
module validates real provider summaries and builds the immutable campaign-level
scientific replay artifact consumed later by I5.8C.

VDA-specific semantics are preserved:
* official streaming-cache state across the ordered registered sequence,
* hardware-resolved production profile/checkpoint,
* one native depth map / latency record per registered view,
* independent per-registered-frame affine alignment into PyCOLMAP SfM gauge,
* no RollingDepth-style sequence-global alignment,
* no promotion of model metric depth or estimated intrinsics to physical/camera
  authority.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import ceil, isfinite
from statistics import mean, median
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_video_depth_anything_robustness import (
    I58A_FROZEN_G3_VDA_STATUSES,
    I58A_PROVIDER_ID,
    I58A_PROVIDER_IMPLEMENTATION,
    I58A_RUNTIME_VARIANTS,
    I58A_SCENARIO_IDS,
)
from core.reconstruction.video_depth_anything_dense_provider import (
    I43D1_PHASE,
    I43D1_SCHEMA_VERSION,
    VideoDepthAnythingProductionConfig,
)

I58B_SCHEMA_VERSION = 1
I58B_PHASE = "I5.8B"
I58B_RECEIPT_PHASE = "I5.8B_REPLAY_RECEIPT"
I58B_I58A_FROZEN_SHA256 = "1d635cdcefe02e1ee3b66dfaa8af4df49fb952e58cbf4446b50c2d562369d90a"
I58B_SCENARIO_IDS = I58A_SCENARIO_IDS

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
            f"{label} SHA256 drifted from the frozen predecessor: "
            f"expected={expected_sha256} actual={binding.sha256}"
        )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _ratio_or_none(numerator: Any, denominator: Any) -> float | None:
    n = _float_or_none(numerator)
    d = _float_or_none(denominator)
    if n is None or d is None or d == 0.0:
        return None
    return n / d


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = max(0, min(len(ordered) - 1, ceil(q * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True, slots=True)
class I58BReplayPolicy:
    scenario_ids: tuple[str, ...] = I58B_SCENARIO_IDS
    provider_id: str = I58A_PROVIDER_ID
    provider_implementation: str = I58A_PROVIDER_IMPLEMENTATION
    temporal_mode: str = "official_streaming_cache"
    stateful_temporal_sequence: bool = True
    source_order_required: bool = True
    stable_pixel_grid_required: bool = True
    warmup_reset_before_user_sequence: bool = True
    worker_model_loads_once_per_sequence: bool = True
    monotonic_worker_frame_ids: bool = True
    canonical_alignment: str = "per_registered_frame_affine_to_authoritative_sfm"
    per_frame_alignment_performed: bool = True
    sequence_global_alignment_performed: bool = False
    runtime_profile_resolution: str = "hardware_resolved_compute_profile_auto"
    checkpoint_must_match_resolved_profile: bool = True
    explicit_provider_selection: bool = True
    fallback_allowed: bool = False
    sequential_scenario_execution: bool = True
    complete_registered_sequence_required: bool = True
    permanent_global_view_cap: int | None = None
    result_driven_retuning: bool = False
    retry_until_pass: bool = False
    provider_comparison: bool = False

    def __post_init__(self) -> None:
        if tuple(self.scenario_ids) != I58B_SCENARIO_IDS:
            raise ReconstructionValidationError("I5.8B scenario matrix drifted from I5.8A.")
        if self.provider_id != I58A_PROVIDER_ID or self.provider_implementation != I58A_PROVIDER_IMPLEMENTATION:
            raise ReconstructionValidationError("I5.8B must execute the admitted VDA production provider.")
        if self.temporal_mode != "official_streaming_cache":
            raise ReconstructionValidationError("I5.8B must preserve official VDA streaming-cache mode.")
        if not all((
            self.stateful_temporal_sequence,
            self.source_order_required,
            self.stable_pixel_grid_required,
            self.warmup_reset_before_user_sequence,
            self.worker_model_loads_once_per_sequence,
            self.monotonic_worker_frame_ids,
        )):
            raise ReconstructionValidationError("I5.8B must preserve stateful ordered VDA streaming semantics.")
        if self.canonical_alignment != "per_registered_frame_affine_to_authoritative_sfm":
            raise ReconstructionValidationError("I5.8B canonical alignment mode drifted.")
        if self.per_frame_alignment_performed is not True or self.sequence_global_alignment_performed is not False:
            raise ReconstructionValidationError("I5.8B cannot replace per-frame SfM alignment with global sequence alignment.")
        if self.runtime_profile_resolution != "hardware_resolved_compute_profile_auto" or self.checkpoint_must_match_resolved_profile is not True:
            raise ReconstructionValidationError("I5.8B must use hardware-resolved production profile/checkpoint binding.")
        if self.explicit_provider_selection is not True or self.fallback_allowed is not False:
            raise ReconstructionValidationError("I5.8B must explicitly select VDA with fallback disabled.")
        if self.sequential_scenario_execution is not True or self.complete_registered_sequence_required is not True:
            raise ReconstructionValidationError("I5.8B must execute each complete admitted sequence exactly once in order.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I5.8B cannot introduce a permanent global view cap.")
        if self.result_driven_retuning is not False or self.retry_until_pass is not False or self.provider_comparison is not False:
            raise ReconstructionValidationError("I5.8B forbids retuning, retry-until-pass, and provider comparison.")

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
            "warmup_reset_before_user_sequence": self.warmup_reset_before_user_sequence,
            "worker_model_loads_once_per_sequence": self.worker_model_loads_once_per_sequence,
            "monotonic_worker_frame_ids": self.monotonic_worker_frame_ids,
            "canonical_alignment": self.canonical_alignment,
            "per_frame_alignment_performed": self.per_frame_alignment_performed,
            "sequence_global_alignment_performed": self.sequence_global_alignment_performed,
            "runtime_profile_resolution": self.runtime_profile_resolution,
            "checkpoint_must_match_resolved_profile": self.checkpoint_must_match_resolved_profile,
            "explicit_provider_selection": self.explicit_provider_selection,
            "fallback_allowed": self.fallback_allowed,
            "sequential_scenario_execution": self.sequential_scenario_execution,
            "complete_registered_sequence_required": self.complete_registered_sequence_required,
            "production_config": VideoDepthAnythingProductionConfig().as_dict(),
            "permanent_global_view_cap": self.permanent_global_view_cap,
            "result_driven_retuning": self.result_driven_retuning,
            "retry_until_pass": self.retry_until_pass,
            "provider_comparison": self.provider_comparison,
        }


def validate_i5_8a_for_i5_8b(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.8A admission")
    _require(payload, "phase", "I5.8A", label="I5.8A admission")
    _require(payload, "status", "video_depth_anything_robustness_scenarios_admitted", label="I5.8A admission")
    _require(payload, "pass", True, label="I5.8A admission")

    scope = _mapping(payload.get("scope"), label="I5.8A scope")
    for key in (
        "video_depth_anything_inference", "rolling_depth_inference", "sfm_rerun",
        "bundle_adjustment_rerun", "mvs_rerun", "provider_comparison_started",
        "source_rematerialized", "thresholds_modified", "production_default_modified",
        "result_driven_retuning", "retry_until_pass", "mp_sfm_started",
        "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.8A scope")

    contract = _mapping(payload.get("provider_contract"), label="I5.8A provider contract")
    _require(contract, "provider_id", I58A_PROVIDER_ID, label="I5.8A provider contract")
    _require(contract, "provider_implementation", I58A_PROVIDER_IMPLEMENTATION, label="I5.8A provider contract")
    _require(contract, "temporal_mode", "official_streaming_cache", label="I5.8A provider contract")
    _require(contract, "stateful_temporal_sequence", True, label="I5.8A provider contract")
    _require(contract, "source_order_required", True, label="I5.8A provider contract")
    _require(contract, "stable_pixel_grid_required", True, label="I5.8A provider contract")
    _require(contract, "worker_model_loads_once_per_sequence", True, label="I5.8A provider contract")
    _require(contract, "warmup_reset_before_user_sequence", True, label="I5.8A provider contract")
    _require(contract, "canonical_alignment", "per_registered_frame_affine_to_authoritative_sfm", label="I5.8A provider contract")
    _require(contract, "per_frame_alignment_performed", True, label="I5.8A provider contract")
    _require(contract, "sequence_global_alignment_performed", False, label="I5.8A provider contract")
    _require(contract, "runtime_profile_resolution", "hardware_resolved_compute_profile_auto", label="I5.8A provider contract")
    _require(contract, "checkpoint_must_match_resolved_profile", True, label="I5.8A provider contract")
    _require(contract, "permanent_global_view_cap", None, label="I5.8A provider contract")
    if dict(_mapping(contract.get("production_config"), label="I5.8A production config")) != VideoDepthAnythingProductionConfig().as_dict():
        raise ReconstructionValidationError("I5.8A VDA production config drifted before I5.8B.")
    variants = _mapping(contract.get("runtime_variants"), label="I5.8A runtime variants")
    if dict(variants) != I58A_RUNTIME_VARIANTS:
        raise ReconstructionValidationError("I5.8A hardware-resolved runtime variant map drifted.")

    matrix = _mapping(payload.get("scenario_matrix"), label="I5.8A scenario matrix")
    if set(matrix) != set(I58B_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.8A must admit exactly the six frozen Castle scenarios.")
    for sid in I58B_SCENARIO_IDS:
        node = _mapping(matrix[sid], label=f"I5.8A {sid}")
        _require(node, "scenario_id", sid, label=f"I5.8A {sid}")
        _require(node, "admitted_for_i5_8b", True, label=f"I5.8A {sid}")
        _require(node, "video_depth_anything_execution_required", True, label=f"I5.8A {sid}")
        _require(node, "scientific_interpretation", _SCENARIO_INTERPRETATION[sid], label=f"I5.8A {sid}")
        _require(node, "source_registered_scope_only", True, label=f"I5.8A {sid}")
        _require(node, "ordered_registered_keyframe_stream", True, label=f"I5.8A {sid}")
        _require(node, "vda_streaming_cache_state_preserved_across_registered_views", True, label=f"I5.8A {sid}")
        _require(node, "per_registered_frame_sfm_alignment_expected", True, label=f"I5.8A {sid}")
        _require(node, "sequence_global_alignment_expected", False, label=f"I5.8A {sid}")
        if int(node.get("registered_images", 0)) < 2:
            raise ReconstructionValidationError(f"I5.8A {sid} registered-view count is invalid.")

    historical = _mapping(payload.get("historical_temporal_anchors"), label="I5.8A historical temporal anchors")
    for sid, expected in I58A_FROZEN_G3_VDA_STATUSES.items():
        node = _mapping(historical.get(sid), label=f"I5.8A {sid}")
        _require(node, "video_depth_anything_production_status", expected, label=f"I5.8A {sid}")
        _require(node, "historical_result_overwritten", False, label=f"I5.8A {sid}")
        _require(node, "rerun_authorized_by_i5_8a", False, label=f"I5.8A {sid}")

    evidence_contract = _mapping(payload.get("i5_8b_evidence_contract"), label="I5.8A I5.8B evidence contract")
    for key in (
        "new_timestamped_attempt_directory_per_real_attempt",
        "failed_interrupted_blocked_rejected_attempts_preserved",
        "runtime_preflight_required",
        "hardware_resolved_profile_recorded",
        "resolved_checkpoint_must_match_probed_checkpoint",
        "source_revision_and_checkpoint_identity_required",
        "source_and_authoritative_sfm_integrity_revalidation_required",
        "ordered_registered_sequence_identity_required",
        "stable_pixel_grid_and_single_camera_model_required",
        "warmup_reset_before_user_sequence_required",
        "single_worker_model_load_per_sequence_required",
        "native_vda_depth_stack_required",
        "native_frame_latency_records_required",
        "per_registered_frame_alignment_evidence_required",
        "sequence_global_alignment_forbidden",
        "wall_time_observation_required",
        "missing_resource_or_performance_metrics_must_remain_missing",
        "pass_degraded_rejected_are_valid_terminal_scientific_outcomes",
        "rejected_requires_explicit_provider_quality_failure_not_runtime_or_integrity_failure",
        "runtime_or_integrity_failure_must_fail_attempt_nonzero",
    ):
        _require(evidence_contract, key, True, label="I5.8A I5.8B evidence contract")
    for key in ("retry_until_pass", "result_driven_retuning", "global_scalar_score_created", "provider_ranking_created", "universal_provider_winner_declared"):
        _require(evidence_contract, key, False, label="I5.8A I5.8B evidence contract")

    authority = _mapping(payload.get("authority"), label="I5.8A authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.8A authority")
    _require(authority, "authoritative_sfm_modified", False, label="I5.8A authority")
    _require(authority, "native_estimated_intrinsics_are_canonical_authority", False, label="I5.8A authority")
    _require(authority, "native_metric_depth_establishes_physical_metric_scale", False, label="I5.8A authority")
    for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated", "provider_selected", "provider_promoted", "provider_ranking_created"):
        _require(authority, key, False, label="I5.8A authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.8A authority")

    decision = _mapping(payload.get("decision"), label="I5.8A decision")
    _require(decision, "i5_7_complete", True, label="I5.8A decision")
    _require(decision, "i5_8_started", True, label="I5.8A decision")
    _require(decision, "i5_8a_complete", True, label="I5.8A decision")
    _require(decision, "i5_8b_authorized", True, label="I5.8A decision")
    _require(decision, "i5_8b_started", False, label="I5.8A decision")
    _require(decision, "i5_9_started", False, label="I5.8A decision")
    return deepcopy(dict(payload))


def validate_i5_8b_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.8B receipt")
    _require(payload, "phase", I58B_RECEIPT_PHASE, label="I5.8B receipt")
    _require(payload, "status", "pass", label="I5.8B receipt")
    _require(payload, "pass", True, label="I5.8B receipt")
    scope = _mapping(payload.get("scope"), label="I5.8B receipt scope")
    for key in (
        "focused_i5_8b_tests", "video_depth_anything_provider_regression",
        "i5_8a_predecessor_regression", "authority_guard_regression",
        "runtime_runner_help_probe",
    ):
        _require(scope, key, True, label="I5.8B receipt scope")
    for key in (
        "video_depth_anything_inference", "sfm_rerun", "mvs_rerun",
        "provider_comparison_started", "thresholds_modified", "production_default_modified",
        "result_driven_retuning", "retry_until_pass", "mp_sfm_started",
        "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.8B receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.8B receipt counts")
    focused = int(counts.get("focused_tests", 0))
    predecessor = int(counts.get("predecessor_total_tests", 0))
    combined = int(counts.get("combined_exercised_tests", 0))
    if focused <= 0 or predecessor <= 0 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.8B receipt test counts are inconsistent.")
    rcs = _mapping(payload.get("return_codes"), label="I5.8B receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if int(rcs.get(key, -1)) != 0:
            raise ReconstructionValidationError(f"I5.8B receipt return code is non-zero: {key}")
    return deepcopy(dict(payload))


def analyze_i5_8b_provider_summary(
    summary: Mapping[str, Any], *, wall_time_s: float | None,
    resource_observation: Mapping[str, Any], expected_runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    _require(summary, "schema_version", I43D1_SCHEMA_VERSION, label="VDA summary")
    _require(summary, "phase", I43D1_PHASE, label="VDA summary")
    status = str(summary.get("status", "")).strip().lower()
    if status not in {"pass", "degraded"}:
        raise ReconstructionValidationError(
            f"I5.8B unsupported persisted VDA summary status: {status!r}. "
            "Runtime/integrity exceptions are non-terminal campaign failures, not silently reclassified scientific results."
        )
    _require(summary, "pass", True, label="VDA summary")

    provider = _mapping(summary.get("provider"), label="VDA provider")
    _require(provider, "provider_id", I58A_PROVIDER_ID, label="VDA provider")
    _require(provider, "implementation", I58A_PROVIDER_IMPLEMENTATION, label="VDA provider")

    request = _mapping(summary.get("selection_request"), label="VDA selection request")
    _require(request, "mode", "explicit_provider", label="VDA selection request")
    _require(request, "authoritative_sfm_available", True, label="VDA selection request")
    _require(request, "requested_provider_id", I58A_PROVIDER_ID, label="VDA selection request")
    _require(request, "allow_fallback", False, label="VDA selection request")
    decision = _mapping(summary.get("selection_decision"), label="VDA selection decision")
    _require(decision, "request_mode", "explicit_provider", label="VDA selection decision")
    _require(decision, "selected_provider_id", I58A_PROVIDER_ID, label="VDA selection decision")
    _require(decision, "selected_implementation", I58A_PROVIDER_IMPLEMENTATION, label="VDA selection decision")
    _require(decision, "automatic", False, label="VDA selection decision")
    _require(decision, "fallback_used", False, label="VDA selection decision")

    execution = _mapping(summary.get("execution_result"), label="VDA execution result")
    expected_execution_status = "completed" if status == "pass" else "degraded"
    _require(execution, "status", expected_execution_status, label="VDA execution result")
    source_views = int(execution.get("source_view_count", 0))
    selected_views = int(execution.get("selected_view_count", 0))
    if source_views < 2 or selected_views != source_views:
        raise ReconstructionValidationError("I5.8B must process every registered VDA source view.")
    _require(execution, "adaptive_view_selection", False, label="VDA execution result")
    _require(execution, "view_selection_reason", "all_registered_authoritative_sfm_views_in_temporal_order", label="VDA execution result")
    _require(execution, "permanent_global_view_cap", None, label="VDA execution result")
    exec_authority = _mapping(execution.get("authority"), label="VDA execution result authority")
    _require(exec_authority, "canonical_geometry_authoritative", True, label="VDA execution result authority")
    _require(exec_authority, "single_authoritative_sfm_frame_preserved", True, label="VDA execution result authority")
    _require(exec_authority, "camera_pose_authority_owner", "pycolmap_sfm", label="VDA execution result authority")
    for key in (
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated", "production_filter_policy_changed", "provider_promotion_performed",
    ):
        _require(exec_authority, key, False, label="VDA execution result authority")

    config = _mapping(summary.get("config"), label="VDA production config")
    if dict(config) != VideoDepthAnythingProductionConfig().as_dict():
        raise ReconstructionValidationError("I5.8B VDA production config drifted.")
    runtime_config = _mapping(summary.get("runtime_config"), label="VDA runtime config")
    if dict(runtime_config) != dict(expected_runtime_config):
        raise ReconstructionValidationError("I5.8B VDA runtime config drifted from the hardware-resolved preflight config.")

    runtime = _mapping(summary.get("runtime"), label="VDA runtime")
    _require(runtime, "source_ready", True, label="VDA runtime")
    _require(runtime, "checkpoint_ready", True, label="VDA runtime")
    _require(runtime, "adapter_importable", True, label="VDA runtime")
    _require(runtime, "upstream_importable", True, label="VDA runtime")
    _require(runtime, "torch_available", True, label="VDA runtime")
    _require(runtime, "cuda_available", True, label="VDA runtime")
    if not str(runtime.get("source_revision", "")).strip():
        raise ReconstructionValidationError("I5.8B VDA runtime source revision is missing.")
    expected_checkpoint = str(expected_runtime_config.get("checkpoint", "")).strip()
    if str(runtime.get("checkpoint_path", "")).strip() != expected_checkpoint:
        raise ReconstructionValidationError("I5.8B VDA probed checkpoint differs from the resolved runtime checkpoint.")

    worker = _mapping(summary.get("worker"), label="VDA worker")
    _require(worker, "schema_version", 1, label="VDA worker")
    _require(worker, "phase", I43D1_PHASE, label="VDA worker")
    _require(worker, "status", "pass", label="VDA worker")
    _require(worker, "model_id", "video-depth-anything", label="VDA worker")
    if str(worker.get("source_revision", "")).strip() != str(runtime.get("source_revision", "")).strip():
        raise ReconstructionValidationError("I5.8B VDA worker/runtime source revision drifted.")
    if int(worker.get("frame_count", 0)) != source_views:
        raise ReconstructionValidationError("I5.8B VDA worker frame count drifted from the complete registered sequence.")
    worker_authority = _mapping(worker.get("authority"), label="VDA worker authority")
    _require(worker_authority, "native_metric_depth_preserved_as_evidence", True, label="VDA worker authority")
    _require(worker_authority, "native_estimated_intrinsics_preserved_as_evidence", True, label="VDA worker authority")
    _require(worker_authority, "native_metric_depth_promoted_to_canonical_metric_scale", False, label="VDA worker authority")
    _require(worker_authority, "native_estimated_intrinsics_promoted_to_canonical_projection", False, label="VDA worker authority")
    _require(worker_authority, "camera_pose_authority", "pycolmap_sfm", label="VDA worker authority")
    worker_elapsed = _float_or_none(worker.get("elapsed_s"))
    if worker_elapsed is None or worker_elapsed < 0.0:
        raise ReconstructionValidationError("I5.8B VDA worker elapsed_s is invalid.")

    temporal = _mapping(summary.get("temporal_sequence"), label="VDA temporal sequence")
    keyframes = [str(v) for v in temporal.get("ordered_keyframe_ids", [])]
    indices = [int(v) for v in temporal.get("ordered_source_sequence_indices", [])]
    timestamps = [int(v) for v in temporal.get("ordered_timestamps_ns", [])]
    if any(len(v) != source_views for v in (keyframes, indices, timestamps)):
        raise ReconstructionValidationError("I5.8B VDA temporal identity count drifted.")
    if len(set(keyframes)) != source_views or indices != sorted(indices) or len(set(indices)) != source_views:
        raise ReconstructionValidationError("I5.8B VDA temporal source order is not unique/increasing.")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ReconstructionValidationError("I5.8B VDA temporal timestamps are not strictly increasing.")
    _require(temporal, "stateful_sequence_required", True, label="VDA temporal sequence")
    _require(temporal, "sequence_reordered_by", "source_sequence_index_then_timestamp", label="VDA temporal sequence")

    frame_records = worker.get("frame_records")
    if not isinstance(frame_records, list) or len(frame_records) != source_views:
        raise ReconstructionValidationError("I5.8B requires one native VDA worker frame record per registered view.")
    latencies: list[float] = []
    for index, record_value in enumerate(frame_records):
        record = _mapping(record_value, label=f"VDA worker frame {index}")
        if int(record.get("order", -1)) != index:
            raise ReconstructionValidationError("I5.8B VDA worker frame/source order drifted.")
        if int(record.get("source_sequence_index", -1)) != indices[index]:
            raise ReconstructionValidationError("I5.8B VDA worker frame/source order drifted.")
        if int(record.get("timestamp_ns", -1)) != timestamps[index]:
            raise ReconstructionValidationError("I5.8B VDA worker timestamp identity drifted.")
        if str(record.get("keyframe_id", "")) != keyframes[index]:
            raise ReconstructionValidationError("I5.8B VDA worker keyframe identity drifted.")
        latency = _float_or_none(record.get("latency_ms"))
        if latency is None or latency < 0.0:
            raise ReconstructionValidationError("I5.8B requires a finite non-negative native frame latency record.")
        latencies.append(latency)
        _require(record, "native_is_metric", True, label=f"VDA worker frame {index}")
        _require(record, "depth_scale_native", "metric", label=f"VDA worker frame {index}")
        _require(record, "intrinsics_source_native", "estimated", label=f"VDA worker frame {index}")

    alignment = _mapping(summary.get("alignment"), label="VDA alignment")
    accepted_views = int(alignment.get("accepted_views", -1))
    rejected_views = int(alignment.get("rejected_views", -1))
    records = alignment.get("records")
    if accepted_views < 0 or rejected_views < 0 or accepted_views + rejected_views != source_views:
        raise ReconstructionValidationError("I5.8B VDA per-frame alignment accounting is invalid.")
    if not isinstance(records, list) or len(records) != source_views:
        raise ReconstructionValidationError("I5.8B requires one per-registered-frame SfM alignment record.")
    rejected_ids = [str(v) for v in alignment.get("rejected_keyframe_ids", [])]
    observed_rejected: list[str] = []
    accepted_medians: list[float] = []
    accepted_p90: list[float] = []
    rejection_codes: dict[str, int] = {}
    for index, record_value in enumerate(records):
        record = _mapping(record_value, label=f"VDA alignment record {index}")
        if int(record.get("source_sequence_index", -1)) != indices[index] or int(record.get("temporal_order_index", -1)) != index:
            raise ReconstructionValidationError("I5.8B VDA per-frame alignment order drifted.")
        _require(record, "estimated_model_intrinsics_consumed_for_canonical_projection", False, label=f"VDA alignment record {index}")
        _require(record, "model_metric_depth_promoted_to_canonical_metric_scale", False, label=f"VDA alignment record {index}")
        _require(record, "metric_scale_claimed", False, label=f"VDA alignment record {index}")
        record_status = str(record.get("status", "")).strip().lower()
        if record_status == "accepted":
            if record.get("alignment") is None:
                raise ReconstructionValidationError("I5.8B accepted VDA alignment record lacks alignment evidence.")
            m = _float_or_none(record.get("median_relative_residual"))
            p90 = _float_or_none(record.get("p90_relative_residual"))
            if m is not None:
                accepted_medians.append(m)
            if p90 is not None:
                accepted_p90.append(p90)
        elif record_status == "rejected":
            kid = str(record.get("keyframe_id", "")).strip()
            observed_rejected.append(kid)
            code = str(record.get("rejection_code", "")).strip()
            if not code:
                raise ReconstructionValidationError("I5.8B rejected VDA alignment record lacks rejection code.")
            rejection_codes[code] = rejection_codes.get(code, 0) + 1
            if record.get("alignment") is not None:
                raise ReconstructionValidationError("I5.8B rejected VDA alignment record cannot carry accepted alignment evidence.")
        else:
            raise ReconstructionValidationError(f"I5.8B VDA alignment record status is invalid: {record_status!r}")
    if len(observed_rejected) != rejected_views or observed_rejected != rejected_ids:
        raise ReconstructionValidationError("I5.8B VDA rejected alignment identity accounting drifted.")
    if status == "pass" and rejected_views != 0:
        raise ReconstructionValidationError("I5.8B pass summary cannot contain rejected VDA alignment views.")

    fusion = _mapping(summary.get("fusion"), label="VDA fusion")
    statistics = _mapping(fusion.get("statistics"), label="VDA fusion statistics")
    point_cloud = _mapping(fusion.get("point_cloud"), label="VDA point cloud")
    point_count = int(point_cloud.get("point_count", 0))
    if point_count <= 0:
        raise ReconstructionValidationError("I5.8B VDA canonical point cloud is empty.")
    capacity_drops = int(statistics.get("dropped_new_voxels_capacity", 0))
    degradation_reasons = [str(v) for v in execution.get("degradation_reasons", [])]
    if status == "degraded" and not degradation_reasons:
        raise ReconstructionValidationError("I5.8B degraded VDA result requires provider-native degradation reasons.")
    if status == "pass" and degradation_reasons:
        raise ReconstructionValidationError("I5.8B pass VDA result cannot carry degradation reasons.")
    if rejected_views > 0 and f"alignment_rejected_views:{rejected_views}" not in degradation_reasons:
        raise ReconstructionValidationError("I5.8B VDA alignment degradation reason drifted.")
    if capacity_drops > 0 and f"fusion_capacity_drops:{capacity_drops}" not in degradation_reasons:
        raise ReconstructionValidationError("I5.8B VDA fusion-capacity degradation reason drifted.")

    integrity = _mapping(summary.get("integrity"), label="VDA integrity")
    _require(integrity, "authoritative_sfm_unchanged", True, label="VDA integrity")
    _require(integrity, "registered_images_unchanged", True, label="VDA integrity")
    _require(integrity, "canonical_pose_layer", "offline_sfm_bundle_adjusted", label="VDA integrity")

    authority = _mapping(summary.get("authority"), label="VDA authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="VDA authority")
    _require(authority, "estimated_model_intrinsics_consumed_for_canonical_projection", False, label="VDA authority")
    _require(authority, "model_metric_depth_promoted_to_canonical_metric_scale", False, label="VDA authority")
    for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated", "provider_promotion_performed"):
        _require(authority, key, False, label="VDA authority")

    wall = _float_or_none(wall_time_s)
    if wall is None or wall < 0.0:
        raise ReconstructionValidationError("I5.8B wall time is invalid.")
    peak_vram = _float_or_none(resource_observation.get("peak_vram_mib"))
    peak_ram = _float_or_none(resource_observation.get("peak_system_ram_used_mib"))

    return {
        "runtime_status": status,
        "provider_execution_completed": True,
        "degradation_reasons": degradation_reasons,
        "native_metrics": {
            "processed_views": source_views,
            "alignment_accepted_views": accepted_views,
            "alignment_rejected_views": rejected_views,
            "alignment_accepted_fraction": accepted_views / float(source_views),
            "alignment_rejection_codes": rejection_codes,
            "accepted_frame_median_relative_residual_median": median(accepted_medians) if accepted_medians else None,
            "accepted_frame_p90_relative_residual_median": median(accepted_p90) if accepted_p90 else None,
            "canonical_point_count": point_count,
            "fusion_capacity_drops": capacity_drops,
            "worker_elapsed_s": worker_elapsed,
            "frame_latency_count": len(latencies),
            "frame_latency_mean_ms": mean(latencies),
            "frame_latency_median_ms": median(latencies),
            "frame_latency_p95_ms": _percentile(latencies, 0.95),
            "frame_latency_min_ms": min(latencies),
            "frame_latency_max_ms": max(latencies),
            "wall_time_s": wall,
            "peak_vram_mib": peak_vram,
            "peak_system_ram_used_mib": peak_ram,
        },
        "resource_metrics_missing": {
            "peak_vram_mib": peak_vram is None,
            "peak_system_ram_used_mib": peak_ram is None,
        },
        "failure": None,
    }


def compare_i5_8b_scenarios(scenario_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if len(scenario_records) != len(I58B_SCENARIO_IDS) or set(scenario_records) != set(I58B_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.8B comparison requires exactly six scenario records.")
    control = _mapping(scenario_records["castle_control"].get("native_metrics"), label="I5.8B control metrics")
    out: dict[str, Any] = {}
    for sid in I58B_SCENARIO_IDS:
        node = _mapping(scenario_records[sid], label=f"I5.8B {sid} record")
        metrics = _mapping(node.get("native_metrics"), label=f"I5.8B {sid} metrics")
        out[sid] = {
            "scenario_id": sid,
            "runtime_status": node.get("runtime_status"),
            "alignment_accepted_fraction_ratio_to_control": _ratio_or_none(
                metrics.get("alignment_accepted_fraction"), control.get("alignment_accepted_fraction")
            ),
            "frame_latency_median_ratio_to_control": _ratio_or_none(
                metrics.get("frame_latency_median_ms"), control.get("frame_latency_median_ms")
            ),
            "wall_time_ratio_to_control": _ratio_or_none(metrics.get("wall_time_s"), control.get("wall_time_s")),
            "peak_vram_ratio_to_control": _ratio_or_none(metrics.get("peak_vram_mib"), control.get("peak_vram_mib")),
            "peak_system_ram_ratio_to_control": _ratio_or_none(
                metrics.get("peak_system_ram_used_mib"), control.get("peak_system_ram_used_mib")
            ),
            "cross_run_absolute_geometry_comparison": False,
            "cross_run_scale_shift_comparison": False,
            "classification_threshold_used": None,
            "global_scalar_score_created": False,
            "provider_ranking_created": False,
        }
    return out


def _validate_scenario_record(node: Mapping[str, Any], *, scenario_id: str) -> dict[str, Any]:
    record = deepcopy(dict(node))
    _require(record, "scenario_id", scenario_id, label=f"I5.8B {scenario_id} record")
    status = str(record.get("runtime_status", "")).strip().lower()
    if status not in {"pass", "degraded", "rejected"}:
        raise ReconstructionValidationError(f"I5.8B invalid runtime_status for {scenario_id}: {status!r}")
    _require(record, "provider_execution_terminal", True, label=f"I5.8B {scenario_id} record")
    source = int(record.get("source_registered_views", 0))
    processed = int(record.get("processed_views", 0))
    if source < 2 or processed != source:
        raise ReconstructionValidationError(f"I5.8B {scenario_id} did not process the complete registered sequence.")
    _require(record, "scientific_interpretation", _SCENARIO_INTERPRETATION[scenario_id], label=f"I5.8B {scenario_id} record")
    metrics = _mapping(record.get("native_metrics"), label=f"I5.8B {scenario_id} metrics")
    if int(metrics.get("processed_views", 0)) != processed:
        raise ReconstructionValidationError(f"I5.8B {scenario_id} metrics view count drifted.")
    if int(metrics.get("frame_latency_count", 0)) != source:
        raise ReconstructionValidationError(f"I5.8B {scenario_id} did not preserve one latency record per registered view.")
    runtime_profile = _mapping(record.get("runtime_profile"), label=f"I5.8B {scenario_id} runtime profile")
    if not str(runtime_profile.get("resolved_profile", "")).strip() or not str(runtime_profile.get("resolved_checkpoint", "")).strip():
        raise ReconstructionValidationError(f"I5.8B {scenario_id} runtime profile/checkpoint evidence is incomplete.")

    authority = _mapping(record.get("authority"), label=f"I5.8B {scenario_id} authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.8B {scenario_id} authority")
    _require(authority, "authoritative_sfm_modified", False, label=f"I5.8B {scenario_id} authority")
    _require(authority, "single_authoritative_sfm_frame_preserved", True, label=f"I5.8B {scenario_id} authority")
    _require(authority, "sfm_rerun", False, label=f"I5.8B {scenario_id} authority")
    _require(authority, "bundle_adjustment_rerun", False, label=f"I5.8B {scenario_id} authority")
    _require(authority, "source_rematerialized", False, label=f"I5.8B {scenario_id} authority")
    _require(authority, "per_frame_alignment_performed", True, label=f"I5.8B {scenario_id} authority")
    _require(authority, "sequence_global_alignment_performed", False, label=f"I5.8B {scenario_id} authority")
    _require(authority, "native_estimated_intrinsics_are_canonical_authority", False, label=f"I5.8B {scenario_id} authority")
    _require(authority, "native_metric_depth_establishes_physical_metric_scale", False, label=f"I5.8B {scenario_id} authority")
    _require(authority, "permanent_global_view_cap", None, label=f"I5.8B {scenario_id} authority")
    for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated", "provider_promotion_performed"):
        _require(authority, key, False, label=f"I5.8B {scenario_id} authority")
    if status == "rejected":
        failure = _mapping(record.get("failure"), label=f"I5.8B {scenario_id} failure")
        _require(failure, "quality_terminal", True, label=f"I5.8B {scenario_id} failure")
    return record


def build_i5_8b_video_depth_anything_robustness_replay(
    *,
    i5_8a_admission: Mapping[str, Any],
    i5_8a_binding: I51EvidenceBinding,
    replay_receipt: Mapping[str, Any],
    replay_receipt_binding: I51EvidenceBinding,
    runtime_preflight: Mapping[str, Any],
    scenario_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    admission = validate_i5_8a_for_i5_8b(i5_8a_admission)
    receipt = validate_i5_8b_receipt(replay_receipt)
    _validate_binding(i5_8a_binding, label="I5.8B I5.8A", expected_sha256=I58B_I58A_FROZEN_SHA256)
    _validate_binding(replay_receipt_binding, label="I5.8B receipt")

    preflight = deepcopy(dict(_mapping(runtime_preflight, label="I5.8B runtime preflight")))
    _require(preflight, "video_depth_anything_runtime_ready", True, label="I5.8B runtime preflight")
    _require(preflight, "provider_id", I58A_PROVIDER_ID, label="I5.8B runtime preflight")
    _require(preflight, "implementation", I58A_PROVIDER_IMPLEMENTATION, label="I5.8B runtime preflight")
    _require(preflight, "source_ready", True, label="I5.8B runtime preflight")
    _require(preflight, "checkpoint_ready", True, label="I5.8B runtime preflight")
    _require(preflight, "cuda_available", True, label="I5.8B runtime preflight")
    if not _valid_sha(preflight.get("checkpoint_sha256")):
        raise ReconstructionValidationError("I5.8B runtime preflight checkpoint SHA256 is missing/malformed.")
    profile = str(preflight.get("resolved_profile", "")).strip()
    if profile not in I58A_RUNTIME_VARIANTS:
        raise ReconstructionValidationError("I5.8B runtime preflight resolved profile is not an admitted VDA variant.")
    expected_variant = I58A_RUNTIME_VARIANTS[profile]
    _require(preflight, "resolved_encoder", expected_variant["encoder"], label="I5.8B runtime preflight")
    if not str(preflight.get("source_revision", "")).strip():
        raise ReconstructionValidationError("I5.8B runtime preflight source revision is missing.")
    if not str(preflight.get("device", "")).startswith("cuda"):
        raise ReconstructionValidationError("I5.8B production-native replay requires a CUDA-resolved VDA profile.")
    if str(preflight.get("precision", "")).strip().lower() not in {"fp16", "fp32"}:
        raise ReconstructionValidationError("I5.8B runtime preflight precision is invalid.")
    _require(preflight, "production_config_unchanged", True, label="I5.8B runtime preflight")
    if dict(_mapping(preflight.get("production_config"), label="I5.8B production config")) != VideoDepthAnythingProductionConfig().as_dict():
        raise ReconstructionValidationError("I5.8B runtime preflight production config drifted.")
    runtime_config = _mapping(preflight.get("runtime_config"), label="I5.8B runtime config")
    if str(runtime_config.get("encoder", "")) != expected_variant["encoder"]:
        raise ReconstructionValidationError("I5.8B resolved runtime encoder does not match admitted profile variant.")
    if int(runtime_config.get("input_size", 0)) != int(expected_variant["input_size"]):
        raise ReconstructionValidationError("I5.8B resolved input_size does not match admitted profile variant.")
    if int(runtime_config.get("max_resolution", 0)) != int(expected_variant["max_resolution"]):
        raise ReconstructionValidationError("I5.8B resolved max_resolution does not match admitted profile variant.")
    _require(runtime_config, "profile", profile, label="I5.8B runtime config")
    _require(runtime_config, "hardware_tier", profile, label="I5.8B runtime config")
    _require(runtime_config, "checkpoint", preflight.get("resolved_checkpoint"), label="I5.8B runtime config")

    if len(scenario_records) != len(I58B_SCENARIO_IDS) or set(scenario_records) != set(I58B_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.8B requires exactly six terminal scenario records.")
    records = {sid: _validate_scenario_record(scenario_records[sid], scenario_id=sid) for sid in I58B_SCENARIO_IDS}

    statuses = [records[sid]["runtime_status"] for sid in I58B_SCENARIO_IDS]
    if "rejected" in statuses:
        quality = "rejected"
    elif "degraded" in statuses:
        quality = "degraded"
    else:
        quality = "pass"

    return {
        "schema_version": I58B_SCHEMA_VERSION,
        "phase": I58B_PHASE,
        "status": "current_frozen_foundation_video_depth_anything_robustness_replayed",
        "pass": True,
        "quality_status": quality,
        "scope": {
            "real_production_video_depth_anything_replay": True,
            "six_admitted_castle_scenarios_attempted": True,
            "sequential_execution": True,
            "full_registered_temporal_sequence_per_scenario": True,
            "hardware_resolved_profile_used": True,
            "adaptive_view_reduction": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "source_rematerialized": False,
            "mvs_rerun": False,
            "rolling_depth_rerun": False,
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
            "i5_8a_admission": i5_8a_binding.as_dict(),
            "i5_8b_replay_receipt": replay_receipt_binding.as_dict(),
            "nested_i5_6a_castle_sfm_admission": deepcopy(
                dict(_mapping(admission["evidence_bindings"]["i5_6a_castle_sfm_admission"], label="I5.8A I5.6A binding"))
            ),
        },
        "runtime_preflight": preflight,
        "execution_policy": I58BReplayPolicy().as_dict(),
        "scenario_records": records,
        "control_relative_observations": compare_i5_8b_scenarios(records),
        "historical_temporal_anchors_preserved": deepcopy(
            dict(_mapping(admission.get("historical_temporal_anchors"), label="I5.8A historical anchors"))
        ),
        "excluded_or_deferred_scenarios_preserved": deepcopy(
            dict(_mapping(admission.get("excluded_or_deferred_scenarios"), label="I5.8A exclusions"))
        ),
        "observations": {
            "classification_threshold_used": None,
            "global_scalar_score_created": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "universal_video_depth_anything_robustness_claimed": False,
            "historical_motion_degraded_can_be_overwritten_by_current_pass": False,
            "cross_run_absolute_geometry_comparison": False,
            "cross_run_scale_shift_comparison": False,
            "ground_truth_dense_depth_available": False,
            "ground_truth_surface_accuracy_available": False,
            "metric_scale_available": False,
            "resource_metrics_are_observations_not_qualification_thresholds": True,
            "frame_latency_metrics_are_observations_not_quality_thresholds": True,
            "quality_interpretation_deferred_to_i5_8c": True,
        },
        "regression_summary": {
            "focused_i5_8b_tests": receipt["counts"].get("focused_tests"),
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
            "i5_8_started": True,
            "i5_8a_complete": True,
            "i5_8b_complete": True,
            "i5_8c_authorized": True,
            "i5_8c_started": False,
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
            "i5_8c_evidence_only_qualification_authorized": True,
            "i5_9_not_started": True,
            "i5_10_mp_sfm_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_auto_deferred": True,
        },
        "next": "I5.8C Video Depth Anything robustness qualification and closure.",
    }


__all__ = [
    "I58B_SCHEMA_VERSION", "I58B_PHASE", "I58B_RECEIPT_PHASE", "I58B_SCENARIO_IDS",
    "I58B_I58A_FROZEN_SHA256", "I58BReplayPolicy", "validate_i5_8a_for_i5_8b",
    "validate_i5_8b_receipt", "analyze_i5_8b_provider_summary", "compare_i5_8b_scenarios",
    "build_i5_8b_video_depth_anything_robustness_replay",
]
