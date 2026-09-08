"""I5.7B real frozen-foundation RollingDepth robustness replay contracts.

This module is evidence/qualification logic only. The real production-native
execution is performed by scripts/run_i5_7b_rolling_depth_robustness_replay.py
through the existing RollingDepthProductionProviderAdapter.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_rolling_depth_robustness import (
    I57AAdmissionPolicy,
    I57A_I56A_FROZEN_SHA256,
    I57A_PHASE,
    I57A_PROVIDER_ID,
    I57A_PROVIDER_IMPLEMENTATION,
    I57A_SCHEMA_VERSION,
    I57A_SCENARIO_IDS,
)
from core.reconstruction.rolling_depth_dense_provider import (
    I43D2_CANONICAL_DEPTH_POLARITY,
    I43D2_NATIVE_DEPTH_POLARITY,
    I43D2_NATIVE_DEPTH_SEMANTICS,
    I43D2_NATIVE_TO_CANONICAL_MULTIPLIER,
    I43D2_PHASE,
    I43D2_SCHEMA_VERSION,
    RollingDepthProductionConfig,
)

I57B_SCHEMA_VERSION = 1
I57B_PHASE = "I5.7B"
I57B_RECEIPT_PHASE = "I5.7B_REPLAY_RECEIPT"
I57B_SCENARIO_IDS = I57A_SCENARIO_IDS
I57B_I57A_FROZEN_SHA256 = "040ee6ce45f32e9f3f51087ce5f5b8764ced9a33e978d8024d04867efa5fae61"
I57B_RECOVERY_SCENARIO_ID = "castle_control"
I57B_RECOVERY_EXCEPTION_MESSAGE = (
    "RollingDepth execution result.metric_scale_claimed must be False; found None."
)
I57B_RECOVERY_MISSING_PERFORMANCE_FIELDS = (
    "wall_time_s", "peak_vram_mib", "peak_system_ram_used_mib"
)

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


def _validate_binding(
    binding: I51EvidenceBinding,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> None:
    if not str(binding.path).strip() or not _valid_sha(binding.sha256):
        raise ReconstructionValidationError(f"{label} evidence binding is malformed.")
    if expected_sha256 is not None and binding.sha256.lower() != expected_sha256.lower():
        raise ReconstructionValidationError(
            f"{label} SHA256 drifted from the frozen predecessor: "
            f"expected={expected_sha256} actual={binding.sha256}"
        )


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _ratio_or_none(numerator: Any, denominator: Any) -> float | None:
    a = _float_or_none(numerator)
    b = _float_or_none(denominator)
    if a is None or b is None or b == 0.0:
        return None
    return a / b


@dataclass(frozen=True, slots=True)
class I57BReplayPolicy:
    scenario_ids: tuple[str, ...] = I57B_SCENARIO_IDS
    provider_id: str = I57A_PROVIDER_ID
    provider_implementation: str = I57A_PROVIDER_IMPLEMENTATION
    production_adapter: str = "RollingDepthProductionProviderAdapter"
    stateful_temporal_sequence: bool = True
    source_order_required: bool = True
    all_registered_views: bool = True
    adaptive_view_reduction: bool = False
    sequence_alignment: str = "one_global_affine_to_authoritative_sfm"
    per_frame_alignment: bool = False
    device: str = "cuda:0"
    precision: str = "fp16"
    processing_resolution: int = 512
    dilations: tuple[int, ...] = (1, 25)
    max_vae_batch_size: int = 1
    unload_snippet: bool = True
    refine_step: int = 0
    sequential_execution: bool = True
    permanent_global_view_cap: int | None = None
    result_driven_retuning: bool = False
    retry_until_pass: bool = False
    provider_comparison: bool = False

    def __post_init__(self) -> None:
        if tuple(self.scenario_ids) != I57B_SCENARIO_IDS:
            raise ReconstructionValidationError("I5.7B must execute exactly the six admitted scenarios.")
        if self.provider_id != I57A_PROVIDER_ID or self.provider_implementation != I57A_PROVIDER_IMPLEMENTATION:
            raise ReconstructionValidationError("I5.7B must use the existing RollingDepth provider.")
        if self.production_adapter != "RollingDepthProductionProviderAdapter":
            raise ReconstructionValidationError("I5.7B cannot substitute a parallel provider architecture.")
        if not self.stateful_temporal_sequence or not self.source_order_required or not self.all_registered_views:
            raise ReconstructionValidationError("I5.7B requires the full ordered registered temporal sequence.")
        if self.adaptive_view_reduction:
            raise ReconstructionValidationError("I5.7B cannot adaptively reduce the admitted registered sequence.")
        if self.sequence_alignment != "one_global_affine_to_authoritative_sfm" or self.per_frame_alignment:
            raise ReconstructionValidationError("I5.7B must preserve one global sequence alignment.")
        if self.device != "cuda:0" or self.precision != "fp16":
            raise ReconstructionValidationError("I5.7B device/precision drifted from production.")
        cfg = RollingDepthProductionConfig()
        if self.processing_resolution != cfg.processing_resolution:
            raise ReconstructionValidationError("I5.7B processing resolution drifted from production.")
        if tuple(self.dilations) != tuple(cfg.dilations):
            raise ReconstructionValidationError("I5.7B dilations drifted from production.")
        if self.max_vae_batch_size != cfg.max_vae_batch_size:
            raise ReconstructionValidationError("I5.7B max VAE batch size drifted from production.")
        if self.unload_snippet != cfg.unload_snippet or self.refine_step != cfg.refine_step:
            raise ReconstructionValidationError("I5.7B memory/refinement defaults drifted from production.")
        if not self.sequential_execution or self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I5.7B must stay sequential with no permanent global cap.")
        if self.result_driven_retuning or self.retry_until_pass or self.provider_comparison:
            raise ReconstructionValidationError("I5.7B forbids retuning, retry-until-pass, and provider comparison.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_ids": list(self.scenario_ids),
            "provider_id": self.provider_id,
            "provider_implementation": self.provider_implementation,
            "production_adapter": self.production_adapter,
            "stateful_temporal_sequence": self.stateful_temporal_sequence,
            "source_order_required": self.source_order_required,
            "all_registered_views": self.all_registered_views,
            "adaptive_view_reduction": self.adaptive_view_reduction,
            "sequence_alignment": self.sequence_alignment,
            "per_frame_alignment": self.per_frame_alignment,
            "device": self.device,
            "precision": self.precision,
            "production_config": RollingDepthProductionConfig().as_dict(),
            "processing_resolution": self.processing_resolution,
            "dilations": list(self.dilations),
            "max_vae_batch_size": self.max_vae_batch_size,
            "unload_snippet": self.unload_snippet,
            "refine_step": self.refine_step,
            "sequential_execution": self.sequential_execution,
            "permanent_global_view_cap": self.permanent_global_view_cap,
            "result_driven_retuning": self.result_driven_retuning,
            "retry_until_pass": self.retry_until_pass,
            "provider_comparison": self.provider_comparison,
        }


def validate_i5_7a_for_i5_7b(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I57A_SCHEMA_VERSION, label="I5.7A admission")
    _require(payload, "phase", I57A_PHASE, label="I5.7A admission")
    _require(payload, "status", "rolling_depth_robustness_admission_complete", label="I5.7A admission")
    _require(payload, "pass", True, label="I5.7A admission")

    scope = _mapping(payload.get("scope"), label="I5.7A scope")
    _require(scope, "admission_and_envelope_design_only", True, label="I5.7A scope")
    for key in (
        "rolling_depth_inference", "sfm_rerun", "bundle_adjustment_rerun", "mvs_rerun",
        "da3_rerun", "video_depth_anything_rerun", "provider_comparison_started",
        "thresholds_modified", "production_default_modified", "result_driven_retuning",
        "retry_until_pass", "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.7A scope")

    provider = _mapping(payload.get("provider_contract"), label="I5.7A provider contract")
    _require(provider, "provider_id", I57A_PROVIDER_ID, label="I5.7A provider contract")
    _require(provider, "implementation", I57A_PROVIDER_IMPLEMENTATION, label="I5.7A provider contract")
    _require(provider, "stateful_sequence", True, label="I5.7A provider contract")
    _require(provider, "ordered_registered_sequence_required", True, label="I5.7A provider contract")
    _require(provider, "sequence_alignment", "global_affine_to_authoritative_sfm", label="I5.7A provider contract")
    _require(provider, "per_frame_alignment", False, label="I5.7A provider contract")
    _require(provider, "sequence_affine_establishes_metric_scale", False, label="I5.7A provider contract")
    _require(provider, "runtime_isolated_external_process", True, label="I5.7A provider contract")
    _require(provider, "runtime_preflight_required_for_i5_7b", True, label="I5.7A provider contract")

    frozen_a = I57AAdmissionPolicy().as_dict()
    policy = _mapping(payload.get("execution_policy"), label="I5.7A execution policy")
    for key in (
        "scenario_ids", "provider_id", "provider_implementation", "stateful_temporal_sequence",
        "source_order_required", "sequence_alignment", "per_frame_alignment", "default_device",
        "default_precision", "production_config", "sequential_execution", "permanent_global_view_cap",
        "result_driven_retuning", "provider_comparison",
    ):
        _require(policy, key, frozen_a[key], label="I5.7A execution policy")

    matrix = _mapping(payload.get("scenario_matrix"), label="I5.7A scenario matrix")
    if len(matrix) != len(I57B_SCENARIO_IDS) or set(matrix) != set(I57B_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.7B requires exactly the six I5.7A scenarios.")
    for sid in I57B_SCENARIO_IDS:
        node = _mapping(matrix.get(sid), label=f"I5.7A {sid}")
        _require(node, "scenario_id", sid, label=f"I5.7A {sid}")
        _require(node, "admitted_for_i5_7b", True, label=f"I5.7A {sid}")
        _require(node, "rolling_depth_execution_required", True, label=f"I5.7A {sid}")
        _require(node, "scientific_interpretation", _SCENARIO_INTERPRETATION[sid], label=f"I5.7A {sid}")
        _require(node, "same_authoritative_pycolmap_sfm_reused", True, label=f"I5.7A {sid}")
        _require(node, "source_rematerialization_authorized", False, label=f"I5.7A {sid}")
        _require(node, "sfm_rerun_authorized", False, label=f"I5.7A {sid}")
        _require(node, "provider_comparison_authorized", False, label=f"I5.7A {sid}")
        if int(node.get("registered_images", 0)) < 2:
            raise ReconstructionValidationError(f"I5.7A {sid} has insufficient registered views.")

    bindings = _mapping(payload.get("evidence_bindings"), label="I5.7A evidence bindings")
    i56a = _mapping(bindings.get("i5_6a_castle_sfm_admission"), label="I5.7A I5.6A binding")
    if not str(i56a.get("path", "")).strip() or str(i56a.get("sha256", "")).lower() != I57A_I56A_FROZEN_SHA256:
        raise ReconstructionValidationError("I5.7A no longer binds the frozen I5.6A SfM admission.")

    contract = _mapping(payload.get("i5_7b_evidence_contract"), label="I5.7A I5.7B evidence contract")
    for key in (
        "new_timestamped_attempt_directory_per_real_attempt",
        "failed_interrupted_blocked_rejected_attempts_preserved",
        "runtime_preflight_and_pinned_source_checkpoint_identity_required",
        "source_and_authoritative_sfm_integrity_revalidation_required",
        "ordered_temporal_input_identity_required", "native_provider_evidence_required",
        "global_sequence_alignment_evidence_required", "wall_time_observation_required",
        "missing_resource_or_performance_metrics_must_remain_missing",
        "pass_degraded_rejected_are_all_valid_terminal_scientific_outcomes",
    ):
        _require(contract, key, True, label="I5.7A I5.7B evidence contract")
    for key in ("global_scalar_score_created", "provider_ranking_created", "universal_provider_winner_declared"):
        _require(contract, key, False, label="I5.7A I5.7B evidence contract")

    authority = _mapping(payload.get("authority"), label="I5.7A authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.7A authority")
    _require(authority, "authoritative_sfm_modified", False, label="I5.7A authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.7A authority")
    for key in (
        "rolling_depth_pose_authority", "metric_scale_claimed", "measurement_authoritative",
        "certified_metrology", "surface_mesh_generated", "provider_selected", "provider_promoted",
        "provider_ranking_created",
    ):
        _require(authority, key, False, label="I5.7A authority")

    decision = _mapping(payload.get("decision"), label="I5.7A decision")
    _require(decision, "i5_7a_complete", True, label="I5.7A decision")
    _require(decision, "i5_7b_authorized", True, label="I5.7A decision")
    _require(decision, "i5_7b_started", False, label="I5.7A decision")
    _require(decision, "i5_8_started", False, label="I5.7A decision")
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized", "mp_sfm_authorized",
        "threshold_relaxation_authorized", "production_default_modified",
    ):
        _require(decision, key, False, label="I5.7A decision")
    return deepcopy(dict(payload))


def validate_i5_7b_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I57B_SCHEMA_VERSION, label="I5.7B receipt")
    _require(payload, "phase", I57B_RECEIPT_PHASE, label="I5.7B receipt")
    _require(payload, "status", "pass", label="I5.7B receipt")
    _require(payload, "pass", True, label="I5.7B receipt")
    scope = _mapping(payload.get("scope"), label="I5.7B receipt scope")
    for key in (
        "focused_i5_7b_tests", "rolling_depth_provider_regression", "i5_7a_predecessor_regression",
        "authority_guard_regression", "runtime_runner_help_probe",
    ):
        _require(scope, key, True, label="I5.7B receipt scope")
    for key in (
        "rolling_depth_inference", "sfm_rerun", "mvs_rerun", "provider_comparison_started",
        "thresholds_modified", "production_default_modified", "result_driven_retuning", "retry_until_pass",
        "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.7B receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.7B receipt counts")
    focused = int(counts.get("focused_tests", 0))
    predecessor = int(counts.get("predecessor_total_tests", 0))
    combined = int(counts.get("combined_exercised_tests", 0))
    if focused <= 0 or predecessor <= 0 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.7B receipt test counts are inconsistent.")
    rcs = _mapping(payload.get("return_codes"), label="I5.7B receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if int(rcs.get(key, -1)) != 0:
            raise ReconstructionValidationError(f"I5.7B receipt return code is non-zero: {key}")
    return deepcopy(dict(payload))


def analyze_i5_7b_provider_summary(
    summary: Mapping[str, Any], *, wall_time_s: float | None, resource_observation: Mapping[str, Any],
    allow_missing_performance_observation: bool = False,
) -> dict[str, Any]:
    _require(summary, "schema_version", I43D2_SCHEMA_VERSION, label="RollingDepth summary")
    _require(summary, "phase", I43D2_PHASE, label="RollingDepth summary")
    status = str(summary.get("status", "")).strip().lower()
    if status not in {"pass", "degraded", "rejected"}:
        raise ReconstructionValidationError(f"I5.7B unsupported RollingDepth status: {status!r}")

    provider = _mapping(summary.get("provider"), label="RollingDepth provider")
    _require(provider, "provider_id", I57A_PROVIDER_ID, label="RollingDepth provider")
    _require(provider, "implementation", I57A_PROVIDER_IMPLEMENTATION, label="RollingDepth provider")

    request = _mapping(summary.get("selection_request"), label="RollingDepth selection request")
    _require(request, "mode", "explicit_provider", label="RollingDepth selection request")
    _require(request, "authoritative_sfm_available", True, label="RollingDepth selection request")
    _require(request, "requested_provider_id", I57A_PROVIDER_ID, label="RollingDepth selection request")
    _require(request, "allow_fallback", False, label="RollingDepth selection request")
    decision = _mapping(summary.get("selection_decision"), label="RollingDepth selection decision")
    _require(decision, "request_mode", "explicit_provider", label="RollingDepth selection decision")
    _require(decision, "selected_provider_id", I57A_PROVIDER_ID, label="RollingDepth selection decision")
    _require(decision, "selected_implementation", I57A_PROVIDER_IMPLEMENTATION, label="RollingDepth selection decision")
    _require(decision, "automatic", False, label="RollingDepth selection decision")
    _require(decision, "fallback_used", False, label="RollingDepth selection decision")

    execution = _mapping(summary.get("execution_result"), label="RollingDepth execution result")
    source_views = int(execution.get("source_view_count", 0))
    selected_views = int(execution.get("selected_view_count", 0))
    if source_views < 2 or selected_views != source_views:
        raise ReconstructionValidationError("I5.7B must process every registered source view.")
    _require(execution, "adaptive_view_selection", False, label="RollingDepth execution result")
    _require(execution, "view_selection_reason", "all_registered_authoritative_sfm_views_in_temporal_order", label="RollingDepth execution result")
    _require(execution, "permanent_global_view_cap", None, label="RollingDepth execution result")
    execution_authority = _mapping(
        execution.get("authority"), label="RollingDepth execution result authority"
    )
    _require(
        execution_authority, "canonical_geometry_authoritative", status != "rejected",
        label="RollingDepth execution result authority",
    )
    _require(
        execution_authority, "single_authoritative_sfm_frame_preserved", True,
        label="RollingDepth execution result authority",
    )
    _require(
        execution_authority, "camera_pose_authority_owner", "pycolmap_sfm",
        label="RollingDepth execution result authority",
    )
    for key in (
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated", "production_filter_policy_changed",
        "provider_promotion_performed",
    ):
        _require(execution_authority, key, False, label="RollingDepth execution result authority")

    config = _mapping(summary.get("config"), label="RollingDepth production config")
    if dict(config) != RollingDepthProductionConfig().as_dict():
        raise ReconstructionValidationError("I5.7B RollingDepth production config drifted.")
    worker = _mapping(summary.get("worker"), label="RollingDepth worker")
    _require(worker, "per_frame_alignment", False, label="RollingDepth worker")
    _require(worker, "depth_semantics", I43D2_NATIVE_DEPTH_SEMANTICS, label="RollingDepth worker")
    canonical = _mapping(summary.get("depth_canonicalization"), label="RollingDepth canonicalization")
    _require(canonical, "native_depth_polarity", I43D2_NATIVE_DEPTH_POLARITY, label="RollingDepth canonicalization")
    _require(canonical, "canonical_depth_polarity", I43D2_CANONICAL_DEPTH_POLARITY, label="RollingDepth canonicalization")
    _require(canonical, "multiplier", I43D2_NATIVE_TO_CANONICAL_MULTIPLIER, label="RollingDepth canonicalization")
    _require(canonical, "sequence_order_changed", False, label="RollingDepth canonicalization")

    temporal = _mapping(summary.get("temporal_sequence"), label="RollingDepth temporal sequence")
    keyframes = list(temporal.get("ordered_keyframe_ids", []))
    indices = [int(v) for v in temporal.get("ordered_source_sequence_indices", [])]
    timestamps = [int(v) for v in temporal.get("ordered_timestamps_ns", [])]
    if any(len(v) != source_views for v in (keyframes, indices, timestamps)):
        raise ReconstructionValidationError("I5.7B temporal identity count drifted.")
    if len(set(keyframes)) != source_views or indices != sorted(indices) or len(set(indices)) != source_views:
        raise ReconstructionValidationError("I5.7B temporal source order is not unique/increasing.")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ReconstructionValidationError("I5.7B temporal timestamps are not strictly increasing.")
    _require(temporal, "native_temporal_coalignment_preserved", True, label="RollingDepth temporal sequence")
    _require(temporal, "canonicalization_preserves_sequence_order", True, label="RollingDepth temporal sequence")

    alignment = _mapping(summary.get("alignment"), label="RollingDepth alignment")
    _require(alignment, "alignment_scope", "one_global_affine_for_entire_temporal_sequence", label="RollingDepth alignment")
    accepted = bool(alignment.get("accepted"))
    if status == "rejected" and accepted:
        raise ReconstructionValidationError("Rejected RollingDepth evidence cannot have accepted alignment.")
    if status != "rejected" and not accepted:
        raise ReconstructionValidationError("Successful/degraded RollingDepth evidence requires accepted alignment.")

    fusion = _mapping(summary.get("fusion"), label="RollingDepth fusion")
    authority = _mapping(summary.get("authority"), label="RollingDepth authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="RollingDepth authority")
    _require(authority, "rolling_depth_pose_authority_promoted", False, label="RollingDepth authority")
    _require(authority, "sequence_affine_establishes_physical_metric_scale", False, label="RollingDepth authority")
    _require(authority, "per_frame_alignment_performed", False, label="RollingDepth authority")
    for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated", "provider_promotion_performed"):
        _require(authority, key, False, label="RollingDepth authority")
    integrity = _mapping(summary.get("integrity"), label="RollingDepth integrity")
    _require(integrity, "authoritative_sfm_unchanged", True, label="RollingDepth integrity")
    _require(integrity, "registered_images_unchanged", True, label="RollingDepth integrity")
    _require(integrity, "canonical_pose_layer", "offline_sfm_bundle_adjusted", label="RollingDepth integrity")

    wall = _float_or_none(wall_time_s)
    if allow_missing_performance_observation:
        if wall_time_s is not None:
            raise ReconstructionValidationError(
                "Recovered I5.7B evidence must preserve the unavailable wrapper wall time as missing."
            )
        wall = None
    elif wall is None or wall < 0.0:
        raise ReconstructionValidationError("I5.7B wall time is invalid.")
    accepted_views = int(fusion.get("accepted_views", 0)) if status != "rejected" else 0
    point_cloud = fusion.get("point_cloud")
    point_count = int(point_cloud.get("point_count", 0)) if status != "rejected" and isinstance(point_cloud, Mapping) else None
    failure = None
    if status == "rejected":
        reason = str(execution.get("failure_reason", "")).strip()
        if not reason.startswith("global_sequence_alignment_rejected:"):
            raise ReconstructionValidationError("I5.7B scientific rejection must come from the frozen global alignment gate.")
        failure = {
            "quality_terminal": True,
            "reason": reason,
            "alignment_message": alignment.get("message"),
            "canonical_geometry_generated": False,
            "retry_until_pass": False,
        }
    return {
        "runtime_status": status,
        "provider_execution_completed": status != "rejected",
        "native_metrics": {
            "processed_views": source_views,
            "alignment_accepted": accepted,
            "initial_correspondences": int(alignment.get("initial_correspondences", 0)),
            "retained_correspondences": int(alignment.get("retained_correspondences", 0)),
            "contributing_views": int(alignment.get("contributing_views", 0)),
            "alignment_median_relative_residual": _float_or_none(alignment.get("median_relative_residual")),
            "alignment_p90_relative_residual": _float_or_none(alignment.get("p90_relative_residual")),
            "accepted_fusion_views": accepted_views,
            "canonical_point_count": point_count,
            "wall_time_s": wall,
            "peak_vram_mib": _float_or_none(resource_observation.get("peak_vram_mib")),
            "peak_system_ram_used_mib": _float_or_none(resource_observation.get("peak_system_ram_used_mib")),
        },
        "performance_observation_missing": bool(allow_missing_performance_observation),
        "failure": failure,
    }


def build_i5_7b_recovered_control_record(
    *,
    provider_summary: Mapping[str, Any],
    artifact_bindings: Mapping[str, Mapping[str, Any]],
    recovery_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the single allowed I5.7B recovery record without re-running RollingDepth.

    This recovery is intentionally narrow: it only admits the preserved castle_control
    execution from the known I5.7B wrapper-schema failure. Performance observations that
    were held only in wrapper memory must remain missing rather than being reconstructed.
    """

    analysis = analyze_i5_7b_provider_summary(
        provider_summary,
        wall_time_s=None,
        resource_observation={},
        allow_missing_performance_observation=True,
    )
    if int(analysis["native_metrics"]["processed_views"]) != 19:
        raise ReconstructionValidationError(
            "I5.7B castle_control recovery requires the complete 19-view registered sequence."
        )

    bindings = deepcopy(dict(_mapping(artifact_bindings, label="I5.7B recovery artifacts")))
    required_bindings = (
        "provider_summary",
        "provider_input_manifest",
        "worker_result_manifest",
        "worker_evidence_npz",
        "canonical_npz",
        "canonical_ply",
        "original_attempt_failure",
        "original_scenario_failure",
        "original_runtime_preflight",
    )
    for name in required_bindings:
        binding = _mapping(bindings.get(name), label=f"I5.7B recovery artifact {name}")
        if not str(binding.get("path", "")).strip() or not _valid_sha(binding.get("sha256")):
            raise ReconstructionValidationError(
                f"I5.7B recovery artifact binding is malformed: {name}"
            )

    provenance = deepcopy(dict(_mapping(recovery_provenance, label="I5.7B recovery provenance")))
    _require(provenance, "scenario_id", I57B_RECOVERY_SCENARIO_ID, label="I5.7B recovery provenance")
    _require(provenance, "original_wrapper_failure_after_provider_execution", True, label="I5.7B recovery provenance")
    _require(provenance, "provider_execution_rerun", False, label="I5.7B recovery provenance")
    _require(provenance, "retry_until_pass", False, label="I5.7B recovery provenance")
    _require(provenance, "thresholds_modified", False, label="I5.7B recovery provenance")
    _require(provenance, "production_default_modified", False, label="I5.7B recovery provenance")
    _require(provenance, "sfm_rerun", False, label="I5.7B recovery provenance")
    _require(provenance, "mvs_rerun", False, label="I5.7B recovery provenance")
    _require(provenance, "manifest_verified", True, label="I5.7B recovery provenance")
    _require(provenance, "console_log_verified", True, label="I5.7B recovery provenance")
    _require(
        provenance,
        "original_exception_message",
        I57B_RECOVERY_EXCEPTION_MESSAGE,
        label="I5.7B recovery provenance",
    )
    if tuple(provenance.get("missing_performance_fields", ())) != I57B_RECOVERY_MISSING_PERFORMANCE_FIELDS:
        raise ReconstructionValidationError(
            "I5.7B recovery must preserve exactly the unavailable wrapper performance fields."
        )

    summary_execution = deepcopy(dict(_mapping(provider_summary.get("execution_result"), label="RollingDepth execution result")))
    return {
        "scenario_id": I57B_RECOVERY_SCENARIO_ID,
        "runtime_status": analysis["runtime_status"],
        "provider_execution_terminal": True,
        "provider_execution_completed": analysis["provider_execution_completed"],
        "source_registered_views": 19,
        "processed_views": analysis["native_metrics"]["processed_views"],
        "scientific_interpretation": _SCENARIO_INTERPRETATION[I57B_RECOVERY_SCENARIO_ID],
        "native_metrics": analysis["native_metrics"],
        "resource_observation": {
            "observation_scope": "unavailable_original_wrapper_observation",
            "qualification_threshold": None,
            "sample_interval_s": None,
            "sample_count": None,
            "baseline_vram_mib": None,
            "peak_vram_mib": None,
            "final_vram_mib": None,
            "baseline_system_ram_used_mib": None,
            "peak_system_ram_used_mib": None,
            "final_system_ram_used_mib": None,
            "missing_metrics_preserved_as_null": True,
            "missing_reason": (
                "original I5.7B wrapper failed after provider execution and before "
                "scenario-record persistence"
            ),
        },
        "artifact_bindings": bindings,
        "failure": analysis["failure"],
        "execution_result": summary_execution,
        "input_integrity_revalidated_after": False,
        "input_integrity_revalidated_during_recovery": True,
        "original_wrapper_post_execution_revalidation_completed": False,
        "recovery": provenance,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "source_rematerialized": False,
            "per_frame_alignment_performed": False,
            "permanent_global_view_cap": None,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_promotion_performed": False,
        },
    }


def compare_i5_7b_scenarios(scenario_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if len(scenario_records) != len(I57B_SCENARIO_IDS) or set(scenario_records) != set(I57B_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.7B comparison requires exactly six scenario records.")
    control = _mapping(scenario_records["castle_control"].get("native_metrics"), label="I5.7B control metrics")
    out: dict[str, Any] = {}
    for sid in I57B_SCENARIO_IDS:
        node = _mapping(scenario_records[sid], label=f"I5.7B {sid} record")
        metrics = _mapping(node.get("native_metrics"), label=f"I5.7B {sid} metrics")
        out[sid] = {
            "scenario_id": sid,
            "runtime_status": node.get("runtime_status"),
            "alignment_median_residual_ratio_to_control": _ratio_or_none(
                metrics.get("alignment_median_relative_residual"), control.get("alignment_median_relative_residual")
            ),
            "retained_correspondence_fraction_ratio_to_control": _ratio_or_none(
                _ratio_or_none(metrics.get("retained_correspondences"), metrics.get("initial_correspondences")),
                _ratio_or_none(control.get("retained_correspondences"), control.get("initial_correspondences")),
            ),
            "accepted_view_fraction_ratio_to_control": _ratio_or_none(
                _ratio_or_none(metrics.get("accepted_fusion_views"), metrics.get("processed_views")),
                _ratio_or_none(control.get("accepted_fusion_views"), control.get("processed_views")),
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
    _require(record, "scenario_id", scenario_id, label=f"I5.7B {scenario_id} record")
    status = str(record.get("runtime_status", "")).strip().lower()
    if status not in {"pass", "degraded", "rejected"}:
        raise ReconstructionValidationError(f"I5.7B invalid runtime_status for {scenario_id}: {status!r}")
    _require(record, "provider_execution_terminal", True, label=f"I5.7B {scenario_id} record")
    source = int(record.get("source_registered_views", 0))
    processed = int(record.get("processed_views", 0))
    if source < 2 or processed != source:
        raise ReconstructionValidationError(f"I5.7B {scenario_id} did not process the complete registered sequence.")
    _require(record, "scientific_interpretation", _SCENARIO_INTERPRETATION[scenario_id], label=f"I5.7B {scenario_id} record")
    metrics = _mapping(record.get("native_metrics"), label=f"I5.7B {scenario_id} metrics")
    if int(metrics.get("processed_views", 0)) != processed:
        raise ReconstructionValidationError(f"I5.7B {scenario_id} metrics view count drifted.")
    authority = _mapping(record.get("authority"), label=f"I5.7B {scenario_id} authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.7B {scenario_id} authority")
    _require(authority, "authoritative_sfm_modified", False, label=f"I5.7B {scenario_id} authority")
    _require(authority, "single_authoritative_sfm_frame_preserved", True, label=f"I5.7B {scenario_id} authority")
    _require(authority, "sfm_rerun", False, label=f"I5.7B {scenario_id} authority")
    _require(authority, "bundle_adjustment_rerun", False, label=f"I5.7B {scenario_id} authority")
    _require(authority, "source_rematerialized", False, label=f"I5.7B {scenario_id} authority")
    _require(authority, "per_frame_alignment_performed", False, label=f"I5.7B {scenario_id} authority")
    _require(authority, "permanent_global_view_cap", None, label=f"I5.7B {scenario_id} authority")
    for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated", "provider_promotion_performed"):
        _require(authority, key, False, label=f"I5.7B {scenario_id} authority")
    if status == "rejected":
        failure = _mapping(record.get("failure"), label=f"I5.7B {scenario_id} failure")
        _require(failure, "quality_terminal", True, label=f"I5.7B {scenario_id} failure")
    recovery = record.get("recovery")
    if recovery is not None:
        if scenario_id != I57B_RECOVERY_SCENARIO_ID:
            raise ReconstructionValidationError("Only castle_control may be recovered in this narrow I5.7B correction.")
        recovery = _mapping(recovery, label="I5.7B castle_control recovery")
        _require(recovery, "provider_execution_rerun", False, label="I5.7B castle_control recovery")
        _require(recovery, "manifest_verified", True, label="I5.7B castle_control recovery")
        _require(recovery, "console_log_verified", True, label="I5.7B castle_control recovery")
        if tuple(recovery.get("missing_performance_fields", ())) != I57B_RECOVERY_MISSING_PERFORMANCE_FIELDS:
            raise ReconstructionValidationError("Recovered castle_control performance-missing contract drifted.")
        for key in I57B_RECOVERY_MISSING_PERFORMANCE_FIELDS:
            if metrics.get(key) is not None:
                raise ReconstructionValidationError(
                    f"Recovered castle_control {key} must remain missing; found {metrics.get(key)!r}."
                )
    return record


def build_i5_7b_rolling_depth_robustness_replay(
    *,
    i5_7a_admission: Mapping[str, Any],
    i5_7a_binding: I51EvidenceBinding,
    replay_receipt: Mapping[str, Any],
    replay_receipt_binding: I51EvidenceBinding,
    runtime_preflight: Mapping[str, Any],
    scenario_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    admission = validate_i5_7a_for_i5_7b(i5_7a_admission)
    receipt = validate_i5_7b_receipt(replay_receipt)
    _validate_binding(i5_7a_binding, label="I5.7B I5.7A", expected_sha256=I57B_I57A_FROZEN_SHA256)
    _validate_binding(replay_receipt_binding, label="I5.7B receipt")

    preflight = deepcopy(dict(_mapping(runtime_preflight, label="I5.7B runtime preflight")))
    _require(preflight, "rolling_depth_runtime_ready", True, label="I5.7B runtime preflight")
    _require(preflight, "provider_id", I57A_PROVIDER_ID, label="I5.7B runtime preflight")
    _require(preflight, "runtime_lock_valid", True, label="I5.7B runtime preflight")
    _require(preflight, "checkpoint_ready", True, label="I5.7B runtime preflight")
    _require(preflight, "cuda_available", True, label="I5.7B runtime preflight")
    _require(preflight, "device", "cuda:0", label="I5.7B runtime preflight")
    _require(preflight, "precision", "fp16", label="I5.7B runtime preflight")
    _require(preflight, "production_config_unchanged", True, label="I5.7B runtime preflight")
    if not str(preflight.get("source_revision", "")).strip():
        raise ReconstructionValidationError("I5.7B runtime preflight lacks a source revision.")

    if len(scenario_records) != len(I57B_SCENARIO_IDS) or set(scenario_records) != set(I57B_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.7B requires exactly six terminal scenario records.")
    records = {sid: _validate_scenario_record(scenario_records[sid], scenario_id=sid) for sid in I57B_SCENARIO_IDS}

    control_status = records["castle_control"]["runtime_status"]
    stress_statuses = [records[sid]["runtime_status"] for sid in I57B_SCENARIO_IDS if sid != "castle_control"]
    if control_status == "rejected":
        quality = "rejected"
    elif control_status == "degraded" or any(status != "pass" for status in stress_statuses):
        quality = "degraded"
    else:
        quality = "pass"

    return {
        "schema_version": I57B_SCHEMA_VERSION,
        "phase": I57B_PHASE,
        "status": "current_frozen_foundation_rolling_depth_robustness_replayed",
        "pass": True,
        "quality_status": quality,
        "scope": {
            "real_production_rolling_depth_replay": True,
            "six_admitted_castle_scenarios_attempted": True,
            "sequential_execution": True,
            "full_registered_temporal_sequence_per_scenario": True,
            "adaptive_view_reduction": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "source_rematerialized": False,
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
            "i5_7a_admission": i5_7a_binding.as_dict(),
            "i5_7b_replay_receipt": replay_receipt_binding.as_dict(),
            "nested_i5_6a_castle_sfm_admission": deepcopy(
                dict(_mapping(admission["evidence_bindings"]["i5_6a_castle_sfm_admission"], label="I5.7A I5.6A binding"))
            ),
        },
        "runtime_preflight": preflight,
        "execution_policy": I57BReplayPolicy().as_dict(),
        "scenario_records": records,
        "control_relative_observations": compare_i5_7b_scenarios(records),
        "historical_temporal_anchors_preserved": deepcopy(
            dict(_mapping(admission.get("historical_temporal_anchors"), label="I5.7A historical anchors"))
        ),
        "excluded_or_deferred_scenarios_preserved": deepcopy(
            dict(_mapping(admission.get("excluded_or_deferred_scenarios"), label="I5.7A exclusions"))
        ),
        "observations": {
            "classification_threshold_used": None,
            "global_scalar_score_created": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "cross_run_absolute_geometry_comparison": False,
            "cross_run_scale_shift_comparison": False,
            "ground_truth_dense_depth_available": False,
            "ground_truth_surface_accuracy_available": False,
            "metric_scale_available": False,
            "resource_metrics_are_observations_not_qualification_thresholds": True,
            "quality_interpretation_deferred_to_i5_7c": True,
        },
        "regression_summary": {
            "focused_i5_7b_tests": receipt["counts"].get("focused_tests"),
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
            "i5_7_started": True,
            "i5_7a_complete": True,
            "i5_7b_complete": True,
            "i5_7c_authorized": True,
            "i5_7c_started": False,
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
            "i5_7c_evidence_only_qualification_authorized": True,
            "i5_8_not_started": True,
            "i5_10_mp_sfm_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_auto_deferred": True,
        },
        "next": "I5.7C RollingDepth robustness qualification and closure.",
    }


__all__ = [
    "I57B_SCHEMA_VERSION", "I57B_PHASE", "I57B_RECEIPT_PHASE", "I57B_SCENARIO_IDS",
    "I57B_I57A_FROZEN_SHA256", "I57BReplayPolicy", "validate_i5_7a_for_i5_7b",
    "validate_i5_7b_receipt", "analyze_i5_7b_provider_summary",
    "build_i5_7b_recovered_control_record", "compare_i5_7b_scenarios",
    "build_i5_7b_rolling_depth_robustness_replay",
    "I57B_RECOVERY_SCENARIO_ID", "I57B_RECOVERY_EXCEPTION_MESSAGE",
    "I57B_RECOVERY_MISSING_PERFORMANCE_FIELDS",
]
