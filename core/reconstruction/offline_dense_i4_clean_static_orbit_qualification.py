"""I4.1.7 clean-static object/orbit qualification helpers.

I4.1.7 is the clean-capture validation gate after I4.1.6A/B demonstrated that
extra dense candidates can be localized into one authoritative SfM frame and
can materially improve dense consistency on the difficult Horse sequence.

This module does not bind a production sampling policy.  It evaluates a real
clean static capture using the unchanged dense gates and scale-invariant orbit
coverage.  The orbit target is explicitly provisional at this step; final
production I4.1 qualification must recompute the target from retained Core
geometry after I4.1.8 production binding/fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_decoupled_dense_ab import compare_dense_plan_to_baseline
from core.reconstruction.offline_dense_i4_orbit_qualification import orbit_coverage_metrics

I4_1_7_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class I417QualificationPolicy:
    minimum_registered_fraction: float = 0.80
    minimum_dense_alignment_fraction: float = 0.80
    minimum_selected_views: int = 10
    minimum_selected_azimuth_coverage_degrees: float = 150.0
    minimum_selected_occupied_azimuth_bins: int = 5
    azimuth_bin_count: int = 12
    minimum_opposed_view_angle_degrees: float = 120.0
    maximum_facing_angle_degrees: float = 60.0
    minimum_target_facing_fraction: float = 0.75
    maximum_standoff_p90_to_p10_ratio: float = 2.50

    def __post_init__(self) -> None:
        for name in (
            "minimum_registered_fraction",
            "minimum_dense_alignment_fraction",
            "minimum_target_facing_fraction",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ReconstructionValidationError(f"I4.1.7 {name} must be finite in [0,1].")
        if int(self.minimum_selected_views) < 3:
            raise ReconstructionValidationError("I4.1.7 minimum_selected_views must be >=3.")
        if int(self.azimuth_bin_count) < 4:
            raise ReconstructionValidationError("I4.1.7 azimuth_bin_count must be >=4.")
        if not 1 <= int(self.minimum_selected_occupied_azimuth_bins) <= int(self.azimuth_bin_count):
            raise ReconstructionValidationError("I4.1.7 occupied azimuth-bin requirement is invalid.")
        for name in (
            "minimum_selected_azimuth_coverage_degrees",
            "minimum_opposed_view_angle_degrees",
            "maximum_facing_angle_degrees",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 180.0:
                raise ReconstructionValidationError(f"I4.1.7 {name} must be finite in [0,180].")
        ratio = float(self.maximum_standoff_p90_to_p10_ratio)
        if not isfinite(ratio) or ratio < 1.0:
            raise ReconstructionValidationError("I4.1.7 standoff ratio must be finite and >=1.")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def validate_prior_decoupled_evidence(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate I4.1.6B guardrails without requiring Horse itself to pass."""
    if not isinstance(summary, Mapping) or str(summary.get("phase")) != "I4.1.6B":
        raise ReconstructionValidationError("I4.1.7 requires an I4.1.6B summary.")
    decision = summary.get("decision")
    if not isinstance(decision, Mapping):
        raise ReconstructionValidationError("I4.1.7 prior I4.1.6B decision is missing.")
    checks = {
        "single_authoritative_sfm_frame": bool(decision.get("single_authoritative_sfm_frame")),
        "production_default_unchanged": not bool(decision.get("production_default_changed")),
        "production_sampling_not_bound": not bool(decision.get("production_sampling_integration_authorized")),
        "dense_thresholds_not_relaxed": not bool(decision.get("dense_consistency_threshold_relaxation_authorized")),
        "metric_scale_not_claimed": not bool(decision.get("metric_scale_claimed")),
        "mesh_not_authorized": not bool(decision.get("mesh_authorization")),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise ReconstructionValidationError(f"I4.1.7 prior guardrails failed: {failed!r}")
    return {
        "guardrails_pass": True,
        "checks": checks,
        "prior_real_dense_ab_validation_pass": bool(decision.get("real_dense_ab_validation_pass")),
        "prior_recommended_plan": str(decision.get("recommended_plan", "")),
    }


def selected_plan_payload(summary: Mapping[str, Any], plan_name: str) -> dict[str, Any]:
    plans = summary.get("plans")
    if not isinstance(plans, list):
        raise ReconstructionValidationError("I4.1.7 I4.1.6B plans are missing.")
    for item in plans:
        if isinstance(item, Mapping) and str(item.get("plan_name")) == str(plan_name):
            return dict(item)
    raise ReconstructionValidationError(f"I4.1.7 plan {plan_name!r} is not present in I4.1.6B evidence.")


def assemble_selected_pose_sequence(
    selected_source_indices: Sequence[int],
    *,
    global_pose_by_source: Mapping[int, object],
    localized_pose_by_source: Mapping[int, object],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Resolve selected observations into one authoritative reconstruction frame."""
    selected = tuple(int(v) for v in selected_source_indices)
    if not selected or len(set(selected)) != len(selected):
        raise ReconstructionValidationError("I4.1.7 selected source indices must be non-empty and unique.")
    matrices: list[np.ndarray] = []
    provenance: list[str] = []
    for source_index in selected:
        if source_index in global_pose_by_source:
            raw = global_pose_by_source[source_index]
            origin = "global_sfm_anchor"
        elif source_index in localized_pose_by_source:
            raw = localized_pose_by_source[source_index]
            origin = "localized_extra_observation"
        else:
            raise ReconstructionValidationError(
                f"I4.1.7 selected source index {source_index} has no authoritative/global or localized pose."
            )
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ReconstructionValidationError(f"I4.1.7 pose for source {source_index} must be finite 4x4.")
        if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
            raise ReconstructionValidationError(f"I4.1.7 pose for source {source_index} has invalid homogeneous row.")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            float(np.linalg.det(rotation)), 1.0, atol=1e-5
        ):
            raise ReconstructionValidationError(f"I4.1.7 pose for source {source_index} is not rigid.")
        matrices.append(matrix)
        provenance.append(origin)
    return np.asarray(matrices, dtype=np.float64), tuple(provenance)


def build_clean_static_orbit_assessment(
    *,
    input_global_views: int,
    registered_global_views: int,
    registered_camera_matrices: object,
    selected_camera_matrices: object,
    selected_source_indices: Sequence[int],
    selected_pose_provenance: Sequence[str],
    provisional_target_reconstruction: object,
    target_diagnostics: Mapping[str, Any],
    provisional_target_mode: str = "provisional_optical_axis_consensus",
    target_correction_evidence: Mapping[str, Any] | None = None,
    plan_payload: Mapping[str, Any],
    baseline_payload: Mapping[str, Any],
    prior_guardrails: Mapping[str, Any],
    capture_confirmed_static: bool,
    policy: I417QualificationPolicy | None = None,
) -> dict[str, Any]:
    """Build the fail-closed I4.1.7 clean-capture qualification decision."""
    p = policy or I417QualificationPolicy()
    target_mode = str(provisional_target_mode).strip()
    if target_mode not in {
        "provisional_optical_axis_consensus",
        "provisional_registered_trajectory_circle_center",
    }:
        raise ReconstructionValidationError(f"I4.1.7 unsupported provisional target mode: {target_mode!r}.")
    if target_correction_evidence is not None and not isinstance(target_correction_evidence, Mapping):
        raise ReconstructionValidationError("I4.1.7 target correction evidence must be a mapping when provided.")
    input_views = int(input_global_views)
    registered_views = int(registered_global_views)
    if input_views <= 0 or registered_views <= 0 or registered_views > input_views:
        raise ReconstructionValidationError("I4.1.7 global SfM view accounting is invalid.")
    registered_fraction = float(registered_views / input_views)

    planned = int(plan_payload.get("planned_candidates", 0))
    aligned = int(plan_payload.get("alignment_accepted", 0))
    alignment_fraction = float(plan_payload.get("alignment_fraction", aligned / max(1, planned)))
    selected_count = int(plan_payload.get("selected_qualified_views", 0))
    selected_sources = tuple(int(v) for v in selected_source_indices)
    if selected_count != len(selected_sources):
        raise ReconstructionValidationError("I4.1.7 selected-view accounting differs from plan evidence.")
    if len(tuple(selected_pose_provenance)) != selected_count:
        raise ReconstructionValidationError("I4.1.7 selected pose provenance is misaligned.")

    pair_fraction = plan_payload.get("pair_pass_fraction")
    baseline_pair = baseline_payload.get("pair_pass_fraction")
    baseline_selected = int(baseline_payload.get("selected_qualified_views", 0))
    if pair_fraction is None or baseline_pair is None:
        comparison = {
            "comparison_pass": False,
            "comparison_rule": "missing_pair_pass_fraction_evidence",
            "selected_view_gain_over_global_only": selected_count - baseline_selected,
            "pair_pass_fraction_gain_over_global_only": None,
            "baseline_pair_pass_fraction_saturated": False,
            "pair_quality_regressed": False,
            "pair_quality_requirement_pass": False,
        }
    else:
        comparison = compare_dense_plan_to_baseline(
            selected_views=selected_count,
            pair_pass_fraction=float(pair_fraction),
            baseline_selected_views=baseline_selected,
            baseline_pair_pass_fraction=float(baseline_pair),
        )
    selected_gain = int(comparison["selected_view_gain_over_global_only"])
    pair_gain = comparison["pair_pass_fraction_gain_over_global_only"]
    real_dense_plan_pass = bool(plan_payload.get("dense_gate_pass") and comparison["comparison_pass"])

    registered = np.asarray(registered_camera_matrices, dtype=np.float64)
    selected = np.asarray(selected_camera_matrices, dtype=np.float64)
    target = np.asarray(provisional_target_reconstruction, dtype=np.float64).reshape(-1)
    if registered.ndim != 3 or registered.shape[1:] != (4, 4) or len(registered) < 3:
        raise ReconstructionValidationError("I4.1.7 registered camera matrices are invalid.")
    if selected_count >= 3:
        if selected.shape != (selected_count, 4, 4):
            raise ReconstructionValidationError("I4.1.7 selected camera matrices are misaligned.")
        selected_metrics = orbit_coverage_metrics(
            selected,
            target,
            azimuth_bin_count=int(p.azimuth_bin_count),
            facing_angle_degrees=float(p.maximum_facing_angle_degrees),
        )
    else:
        selected_metrics = None
    registered_metrics = orbit_coverage_metrics(
        registered,
        target,
        azimuth_bin_count=int(p.azimuth_bin_count),
        facing_angle_degrees=float(p.maximum_facing_angle_degrees),
    )

    selected_orbit_gates = {
        "selected_azimuth_coverage": bool(
            selected_metrics is not None
            and float(selected_metrics["azimuth_coverage_degrees"]) >= float(p.minimum_selected_azimuth_coverage_degrees)
        ),
        "selected_azimuth_bins": bool(
            selected_metrics is not None
            and int(selected_metrics["occupied_azimuth_bins"]) >= int(p.minimum_selected_occupied_azimuth_bins)
        ),
        "selected_opposed_view_diversity": bool(
            selected_metrics is not None
            and float(selected_metrics["maximum_opposed_view_angle_degrees"]) >= float(p.minimum_opposed_view_angle_degrees)
        ),
        "selected_target_facing": bool(
            selected_metrics is not None
            and float(selected_metrics["target_facing_fraction"]) >= float(p.minimum_target_facing_fraction)
        ),
        "selected_standoff_consistency": bool(
            selected_metrics is not None
            and float(selected_metrics["standoff_p90_to_p10_ratio"]) <= float(p.maximum_standoff_p90_to_p10_ratio)
        ),
    }
    gates = {
        "prior_decoupled_guardrails": bool(prior_guardrails.get("guardrails_pass")),
        "capture_confirmed_static": bool(capture_confirmed_static),
        "global_sfm_registered_fraction": registered_fraction >= float(p.minimum_registered_fraction),
        "full_span_candidate_plan": bool(plan_payload.get("full_span")),
        "dense_alignment_fraction": alignment_fraction >= float(p.minimum_dense_alignment_fraction),
        "selected_view_count": selected_count >= int(p.minimum_selected_views),
        "real_dense_plan_improves_over_global_only": real_dense_plan_pass,
        **selected_orbit_gates,
    }
    passed = all(bool(value) for value in gates.values())
    return {
        "pass": passed,
        "policy": p.as_dict(),
        "plan_name": str(plan_payload.get("plan_name", "")),
        "counts": {
            "input_global_views": input_views,
            "registered_global_views": registered_views,
            "registered_fraction": registered_fraction,
            "planned_dense_candidates": planned,
            "dense_alignment_accepted_views": aligned,
            "dense_alignment_fraction": alignment_fraction,
            "selected_qualified_views": selected_count,
            "baseline_selected_qualified_views": baseline_selected,
            "selected_view_gain_over_global_only": selected_gain,
            "pair_pass_fraction": pair_fraction,
            "baseline_pair_pass_fraction": baseline_pair,
            "pair_pass_fraction_gain_over_global_only": pair_gain,
            "dense_baseline_comparison_rule": comparison["comparison_rule"],
            "baseline_pair_pass_fraction_saturated": comparison["baseline_pair_pass_fraction_saturated"],
            "pair_quality_regressed": comparison["pair_quality_regressed"],
            "pair_quality_requirement_pass": comparison["pair_quality_requirement_pass"],
        },
        "target": {
            "mode": target_mode,
            "target_reconstruction": [float(v) for v in target],
            "diagnostics": dict(target_diagnostics),
            "correction_evidence": None if target_correction_evidence is None else dict(target_correction_evidence),
            "production_target_authority": False,
            "i4_1_8_recompute_from_retained_core_required": True,
        },
        "registered_orbit": registered_metrics,
        "selected_orbit": selected_metrics,
        "selected_source_sequence_indices": list(selected_sources),
        "selected_pose_provenance": list(selected_pose_provenance),
        "gates": gates,
        "authority": {
            "single_authoritative_sfm_frame": True,
            "production_sampling_integration_authorized": False,
            "production_default_changed": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
            "mesh_authorization": False,
        },
        "decision": {
            "i4_1_8_production_binding_authorized": passed,
            "i4_2_detail_preservation_authorized": False,
            "mesh_authorization": False,
        },
    }


__all__ = [
    "I4_1_7_SCHEMA_VERSION",
    "I417QualificationPolicy",
    "assemble_selected_pose_sequence",
    "build_clean_static_orbit_assessment",
    "selected_plan_payload",
    "validate_prior_decoupled_evidence",
]
