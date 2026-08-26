"""I4.1.8 bounded production binding and retained-Core final qualification.

This module contains the pure, dependency-light policy used by I4.1.8.  The
heavy runner consumes already-qualified I4.1.7B evidence, binds the winning
decoupled dense plan under the unchanged final fusion cap, then reruns the
existing fusion/F/G/I2 production kernels.  Final I4.1 orbit qualification is
recomputed from the *retained Core geometry* produced by that bound scene.

Authority remains fail-closed: arbitrary SfM gauge only, no metric or
measurement promotion, no mesh, no threshold relaxation, and no silent
production-default change.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_qualification import (
    I4OrbitQualificationPolicy,
    orbit_coverage_metrics,
    robust_object_target,
)

I4_1_8_SCHEMA_VERSION = 1
DEFAULT_FINAL_FUSION_CAP = 40


@dataclass(frozen=True, slots=True)
class I418ProductionBinding:
    """Immutable description of the bounded production view binding."""

    plan_name: str
    qualified_source_count: int
    final_fusion_cap: int
    selected_positions: tuple[int, ...]
    selected_source_sequence_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not str(self.plan_name).strip():
            raise ReconstructionValidationError("I4.1.8 production plan name must be non-empty.")
        if int(self.qualified_source_count) < 10:
            raise ReconstructionValidationError("I4.1.8 requires at least 10 qualified plan views.")
        if int(self.final_fusion_cap) < 10:
            raise ReconstructionValidationError("I4.1.8 final fusion cap must be at least 10.")
        if len(self.selected_positions) != len(self.selected_source_sequence_indices):
            raise ReconstructionValidationError("I4.1.8 production binding positions/sources are misaligned.")
        if not self.selected_positions:
            raise ReconstructionValidationError("I4.1.8 production binding selected no views.")
        if len(self.selected_positions) > int(self.final_fusion_cap):
            raise ReconstructionValidationError("I4.1.8 production binding exceeds the final fusion cap.")
        if tuple(sorted(self.selected_positions)) != self.selected_positions:
            raise ReconstructionValidationError("I4.1.8 selected positions must be strictly ordered.")
        if len(set(self.selected_positions)) != len(self.selected_positions):
            raise ReconstructionValidationError("I4.1.8 selected positions must be unique.")
        if tuple(sorted(self.selected_source_sequence_indices)) != self.selected_source_sequence_indices:
            raise ReconstructionValidationError("I4.1.8 selected source indices must be monotonic.")
        if len(set(self.selected_source_sequence_indices)) != len(self.selected_source_sequence_indices):
            raise ReconstructionValidationError("I4.1.8 selected source indices must be unique.")
        if self.selected_positions[0] != 0 or self.selected_positions[-1] != self.qualified_source_count - 1:
            raise ReconstructionValidationError("I4.1.8 bounded binding must preserve both qualified path endpoints.")

    @property
    def selected_view_count(self) -> int:
        return len(self.selected_positions)

    @property
    def source_span_preserved(self) -> bool:
        return self.selected_positions[0] == 0 and self.selected_positions[-1] == self.qualified_source_count - 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "qualified_source_count": int(self.qualified_source_count),
            "final_fusion_cap": int(self.final_fusion_cap),
            "selected_view_count": int(self.selected_view_count),
            "selected_positions": list(self.selected_positions),
            "selected_source_sequence_indices": list(self.selected_source_sequence_indices),
            "selection_policy": "deterministic_uniform_over_qualified_path_preserve_endpoints",
            "source_span_preserved": bool(self.source_span_preserved),
        }


def _strict_monotonic_sources(values: Sequence[int]) -> tuple[int, ...]:
    sources = tuple(int(v) for v in values)
    if len(sources) < 10:
        raise ReconstructionValidationError("I4.1.8 qualified plan must contain at least 10 sources.")
    if tuple(sorted(sources)) != sources or len(set(sources)) != len(sources):
        raise ReconstructionValidationError("I4.1.8 qualified plan source indices must be strictly increasing.")
    return sources


def deterministic_capped_positions(count: int, cap: int) -> tuple[int, ...]:
    """Uniformly downselect an ordered path while preserving both endpoints.

    This selector is intentionally geometry-agnostic.  It never searches for a
    passing subset; the already-qualified path is simply bounded to the
    unchanged production fusion cap in a deterministic, full-span way.
    """

    count = int(count)
    cap = int(cap)
    if count < 10:
        raise ReconstructionValidationError("I4.1.8 requires at least 10 qualified views before capping.")
    if cap < 10:
        raise ReconstructionValidationError("I4.1.8 final fusion cap must be >=10.")
    if count <= cap:
        return tuple(range(count))

    # Rounded linspace is deterministic and preserves endpoints.  Resolve the
    # extremely rare duplicate produced by rounding by monotonic forward fill,
    # then backwards tighten while retaining endpoint coverage.
    raw = np.rint(np.linspace(0, count - 1, cap, dtype=np.float64)).astype(np.int64)
    raw[0] = 0
    raw[-1] = count - 1
    for idx in range(1, len(raw)):
        raw[idx] = max(raw[idx], raw[idx - 1] + 1)
    raw[-1] = count - 1
    for idx in range(len(raw) - 2, -1, -1):
        raw[idx] = min(raw[idx], raw[idx + 1] - 1)
    raw[0] = 0
    if np.any(np.diff(raw) <= 0) or int(raw[-1]) != count - 1:
        raise ReconstructionValidationError("I4.1.8 could not construct a strictly ordered capped subset.")
    return tuple(int(v) for v in raw.tolist())


def build_production_binding(
    *,
    i4_1_7b_summary: Mapping[str, Any],
    plan_name: str,
    qualified_source_sequence_indices: Sequence[int],
    final_fusion_cap: int = DEFAULT_FINAL_FUSION_CAP,
) -> I418ProductionBinding:
    """Validate I4.1.7B authorization and bind a capped production subset."""

    if str(i4_1_7b_summary.get("phase")) != "I4.1.7B":
        raise ReconstructionValidationError("I4.1.8 requires an I4.1.7B summary.")
    if str(i4_1_7b_summary.get("status")) != "qualified_for_i4_1_8":
        raise ReconstructionValidationError("I4.1.8 requires I4.1.7B status=qualified_for_i4_1_8.")
    decision = i4_1_7b_summary.get("decision")
    if not isinstance(decision, Mapping) or not bool(decision.get("i4_1_8_production_binding_authorized")):
        raise ReconstructionValidationError("I4.1.8 requires explicit I4.1.7B production-binding authorization.")
    authority = i4_1_7b_summary.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.1.8 I4.1.7B authority evidence is missing.")
    forbidden_true = (
        "metric_scale_claimed",
        "measurement_authoritative",
        "surface_mesh_generated",
        "dense_consistency_threshold_relaxation_authorized",
    )
    for name in forbidden_true:
        if bool(authority.get(name)):
            raise ReconstructionValidationError(f"I4.1.8 refuses I4.1.7B authority drift: {name}=True.")
    if not bool(decision.get("i4_1_8_core_target_requalification_required")):
        raise ReconstructionValidationError("I4.1.8 requires retained-Core target requalification to remain mandatory.")

    sources = _strict_monotonic_sources(qualified_source_sequence_indices)
    positions = deterministic_capped_positions(len(sources), int(final_fusion_cap))
    selected_sources = tuple(sources[pos] for pos in positions)
    return I418ProductionBinding(
        plan_name=str(plan_name).strip(),
        qualified_source_count=len(sources),
        final_fusion_cap=int(final_fusion_cap),
        selected_positions=positions,
        selected_source_sequence_indices=selected_sources,
    )


def _pose_array(value: object, *, name: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 3 or not np.isfinite(poses).all():
        raise ReconstructionValidationError(f"I4.1.8 {name} poses must be finite Nx4x4 with N>=3.")
    if not np.allclose(poses[:, 3, :], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ReconstructionValidationError(f"I4.1.8 {name} poses have invalid homogeneous rows.")
    return poses


def build_final_i4_1_assessment(
    *,
    input_global_views: int,
    registered_global_views: int,
    production_alignment_accepted_views: int,
    registered_camera_matrices: object,
    production_selected_camera_matrices: object,
    retained_core_points: object,
    i2_summary: Mapping[str, Any],
    binding: I418ProductionBinding,
    policy: I4OrbitQualificationPolicy | None = None,
) -> dict[str, Any]:
    """Perform final I4.1 qualification from the bound scene's retained Core.

    Unlike I4.1.7, there is no A/B-improvement gate here: that experiment is a
    prerequisite already proven by I4.1.7B.  I4.1.8 validates the actual bound
    production subset after fusion/F/G/I2 and recomputes the object target from
    the resulting Core geometry.
    """

    p = policy or I4OrbitQualificationPolicy()
    input_global_views = int(input_global_views)
    registered_global_views = int(registered_global_views)
    production_alignment_accepted_views = int(production_alignment_accepted_views)
    if input_global_views <= 0 or registered_global_views <= 0:
        raise ReconstructionValidationError("I4.1.8 global SfM counts must be positive.")
    if production_alignment_accepted_views <= 0:
        raise ReconstructionValidationError("I4.1.8 production alignment count must be positive.")
    if str(i2_summary.get("phase")) != "I2":
        raise ReconstructionValidationError("I4.1.8 requires an I2 summary for the bound production scene.")

    registered = _pose_array(registered_camera_matrices, name="registered")
    selected = _pose_array(production_selected_camera_matrices, name="production-selected")
    if len(registered) != registered_global_views:
        raise ReconstructionValidationError("I4.1.8 registered pose count disagrees with SfM evidence.")
    if len(selected) != binding.selected_view_count:
        raise ReconstructionValidationError("I4.1.8 production-selected pose count disagrees with binding.")
    if production_alignment_accepted_views != binding.selected_view_count:
        raise ReconstructionValidationError("I4.1.8 production alignment count must equal the bound fusion-view count.")

    target = robust_object_target(retained_core_points)
    registered_metrics = orbit_coverage_metrics(
        registered,
        target,
        azimuth_bin_count=int(p.azimuth_bin_count),
        facing_angle_degrees=float(p.maximum_facing_angle_degrees),
    )
    selected_metrics = orbit_coverage_metrics(
        selected,
        target,
        azimuth_bin_count=int(p.azimuth_bin_count),
        facing_angle_degrees=float(p.maximum_facing_angle_degrees),
    )
    registered_fraction = registered_global_views / float(input_global_views)
    # The bound candidates are all pre-qualified aligned observations.  The
    # denominator is therefore the bound production view set rather than the
    # number of global SfM anchors.
    alignment_fraction = production_alignment_accepted_views / float(binding.selected_view_count)
    i2_pass = bool(i2_summary.get("automated_pass"))

    gates = {
        "i4_1_7b_production_binding_authorized": True,
        "production_binding_preserves_full_qualified_span": bool(binding.source_span_preserved),
        "final_fusion_cap_respected": binding.selected_view_count <= int(binding.final_fusion_cap),
        "ordinary_i2_bound_scene_quality": i2_pass,
        "registered_fraction": registered_fraction >= float(p.minimum_registered_fraction),
        "dense_alignment_fraction": alignment_fraction >= float(p.minimum_dense_alignment_fraction),
        "selected_view_count": binding.selected_view_count >= int(p.minimum_selected_views),
        "selected_azimuth_coverage": float(selected_metrics["azimuth_coverage_degrees"])
        >= float(p.minimum_selected_azimuth_coverage_degrees),
        "selected_azimuth_bins": int(selected_metrics["occupied_azimuth_bins"])
        >= int(p.minimum_selected_occupied_azimuth_bins),
        "selected_opposed_view_diversity": float(selected_metrics["maximum_opposed_view_angle_degrees"])
        >= float(p.minimum_opposed_view_angle_degrees),
        "selected_target_facing": float(selected_metrics["target_facing_fraction"])
        >= float(p.minimum_target_facing_fraction),
        "selected_standoff_consistency": float(selected_metrics["standoff_p90_to_p10_ratio"])
        <= float(p.maximum_standoff_p90_to_p10_ratio),
    }
    passed = all(bool(v) for v in gates.values())
    return {
        "pass": bool(passed),
        "policy": p.as_dict(),
        "binding": binding.as_dict(),
        "counts": {
            "input_global_views": input_global_views,
            "registered_global_views": registered_global_views,
            "production_alignment_accepted_views": production_alignment_accepted_views,
            "production_selected_views": binding.selected_view_count,
            "registered_fraction": float(registered_fraction),
            "dense_alignment_fraction_of_bound_views": float(alignment_fraction),
        },
        "target": {
            "mode": "retained_core_robust_median",
            "target_reconstruction": [float(v) for v in target],
            "production_target_authority": True,
            "source": "I4.1.8 bound production F retained Core geometry",
        },
        "registered_orbit": registered_metrics,
        "selected_orbit": selected_metrics,
        "gates": gates,
        "failed_gates": sorted(name for name, value in gates.items() if not bool(value)),
    }


def build_i4_1_8_decision(
    *,
    binding: I418ProductionBinding,
    final_assessment: Mapping[str, Any],
    fusion_pass: bool,
    filtering_pass: bool,
    persistence_pass: bool,
    i2_automated_pass: bool,
) -> dict[str, Any]:
    """Produce fail-closed I4.1.8 authority/decision metadata."""

    stages_pass = bool(fusion_pass and filtering_pass and persistence_pass and i2_automated_pass)
    final_pass = bool(stages_pass and final_assessment.get("pass"))
    return {
        "phase": "I4.1.8",
        "schema_version": I4_1_8_SCHEMA_VERSION,
        "status": "ready_for_manual_orbit_review" if final_pass else "degraded",
        "automated_i4_1_final_qualification_pass": final_pass,
        "manual_orbit_review_required": True,
        "binding": binding.as_dict(),
        "stage_gates": {
            "production_fusion": bool(fusion_pass),
            "formal_filtering": bool(filtering_pass),
            "canonical_persistence": bool(persistence_pass),
            "ordinary_i2_bound_scene_quality": bool(i2_automated_pass),
        },
        "final_assessment": dict(final_assessment),
        "authority": {
            "single_authoritative_sfm_frame": True,
            "production_binding_executed": bool(stages_pass),
            "production_sampling_integration_authorized": bool(final_pass),
            "production_default_changed": False,
            "final_target_recomputed_from_retained_core": True,
            "provisional_i4_1_7_target_is_final_authority": False,
            "orbit_thresholds_changed": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "filter_policy_changed": False,
            "persistence_policy_changed": False,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "final_i4_1_automated_qualification_pass": bool(final_pass),
            "authorizes_manual_orbit_review": bool(final_pass),
            "i4_2_detail_preservation_authorized": False,
            "mesh_authorization": False,
            "next_action": (
                "Review the bound Dense RGB scene together with the retained-Core target and selected camera orbit. "
                "If manual orbit review passes, close I4.1 and proceed to I4.2 thin-structure/detail preservation."
                if final_pass
                else "Diagnose the failed bound-scene stage or retained-Core orbit gate; do not relax thresholds."
            ),
        },
    }


__all__ = [
    "DEFAULT_FINAL_FUSION_CAP",
    "I4_1_8_SCHEMA_VERSION",
    "I418ProductionBinding",
    "build_final_i4_1_assessment",
    "build_i4_1_8_decision",
    "build_production_binding",
    "deterministic_capped_positions",
]
