"""I4.1.9 visual-reconstruction acceptance separated from geometry qualification.

This module deliberately keeps the existing I4.1.7/I4.1.8 geometric gates
unchanged.  It defines an additive visual path that may bind a broad set of
already alignment-qualified dense observations for fusion/rendering even when
the strict dense pair-consistency/orbit qualification remains degraded.

The visual path never promotes metric scale, measurement authority, certified
metrology, or mesh authority.  Geometric accuracy can therefore be refined
later (for example with calibrated intrinsics and ChArUco evidence) without
misrepresenting the current reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_production_binding import deterministic_capped_positions

I4_1_9_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class I419VisualBinding:
    plan_name: str
    planned_candidate_count: int
    alignment_accepted_count: int
    alignment_fraction: float
    aligned_source_span_fraction: float
    visual_fusion_cap: int
    selected_positions: tuple[int, ...]
    selected_source_sequence_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.plan_name:
            raise ReconstructionValidationError("I4.1.9 visual binding requires a plan name.")
        if self.planned_candidate_count < 1:
            raise ReconstructionValidationError("I4.1.9 planned candidate count must be positive.")
        if not 1 <= self.alignment_accepted_count <= self.planned_candidate_count:
            raise ReconstructionValidationError("I4.1.9 aligned candidate accounting is invalid.")
        if not isfinite(float(self.alignment_fraction)) or not 0.0 <= float(self.alignment_fraction) <= 1.0:
            raise ReconstructionValidationError("I4.1.9 alignment fraction is invalid.")
        if not isfinite(float(self.aligned_source_span_fraction)) or not 0.0 <= float(self.aligned_source_span_fraction) <= 1.0:
            raise ReconstructionValidationError("I4.1.9 aligned source-span fraction is invalid.")
        if self.visual_fusion_cap < 10:
            raise ReconstructionValidationError("I4.1.9 visual fusion cap must be at least 10.")
        if not self.selected_positions or len(self.selected_positions) != len(self.selected_source_sequence_indices):
            raise ReconstructionValidationError("I4.1.9 selected visual binding is empty or inconsistent.")
        if len(self.selected_positions) > self.visual_fusion_cap:
            raise ReconstructionValidationError("I4.1.9 selected visual binding exceeds the fusion cap.")
        if any(b <= a for a, b in zip(self.selected_positions, self.selected_positions[1:])):
            raise ReconstructionValidationError("I4.1.9 selected positions must be strictly increasing.")

    @property
    def selected_view_count(self) -> int:
        return len(self.selected_positions)

    @property
    def source_span_preserved_within_aligned_pool(self) -> bool:
        return bool(
            self.selected_positions[0] == 0
            and self.selected_positions[-1] == self.alignment_accepted_count - 1
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "planned_candidate_count": int(self.planned_candidate_count),
            "alignment_accepted_count": int(self.alignment_accepted_count),
            "alignment_fraction": float(self.alignment_fraction),
            "aligned_source_span_fraction": float(self.aligned_source_span_fraction),
            "visual_fusion_cap": int(self.visual_fusion_cap),
            "selected_view_count": int(self.selected_view_count),
            "selected_positions": list(self.selected_positions),
            "selected_source_sequence_indices": list(self.selected_source_sequence_indices),
            "source_span_preserved_within_aligned_pool": self.source_span_preserved_within_aligned_pool,
        }


def _candidate_rows(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = plan.get("candidate_rows")
    if not isinstance(rows, list) or not rows:
        raise ReconstructionValidationError("I4.1.9 visual plan requires non-empty candidate_rows.")
    result: list[dict[str, Any]] = []
    seen_orders: set[int] = set()
    seen_sources: set[int] = set()
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping):
            raise ReconstructionValidationError("I4.1.9 candidate_rows must contain objects.")
        row = dict(value)
        order = int(row.get("plan_order", index))
        source = int(row.get("source_sequence_index", -1))
        if source < 0 or order < 0:
            raise ReconstructionValidationError("I4.1.9 candidate row has invalid source/order provenance.")
        if order in seen_orders or source in seen_sources:
            raise ReconstructionValidationError("I4.1.9 candidate plan contains duplicate order/source provenance.")
        seen_orders.add(order)
        seen_sources.add(source)
        row["plan_order"] = order
        row["source_sequence_index"] = source
        result.append(row)
    result.sort(key=lambda item: int(item["plan_order"]))
    return tuple(result)


def assess_visual_plan(
    plan: Mapping[str, Any],
    *,
    minimum_alignment_fraction: float = 0.80,
    minimum_aligned_views: int = 10,
    minimum_aligned_source_span_fraction: float = 0.80,
) -> dict[str, Any]:
    """Assess a dense plan for visual fusion without using pair-consistency gates."""
    if not 0.0 < float(minimum_alignment_fraction) <= 1.0:
        raise ReconstructionValidationError("I4.1.9 minimum alignment fraction must be in (0,1].")
    if int(minimum_aligned_views) < 10:
        raise ReconstructionValidationError("I4.1.9 minimum aligned visual views must be at least 10.")
    if not 0.0 < float(minimum_aligned_source_span_fraction) <= 1.0:
        raise ReconstructionValidationError("I4.1.9 minimum source-span fraction must be in (0,1].")

    name = str(plan.get("plan_name", ""))
    if not name:
        raise ReconstructionValidationError("I4.1.9 visual plan has no plan_name.")
    rows = _candidate_rows(plan)
    planned = int(plan.get("planned_candidates", len(rows)))
    if planned != len(rows):
        raise ReconstructionValidationError("I4.1.9 planned candidate count disagrees with candidate_rows.")
    aligned_rows = tuple(row for row in rows if bool(row.get("alignment_accepted")))
    aligned = int(plan.get("alignment_accepted", len(aligned_rows)))
    if aligned != len(aligned_rows):
        raise ReconstructionValidationError("I4.1.9 aligned candidate count disagrees with candidate_rows.")

    candidate_sources = tuple(int(row["source_sequence_index"]) for row in rows)
    aligned_sources = tuple(int(row["source_sequence_index"]) for row in aligned_rows)
    candidate_min, candidate_max = min(candidate_sources), max(candidate_sources)
    if aligned_sources:
        aligned_min, aligned_max = min(aligned_sources), max(aligned_sources)
    else:
        aligned_min = aligned_max = candidate_min
    denominator = max(1, candidate_max - candidate_min)
    span_fraction = float((aligned_max - aligned_min) / denominator) if candidate_max > candidate_min else 1.0
    alignment_fraction = float(aligned / max(1, planned))
    full_span = bool(plan.get("full_span", False))
    gates = {
        "candidate_plan_full_span": full_span,
        "minimum_alignment_fraction": alignment_fraction >= float(minimum_alignment_fraction),
        "minimum_aligned_visual_views": aligned >= int(minimum_aligned_views),
        "minimum_aligned_source_span_fraction": span_fraction >= float(minimum_aligned_source_span_fraction),
    }
    visual_gate_pass = all(gates.values())
    return {
        "plan_name": name,
        "planned_candidates": planned,
        "alignment_accepted": aligned,
        "alignment_fraction": alignment_fraction,
        "aligned_source_span_fraction": span_fraction,
        "aligned_source_sequence_indices": list(aligned_sources),
        "full_span": full_span,
        "visual_gate_pass": visual_gate_pass,
        "gates": gates,
        "strict_pair_consistency_required": False,
        "pair_pass_fraction": plan.get("pair_pass_fraction"),
        "strict_selected_qualified_views": int(plan.get("selected_qualified_views", 0)),
        "note": (
            "Visual acceptance uses only already alignment-qualified observations plus broad source coverage; "
            "the existing strict pair-consistency/orbit qualification remains unchanged and separately reported."
        ),
    }


def choose_visual_plan(
    dense_ab_summary: Mapping[str, Any],
    *,
    minimum_alignment_fraction: float = 0.80,
    minimum_aligned_views: int = 10,
    minimum_aligned_source_span_fraction: float = 0.80,
    visual_fusion_cap: int = 40,
) -> dict[str, Any]:
    """Choose the strongest visual plan without consulting pair-PASS ranking."""
    if str(dense_ab_summary.get("phase")) != "I4.1.6B":
        raise ReconstructionValidationError("I4.1.9 requires I4.1.6B dense evidence.")
    decision = dense_ab_summary.get("decision")
    if not isinstance(decision, Mapping):
        raise ReconstructionValidationError("I4.1.9 requires I4.1.6B decision evidence.")
    if not bool(decision.get("single_authoritative_sfm_frame")):
        raise ReconstructionValidationError("I4.1.9 refuses evidence outside the single authoritative SfM frame.")
    if bool(decision.get("metric_scale_claimed")) or bool(decision.get("measurement_authoritative")):
        raise ReconstructionValidationError("I4.1.9 refuses metric/measurement promotion in visual acceptance.")
    if bool(decision.get("surface_mesh_generated")):
        raise ReconstructionValidationError("I4.1.9 refuses mesh promotion in visual acceptance.")
    if bool(decision.get("production_default_changed")):
        raise ReconstructionValidationError("I4.1.9 refuses a changed production default.")
    thresholds = dense_ab_summary.get("thresholds")
    if not isinstance(thresholds, Mapping) or bool(thresholds.get("threshold_relaxation")):
        raise ReconstructionValidationError("I4.1.9 requires unchanged strict dense-consistency thresholds.")
    if int(visual_fusion_cap) < 10:
        raise ReconstructionValidationError("I4.1.9 visual fusion cap must be at least 10.")

    raw_plans = dense_ab_summary.get("plans")
    if not isinstance(raw_plans, list) or not raw_plans:
        raise ReconstructionValidationError("I4.1.9 dense evidence has no plans.")
    assessments = [
        assess_visual_plan(
            plan,
            minimum_alignment_fraction=minimum_alignment_fraction,
            minimum_aligned_views=minimum_aligned_views,
            minimum_aligned_source_span_fraction=minimum_aligned_source_span_fraction,
        )
        for plan in raw_plans
        if isinstance(plan, Mapping)
    ]
    if not assessments:
        raise ReconstructionValidationError("I4.1.9 found no valid visual-plan evidence.")

    def rank(item: Mapping[str, Any]) -> tuple[int, int, float, float, int, str]:
        aligned = int(item["alignment_accepted"])
        return (
            1 if bool(item["visual_gate_pass"]) else 0,
            min(aligned, int(visual_fusion_cap)),
            float(item["alignment_fraction"]),
            float(item["aligned_source_span_fraction"]),
            aligned,
            str(item["plan_name"]),
        )

    winner = max(assessments, key=rank)
    return {
        "selected_plan": str(winner["plan_name"]),
        "selected_plan_assessment": dict(winner),
        "plan_assessments": assessments,
        "visual_plan_available": bool(winner["visual_gate_pass"]),
        "selection_rule": (
            "visual_gate_then_capped_view_count_then_alignment_fraction_then_source_span; "
            "pair-consistency score is diagnostic only"
        ),
        "strict_dense_ab_recommended_plan": str(decision.get("recommended_plan", "")),
        "strict_dense_ab_validation_pass": bool(decision.get("real_dense_ab_validation_pass")),
        "strict_pair_consistency_required": False,
        "dense_consistency_thresholds_changed": False,
    }


def build_visual_binding(
    visual_plan_selection: Mapping[str, Any],
    *,
    visual_fusion_cap: int,
) -> I419VisualBinding:
    assessment = visual_plan_selection.get("selected_plan_assessment")
    if not isinstance(assessment, Mapping) or not bool(assessment.get("visual_gate_pass")):
        raise ReconstructionValidationError("I4.1.9 visual binding requires a passing visual-plan assessment.")
    sources = tuple(int(v) for v in assessment.get("aligned_source_sequence_indices", ()))
    if len(sources) < 10:
        raise ReconstructionValidationError("I4.1.9 visual binding requires at least 10 aligned sources.")
    positions = deterministic_capped_positions(len(sources), int(visual_fusion_cap))
    selected_sources = tuple(sources[pos] for pos in positions)
    return I419VisualBinding(
        plan_name=str(assessment["plan_name"]),
        planned_candidate_count=int(assessment["planned_candidates"]),
        alignment_accepted_count=int(assessment["alignment_accepted"]),
        alignment_fraction=float(assessment["alignment_fraction"]),
        aligned_source_span_fraction=float(assessment["aligned_source_span_fraction"]),
        visual_fusion_cap=int(visual_fusion_cap),
        selected_positions=positions,
        selected_source_sequence_indices=selected_sources,
    )


def validate_i4_1_7_visual_source(source_summary: Mapping[str, Any]) -> dict[str, Any]:
    if str(source_summary.get("phase")) != "I4.1.7":
        raise ReconstructionValidationError("I4.1.9 requires an I4.1.7 source summary.")
    authority = source_summary.get("authority")
    decision = source_summary.get("decision")
    if not isinstance(authority, Mapping) or not isinstance(decision, Mapping):
        raise ReconstructionValidationError("I4.1.9 source summary lacks authority/decision evidence.")
    if not bool(authority.get("single_authoritative_sfm_frame")):
        raise ReconstructionValidationError("I4.1.9 requires the single authoritative SfM frame.")
    if bool(decision.get("dense_consistency_threshold_relaxation_authorized")):
        raise ReconstructionValidationError("I4.1.9 refuses relaxed dense-consistency thresholds.")
    if bool(authority.get("metric_scale_claimed")) or bool(authority.get("measurement_authoritative")):
        raise ReconstructionValidationError("I4.1.9 refuses metric/measurement authority promotion.")
    if bool(authority.get("certified_metrology")) or bool(authority.get("surface_mesh_generated")):
        raise ReconstructionValidationError("I4.1.9 refuses certified metrology or mesh promotion.")
    return {
        "strict_geometry_qualification_passed": bool(decision.get("clean_static_orbit_qualified")),
        "strict_i4_1_8_binding_authorized": bool(decision.get("i4_1_8_production_binding_authorized")),
        "single_authoritative_sfm_frame": True,
        "dense_consistency_threshold_relaxation_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
    }


def build_visual_acceptance_decision(
    *,
    source_summary: Mapping[str, Any],
    visual_plan_selection: Mapping[str, Any],
    binding: I419VisualBinding | None,
    fusion_pass: bool,
    filtering_pass: bool,
    persistence_pass: bool,
    i2_automated_pass: bool,
) -> dict[str, Any]:
    source = validate_i4_1_7_visual_source(source_summary)
    plan_pass = bool(visual_plan_selection.get("visual_plan_available"))
    binding_pass = bool(
        binding is not None
        and binding.selected_view_count >= 10
        and binding.source_span_preserved_within_aligned_pool
    )
    gates = {
        "visual_plan_available": plan_pass,
        "visual_binding_minimum_views_and_span": binding_pass,
        "visual_fusion": bool(fusion_pass),
        "formal_filtering": bool(filtering_pass),
        "canonical_persistence": bool(persistence_pass),
        "ordinary_i2_visual_spatial_quality": bool(i2_automated_pass),
    }
    automated_pass = all(gates.values())
    strict_geometry_pass = bool(source["strict_geometry_qualification_passed"])
    return {
        "phase": "I4.1.9",
        "schema_version": I4_1_9_SCHEMA_VERSION,
        "status": "ready_for_manual_visual_review" if automated_pass else "degraded",
        "automated_visual_reconstruction_pass": automated_pass,
        "manual_visual_review_required": True,
        "visual_gates": gates,
        "visual_plan_selection": dict(visual_plan_selection),
        "visual_binding": None if binding is None else binding.as_dict(),
        "geometry_qualification": {
            "strict_geometry_qualification_passed": strict_geometry_pass,
            "geometry_accuracy_qualified": strict_geometry_pass,
            "geometry_accuracy_deferred": not strict_geometry_pass,
            "strict_pair_consistency_gate_changed": False,
            "strict_orbit_gate_changed": False,
            "dense_consistency_thresholds_changed": False,
            "note": (
                "A canonical arbitrary-gauge geometry may be used for rendering/picking while accuracy qualification "
                "remains separately degraded/deferred."
            ),
        },
        "authority": {
            "single_authoritative_sfm_frame": True,
            "visual_reconstruction_authorized_for_manual_review": automated_pass,
            "canonical_coordinate_frame_for_rendering": bool(persistence_pass),
            "production_sampling_integration_authorized": False,
            "production_default_changed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "i4_1_visual_reconstruction_closure_authorized": automated_pass,
            "i4_2_detail_preservation_authorized": False,
            "geometry_accuracy_followup_required": not strict_geometry_pass,
            "camera_calibration_metric_refinement_deferred": not strict_geometry_pass,
            "mesh_authorization": False,
        },
    }


def finalize_visual_acceptance(
    automated_summary: Mapping[str, Any],
    *,
    manual_visual_review_pass: bool,
) -> dict[str, Any]:
    if str(automated_summary.get("phase")) != "I4.1.9":
        raise ReconstructionValidationError("I4.1.9 closure requires an I4.1.9 automated summary.")
    if not bool(automated_summary.get("automated_visual_reconstruction_pass")):
        raise ReconstructionValidationError("I4.1.9 closure requires passing automated visual reconstruction gates.")
    if not bool(manual_visual_review_pass):
        raise ReconstructionValidationError("I4.1.9 closure requires explicit manual visual review PASS.")
    geometry = automated_summary.get("geometry_qualification")
    if not isinstance(geometry, Mapping):
        raise ReconstructionValidationError("I4.1.9 closure lacks geometry-qualification separation evidence.")
    strict_geometry_pass = bool(geometry.get("strict_geometry_qualification_passed"))
    return {
        "phase": "I4.1.9-CLOSURE",
        "schema_version": I4_1_9_SCHEMA_VERSION,
        "status": "visual_reconstruction_qualified_geometry_deferred" if not strict_geometry_pass else "visual_and_geometry_qualified",
        "visual_reconstruction_qualified": True,
        "manual_visual_review": "pass",
        "strict_geometry_qualification_passed": strict_geometry_pass,
        "geometry_accuracy_deferred": not strict_geometry_pass,
        "authority": {
            "single_authoritative_sfm_frame": True,
            "canonical_coordinate_frame_for_rendering": True,
            "visual_reconstruction_qualified": True,
            "production_sampling_integration_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "i4_1_visual_reconstruction_closed": True,
            "i4_2_detail_preservation_authorized": True,
            "geometry_accuracy_followup_required": not strict_geometry_pass,
            "calibration_and_metric_refinement_remain_future_work": True,
            "mesh_authorization": False,
        },
        "next": (
            "Proceed to I4.2 thin-structure/detail preservation.  Keep strict geometry/metric qualification separate; "
            "later camera calibration and ChArUco/reference evidence may refine intrinsics, scale, and measurement authority."
        ),
    }


__all__ = [
    "I4_1_9_SCHEMA_VERSION",
    "I419VisualBinding",
    "assess_visual_plan",
    "build_visual_acceptance_decision",
    "build_visual_binding",
    "choose_visual_plan",
    "finalize_visual_acceptance",
    "validate_i4_1_7_visual_source",
]
