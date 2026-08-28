"""I4.3G1 controlled cross-provider qualification framework.

G1 is methodology-only. It freezes the comparison scenarios, evidence schema,
metric registry, fairness rules, and fail-closed predecessor gates for G2-G5.
It deliberately does not execute a provider, score a winner, implement Auto,
authorize hybrid fusion, or modify the authoritative PyCOLMAP SfM frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence

from core.reconstruction.dense_provider_architecture import (
    I43A_AUTHORITATIVE_FRAME_OWNER,
    DenseProviderFamily,
    DenseProviderImplementation,
    dense_provider_implementation_contracts,
)
from core.reconstruction.errors import ReconstructionValidationError

I43G1_SCHEMA_VERSION = 1
I43G1_PHASE = "I4.3G1"


class ComparisonMode(str, Enum):
    PRODUCTION_NATIVE = "production_native"
    MATCHED_VIEW_DIAGNOSTIC = "matched_view_diagnostic"
    HISTORICAL_READ_ONLY = "historical_read_only"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    BOOLEAN_GATE_TRUE = "boolean_gate_true"
    INFORMATIONAL = "informational"


class MetricCategory(str, Enum):
    AUTHORITY = "authority"
    QUALITY = "quality"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ARTIFACTS = "artifacts"
    RESOURCES = "resources"
    RUNTIME = "runtime"
    COVERAGE = "coverage"
    PROVENANCE = "provenance"


@dataclass(frozen=True, slots=True)
class QualificationMetricSpec:
    metric_id: str
    category: MetricCategory
    direction: MetricDirection
    unit: str
    required: bool
    description: str
    comparable_across_families: bool = True

    def __post_init__(self) -> None:
        metric_id = str(self.metric_id).strip()
        unit = str(self.unit).strip()
        description = str(self.description).strip()
        if not metric_id or not description:
            raise ReconstructionValidationError("G1 metric id/description must be non-empty.")
        if not isinstance(self.category, MetricCategory):
            raise TypeError("category must be MetricCategory.")
        if not isinstance(self.direction, MetricDirection):
            raise TypeError("direction must be MetricDirection.")
        if not isinstance(self.required, bool) or not isinstance(self.comparable_across_families, bool):
            raise TypeError("metric booleans must be bool values.")
        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "unit", unit or "none")
        object.__setattr__(self, "description", description)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "category": self.category.value,
            "direction": self.direction.value,
            "unit": self.unit,
            "required": self.required,
            "description": self.description,
            "comparable_across_families": self.comparable_across_families,
        }


@dataclass(frozen=True, slots=True)
class QualificationScenarioSpec:
    scenario_id: str
    substep: str
    label: str
    provider_implementations: tuple[DenseProviderImplementation, ...]
    source_kind: str
    comparison_modes: tuple[ComparisonMode, ...]
    required: bool
    new_real_execution_required: bool
    manual_review_required: bool
    temporal_order_required: bool
    texture_regime: str
    overlap_regime: str
    purpose: str
    manual_review_dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id).strip()
        substep = str(self.substep).strip()
        label = str(self.label).strip()
        source_kind = str(self.source_kind).strip()
        purpose = str(self.purpose).strip()
        texture = str(self.texture_regime).strip()
        overlap = str(self.overlap_regime).strip()
        if not all((scenario_id, substep, label, source_kind, purpose, texture, overlap)):
            raise ReconstructionValidationError("G1 scenario strings must be non-empty.")
        if substep not in {"I4.3G2", "I4.3G3", "I4.3G4"}:
            raise ReconstructionValidationError("G1 scenarios must belong to G2, G3, or G4.")
        if len(self.provider_implementations) < 2:
            raise ReconstructionValidationError("Cross-provider scenarios require at least two providers.")
        if len(set(self.provider_implementations)) != len(self.provider_implementations):
            raise ReconstructionValidationError("Scenario provider implementations must be unique.")
        if any(not isinstance(item, DenseProviderImplementation) for item in self.provider_implementations):
            raise TypeError("provider_implementations must contain DenseProviderImplementation values.")
        if not self.comparison_modes or any(not isinstance(item, ComparisonMode) for item in self.comparison_modes):
            raise ReconstructionValidationError("Each scenario requires valid comparison modes.")
        if ComparisonMode.HISTORICAL_READ_ONLY in self.comparison_modes and self.new_real_execution_required:
            raise ReconstructionValidationError("Historical-only scenarios cannot require a new real execution.")
        temporal_impls = {
            DenseProviderImplementation.ROLLING_DEPTH,
            DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
        }
        if temporal_impls.intersection(self.provider_implementations) and not self.temporal_order_required:
            raise ReconstructionValidationError("Temporal-provider scenarios must preserve source order.")
        if self.temporal_order_required and not temporal_impls.intersection(self.provider_implementations):
            raise ReconstructionValidationError("temporal_order_required is only valid for temporal providers.")
        if self.manual_review_required and not self.manual_review_dimensions:
            raise ReconstructionValidationError("Manual-review scenarios require frozen review dimensions.")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "substep", substep)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "texture_regime", texture)
        object.__setattr__(self, "overlap_regime", overlap)
        object.__setattr__(self, "manual_review_dimensions", tuple(str(v).strip() for v in self.manual_review_dimensions))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "substep": self.substep,
            "label": self.label,
            "provider_implementations": [item.value for item in self.provider_implementations],
            "source_kind": self.source_kind,
            "comparison_modes": [item.value for item in self.comparison_modes],
            "required": self.required,
            "new_real_execution_required": self.new_real_execution_required,
            "manual_review_required": self.manual_review_required,
            "temporal_order_required": self.temporal_order_required,
            "texture_regime": self.texture_regime,
            "overlap_regime": self.overlap_regime,
            "purpose": self.purpose,
            "manual_review_dimensions": list(self.manual_review_dimensions),
        }


_STATIC_MANUAL = (
    "global_completeness",
    "legitimate_hole_recovery",
    "false_geometry_and_floaters",
    "surface_noise_and_roughness",
    "thin_structure_preservation",
    "edge_definition",
    "small_detail_preservation",
    "false_bridging_or_thickening",
)
_TEMPORAL_MANUAL = (
    "temporal_flicker",
    "temporal_drift",
    "static_geometry_consistency",
    "motion_boundary_stability",
    "transient_artifacts",
    "textureless_region_stability",
)
_SINGLE_IMAGE_MANUAL = (
    "global_completeness",
    "local_shape_plausibility",
    "edge_definition",
    "small_detail_preservation",
    "texture_sensitivity",
    "cross_view_stability_after_sfm_alignment",
    "false_geometry_and_floaters",
)


def qualification_metric_registry() -> tuple[QualificationMetricSpec, ...]:
    """Frozen common metric vocabulary for G2-G5.

    G1 intentionally avoids a universal scalar score. Metrics retain their own
    direction and semantics so G5 can produce scenario-specific operating
    envelopes rather than a leaderboard that hides trade-offs.
    """

    return (
        QualificationMetricSpec(
            "execution_completed", MetricCategory.PROVENANCE, MetricDirection.BOOLEAN_GATE_TRUE,
            "bool", True, "Provider execution and canonical finalization completed without fallback substitution."
        ),
        QualificationMetricSpec(
            "source_input_identity_preserved", MetricCategory.PROVENANCE, MetricDirection.BOOLEAN_GATE_TRUE,
            "bool", True, "Input asset hashes match the frozen scenario manifest."
        ),
        QualificationMetricSpec(
            "authoritative_sfm_preserved", MetricCategory.AUTHORITY, MetricDirection.BOOLEAN_GATE_TRUE,
            "bool", True, "Authoritative PyCOLMAP cameras/frame and frozen sparse evidence remain unchanged."
        ),
        QualificationMetricSpec(
            "adaptive_plan_bound", MetricCategory.PROVENANCE, MetricDirection.BOOLEAN_GATE_TRUE,
            "bool", True, "Execution is bound to the persisted F adaptive-view plan or the frozen matched-view diagnostic plan."
        ),
        QualificationMetricSpec(
            "provider_provenance_complete", MetricCategory.PROVENANCE, MetricDirection.BOOLEAN_GATE_TRUE,
            "bool", True, "Provider implementation, backend/checkpoint identity, parameters and artifact hashes are complete."
        ),
        QualificationMetricSpec(
            "forbidden_authority_claims_absent", MetricCategory.AUTHORITY, MetricDirection.BOOLEAN_GATE_TRUE,
            "bool", True, "No metric-scale, measurement, metrology, mesh, Auto, hybrid, or provider-promotion authority is introduced."
        ),
        QualificationMetricSpec(
            "wall_time_s", MetricCategory.RUNTIME, MetricDirection.LOWER_IS_BETTER,
            "s", True, "Measured wall time from provider start through canonical finalization."
        ),
        QualificationMetricSpec(
            "peak_vram_mib", MetricCategory.RESOURCES, MetricDirection.LOWER_IS_BETTER,
            "MiB", True, "Best available measured peak GPU memory evidence, with measurement scope persisted."
        ),
        QualificationMetricSpec(
            "peak_host_memory_mib", MetricCategory.RESOURCES, MetricDirection.LOWER_IS_BETTER,
            "MiB", False, "Best available measured peak host-memory evidence, with measurement scope persisted."
        ),
        QualificationMetricSpec(
            "selected_view_count", MetricCategory.COVERAGE, MetricDirection.INFORMATIONAL,
            "views", True, "Number of views selected by the frozen plan for the provider execution."
        ),
        QualificationMetricSpec(
            "processed_view_count", MetricCategory.COVERAGE, MetricDirection.INFORMATIONAL,
            "views", True, "Number of views actually consumed successfully by the provider."
        ),
        QualificationMetricSpec(
            "aligned_view_fraction", MetricCategory.CONSISTENCY, MetricDirection.HIGHER_IS_BETTER,
            "fraction", True, "Fraction of selected views with valid provider-to-authoritative-SfM alignment evidence."
        ),
        QualificationMetricSpec(
            "sparse_correspondence_coverage", MetricCategory.COVERAGE, MetricDirection.HIGHER_IS_BETTER,
            "fraction", False, "Fraction of eligible authoritative sparse correspondences participating in provider alignment/validation."
        ),
        QualificationMetricSpec(
            "alignment_median_relative_residual", MetricCategory.CONSISTENCY, MetricDirection.LOWER_IS_BETTER,
            "ratio", False, "Median relative residual after the provider's authorized per-view/sequence alignment into the SfM gauge."
        ),
        QualificationMetricSpec(
            "alignment_p95_relative_residual", MetricCategory.CONSISTENCY, MetricDirection.LOWER_IS_BETTER,
            "ratio", False, "95th-percentile relative alignment residual in the SfM gauge."
        ),
        QualificationMetricSpec(
            "inter_view_consistency_ratio", MetricCategory.CONSISTENCY, MetricDirection.HIGHER_IS_BETTER,
            "fraction", False, "Common-view consistency support ratio after canonical alignment."
        ),
        QualificationMetricSpec(
            "inter_view_median_relative_residual", MetricCategory.CONSISTENCY, MetricDirection.LOWER_IS_BETTER,
            "ratio", False, "Median common-view relative depth/geometry consistency residual."
        ),
        QualificationMetricSpec(
            "registered_view_surface_coverage", MetricCategory.COMPLETENESS, MetricDirection.HIGHER_IS_BETTER,
            "fraction", False, "Fraction of authoritative registered views receiving usable canonical dense support."
        ),
        QualificationMetricSpec(
            "canonical_point_count", MetricCategory.COMPLETENESS, MetricDirection.INFORMATIONAL,
            "points", False, "Canonical point count before any later mesh stage; never interpreted alone as quality."
        ),
        QualificationMetricSpec(
            "detached_component_fraction", MetricCategory.ARTIFACTS, MetricDirection.LOWER_IS_BETTER,
            "fraction", False, "Fraction of output geometry classified as detached/isolated under the frozen diagnostic policy."
        ),
        QualificationMetricSpec(
            "unsupported_geometry_fraction", MetricCategory.ARTIFACTS, MetricDirection.LOWER_IS_BETTER,
            "fraction", False, "Fraction of canonical geometry lacking the frozen minimum support evidence."
        ),
    )


def _providers(*items: DenseProviderImplementation) -> tuple[DenseProviderImplementation, ...]:
    return tuple(items)


def qualification_scenario_matrix() -> tuple[QualificationScenarioSpec, ...]:
    """Return the frozen G2-G4 scenario matrix.

    Logical dataset/scenario IDs are frozen here, while concrete filesystem
    paths and hashes are bound by each execution manifest. G1 therefore does
    not invent a user dataset path or treat historical evidence as a new run.
    """

    singles = _providers(
        DenseProviderImplementation.METRIC3D_V2,
        DenseProviderImplementation.UNIDEPTH_V2,
        DenseProviderImplementation.MOGE_2,
        DenseProviderImplementation.DEPTH_PRO,
        DenseProviderImplementation.DEPTH_ANYTHING_V2,
        DenseProviderImplementation.MARIGOLD,
    )
    mvs_da3 = _providers(DenseProviderImplementation.COLMAP_MVS, DenseProviderImplementation.DA3)
    temporal = _providers(
        DenseProviderImplementation.ROLLING_DEPTH,
        DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
    )
    return (
        QualificationScenarioSpec(
            "g2_complete_room_mvs_da3", "I4.3G2", "Complete-room MVS vs DA3",
            mvs_da3, "static_multiview_complete_room",
            (ComparisonMode.PRODUCTION_NATIVE, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC),
            True, True, True, False, "mixed", "normal",
            "Primary complete-room operating-envelope comparison required by I4.3A.", _STATIC_MANUAL,
        ),
        QualificationScenarioSpec(
            "g2_temple_ring_mvs_da3", "I4.3G2", "TempleRing MVS vs DA3 detail transfer",
            mvs_da3, "middlebury_temple_ring",
            (ComparisonMode.PRODUCTION_NATIVE, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC),
            True, True, True, False, "textured_object", "controlled_ring",
            "Thin-structure/detail comparison while preserving the qualified TempleRing authority chain.", _STATIC_MANUAL,
        ),
        QualificationScenarioSpec(
            "g2_house_mvs_da3_historical", "I4.3G2", "House historical MVS vs DA3 positive-control context",
            mvs_da3, "historical_house_positive_control",
            (ComparisonMode.HISTORICAL_READ_ONLY,),
            True, False, True, False, "mixed", "controlled_orbit",
            "Retain frozen House A/B evidence as historical context without misreporting it as a new G2 execution.", _STATIC_MANUAL,
        ),
        QualificationScenarioSpec(
            "g2_low_texture_mvs_da3", "I4.3G2", "Low-texture static MVS vs DA3",
            mvs_da3, "static_multiview_low_texture",
            (ComparisonMode.PRODUCTION_NATIVE, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC),
            True, True, True, False, "low", "normal",
            "Probe provider behavior where photometric matching is weak but authoritative SfM remains valid.", _STATIC_MANUAL,
        ),
        QualificationScenarioSpec(
            "g2_weak_overlap_mvs_da3", "I4.3G2", "Weak-overlap static MVS vs DA3",
            mvs_da3, "static_multiview_weak_overlap",
            (ComparisonMode.PRODUCTION_NATIVE, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC),
            True, True, True, False, "mixed", "weak",
            "Probe controlled weak-overlap failure/degradation behavior without weakening SfM authority gates.", _STATIC_MANUAL,
        ),
        QualificationScenarioSpec(
            "g3_temporal_static_sweep", "I4.3G3", "Temporal static-scene camera sweep",
            temporal, "ordered_video_static_scene",
            (ComparisonMode.PRODUCTION_NATIVE, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC),
            True, True, True, True, "mixed", "temporal_contiguous",
            "Compare temporal consistency/drift on an ordered sweep of a predominantly static scene.", _TEMPORAL_MANUAL,
        ),
        QualificationScenarioSpec(
            "g3_temporal_motion", "I4.3G3", "Temporal motion/dynamics sequence",
            temporal, "ordered_video_dynamic_scene",
            (ComparisonMode.PRODUCTION_NATIVE, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC),
            True, True, True, True, "mixed", "temporal_contiguous",
            "Compare transient artifacts and motion-boundary stability under real scene motion.", _TEMPORAL_MANUAL,
        ),
        QualificationScenarioSpec(
            "g3_temporal_low_texture", "I4.3G3", "Temporal low-texture sequence",
            temporal, "ordered_video_low_texture",
            (ComparisonMode.PRODUCTION_NATIVE, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC),
            True, True, True, True, "low", "temporal_contiguous",
            "Compare flicker/drift and reconstruction usefulness in low-texture temporal content.", _TEMPORAL_MANUAL,
        ),
        QualificationScenarioSpec(
            "g4_single_image_complete_room", "I4.3G4", "Single-image providers on complete-room SfM",
            singles, "static_multiview_complete_room",
            (ComparisonMode.PRODUCTION_NATIVE, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC),
            True, True, True, False, "mixed", "normal",
            "Compare all six single-image depth providers after their individually authorized SfM-gauge alignment.", _SINGLE_IMAGE_MANUAL,
        ),
        QualificationScenarioSpec(
            "g4_single_image_low_texture", "I4.3G4", "Single-image providers on low-texture SfM",
            singles, "static_multiview_low_texture",
            (ComparisonMode.PRODUCTION_NATIVE, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC),
            True, True, True, False, "low", "normal",
            "Compare depth stability/completeness where image texture is weak.", _SINGLE_IMAGE_MANUAL,
        ),
        QualificationScenarioSpec(
            "g4_single_image_weak_overlap", "I4.3G4", "Single-image providers on weak-overlap SfM",
            singles, "static_multiview_weak_overlap",
            (ComparisonMode.PRODUCTION_NATIVE, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC),
            True, True, True, False, "mixed", "weak",
            "Compare aligned single-image depth evidence when sparse cross-view support is limited.", _SINGLE_IMAGE_MANUAL,
        ),
    )


def qualification_decision_rules() -> dict[str, Any]:
    return {
        "scenario_specific_verdicts_required": True,
        "global_scalar_score_forbidden": True,
        "universal_provider_winner_assumed": False,
        "quality_thresholds_may_not_be_relaxed_after_results_are_seen": True,
        "infrastructure_failure_may_be_rerun_with_attempt_provenance": True,
        "quality_failure_may_not_be_erased_by_untracked_retuning": True,
        "production_native_plan_is_primary_operating_envelope": True,
        "matched_view_diagnostic_is_secondary_causal_diagnostic": True,
        "matched_view_mode_must_preserve_provider_semantics": True,
        "temporal_matched_view_mode_requires_contiguous_source_order": True,
        "same_authoritative_sfm_required_within_static_scenario": True,
        "same_source_asset_identity_required_within_scenario": True,
        "same_manual_review_camera_and_framing_required": True,
        "independent_visual_auto_fit_forbidden": True,
        "manual_ab_labels_should_be_blinded_when_practical": True,
        "provider_specific_native_evidence_may_be_reported_but_not_used_as_unfair_common_metric": True,
        "missing_optional_metric_requires_reason": True,
        "hybrid_experiment_authorized": False,
        "auto_policy_implemented": False,
        "provider_promotion_performed": False,
    }


def qualification_run_evidence_schema() -> dict[str, Any]:
    """Human-readable schema contract for G2-G4 run evidence manifests."""

    return {
        "required_top_level": [
            "schema_version",
            "scenario_id",
            "provider_implementation",
            "comparison_mode",
            "attempt",
            "status",
            "source_identity",
            "authoritative_sfm_identity",
            "adaptive_plan_identity",
            "provider_provenance",
            "runtime_resources",
            "metrics",
            "artifacts",
            "authority",
        ],
        "status_values": ["pass", "degraded", "failed", "cancelled", "unavailable"],
        "source_identity_required": ["manifest_path", "manifest_sha256", "asset_set_sha256"],
        "authoritative_sfm_identity_required": ["snapshot_path", "snapshot_sha256", "frame_id"],
        "adaptive_plan_identity_required": ["plan_path", "plan_sha256", "selected_view_ids"],
        "provider_provenance_required": [
            "provider_id",
            "implementation",
            "backend_version",
            "adapter_version",
            "effective_profile",
            "checkpoint_or_model_identity",
            "parameters",
        ],
        "runtime_resources_required": ["wall_time_s", "peak_vram_mib", "measurement_scope"],
        "artifact_integrity_required": True,
        "authority_flags_must_remain_false": [
            "authoritative_sfm_modified",
            "provider_promoted",
            "auto_policy_implemented",
            "hybrid_authorized",
            "metric_scale_claimed",
            "measurement_authoritative",
            "certified_metrology",
            "surface_mesh_generated",
        ],
    }


def _require_pass(snapshot: Mapping[str, Any], *, phase: str) -> None:
    if not isinstance(snapshot, Mapping):
        raise TypeError(f"{phase} snapshot must be a mapping.")
    if snapshot.get("phase") != phase or snapshot.get("status") != "pass" or snapshot.get("pass") is not True:
        raise ReconstructionValidationError(f"I4.3G1 requires a passing {phase} snapshot.")


def validate_i4_3g1_predecessors(
    *,
    i4_3a_snapshot: Mapping[str, Any],
    i4_3f3_snapshot: Mapping[str, Any],
) -> None:
    _require_pass(i4_3a_snapshot, phase="I4.3A")
    _require_pass(i4_3f3_snapshot, phase="I4.3F3")

    architecture = i4_3a_snapshot.get("architecture")
    plan = i4_3a_snapshot.get("qualification_plan")
    a_authority = i4_3a_snapshot.get("authority")
    f_decision = i4_3f3_snapshot.get("decision")
    f_authority = i4_3f3_snapshot.get("authority")
    if not all(isinstance(value, Mapping) for value in (architecture, plan, a_authority, f_decision, f_authority)):
        raise ReconstructionValidationError("I4.3G1 predecessor evidence is missing architecture/authority sections.")

    expected_impls = [item.implementation.value for item in dense_provider_implementation_contracts()]
    if architecture.get("provider_implementations") != expected_impls:
        raise ReconstructionValidationError("I4.3G1 requires the exact frozen 10-provider I4.3A implementation order.")
    if architecture.get("scenario_specific_qualification_required") is not True:
        raise ReconstructionValidationError("I4.3G1 requires scenario-specific qualification.")
    if architecture.get("universal_provider_winner_assumed") is not False:
        raise ReconstructionValidationError("I4.3G1 refuses a predecessor universal-winner assumption.")
    for key in (
        "i4_3g_required",
        "complete_room_mvs_vs_da3_required",
        "temple_ring_mvs_vs_ai_where_useful",
        "house_mvs_vs_da3_evidence_retained",
        "temporal_video_comparison_required",
        "low_texture_comparison_required",
        "weak_overlap_comparison_required",
    ):
        if plan.get(key) is not True:
            raise ReconstructionValidationError(f"I4.3G1 qualification-plan prerequisite is not frozen: {key}")
    if plan.get("hybrid_implementation_authorized") is not False:
        raise ReconstructionValidationError("I4.3G1 refuses predecessor hybrid authorization.")

    if f_decision.get("i4_3f_closed") is not True or f_decision.get("i4_3g_cross_provider_qualification_authorized") is not True:
        raise ReconstructionValidationError("I4.3G1 requires F3 closure and explicit G authorization.")
    if f_decision.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G1 refuses a permanent global view cap.")
    validated = i4_3f3_snapshot.get("validated_provider_implementations")
    if validated != expected_impls:
        raise ReconstructionValidationError("I4.3G1 requires F3 validation coverage for the exact 10 providers.")
    if f_authority.get("camera_pose_authority_owner") != I43A_AUTHORITATIVE_FRAME_OWNER:
        raise ReconstructionValidationError("I4.3G1 requires PyCOLMAP camera/frame authority.")

    forbidden_a = (
        "backend_started",
        "provider_selected",
        "provider_promoted",
        "auto_policy_implemented",
        "hybrid_implementation_authorized",
        "authoritative_sparse_modified",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    )
    drifted_a = [name for name in forbidden_a if bool(a_authority.get(name, False))]
    if drifted_a:
        raise ReconstructionValidationError(f"I4.3G1 I4.3A authority drift: {drifted_a}")
    forbidden_f = (
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
    drifted_f = [name for name in forbidden_f if bool(f_authority.get(name, False))]
    if drifted_f:
        raise ReconstructionValidationError(f"I4.3G1 F3 authority drift: {drifted_f}")


def _validate_framework_internal_consistency() -> None:
    metrics = qualification_metric_registry()
    metric_ids = [item.metric_id for item in metrics]
    if len(metric_ids) != len(set(metric_ids)):
        raise ReconstructionValidationError("I4.3G1 metric registry contains duplicate IDs.")
    scenarios = qualification_scenario_matrix()
    scenario_ids = [item.scenario_id for item in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ReconstructionValidationError("I4.3G1 scenario matrix contains duplicate IDs.")
    expected = {item.implementation for item in dense_provider_implementation_contracts()}
    covered = {impl for scenario in scenarios for impl in scenario.provider_implementations}
    if covered != expected:
        missing = sorted(item.value for item in expected - covered)
        extra = sorted(item.value for item in covered - expected)
        raise ReconstructionValidationError(f"I4.3G1 provider coverage mismatch: missing={missing} extra={extra}")
    required_groups = {"I4.3G2", "I4.3G3", "I4.3G4"}
    if {item.substep for item in scenarios if item.required} != required_groups:
        raise ReconstructionValidationError("I4.3G1 required scenario coverage must span G2, G3, and G4.")


def build_i4_3g1_framework_snapshot(
    *,
    i4_3a_snapshot: Mapping[str, Any],
    i4_3f3_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    validate_i4_3g1_predecessors(i4_3a_snapshot=i4_3a_snapshot, i4_3f3_snapshot=i4_3f3_snapshot)
    _validate_framework_internal_consistency()
    scenarios = qualification_scenario_matrix()
    metrics = qualification_metric_registry()
    return {
        "schema_version": I43G1_SCHEMA_VERSION,
        "phase": I43G1_PHASE,
        "status": "pass",
        "pass": True,
        "scope": {
            "methodology_freeze_only": True,
            "real_provider_execution_started": False,
            "quality_results_consumed": False,
            "scenario_matrix_frozen": True,
            "metric_registry_frozen": True,
            "run_evidence_schema_frozen": True,
            "decision_rules_frozen": True,
            "g2_g3_g4_real_execution_deferred": True,
            "g5_synthesis_deferred": True,
        },
        "scenario_matrix": [item.as_dict() for item in scenarios],
        "metric_registry": [item.as_dict() for item in metrics],
        "run_evidence_schema": qualification_run_evidence_schema(),
        "decision_rules": qualification_decision_rules(),
        "coverage": {
            "scenario_count": len(scenarios),
            "metric_count": len(metrics),
            "provider_implementation_count": 10,
            "provider_implementations": [item.implementation.value for item in dense_provider_implementation_contracts()],
            "g2_scenarios": sum(item.substep == "I4.3G2" for item in scenarios),
            "g3_scenarios": sum(item.substep == "I4.3G3" for item in scenarios),
            "g4_scenarios": sum(item.substep == "I4.3G4" for item in scenarios),
        },
        "fairness": {
            "production_native_comparison_primary": True,
            "matched_view_diagnostic_secondary": True,
            "same_authoritative_sfm_within_static_scenario": True,
            "same_source_identity_within_scenario": True,
            "provider_specific_adaptive_plans_persisted": True,
            "temporal_contiguous_source_order_required": True,
            "manual_review_same_camera_and_framing": True,
            "independent_visual_auto_fit_forbidden": True,
            "historical_evidence_never_reported_as_new_execution": True,
        },
        "authority": {
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i4_3g1_framework_closed": True,
            "i4_3g2_mvs_da3_qualification_authorized": True,
            "universal_provider_winner_assumed": False,
            "global_scalar_score_forbidden": True,
            "provider_selected": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "next": "I4.3G2 controlled MVS vs DA3 production comparison.",
    }


def validate_metric_value(spec: QualificationMetricSpec, value: Any) -> None:
    """Small schema helper used by later G runners and G1 tests."""

    if spec.direction is MetricDirection.BOOLEAN_GATE_TRUE:
        if value is not True:
            raise ReconstructionValidationError(f"Metric {spec.metric_id} requires boolean true.")
        return
    if value is None:
        if spec.required:
            raise ReconstructionValidationError(f"Required metric {spec.metric_id} is missing.")
        return
    if isinstance(value, bool):
        raise ReconstructionValidationError(f"Numeric/informational metric {spec.metric_id} cannot be bool.")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not isfinite(numeric):
            raise ReconstructionValidationError(f"Metric {spec.metric_id} must be finite.")
        return
    if spec.direction is MetricDirection.INFORMATIONAL and isinstance(value, str) and value.strip():
        return
    raise ReconstructionValidationError(f"Metric {spec.metric_id} has unsupported value type.")


__all__ = [
    "I43G1_SCHEMA_VERSION",
    "I43G1_PHASE",
    "ComparisonMode",
    "MetricDirection",
    "MetricCategory",
    "QualificationMetricSpec",
    "QualificationScenarioSpec",
    "qualification_metric_registry",
    "qualification_scenario_matrix",
    "qualification_decision_rules",
    "qualification_run_evidence_schema",
    "validate_i4_3g1_predecessors",
    "build_i4_3g1_framework_snapshot",
    "validate_metric_value",
]
