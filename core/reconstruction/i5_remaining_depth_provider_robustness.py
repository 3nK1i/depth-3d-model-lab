"""I5.9A remaining single-image depth-provider robustness admission.

I5.9A is execution-free.  It binds the formally closed I5.8 campaign, the
frozen six-scenario Castle/PyCOLMAP foundation from I5.6A, and the historical
I4.3G4C single-image qualification closure.  The step inventories the six
remaining production single-image depth providers, freezes their existing
production/alignment semantics, and admits provider-specific real robustness
replays without changing thresholds, defaults, fusion capacity, camera
authority, or metric/metrology claims.

The real replay is deliberately split into two scientifically distinct batches:
I5.9B1 covers the four providers whose native metric-like depth is aligned by
per-view sparse-correspondence scale; I5.9B2 covers the two relative-depth
providers whose existing production paths use per-view affine alignment.  The
split also preserves the historical G4 low-texture result, where the B2 pair
(Depth Anything V2 and Marigold) were genuine production rejections.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from core.reconstruction.depth_anything_v2_dense_provider import (
    I43E5_PROVIDER_ID,
    DepthAnythingV2ProductionConfig,
)
from core.reconstruction.depth_pro_dense_provider import I43E4_PROVIDER_ID, DepthProProductionConfig
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_mvs_robustness import I56A_SCENARIO_IDS, validate_i5_6a_for_i5_6b
from core.reconstruction.marigold_dense_provider import I43E6_PROVIDER_ID, MarigoldProductionConfig
from core.reconstruction.metric3d_v2_dense_provider import I43E1_PROVIDER_ID, Metric3DV2ProductionConfig
from core.reconstruction.moge2_dense_provider import I43E3_PROVIDER_ID, MoGe2ProductionConfig
from core.reconstruction.unidepth_v2_dense_provider import I43E2_PROVIDER_ID, UniDepthV2ProductionConfig

I59A_SCHEMA_VERSION = 1
I59A_PHASE = "I5.9A"
I59A_RECEIPT_PHASE = "I5.9A_ADMISSION_RECEIPT"
I59A_I58C_FROZEN_SHA256 = "741ecf17db05572b86102b410c219559edaacd7a64bf7ea4108eac8bda57daa2"
I59A_I56A_FROZEN_SHA256 = "506fb778d82ddc46350cbe4922ebe8a2ba43e0df4fa7818687a5768ebe0afc7a"
I59A_G4C_FROZEN_SHA256 = "577dba2b6f4549e122ac1d6892d537d7ecac49049410be819fbf1349f7d4abd2"
I59A_SCENARIO_IDS = I56A_SCENARIO_IDS
I59A_PROVIDER_IMPLEMENTATIONS = (
    "metric3d_v2",
    "unidepth_v2",
    "moge_2",
    "depth_pro",
    "depth_anything_v2",
    "marigold",
)
I59A_B1_PROVIDERS = ("metric3d_v2", "unidepth_v2", "moge_2", "depth_pro")
I59A_B2_PROVIDERS = ("depth_anything_v2", "marigold")
I59A_G4_SCENARIOS = (
    "g4_single_image_complete_room",
    "g4_single_image_low_texture",
    "g4_single_image_weak_overlap",
)
I59A_FROZEN_G4_STATUSES = {
    "g4_single_image_complete_room": {p: "pass" for p in I59A_PROVIDER_IMPLEMENTATIONS},
    "g4_single_image_low_texture": {
        "metric3d_v2": "pass",
        "unidepth_v2": "pass",
        "moge_2": "pass",
        "depth_pro": "pass",
        "depth_anything_v2": "rejected",
        "marigold": "rejected",
    },
    "g4_single_image_weak_overlap": {p: "pass" for p in I59A_PROVIDER_IMPLEMENTATIONS},
}

_SCENARIO_INTERPRETATION = {
    "castle_control": "stateless_registered_view_single_image_dense_control",
    "castle_low_overlap": "reduced_registered_view_support_alignment_and_fusion_robustness_not_mvs_overlap_equivalence",
    "castle_weak_parallax": "narrow_baseline_sparse_sfm_alignment_support_robustness",
    "castle_overexposure": "photometric_overexposure_stateless_single_frame_depth_robustness",
    "castle_symmetry": "repetitive_structure_and_sparse_support_stateless_alignment_robustness",
    "castle_dynamic_proxy": "independent_frame_dynamic_contamination_robustness_no_temporal_stability_claim",
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


def provider_robustness_specs() -> dict[str, dict[str, Any]]:
    """Return the exact existing production semantics for the six providers."""

    return {
        "metric3d_v2": {
            "provider_id": I43E1_PROVIDER_ID,
            "family": "single_image_depth",
            "batch": "I5.9B1",
            "native_depth_scale": "metric",
            "alignment_semantics": "per_view_sparse_correspondence_scale_to_authoritative_sfm",
            "production_config": Metric3DV2ProductionConfig().as_dict(),
            "runtime_resolution": "existing_metric3d_v2_production_runtime",
            "native_metric_promoted_to_physical_metric_scale": False,
            "model_intrinsics_are_canonical_authority": False,
            "fusion_capacity_drops_change_status": False,
            "permanent_global_view_cap": None,
        },
        "unidepth_v2": {
            "provider_id": I43E2_PROVIDER_ID,
            "family": "single_image_depth",
            "batch": "I5.9B1",
            "native_depth_scale": "metric",
            "alignment_semantics": "per_view_sparse_correspondence_scale_to_authoritative_sfm",
            "production_config": UniDepthV2ProductionConfig().as_dict(),
            "runtime_resolution": "existing_unidepth_v2_hardware_resolved_variant",
            "native_metric_promoted_to_physical_metric_scale": False,
            "model_intrinsics_are_canonical_authority": False,
            "fusion_capacity_drops_change_status": False,
            "permanent_global_view_cap": None,
        },
        "moge_2": {
            "provider_id": I43E3_PROVIDER_ID,
            "family": "single_image_depth",
            "batch": "I5.9B1",
            "native_depth_scale": "metric",
            "alignment_semantics": "per_view_sparse_correspondence_scale_to_authoritative_sfm",
            "production_config": MoGe2ProductionConfig().as_dict(),
            "runtime_resolution": "existing_moge_2_hardware_resolved_variant",
            "native_metric_promoted_to_physical_metric_scale": False,
            "model_intrinsics_are_canonical_authority": False,
            "fusion_capacity_drops_change_status": False,
            "permanent_global_view_cap": None,
        },
        "depth_pro": {
            "provider_id": I43E4_PROVIDER_ID,
            "family": "single_image_depth",
            "batch": "I5.9B1",
            "native_depth_scale": "metric",
            "alignment_semantics": "per_view_sparse_correspondence_scale_to_authoritative_sfm",
            "production_config": DepthProProductionConfig().as_dict(),
            "runtime_resolution": "existing_depth_pro_production_runtime",
            "native_metric_promoted_to_physical_metric_scale": False,
            "model_intrinsics_are_canonical_authority": False,
            "fusion_capacity_drops_change_status": False,
            "permanent_global_view_cap": None,
        },
        "depth_anything_v2": {
            "provider_id": I43E5_PROVIDER_ID,
            "family": "single_image_depth",
            "batch": "I5.9B2",
            "native_depth_scale": "relative",
            "native_depth_representation": "relative_distance_from_inverse_depth",
            "alignment_semantics": "per_view_affine_inverse_depth_to_authoritative_sfm",
            "production_config": DepthAnythingV2ProductionConfig().as_dict(),
            "runtime_resolution": "existing_depth_anything_v2_hardware_resolved_variant",
            "native_metric_promoted_to_physical_metric_scale": False,
            "model_intrinsics_are_canonical_authority": False,
            "fusion_capacity_drops_change_status": False,
            "permanent_global_view_cap": None,
        },
        "marigold": {
            "provider_id": I43E6_PROVIDER_ID,
            "family": "single_image_depth",
            "batch": "I5.9B2",
            "native_depth_scale": "relative",
            "native_depth_representation": "affine_invariant_relative_depth_or_disparity",
            "alignment_semantics": "per_view_affine_native_domain_to_authoritative_sfm",
            "production_config": MarigoldProductionConfig().as_dict(),
            "runtime_resolution": "existing_marigold_hardware_resolved_profile",
            "native_metric_promoted_to_physical_metric_scale": False,
            "model_intrinsics_are_canonical_authority": False,
            "fusion_capacity_drops_change_status": False,
            "permanent_global_view_cap": None,
        },
    }


@dataclass(frozen=True, slots=True)
class I59AAdmissionPolicy:
    scenario_ids: tuple[str, ...] = I59A_SCENARIO_IDS
    b1_providers: tuple[str, ...] = I59A_B1_PROVIDERS
    b2_providers: tuple[str, ...] = I59A_B2_PROVIDERS
    stateless_single_image: bool = True
    all_registered_views_available_by_default: bool = True
    adaptive_view_selection_deferred_to: str = "I4.3F"
    sequential_provider_scenario_execution: bool = True
    production_max_voxels: int = 1_500_000
    minimum_accepted_views: int = 2
    minimum_correspondences: int = 8
    maximum_relative_residual: float = 0.35
    permanent_global_view_cap: int | None = None
    provider_comparison: bool = False
    result_driven_retuning: bool = False
    retry_until_pass: bool = False

    def __post_init__(self) -> None:
        if tuple(self.scenario_ids) != I59A_SCENARIO_IDS:
            raise ReconstructionValidationError("I5.9A is frozen to the six admitted Castle/PyCOLMAP scenarios.")
        if tuple(self.b1_providers) != I59A_B1_PROVIDERS or tuple(self.b2_providers) != I59A_B2_PROVIDERS:
            raise ReconstructionValidationError("I5.9A provider batches are frozen to the inspected B1/B2 partition.")
        if set(self.b1_providers).intersection(self.b2_providers) or set(self.b1_providers + self.b2_providers) != set(I59A_PROVIDER_IMPLEMENTATIONS):
            raise ReconstructionValidationError("I5.9A provider batches must partition the six remaining providers exactly.")
        if self.stateless_single_image is not True or self.all_registered_views_available_by_default is not True:
            raise ReconstructionValidationError("I5.9A must preserve the existing stateless single-image provider contract.")
        if self.adaptive_view_selection_deferred_to != "I4.3F":
            raise ReconstructionValidationError("I5.9A cannot replace the existing I4.3F adaptive-selection authority.")
        if self.sequential_provider_scenario_execution is not True:
            raise ReconstructionValidationError("I5.9A real replay batches are sequential by design.")
        if self.production_max_voxels != 1_500_000:
            raise ReconstructionValidationError("I5.9A cannot modify the existing 1,500,000-voxel production fusion capacity.")
        if self.minimum_accepted_views != 2 or self.minimum_correspondences != 8 or self.maximum_relative_residual != 0.35:
            raise ReconstructionValidationError("I5.9A cannot modify existing single-image alignment/acceptance defaults.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I5.9A cannot introduce a permanent global view cap.")
        if self.provider_comparison or self.result_driven_retuning or self.retry_until_pass:
            raise ReconstructionValidationError("I5.9A forbids provider comparison, result-driven retuning, and retry-until-pass.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_ids": list(self.scenario_ids),
            "providers": list(I59A_PROVIDER_IMPLEMENTATIONS),
            "stateless_single_image": self.stateless_single_image,
            "all_registered_views_available_by_default": self.all_registered_views_available_by_default,
            "adaptive_view_selection_deferred_to": self.adaptive_view_selection_deferred_to,
            "sequential_provider_scenario_execution": self.sequential_provider_scenario_execution,
            "production_max_voxels": self.production_max_voxels,
            "minimum_accepted_views": self.minimum_accepted_views,
            "minimum_correspondences": self.minimum_correspondences,
            "maximum_relative_residual": self.maximum_relative_residual,
            "permanent_global_view_cap": self.permanent_global_view_cap,
            "provider_comparison": self.provider_comparison,
            "result_driven_retuning": self.result_driven_retuning,
            "retry_until_pass": self.retry_until_pass,
            "batches": {
                "execution_order": ["I5.9B1", "I5.9B2"],
                "I5.9B1": {
                    "providers": list(self.b1_providers),
                    "alignment_family": "per_view_sparse_correspondence_scale_to_authoritative_sfm",
                },
                "I5.9B2": {
                    "providers": list(self.b2_providers),
                    "alignment_family": "provider_specific_per_view_affine_relative_depth_to_authoritative_sfm",
                },
            },
        }


def validate_i5_8c_for_i5_9(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.8C closure")
    _require(payload, "phase", "I5.8C", label="I5.8C closure")
    _require(payload, "status", "video_depth_anything_robustness_qualified_and_closed", label="I5.8C closure")
    _require(payload, "pass", True, label="I5.8C closure")
    _require(payload, "quality_status", "degraded", label="I5.8C closure")
    authority = _mapping(payload.get("authority"), label="I5.8C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.8C authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.8C authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "provider_ranking_created",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.8C authority")
    decision = _mapping(payload.get("decision"), label="I5.8C decision")
    _require(decision, "i5_8_complete", True, label="I5.8C decision")
    _require(decision, "i5_9_authorized", True, label="I5.8C decision")
    _require(decision, "i5_9_started", False, label="I5.8C decision")
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized", "mp_sfm_authorized",
        "fusion_capacity_increase_authorized", "production_default_modified",
    ):
        _require(decision, key, False, label="I5.8C decision")
    _require(payload, "next", "I5.9 Remaining depth-provider robustness.", label="I5.8C closure")
    return {
        "i5_8_complete": True,
        "i5_9_authorized": True,
        "i5_9_started": False,
        "quality_status": "degraded",
    }


def validate_i4_3g4c_for_i5_9(payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    _require(payload, "phase", "I4.3G4C", label="I4.3G4C closure")
    _require(payload, "status", "formally_closed", label="I4.3G4C closure")
    _require(payload, "pass", True, label="I4.3G4C closure")
    _require(payload, "scenario_count", 3, label="I4.3G4C closure")
    for key in (
        "all_required_single_image_scenarios_accounted",
        "all_required_manual_reviews_complete",
        "all_six_provider_outcomes_accounted_per_scenario",
    ):
        _require(payload, key, True, label="I4.3G4C closure")
    synthesis = _mapping(payload.get("scenario_synthesis"), label="I4.3G4C scenario synthesis")
    if set(synthesis) != set(I59A_G4_SCENARIOS):
        raise ReconstructionValidationError("I5.9A requires exactly the three frozen I4.3G4C single-image scenarios.")
    compact: dict[str, dict[str, str]] = {}
    aliases = {
        "g4_single_image_complete_room": "complete_room",
        "g4_single_image_low_texture": "low_texture",
        "g4_single_image_weak_overlap": "weak_overlap",
    }
    for sid in I59A_G4_SCENARIOS:
        node = _mapping(synthesis.get(sid), label=f"I4.3G4C {sid}")
        statuses = _mapping(node.get("production_statuses"), label=f"I4.3G4C {sid} statuses")
        if set(statuses) != set(I59A_PROVIDER_IMPLEMENTATIONS):
            raise ReconstructionValidationError(f"I4.3G4C {sid} provider inventory drifted.")
        expected = I59A_FROZEN_G4_STATUSES[sid]
        for provider, status in expected.items():
            _require(statuses, provider, status, label=f"I4.3G4C {sid} statuses")
        _require(node, "manual_review_complete", True, label=f"I4.3G4C {sid}")
        _require(node, "matched_view_diagnostic_secondary", True, label=f"I4.3G4C {sid}")
        _require(node, "production_rejection_overridden", False, label=f"I4.3G4C {sid}")
        _require(node, "provider_ranking_created", False, label=f"I4.3G4C {sid}")
        _require(node, "global_provider_winner_declared", False, label=f"I4.3G4C {sid}")
        compact[aliases[sid]] = dict(statuses)
    decision = _mapping(payload.get("decision"), label="I4.3G4C decision")
    _require(decision, "production_native_evidence_primary", True, label="I4.3G4C decision")
    _require(decision, "scenario_specific_verdicts_preserved", True, label="I4.3G4C decision")
    _require(decision, "quality_rejections_preserved", True, label="I4.3G4C decision")
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(decision, key, False, label="I4.3G4C decision")
    authority = _mapping(payload.get("authority"), label="I4.3G4C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I4.3G4C authority")
    _require(authority, "permanent_global_view_cap", None, label="I4.3G4C authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented",
        "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I4.3G4C authority")
    return compact


def validate_i5_9a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I59A_SCHEMA_VERSION, label="I5.9A receipt")
    _require(payload, "phase", I59A_RECEIPT_PHASE, label="I5.9A receipt")
    _require(payload, "status", "pass", label="I5.9A receipt")
    _require(payload, "pass", True, label="I5.9A receipt")
    scope = _mapping(payload.get("scope"), label="I5.9A receipt scope")
    for key in (
        "focused_i5_9a_tests", "single_image_provider_regression", "historical_g4_regression",
        "i5_6a_source_binding_regression", "i5_8c_authorization_regression", "authority_guard_regression",
        "roadmap_guard_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.9A receipt scope")
    for key in (
        "single_image_provider_inference", "sfm_rerun", "mvs_rerun", "provider_comparison_started",
        "thresholds_modified", "production_default_modified", "result_driven_retuning", "retry_until_pass",
        "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.9A receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.9A receipt counts")
    focused = int(counts.get("focused_tests", 0))
    predecessor = int(counts.get("predecessor_total_tests", 0))
    combined = int(counts.get("combined_exercised_tests", 0))
    if focused <= 0 or predecessor <= 0 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.9A receipt test counts are inconsistent.")
    rcs = _mapping(payload.get("return_codes"), label="I5.9A receipt return codes")
    for key in (
        "focused", "predecessor_unit", "predecessor_integration", "predecessor_stress",
        "import_probe", "runner_help_probe",
    ):
        if int(rcs.get(key, -1)) != 0:
            raise ReconstructionValidationError(f"I5.9A receipt return code is non-zero: {key}")
    return deepcopy(dict(payload))


def build_i5_9a_remaining_depth_provider_robustness_admission(
    *,
    i5_8c_closure: Mapping[str, Any],
    i5_8c_binding: I51EvidenceBinding,
    i5_6a_admission: Mapping[str, Any],
    i5_6a_binding: I51EvidenceBinding,
    i4_3g4c_closure: Mapping[str, Any],
    i4_3g4c_binding: I51EvidenceBinding,
    admission_receipt: Mapping[str, Any],
    admission_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    validate_i5_8c_for_i5_9(i5_8c_closure)
    i56a = validate_i5_6a_for_i5_6b(i5_6a_admission)
    g4 = validate_i4_3g4c_for_i5_9(i4_3g4c_closure)
    receipt = validate_i5_9a_receipt(admission_receipt)
    _validate_binding(i5_8c_binding, label="I5.9A I5.8C", expected_sha256=I59A_I58C_FROZEN_SHA256)
    _validate_binding(i5_6a_binding, label="I5.9A I5.6A", expected_sha256=I59A_I56A_FROZEN_SHA256)
    _validate_binding(i4_3g4c_binding, label="I5.9A I4.3G4C", expected_sha256=I59A_G4C_FROZEN_SHA256)
    _validate_binding(admission_receipt_binding, label="I5.9A admission receipt")

    policy = I59AAdmissionPolicy().as_dict()
    specs = provider_robustness_specs()
    for provider, spec in specs.items():
        cfg = _mapping(spec["production_config"], label=f"{provider} production config")
        _require(cfg, "max_voxels", policy["production_max_voxels"], label=f"{provider} production config")
        _require(cfg, "minimum_accepted_views", policy["minimum_accepted_views"], label=f"{provider} production config")
        _require(cfg, "minimum_correspondences", policy["minimum_correspondences"], label=f"{provider} production config")
        _require(cfg, "maximum_relative_residual", policy["maximum_relative_residual"], label=f"{provider} production config")
        _require(cfg, "all_registered_views_selected", True, label=f"{provider} production config")
        _require(cfg, "adaptive_view_selection_deferred_to", "I4.3F", label=f"{provider} production config")
        _require(cfg, "permanent_global_view_cap", None, label=f"{provider} production config")

    source_matrix = _mapping(i56a.get("scenario_matrix"), label="I5.6A scenario matrix")
    if set(source_matrix) != set(I59A_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.9A requires exactly the six frozen I5.6A Castle scenarios.")
    scenario_matrix: dict[str, Any] = {}
    for sid in I59A_SCENARIO_IDS:
        src = _mapping(source_matrix.get(sid), label=f"I5.6A {sid}")
        registered = int(src.get("registered_images", src.get("registered_image_count", 0)))
        input_images = int(src.get("input_images", 0))
        if registered < policy["minimum_accepted_views"]:
            raise ReconstructionValidationError(f"I5.9A {sid} has fewer than two authoritative registered views.")
        scenario_matrix[sid] = {
            "scenario_id": sid,
            "admitted_for_i5_9b1": True,
            "admitted_for_i5_9b2": True,
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
            "stateless_single_image_views": True,
            "temporal_order_semantics_required": False,
            "mvs_specific_metrics_are_single_image_provider_criteria": False,
        }

    exclusions = {
        "castle_blur": {
            "classification": "blocked_upstream",
            "reason": "authoritative_pycolmap_sfm_unavailable_after_blur_stress",
            "remaining_provider_failure": False,
        },
        "castle_underexposure": {
            "classification": "blocked_upstream",
            "reason": "authoritative_pycolmap_sfm_unavailable_after_pre_sfm_ingestion_rejection",
            "remaining_provider_failure": False,
        },
        "castle_clutter": {
            "classification": "not_formally_admitted",
            "reason": "clutter_semantic_stress_not_revalidated_in_frozen_predecessor",
            "remaining_provider_failure": False,
        },
        "kitti_forward_motion": {
            "classification": "blocked_upstream",
            "reason": "incremental_mapping_produced_no_authoritative_sparse_model_or_registered_cameras",
            "remaining_provider_failure": False,
            "mp_sfm_not_pulled_forward": True,
        },
        "difficult_handheld_horse": {
            "classification": "deferred_not_admitted",
            "reason": "exact_authoritative_sfm_binding_not_present_in_i5_6a_admission_matrix",
            "remaining_provider_failure": False,
            "provenance_not_invented": True,
        },
    }

    batch_plan = {
        "execution_order": ["I5.9B1", "I5.9B2"],
        "I5.9B1": {
            "providers": list(I59A_B1_PROVIDERS),
            "purpose": "metric_like_single_image_providers_shared_sparse_correspondence_scale_alignment",
            "historical_low_texture_anchor": "all_four_pass",
            "real_provider_scenario_runs": len(I59A_B1_PROVIDERS) * len(I59A_SCENARIO_IDS),
            "authorized": True,
            "started": False,
        },
        "I5.9B2": {
            "providers": list(I59A_B2_PROVIDERS),
            "purpose": "relative_depth_single_image_providers_provider_specific_affine_alignment",
            "historical_low_texture_anchor": "both_rejected",
            "real_provider_scenario_runs": len(I59A_B2_PROVIDERS) * len(I59A_SCENARIO_IDS),
            "authorized": True,
            "started": False,
            "execute_after_i5_9b1_freeze": True,
        },
    }

    return {
        "schema_version": I59A_SCHEMA_VERSION,
        "phase": I59A_PHASE,
        "status": "remaining_depth_provider_robustness_inventory_and_scenarios_admitted",
        "pass": True,
        "scope": {
            "inventory_and_scenario_admission_only": True,
            "i5_8_closed_predecessor_reused": True,
            "i5_6a_castle_sfm_matrix_reused": True,
            "historical_i4_3g4c_single_image_closure_reused": True,
            "single_image_provider_inference": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "mvs_rerun": False,
            "provider_comparison_started": False,
            "source_rematerialized": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "fusion_capacity_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "evidence_bindings": {
            "i5_8c_video_depth_anything_closure": i5_8c_binding.as_dict(),
            "i5_6a_castle_sfm_admission": i5_6a_binding.as_dict(),
            "i4_3g4c_single_image_closure": i4_3g4c_binding.as_dict(),
            "i5_9a_admission_receipt": admission_receipt_binding.as_dict(),
        },
        "provider_inventory": deepcopy(specs),
        "admission_policy": policy,
        "batch_plan": batch_plan,
        "scenario_matrix": scenario_matrix,
        "historical_g4_anchors": g4,
        "exclusions": exclusions,
        "i5_9b_evidence_contract": {
            "fresh_timestamped_attempt_directory_per_provider_scenario": True,
            "failed_interrupted_blocked_rejected_attempts_preserved": True,
            "runtime_preflight_required_per_provider_batch": True,
            "existing_hardware_resolved_variant_or_profile_must_be_preserved": True,
            "resolved_source_and_checkpoint_identity_required": True,
            "source_and_authoritative_sfm_integrity_revalidation_required": True,
            "existing_i4_3f_adaptive_plan_must_be_resolved_before_result": True,
            "result_driven_adaptive_plan_change_forbidden": True,
            "all_selected_registered_views_are_independent_single_image_inference_items": True,
            "provider_specific_alignment_evidence_required_per_selected_view": True,
            "canonical_fusion_evidence_required_when_minimum_alignment_support_is_met": True,
            "fusion_capacity_drops_recorded_but_do_not_change_existing_single_image_status_semantics": True,
            "pass_degraded_rejected_are_valid_terminal_scientific_outcomes": True,
            "runtime_or_integrity_failure_must_fail_attempt_nonzero": True,
            "historical_g4_rejections_cannot_be_overwritten": True,
            "provider_comparison_forbidden": True,
            "provider_ranking_created": False,
            "global_scalar_score_created": False,
            "universal_provider_winner_declared": False,
            "retry_until_pass": False,
            "result_driven_retuning": False,
        },
        "known_qualification_gaps": {
            "historical_depth_anything_v2_low_texture_rejection_remains_active": True,
            "historical_marigold_low_texture_rejection_remains_active": True,
            "runtime_performance_envelope_complete": False,
            "resource_envelope_complete": False,
            "rtx_4090_qualified": False,
            "rtx_5090_qualified": False,
            "cross_gpu_equivalence_qualified": False,
            "ground_truth_dense_depth_available": False,
            "ground_truth_surface_accuracy_available": False,
            "metric_scale_available": False,
            "certified_metrology_available": False,
        },
        "regression_summary": {
            "focused_i5_9a_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "single_image_provider_pose_authority": False,
            "model_intrinsics_are_canonical_authority": False,
            "native_metric_depth_establishes_physical_metric_scale": False,
            "relative_depth_establishes_physical_metric_scale": False,
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
            "i5_8_complete": True,
            "i5_9_started": True,
            "i5_9a_complete": True,
            "i5_9b1_authorized": True,
            "i5_9b1_started": False,
            "i5_9b2_authorized": True,
            "i5_9b2_started": False,
            "i5_9c_started": False,
            "i5_10_started": False,
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
            "i5_9b1_metric_like_single_image_replay_authorized_not_started": True,
            "i5_9b2_relative_single_image_replay_authorized_not_started": True,
            "i5_9c_qualification_deferred_until_b1_b2_complete": True,
            "i5_10_mp_sfm_deferred": True,
            "i5_11_oom_disk_crash_recovery_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_auto_deferred": True,
            "i5_14_final_i5_closure_deferred": True,
        },
        "next": "I5.9B1 metric-like single-image provider robustness replay.",
    }


__all__ = [
    "I59A_SCHEMA_VERSION", "I59A_PHASE", "I59A_RECEIPT_PHASE",
    "I59A_I58C_FROZEN_SHA256", "I59A_I56A_FROZEN_SHA256", "I59A_G4C_FROZEN_SHA256",
    "I59A_SCENARIO_IDS", "I59A_PROVIDER_IMPLEMENTATIONS", "I59A_B1_PROVIDERS", "I59A_B2_PROVIDERS",
    "I59A_G4_SCENARIOS", "I59A_FROZEN_G4_STATUSES", "I59AAdmissionPolicy",
    "provider_robustness_specs", "validate_i5_8c_for_i5_9", "validate_i4_3g4c_for_i5_9",
    "validate_i5_9a_receipt", "build_i5_9a_remaining_depth_provider_robustness_admission",
]
