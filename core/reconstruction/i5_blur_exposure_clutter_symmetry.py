"""I5.3A blur / exposure / clutter / symmetry scenario admission.

This module is deliberately execution-free. It binds the frozen I5.2C closure,
revalidates the exact control source identity, and freezes deterministic,
source-derived stress recipes for the later I5.3B real replay. No transformed
images are materialized here and no SfM/provider execution occurs here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import validate_ordered_source_identity

I53A_SCHEMA_VERSION = 1
I53A_PHASE = "I5.3A"
I53A_RECEIPT_PHASE = "I5.3A_ADMISSION_RECEIPT"


@dataclass(frozen=True, slots=True)
class I53AStressDesignPolicy:
    """Frozen, result-independent recipes for the I5.3B controlled replay."""

    blur_kernel_size: int = 13
    blur_sigma: float = 3.0
    underexposure_gain: float = 0.35
    overexposure_gain: float = 1.55
    overexposure_bias: float = 32.0
    clutter_band_fraction: float = 0.18
    clutter_source_offset: int = 7
    symmetry_band_fraction: float = 0.18

    def __post_init__(self) -> None:
        if not isinstance(self.blur_kernel_size, int) or self.blur_kernel_size < 3 or self.blur_kernel_size % 2 == 0:
            raise ReconstructionValidationError("I5.3A blur_kernel_size must be an odd integer >= 3.")
        for name in (
            "blur_sigma",
            "underexposure_gain",
            "overexposure_gain",
            "overexposure_bias",
            "clutter_band_fraction",
            "symmetry_band_fraction",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ReconstructionValidationError(f"I5.3A {name} must be finite.")
            object.__setattr__(self, name, value)
        if self.blur_sigma <= 0.0:
            raise ReconstructionValidationError("I5.3A blur_sigma must be positive.")
        if not 0.0 < self.underexposure_gain < 1.0:
            raise ReconstructionValidationError("I5.3A underexposure_gain must be strictly between 0 and 1.")
        if self.overexposure_gain <= 1.0:
            raise ReconstructionValidationError("I5.3A overexposure_gain must be greater than 1.")
        if not 0.0 <= self.overexposure_bias <= 255.0:
            raise ReconstructionValidationError("I5.3A overexposure_bias must be between 0 and 255.")
        if not 0.05 <= self.clutter_band_fraction < 0.5:
            raise ReconstructionValidationError("I5.3A clutter_band_fraction must be in [0.05, 0.5).")
        if not isinstance(self.clutter_source_offset, int) or self.clutter_source_offset <= 0:
            raise ReconstructionValidationError("I5.3A clutter_source_offset must be a positive integer.")
        if not 0.05 <= self.symmetry_band_fraction < 0.5:
            raise ReconstructionValidationError("I5.3A symmetry_band_fraction must be in [0.05, 0.5).")

    def as_dict(self) -> dict[str, Any]:
        return {
            "blur_kernel_size": self.blur_kernel_size,
            "blur_sigma": self.blur_sigma,
            "underexposure_gain": self.underexposure_gain,
            "overexposure_gain": self.overexposure_gain,
            "overexposure_bias": self.overexposure_bias,
            "clutter_band_fraction": self.clutter_band_fraction,
            "clutter_source_offset": self.clutter_source_offset,
            "symmetry_band_fraction": self.symmetry_band_fraction,
        }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{label} must be an object.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"{label}.{key} must be {expected!r}; found {mapping.get(key)!r}."
        )


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def validate_i5_2c_for_i5_3(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.2C closure")
    _require(payload, "phase", "I5.2C", label="I5.2C closure")
    _require(payload, "status", "low_overlap_weak_parallax_robustness_qualified", label="I5.2C closure")
    _require(payload, "pass", True, label="I5.2C closure")

    scope = _mapping(payload.get("scope"), label="I5.2C scope")
    for key in (
        "sfm_rerun",
        "dense_provider_rerun",
        "quality_replay_rerun",
        "provider_specific_robustness_started",
        "mp_sfm_strategy_evaluation_started",
        "thresholds_modified",
        "production_default_modified",
        "auto_policy_work_started",
        "hybrid_experiment_started",
    ):
        _require(scope, key, False, label="I5.2C scope")

    qualification = _mapping(payload.get("qualification"), label="I5.2C qualification")
    source_sha = qualification.get("source_asset_set_sha256")
    if not _valid_sha256(source_sha):
        raise ReconstructionValidationError("I5.3A requires a valid frozen I5.2 source asset-set SHA256.")
    _require(qualification, "retry_until_quality_pass", False, label="I5.2C qualification")
    _require(qualification, "result_driven_subset_tuning", False, label="I5.2C qualification")

    authority = _mapping(payload.get("authority"), label="I5.2C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.2C authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.2C authority")
    for key in (
        "authoritative_sfm_modified",
        "provider_selected",
        "provider_promoted",
        "auto_policy_implemented",
        "hybrid_authorized",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.2C authority")

    decision = _mapping(payload.get("decision"), label="I5.2C decision")
    for key in (
        "i4_closed",
        "i5_authorized",
        "i5_1_complete",
        "i5_2_complete",
        "i5_2a_complete",
        "i5_2b_complete",
        "i5_2c_complete",
        "i5_3_authorized",
    ):
        _require(decision, key, True, label="I5.2C decision")
    _require(decision, "i5_3_started", False, label="I5.2C decision")
    _require(decision, "mp_sfm_authorized", False, label="I5.2C decision")
    _require(decision, "production_default_modified", False, label="I5.2C decision")
    _require(decision, "threshold_relaxation_authorized", False, label="I5.2C decision")
    if payload.get("next") != "I5.3 blur / exposure / clutter / symmetry.":
        raise ReconstructionValidationError("I5.2C next-step boundary does not authorize I5.3.")
    return deepcopy(dict(payload))


def _scenario_record(
    names: Sequence[str],
    *,
    kind: str,
    stress_proxy: bool,
    transform: Mapping[str, Any],
    revalidation: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = [str(name) for name in names]
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ReconstructionValidationError("I5.3A scenarios require a unique ordered source membership.")
    return {
        "kind": kind,
        "names": ordered,
        "image_count": len(ordered),
        "source_order_preserved": True,
        "full_source_membership_preserved": True,
        "stress_proxy": bool(stress_proxy),
        "source_derived": bool(stress_proxy),
        "stress_images_materialized": False,
        "semantic_stress_authoritative": not stress_proxy,
        "i5_3b_revalidation_required": bool(stress_proxy),
        "transform": deepcopy(dict(transform)),
        "required_i5_3b_revalidation": deepcopy(dict(revalidation)),
    }


def build_i5_3a_stress_plan(
    source_identity: Mapping[str, Any],
    *,
    policy: I53AStressDesignPolicy = I53AStressDesignPolicy(),
) -> dict[str, Any]:
    identity = validate_ordered_source_identity(source_identity)
    names = tuple(str(name) for name in identity["ordered_names"])
    count = len(names)
    if policy.clutter_source_offset >= count:
        raise ReconstructionValidationError(
            "I5.3A clutter_source_offset must be smaller than the bound source image count."
        )

    return {
        "control": _scenario_record(
            names,
            kind="full_source_control",
            stress_proxy=False,
            transform={"algorithm": "identity"},
            revalidation={"required": False},
        ),
        "blur_proxy": _scenario_record(
            names,
            kind="controlled_gaussian_blur_proxy",
            stress_proxy=True,
            transform={
                "algorithm": "gaussian_blur",
                "kernel_size": policy.blur_kernel_size,
                "sigma": policy.blur_sigma,
                "border_mode": "reflect_101",
            },
            revalidation={
                "image_domain": "median_laplacian_variance_must_decrease_vs_control",
                "sfm_outcome": "registration_and_sparse_health_observed_without_threshold_relaxation",
            },
        ),
        "underexposure_proxy": _scenario_record(
            names,
            kind="controlled_underexposure_proxy",
            stress_proxy=True,
            transform={
                "algorithm": "linear_intensity",
                "gain": policy.underexposure_gain,
                "bias": 0.0,
                "clip_uint8": True,
            },
            revalidation={
                "image_domain": "median_luma_must_decrease_vs_control",
                "quality_diagnostic": "underexposed_or_clipping_burden_must_not_improve",
                "sfm_outcome": "registration_and_sparse_health_observed_without_threshold_relaxation",
            },
        ),
        "overexposure_proxy": _scenario_record(
            names,
            kind="controlled_overexposure_proxy",
            stress_proxy=True,
            transform={
                "algorithm": "linear_intensity",
                "gain": policy.overexposure_gain,
                "bias": policy.overexposure_bias,
                "clip_uint8": True,
            },
            revalidation={
                "image_domain": "median_luma_must_increase_vs_control",
                "quality_diagnostic": "overexposed_or_clipping_burden_must_not_improve",
                "sfm_outcome": "registration_and_sparse_health_observed_without_threshold_relaxation",
            },
        ),
        "clutter_proxy": _scenario_record(
            names,
            kind="controlled_source_derived_inconsistent_distractor_clutter_proxy",
            stress_proxy=True,
            transform={
                "algorithm": "source_derived_distractor_band",
                "band_fraction": policy.clutter_band_fraction,
                "band_location": "bottom",
                "source_offset": policy.clutter_source_offset,
                "source_index_rule": "(current_index + source_offset) mod image_count",
                "resize_mode": "cover_then_center_crop",
            },
            revalidation={
                "image_domain": "distractor_band_feature_burden_must_increase_vs_control_band",
                "semantic_guard": "proxy_is_not_natural_scene_clutter_ground_truth",
                "sfm_outcome": "geometric_verification_and_sparse_health_observed_without_threshold_relaxation",
            },
        ),
        "symmetry_proxy": _scenario_record(
            names,
            kind="controlled_mirrored_repeated_structure_symmetry_proxy",
            stress_proxy=True,
            transform={
                "algorithm": "mirrored_repeated_side_bands",
                "band_fraction": policy.symmetry_band_fraction,
                "source_region": "center_vertical_band",
                "placements": ["left", "right"],
                "right_band_mirrored": True,
            },
            revalidation={
                "image_domain": "repeated_region_self_similarity_or_descriptor_ambiguity_must_increase_vs_control",
                "semantic_guard": "proxy_is_not_natural_scene_symmetry_ground_truth",
                "sfm_outcome": "geometric_verification_and_sparse_health_observed_without_threshold_relaxation",
            },
        ),
    }


def validate_i5_3a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.3A receipt")
    _require(payload, "phase", I53A_RECEIPT_PHASE, label="I5.3A receipt")
    _require(payload, "status", "pass", label="I5.3A receipt")
    _require(payload, "pass", True, label="I5.3A receipt")
    scope = _mapping(payload.get("scope"), label="I5.3A receipt scope")
    for key in (
        "focused_i5_3a_tests",
        "predecessor_i5_2c_regression",
        "source_identity_regression",
        "stress_design_regression",
        "import_probe",
        "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.3A receipt scope")
    for key in (
        "stress_images_materialized",
        "sfm_rerun",
        "dense_provider_rerun",
        "thresholds_modified",
        "production_default_modified",
        "mp_sfm_started",
        "auto_policy_implemented",
        "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.3A receipt scope")

    counts = _mapping(payload.get("counts"), label="I5.3A receipt counts")
    if int(counts.get("focused_tests", 0)) < 10:
        raise ReconstructionValidationError("I5.3A focused test coverage is below the required admission floor.")
    if int(counts.get("combined_exercised_tests", 0)) < 40:
        raise ReconstructionValidationError("I5.3A receipt coverage is below the required admission floor.")
    codes = _mapping(payload.get("return_codes"), label="I5.3A receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        _require(codes, key, 0, label="I5.3A receipt return_codes")
    return deepcopy(dict(payload))


def build_i5_3a_blur_exposure_clutter_symmetry_admission(
    *,
    i5_2c_result: Mapping[str, Any],
    i5_2c_binding: I51EvidenceBinding,
    source_identity: Mapping[str, Any],
    controlled_source_confirmation: bool,
    admission_receipt: Mapping[str, Any],
    admission_receipt_binding: I51EvidenceBinding,
    policy: I53AStressDesignPolicy = I53AStressDesignPolicy(),
) -> dict[str, Any]:
    predecessor = validate_i5_2c_for_i5_3(i5_2c_result)
    receipt = validate_i5_3a_receipt(admission_receipt)
    identity = validate_ordered_source_identity(source_identity)
    if not controlled_source_confirmation:
        raise ReconstructionValidationError(
            "I5.3A requires explicit confirmation that the rebound source is the frozen ordered static control capture used for source-derived stress design."
        )

    qualification = _mapping(predecessor.get("qualification"), label="I5.2C qualification")
    frozen_source_sha = str(qualification.get("source_asset_set_sha256", "")).lower()
    rebound_source_sha = str(identity.get("asset_set_sha256", "")).lower()
    if rebound_source_sha != frozen_source_sha:
        raise ReconstructionValidationError(
            "I5.3A source asset identity does not match the frozen I5.2 control source."
        )

    plan = build_i5_3a_stress_plan(identity, policy=policy)
    authority = deepcopy(dict(_mapping(predecessor.get("authority"), label="I5.2C authority")))
    return {
        "schema_version": I53A_SCHEMA_VERSION,
        "phase": I53A_PHASE,
        "status": "blur_exposure_clutter_symmetry_scenarios_admitted",
        "pass": True,
        "scope": {
            "scenario_admission_only": True,
            "same_control_source_as_i5_2": True,
            "deterministic_source_derived_stress_recipes_only": True,
            "stress_images_materialized": False,
            "sfm_rerun": False,
            "dense_provider_rerun": False,
            "provider_specific_robustness_started": False,
            "mp_sfm_strategy_evaluation_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
        },
        "evidence_bindings": {
            "i5_2c_closure": i5_2c_binding.as_dict(),
            "i5_3a_admission_receipt": admission_receipt_binding.as_dict(),
        },
        "predecessor": {
            "i5_2_status": predecessor.get("status"),
            "i5_2_quality_status": qualification.get("quality_status"),
            "i5_2_operating_envelope": qualification.get("operating_envelope"),
            "i5_2_source_asset_set_sha256": frozen_source_sha,
            "i5_3_authorized": True,
            "i5_3_started_before_i5_3a": False,
        },
        "source_identity": identity,
        "stress_plan": plan,
        "stress_design": {
            "policy": policy.as_dict(),
            "stress_recipe_count": 5,
            "exposure_subcases": ["underexposure_proxy", "overexposure_proxy"],
            "all_stresses_share_exact_control_source_membership": True,
            "all_stress_images_deferred_to_i5_3b_materialization": True,
            "semantic_stress_labels_require_i5_3b_revalidation": True,
            "failed_semantic_revalidation_is_legitimate_evidence": True,
            "result_driven_recipe_tuning_authorized": False,
            "retry_until_quality_pass_authorized": False,
            "natural_clutter_ground_truth_claimed": False,
            "natural_symmetry_ground_truth_claimed": False,
            "image_quality_threshold_relaxation_authorized": False,
        },
        "regression_summary": {
            "focused_i5_3a_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "all_required_suites_passed": True,
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": authority,
        "decision": {
            "i4_closed": True,
            "i5_authorized": True,
            "i5_1_complete": True,
            "i5_2_complete": True,
            "i5_3_started": True,
            "i5_3a_complete": True,
            "i5_3b_real_replay_authorized": True,
            "i5_3_complete": False,
            "i5_4_authorized": False,
            "provider_specific_robustness_started": False,
            "mp_sfm_authorized": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "threshold_relaxation_authorized": False,
            "production_default_modified": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "roadmap_guard": {
            "i5_3b_real_replay_authorized_not_started": True,
            "i5_4_forward_motion_kitti_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.3B current frozen-foundation blur / exposure / clutter / symmetry replay.",
    }


__all__ = [
    "I53A_SCHEMA_VERSION",
    "I53A_PHASE",
    "I53A_RECEIPT_PHASE",
    "I53AStressDesignPolicy",
    "validate_i5_2c_for_i5_3",
    "build_i5_3a_stress_plan",
    "validate_i5_3a_receipt",
    "build_i5_3a_blur_exposure_clutter_symmetry_admission",
]
