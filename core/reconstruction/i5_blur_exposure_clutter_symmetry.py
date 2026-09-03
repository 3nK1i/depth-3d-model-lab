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
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import analyze_i5_2b_sfm_report, validate_ordered_source_identity
from core.reconstruction.quality import analyze_frame_quality

I53A_SCHEMA_VERSION = 1
I53A_PHASE = "I5.3A"
I53A_RECEIPT_PHASE = "I5.3A_ADMISSION_RECEIPT"
I53B_SCHEMA_VERSION = 1
I53B_PHASE = "I5.3B"
I53B_RECEIPT_PHASE = "I5.3B_REPLAY_RECEIPT"


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



def validate_i5_3a_for_i5_3b(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen I5.3A handoff before any I5.3B materialization."""

    _require(payload, "schema_version", I53A_SCHEMA_VERSION, label="I5.3A admission")
    _require(payload, "phase", I53A_PHASE, label="I5.3A admission")
    _require(payload, "status", "blur_exposure_clutter_symmetry_scenarios_admitted", label="I5.3A admission")
    _require(payload, "pass", True, label="I5.3A admission")

    scope = _mapping(payload.get("scope"), label="I5.3A scope")
    _require(scope, "scenario_admission_only", True, label="I5.3A scope")
    _require(scope, "same_control_source_as_i5_2", True, label="I5.3A scope")
    _require(scope, "deterministic_source_derived_stress_recipes_only", True, label="I5.3A scope")
    for key in (
        "stress_images_materialized",
        "sfm_rerun",
        "dense_provider_rerun",
        "provider_specific_robustness_started",
        "mp_sfm_strategy_evaluation_started",
        "thresholds_modified",
        "production_default_modified",
        "auto_policy_work_started",
        "hybrid_experiment_started",
    ):
        _require(scope, key, False, label="I5.3A scope")

    source = validate_ordered_source_identity(_mapping(payload.get("source_identity"), label="I5.3A source_identity"))
    predecessor = _mapping(payload.get("predecessor"), label="I5.3A predecessor")
    frozen_source_sha = str(predecessor.get("i5_2_source_asset_set_sha256", "")).lower()
    if str(source.get("asset_set_sha256", "")).lower() != frozen_source_sha:
        raise ReconstructionValidationError("I5.3A source identity drifted from the frozen I5.2 control source.")
    plan = _mapping(payload.get("stress_plan"), label="I5.3A stress_plan")
    expected = {
        "control",
        "blur_proxy",
        "underexposure_proxy",
        "overexposure_proxy",
        "clutter_proxy",
        "symmetry_proxy",
    }
    if set(plan) != expected:
        raise ReconstructionValidationError("I5.3B requires the exact frozen six-scenario I5.3A plan.")
    source_names = [str(value) for value in source["ordered_names"]]
    for key in sorted(expected):
        item = _mapping(plan.get(key), label=f"I5.3A stress_plan.{key}")
        if list(item.get("names", [])) != source_names or int(item.get("image_count", -1)) != len(source_names):
            raise ReconstructionValidationError(f"I5.3A {key} membership drifted from the frozen control source.")
        _require(item, "stress_images_materialized", False, label=f"I5.3A stress_plan.{key}")
        if key == "control":
            _require(item, "stress_proxy", False, label=f"I5.3A stress_plan.{key}")
            _require(item, "semantic_stress_authoritative", True, label=f"I5.3A stress_plan.{key}")
            _require(item, "i5_3b_revalidation_required", False, label=f"I5.3A stress_plan.{key}")
        else:
            _require(item, "stress_proxy", True, label=f"I5.3A stress_plan.{key}")
            _require(item, "semantic_stress_authoritative", False, label=f"I5.3A stress_plan.{key}")
            _require(item, "i5_3b_revalidation_required", True, label=f"I5.3A stress_plan.{key}")

    design = _mapping(payload.get("stress_design"), label="I5.3A stress_design")
    _require(design, "stress_recipe_count", 5, label="I5.3A stress_design")
    _require(design, "semantic_stress_labels_require_i5_3b_revalidation", True, label="I5.3A stress_design")
    _require(design, "failed_semantic_revalidation_is_legitimate_evidence", True, label="I5.3A stress_design")
    _require(design, "result_driven_recipe_tuning_authorized", False, label="I5.3A stress_design")
    _require(design, "retry_until_quality_pass_authorized", False, label="I5.3A stress_design")
    _require(design, "natural_clutter_ground_truth_claimed", False, label="I5.3A stress_design")
    _require(design, "natural_symmetry_ground_truth_claimed", False, label="I5.3A stress_design")
    _require(design, "image_quality_threshold_relaxation_authorized", False, label="I5.3A stress_design")

    authority = _mapping(payload.get("authority"), label="I5.3A authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.3A authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.3A authority")
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
        _require(authority, key, False, label="I5.3A authority")

    decision = _mapping(payload.get("decision"), label="I5.3A decision")
    for key in ("i4_closed", "i5_authorized", "i5_1_complete", "i5_2_complete", "i5_3_started", "i5_3a_complete", "i5_3b_real_replay_authorized"):
        _require(decision, key, True, label="I5.3A decision")
    for key in ("i5_3_complete", "i5_4_authorized", "mp_sfm_authorized", "auto_policy_implemented", "hybrid_authorized", "production_default_modified"):
        _require(decision, key, False, label="I5.3A decision")
    if payload.get("next") != "I5.3B current frozen-foundation blur / exposure / clutter / symmetry replay.":
        raise ReconstructionValidationError("I5.3A next-step boundary does not authorize I5.3B.")
    return deepcopy(dict(payload))


def validate_i5_3b_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.3B receipt")
    _require(payload, "phase", I53B_RECEIPT_PHASE, label="I5.3B receipt")
    _require(payload, "status", "pass", label="I5.3B receipt")
    _require(payload, "pass", True, label="I5.3B receipt")
    scope = _mapping(payload.get("scope"), label="I5.3B receipt scope")
    for key in (
        "focused_i5_3b_tests",
        "predecessor_i5_3a_regression",
        "production_native_sfm_contract_regression",
        "image_domain_stress_regression",
        "import_probe",
        "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.3B receipt scope")
    for key in (
        "real_sfm_executed_by_test_receipt",
        "dense_provider_rerun",
        "thresholds_modified",
        "production_default_modified",
        "mp_sfm_started",
        "auto_policy_implemented",
        "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.3B receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.3B receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(value, int) and value >= 0 for value in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.3B receipt test counts must be non-negative integers.")
    if focused < 15 or predecessor < 35 or combined < 50 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.3B receipt coverage is below the required replay floor.")
    codes = _mapping(payload.get("return_codes"), label="I5.3B receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        _require(codes, key, 0, label="I5.3B receipt return_codes")
    return deepcopy(dict(payload))


def _cover_then_center_crop(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ReconstructionValidationError("I5.3B transform input must be uint8 BGR.")
    ih, iw = image.shape[:2]
    if ih <= 0 or iw <= 0 or height <= 0 or width <= 0:
        raise ReconstructionValidationError("I5.3B transform dimensions must be positive.")
    scale = max(width / iw, height / ih)
    resized = cv2.resize(image, (max(width, int(round(iw * scale))), max(height, int(round(ih * scale)))), interpolation=cv2.INTER_LINEAR)
    rh, rw = resized.shape[:2]
    x0 = max(0, (rw - width) // 2)
    y0 = max(0, (rh - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width].copy()


def apply_i5_3b_transform(
    image: np.ndarray,
    transform: Mapping[str, Any],
    *,
    distractor_image: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one exact I5.3A recipe without mutating the source image."""

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ReconstructionValidationError("I5.3B source image must be HxWx3 uint8 BGR.")
    algorithm = str(transform.get("algorithm", ""))
    if algorithm == "identity":
        return image.copy()
    if algorithm == "gaussian_blur":
        kernel = int(transform.get("kernel_size", 0))
        sigma = float(transform.get("sigma", 0.0))
        if kernel < 3 or kernel % 2 == 0 or sigma <= 0.0:
            raise ReconstructionValidationError("I5.3B Gaussian blur recipe is invalid.")
        if transform.get("border_mode") != "reflect_101":
            raise ReconstructionValidationError("I5.3B Gaussian blur border-mode drifted from I5.3A.")
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    if algorithm == "linear_intensity":
        gain = float(transform.get("gain", 0.0))
        bias = float(transform.get("bias", 0.0))
        if gain <= 0.0 or transform.get("clip_uint8") is not True:
            raise ReconstructionValidationError("I5.3B linear-intensity recipe is invalid.")
        values = image.astype(np.float32) * gain + bias
        return np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)
    if algorithm == "source_derived_distractor_band":
        if distractor_image is None:
            raise ReconstructionValidationError("I5.3B clutter transform requires its frozen offset source image.")
        fraction = float(transform.get("band_fraction", 0.0))
        if not 0.05 <= fraction < 0.5 or transform.get("band_location") != "bottom" or transform.get("resize_mode") != "cover_then_center_crop":
            raise ReconstructionValidationError("I5.3B clutter recipe drifted from I5.3A.")
        height, width = image.shape[:2]
        distractor = _cover_then_center_crop(distractor_image, width=width, height=height)
        band = max(1, int(round(height * fraction)))
        result = image.copy()
        result[height - band:height] = distractor[height - band:height]
        return result
    if algorithm == "mirrored_repeated_side_bands":
        fraction = float(transform.get("band_fraction", 0.0))
        if not 0.05 <= fraction < 0.5:
            raise ReconstructionValidationError("I5.3B symmetry band fraction drifted from I5.3A.")
        if transform.get("source_region") != "center_vertical_band" or transform.get("placements") != ["left", "right"] or transform.get("right_band_mirrored") is not True:
            raise ReconstructionValidationError("I5.3B symmetry recipe drifted from I5.3A.")
        height, width = image.shape[:2]
        band = max(1, int(round(width * fraction)))
        center = width // 2
        x0 = max(0, min(width - band, center - band // 2))
        source_band = image[:, x0:x0 + band].copy()
        result = image.copy()
        result[:, :band] = source_band
        result[:, width - band:width] = cv2.flip(source_band, 1)
        return result
    raise ReconstructionValidationError(f"I5.3B unsupported frozen transform algorithm: {algorithm!r}")


def analyze_i5_3b_image(
    image: np.ndarray,
    *,
    band_fraction: float,
    symmetry_band_fraction: float | None = None,
) -> dict[str, Any]:
    if not 0.05 <= float(band_fraction) < 0.5:
        raise ReconstructionValidationError("I5.3B diagnostic band_fraction must be in [0.05, 0.5).")
    symmetry_fraction = float(band_fraction if symmetry_band_fraction is None else symmetry_band_fraction)
    if not 0.05 <= symmetry_fraction < 0.5:
        raise ReconstructionValidationError("I5.3B symmetry diagnostic band fraction must be in [0.05, 0.5).")
    quality = analyze_frame_quality(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    bottom_height = max(1, int(round(height * float(band_fraction))))
    bottom = gray[height - bottom_height:height]
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints = orb.detect(bottom, None) or ()
    megapixels = max(float(bottom.size) / 1_000_000.0, 1e-12)
    side_width = max(1, int(round(width * symmetry_fraction)))
    left = gray[:, :side_width].astype(np.float32)
    right = cv2.flip(gray[:, width - side_width:width], 1).astype(np.float32)
    similarity = float(np.clip(1.0 - np.mean(np.abs(left - right)) / 255.0, 0.0, 1.0))
    payload = quality.as_dict()
    payload.update({
        "bottom_band_orb_keypoints": int(len(keypoints)),
        "bottom_band_orb_feature_density_per_megapixel": float(len(keypoints) / megapixels),
        "mirrored_side_similarity": similarity,
        "diagnostic_band_fraction": float(band_fraction),
        "symmetry_diagnostic_band_fraction": float(band_fraction if symmetry_band_fraction is None else symmetry_band_fraction),
    })
    return payload


def analyze_i5_3b_image_set(
    directory: str | Path,
    *,
    names: Sequence[str],
    band_fraction: float,
    symmetry_band_fraction: float | None = None,
) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"I5.3B image-set directory is missing: {root}")
    records: list[dict[str, Any]] = []
    for name in names:
        path = root / str(name)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ReconstructionValidationError(f"I5.3B cannot decode staged image: {path}")
        record = analyze_i5_3b_image(image, band_fraction=band_fraction, symmetry_band_fraction=symmetry_band_fraction)
        record["name"] = str(name)
        records.append(record)
    if not records:
        raise ReconstructionValidationError("I5.3B image-set diagnostics require at least one image.")

    def med(key: str) -> float:
        return float(median(float(record[key]) for record in records))

    return {
        "image_count": len(records),
        "blur_variance_p50": med("blur_variance"),
        "contrast_stddev_p50": med("contrast_stddev"),
        "mean_luma_p50": med("mean_luma"),
        "underexposed_fraction_p50": med("underexposed_fraction"),
        "overexposed_fraction_p50": med("overexposed_fraction"),
        "quality_score_p50": med("quality_score"),
        "accepted_fraction": float(sum(bool(record["accepted"]) for record in records) / len(records)),
        "bottom_band_orb_feature_density_p50": med("bottom_band_orb_feature_density_per_megapixel"),
        "mirrored_side_similarity_p50": med("mirrored_side_similarity"),
        "diagnostic_band_fraction": float(band_fraction),
        "symmetry_diagnostic_band_fraction": float(band_fraction if symmetry_band_fraction is None else symmetry_band_fraction),
        "per_image": records,
    }


def _finite_metric(mapping: Mapping[str, Any], key: str, *, label: str) -> float:
    value = float(mapping.get(key))
    if not isfinite(value):
        raise ReconstructionValidationError(f"I5.3B {label}.{key} must be finite.")
    return value


def _directional_ratio(stress: float, control: float) -> float | None:
    if control == 0.0:
        return None
    return float(stress / control)


def compare_i5_3b_image_domain(*, control: Mapping[str, Any], stresses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = {"blur", "underexposure", "overexposure", "clutter", "symmetry"}
    if set(stresses) != required:
        raise ReconstructionValidationError("I5.3B image-domain comparison requires exactly five stress scenarios.")
    c_blur = _finite_metric(control, "blur_variance_p50", label="control")
    c_luma = _finite_metric(control, "mean_luma_p50", label="control")
    c_under = _finite_metric(control, "underexposed_fraction_p50", label="control")
    c_over = _finite_metric(control, "overexposed_fraction_p50", label="control")
    c_quality = _finite_metric(control, "quality_score_p50", label="control")
    c_clutter = _finite_metric(control, "bottom_band_orb_feature_density_p50", label="control")
    c_sym = _finite_metric(control, "mirrored_side_similarity_p50", label="control")

    b = stresses["blur"]
    u = stresses["underexposure"]
    o = stresses["overexposure"]
    c = stresses["clutter"]
    s = stresses["symmetry"]
    b_blur = _finite_metric(b, "blur_variance_p50", label="blur")
    u_luma = _finite_metric(u, "mean_luma_p50", label="underexposure")
    u_under = _finite_metric(u, "underexposed_fraction_p50", label="underexposure")
    u_quality = _finite_metric(u, "quality_score_p50", label="underexposure")
    o_luma = _finite_metric(o, "mean_luma_p50", label="overexposure")
    o_over = _finite_metric(o, "overexposed_fraction_p50", label="overexposure")
    o_quality = _finite_metric(o, "quality_score_p50", label="overexposure")
    c_density = _finite_metric(c, "bottom_band_orb_feature_density_p50", label="clutter")
    s_similarity = _finite_metric(s, "mirrored_side_similarity_p50", label="symmetry")

    blur_ok = b_blur < c_blur
    under_burden = u_under >= c_under or u_quality <= c_quality
    under_ok = u_luma < c_luma and under_burden
    over_burden = o_over >= c_over or o_quality <= c_quality
    over_ok = o_luma > c_luma and over_burden
    clutter_ok = c_density > c_clutter
    symmetry_ok = s_similarity > c_sym
    return {
        "directional_only": True,
        "threshold_relaxation_used": False,
        "blur": {
            "control_blur_variance_p50": c_blur,
            "stress_blur_variance_p50": b_blur,
            "stress_to_control_ratio": _directional_ratio(b_blur, c_blur),
            "semantic_stress_revalidated": bool(blur_ok),
        },
        "underexposure": {
            "control_mean_luma_p50": c_luma,
            "stress_mean_luma_p50": u_luma,
            "control_underexposed_fraction_p50": c_under,
            "stress_underexposed_fraction_p50": u_under,
            "control_quality_score_p50": c_quality,
            "stress_quality_score_p50": u_quality,
            "exposure_or_clipping_burden_not_improved": bool(under_burden),
            "semantic_stress_revalidated": bool(under_ok),
        },
        "overexposure": {
            "control_mean_luma_p50": c_luma,
            "stress_mean_luma_p50": o_luma,
            "control_overexposed_fraction_p50": c_over,
            "stress_overexposed_fraction_p50": o_over,
            "control_quality_score_p50": c_quality,
            "stress_quality_score_p50": o_quality,
            "exposure_or_clipping_burden_not_improved": bool(over_burden),
            "semantic_stress_revalidated": bool(over_ok),
        },
        "clutter": {
            "control_bottom_band_orb_feature_density_p50": c_clutter,
            "stress_bottom_band_orb_feature_density_p50": c_density,
            "stress_to_control_ratio": _directional_ratio(c_density, c_clutter),
            "semantic_stress_revalidated": bool(clutter_ok),
            "natural_scene_ground_truth": False,
        },
        "symmetry": {
            "control_mirrored_side_similarity_p50": c_sym,
            "stress_mirrored_side_similarity_p50": s_similarity,
            "semantic_stress_revalidated": bool(symmetry_ok),
            "natural_scene_ground_truth": False,
        },
        "all_semantic_stresses_revalidated": bool(blur_ok and under_ok and over_ok and clutter_ok and symmetry_ok),
    }


def _validate_i5_3b_image_domain_comparison(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "directional_only", True, label="I5.3B image comparison")
    _require(payload, "threshold_relaxation_used", False, label="I5.3B image comparison")
    blur = _mapping(payload.get("blur"), label="I5.3B image comparison blur")
    under = _mapping(payload.get("underexposure"), label="I5.3B image comparison underexposure")
    over = _mapping(payload.get("overexposure"), label="I5.3B image comparison overexposure")
    clutter = _mapping(payload.get("clutter"), label="I5.3B image comparison clutter")
    symmetry = _mapping(payload.get("symmetry"), label="I5.3B image comparison symmetry")
    expected = [
        _finite_metric(blur, "stress_blur_variance_p50", label="comparison.blur") < _finite_metric(blur, "control_blur_variance_p50", label="comparison.blur"),
        _finite_metric(under, "stress_mean_luma_p50", label="comparison.under") < _finite_metric(under, "control_mean_luma_p50", label="comparison.under") and (
            _finite_metric(under, "stress_underexposed_fraction_p50", label="comparison.under") >= _finite_metric(under, "control_underexposed_fraction_p50", label="comparison.under")
            or _finite_metric(under, "stress_quality_score_p50", label="comparison.under") <= _finite_metric(under, "control_quality_score_p50", label="comparison.under")
        ),
        _finite_metric(over, "stress_mean_luma_p50", label="comparison.over") > _finite_metric(over, "control_mean_luma_p50", label="comparison.over") and (
            _finite_metric(over, "stress_overexposed_fraction_p50", label="comparison.over") >= _finite_metric(over, "control_overexposed_fraction_p50", label="comparison.over")
            or _finite_metric(over, "stress_quality_score_p50", label="comparison.over") <= _finite_metric(over, "control_quality_score_p50", label="comparison.over")
        ),
        _finite_metric(clutter, "stress_bottom_band_orb_feature_density_p50", label="comparison.clutter") > _finite_metric(clutter, "control_bottom_band_orb_feature_density_p50", label="comparison.clutter"),
        _finite_metric(symmetry, "stress_mirrored_side_similarity_p50", label="comparison.symmetry") > _finite_metric(symmetry, "control_mirrored_side_similarity_p50", label="comparison.symmetry"),
    ]
    nodes = (blur, under, over, clutter, symmetry)
    for node, value in zip(nodes, expected):
        _require(node, "semantic_stress_revalidated", bool(value), label="I5.3B image comparison")
    _require(payload, "all_semantic_stresses_revalidated", bool(all(expected)), label="I5.3B image comparison")
    _require(clutter, "natural_scene_ground_truth", False, label="I5.3B image comparison clutter")
    _require(symmetry, "natural_scene_ground_truth", False, label="I5.3B image comparison symmetry")
    return deepcopy(dict(payload))


def analyze_i5_3b_sfm_report(
    report: Mapping[str, Any],
    *,
    scenario_plan: Mapping[str, Any],
    staged_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    adapted = deepcopy(dict(scenario_plan))
    count = adapted.get("image_count")
    if not isinstance(count, int) or count < 3:
        raise ReconstructionValidationError("I5.3B scenario image_count must be at least three.")
    source_names_raw = adapted.get("names")
    if not isinstance(source_names_raw, Sequence) or isinstance(source_names_raw, (str, bytes)) or len(source_names_raw) != count:
        raise ReconstructionValidationError("I5.3B scenario names do not match image_count.")
    source_names = tuple(str(value) for value in source_names_raw)
    if staged_names is None:
        materialized_names = source_names
    else:
        materialized_names = tuple(str(value) for value in staged_names)
        if len(materialized_names) != count or len(set(materialized_names)) != count:
            raise ReconstructionValidationError("I5.3B staged scenario names must be unique and match image_count.")
    adapted["names"] = list(materialized_names)
    adapted["indices"] = list(range(count))
    analysis = analyze_i5_2b_sfm_report(report, scenario_plan=adapted)
    analysis["registration_attempted"] = True
    analysis["sfm_executed"] = True
    analysis["pre_sfm_ingestion_rejected"] = False
    analysis["rejection_stage"] = None
    analysis["rejection_reason"] = None
    if materialized_names != source_names:
        mapping = dict(zip(materialized_names, source_names))
        registered_staged = [str(value) for value in analysis.get("registered_names", [])]
        analysis["registered_staged_names"] = registered_staged
        analysis["registered_names"] = [mapping[value] for value in registered_staged]
        analysis["source_membership_names"] = list(source_names)
        analysis["staged_membership_names"] = list(materialized_names)
        analysis["lossless_staging_name_mapping_preserved"] = True
    return analysis


def analyze_i5_3b_pre_sfm_ingestion_rejection(
    *,
    scenario_plan: Mapping[str, Any],
    staged_names: Sequence[str],
    reason: str,
) -> dict[str, Any]:
    """Preserve a production ingestion rejection as structured I5.3B evidence.

    The production-native runner can reject a severe visual stress before PyCOLMAP
    mapping starts when fewer than three keyframes survive ingestion. That outcome
    is a legitimate robustness result, not an infrastructure failure.
    """
    adapted = deepcopy(dict(scenario_plan))
    count = adapted.get("image_count")
    if not isinstance(count, int) or count < 3:
        raise ReconstructionValidationError("I5.3B scenario image_count must be at least three.")
    source_names_raw = adapted.get("names")
    if not isinstance(source_names_raw, Sequence) or isinstance(source_names_raw, (str, bytes)) or len(source_names_raw) != count:
        raise ReconstructionValidationError("I5.3B scenario names do not match image_count.")
    materialized_names = tuple(str(value) for value in staged_names)
    if len(materialized_names) != count or len(set(materialized_names)) != count:
        raise ReconstructionValidationError("I5.3B staged scenario names must be unique and match image_count.")
    source_names = tuple(str(value) for value in source_names_raw)
    message = str(reason).strip()
    if not message:
        raise ReconstructionValidationError("I5.3B pre-SfM ingestion rejection requires a reason.")
    return {
        "scenario_kind": str(adapted.get("scenario_kind", "")),
        "input_images": count,
        "registered_images": 0,
        "registration_ratio": 0.0,
        "full_input_scope_registered": False,
        "partial_registration_preserved": False,
        "registered_names": [],
        "registered_staged_names": [],
        "source_membership_names": list(source_names),
        "staged_membership_names": list(materialized_names),
        "lossless_staging_name_mapping_preserved": True,
        "quality_status": "rejected",
        "quality_assessment": {
            "status": "rejected",
            "usable": False,
            "reasons": [message],
        },
        "statistics": {
            "input_keyframes": count,
            "registered_keyframes": 0,
            "landmarks": 0,
            "observations": 0,
            "mean_reprojection_error_px": None,
        },
        "geometry": {
            "available": False,
            "reason": "pre_sfm_ingestion_rejection",
        },
        "registration_attempted": False,
        "sfm_executed": False,
        "pre_sfm_ingestion_rejected": True,
        "rejection_stage": "ingestion",
        "rejection_reason": message,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "per_scenario_gauge": "unavailable_no_sfm_model",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
    }


def build_i5_3b_blur_exposure_clutter_symmetry_replay(
    *,
    i5_3a_result: Mapping[str, Any],
    i5_3a_binding: I51EvidenceBinding,
    replay_receipt: Mapping[str, Any],
    replay_receipt_binding: I51EvidenceBinding,
    image_domain_comparison: Mapping[str, Any],
    scenario_analyses: Mapping[str, Mapping[str, Any]],
    source_reverified_before: bool,
    source_reverified_after: bool,
    staged_inputs_reverified: bool,
) -> dict[str, Any]:
    predecessor = validate_i5_3a_for_i5_3b(i5_3a_result)
    receipt = validate_i5_3b_receipt(replay_receipt)
    comparison = _validate_i5_3b_image_domain_comparison(image_domain_comparison)
    if source_reverified_before is not True or source_reverified_after is not True or staged_inputs_reverified is not True:
        raise ReconstructionValidationError("I5.3B requires source and staged-input revalidation around the real replay.")
    expected = {"control", "blur", "underexposure", "overexposure", "clutter", "symmetry"}
    if set(scenario_analyses) != expected:
        raise ReconstructionValidationError("I5.3B requires exactly six scenario analyses.")
    scenarios: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    for key in sorted(expected):
        analysis = deepcopy(dict(_mapping(scenario_analyses[key], label=f"I5.3B scenario {key}")))
        status = str(analysis.get("quality_status", ""))
        if status not in {"accepted", "degraded", "rejected"}:
            raise ReconstructionValidationError(f"I5.3B scenario {key} has an invalid quality status.")
        authority = _mapping(analysis.get("authority"), label=f"I5.3B scenario {key} authority")
        _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.3B scenario {key} authority")
        statuses[key] = status
        scenarios[key] = analysis

    if statuses["control"] == "rejected":
        quality_status = "rejected"
    elif comparison["all_semantic_stresses_revalidated"] and all(value == "accepted" for value in statuses.values()):
        quality_status = "pass"
    else:
        quality_status = "degraded"

    authority = deepcopy(dict(_mapping(predecessor.get("authority"), label="I5.3A authority")))
    return {
        "schema_version": I53B_SCHEMA_VERSION,
        "phase": I53B_PHASE,
        "status": "blur_exposure_clutter_symmetry_replay_evidence_complete",
        "pass": True,
        "quality_status": quality_status,
        "scope": {
            "current_frozen_foundation_real_replay": True,
            "stress_images_materialized": True,
            "control_replay_attempted": True,
            "blur_replay_attempted": True,
            "underexposure_replay_attempted": True,
            "overexposure_replay_attempted": True,
            "clutter_replay_attempted": True,
            "symmetry_replay_attempted": True,
            "control_sfm_rerun": bool(scenarios["control"].get("sfm_executed", True)),
            "blur_sfm_rerun": bool(scenarios["blur"].get("sfm_executed", True)),
            "underexposure_sfm_rerun": bool(scenarios["underexposure"].get("sfm_executed", True)),
            "overexposure_sfm_rerun": bool(scenarios["overexposure"].get("sfm_executed", True)),
            "clutter_sfm_rerun": bool(scenarios["clutter"].get("sfm_executed", True)),
            "symmetry_sfm_rerun": bool(scenarios["symmetry"].get("sfm_executed", True)),
            "dense_provider_rerun": False,
            "provider_specific_robustness_started": False,
            "mp_sfm_strategy_evaluation_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
        },
        "evidence_bindings": {
            "i5_3a_admission": i5_3a_binding.as_dict(),
            "i5_3b_replay_receipt": replay_receipt_binding.as_dict(),
        },
        "provenance": {
            "frozen_source_asset_set_sha256": predecessor["source_identity"]["asset_set_sha256"],
            "source_reverified_before_replay": True,
            "source_reverified_after_replay": True,
            "staged_inputs_reverified_after_replay": True,
            "same_matching_policy_all_scenarios": "exhaustive",
            "same_production_native_sfm_runner_all_scenarios": True,
            "stress_recipes_reused_exactly_from_i5_3a": True,
            "result_driven_recipe_tuning": False,
            "retry_until_quality_pass": False,
            "failed_semantic_revalidation_preserved": not bool(comparison["all_semantic_stresses_revalidated"]),
            "historical_result_overwritten": False,
        },
        "image_domain_revalidation": comparison,
        "scenarios": scenarios,
        "regression_summary": {
            "focused_i5_3b_tests": receipt["counts"]["focused_tests"],
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
            "i5_3b_complete": True,
            "i5_3c_qualification_closure_authorized": True,
            "i5_3_complete": False,
            "i5_4_authorized": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "mp_sfm_authorized": False,
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
            "i5_3c_qualification_closure_authorized_not_started": True,
            "i5_4_forward_motion_kitti_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.3C blur / exposure / clutter / symmetry robustness qualification and closure.",
    }


# ---------------------------------------------------------------------------
# I5.3C — blur / exposure / clutter / symmetry robustness qualification + closure
# ---------------------------------------------------------------------------

I53C_SCHEMA_VERSION = 1
I53C_PHASE = "I5.3C"
I53C_RECEIPT_PHASE = "I5.3C_QUALIFICATION_RECEIPT"


def _i53c_registration_string(scenario: Mapping[str, Any], *, label: str) -> str:
    input_images = scenario.get("input_images")
    registered_images = scenario.get("registered_images")
    if not isinstance(input_images, int) or input_images <= 0:
        raise ReconstructionValidationError(f"I5.3C {label} input_images is invalid.")
    if not isinstance(registered_images, int) or registered_images < 0 or registered_images > input_images:
        raise ReconstructionValidationError(f"I5.3C {label} registered_images is invalid.")
    ratio = scenario.get("registration_ratio")
    expected_ratio = registered_images / input_images
    if not isinstance(ratio, (int, float)) or not isfinite(float(ratio)) or abs(float(ratio) - expected_ratio) > 1e-9:
        raise ReconstructionValidationError(f"I5.3C {label} registration ratio is inconsistent with image counts.")
    full = scenario.get("full_input_scope_registered")
    partial = scenario.get("partial_registration_preserved")
    if full is not (registered_images == input_images):
        raise ReconstructionValidationError(f"I5.3C {label} full-input registration flag is inconsistent.")
    if partial is not (0 < registered_images < input_images):
        raise ReconstructionValidationError(f"I5.3C {label} partial-registration flag is inconsistent.")
    return f"{registered_images}/{input_images}"


def validate_i5_3b_for_i5_3c(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact evidence-complete I5.3B replay being closed by I5.3C."""
    if payload.get("phase") != I53B_PHASE or payload.get("status") != "blur_exposure_clutter_symmetry_replay_evidence_complete" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.3C requires the evidence-complete I5.3B replay record.")
    quality_status = str(payload.get("quality_status", ""))
    if quality_status not in {"pass", "degraded", "rejected"}:
        raise ReconstructionValidationError("I5.3C I5.3B quality status is invalid.")

    scope = _mapping(payload.get("scope"), label="I5.3B scope")
    for key in (
        "current_frozen_foundation_real_replay", "stress_images_materialized",
        "control_replay_attempted", "blur_replay_attempted", "underexposure_replay_attempted",
        "overexposure_replay_attempted", "clutter_replay_attempted", "symmetry_replay_attempted",
    ):
        _require(scope, key, True, label="I5.3B scope")
    for key in (
        "dense_provider_rerun", "provider_specific_robustness_started", "mp_sfm_strategy_evaluation_started",
        "thresholds_modified", "production_default_modified", "auto_policy_work_started", "hybrid_experiment_started",
    ):
        _require(scope, key, False, label="I5.3B scope")

    provenance = _mapping(payload.get("provenance"), label="I5.3B provenance")
    for key in (
        "source_reverified_before_replay", "source_reverified_after_replay", "staged_inputs_reverified_after_replay",
        "same_production_native_sfm_runner_all_scenarios", "stress_recipes_reused_exactly_from_i5_3a",
    ):
        _require(provenance, key, True, label="I5.3B provenance")
    _require(provenance, "same_matching_policy_all_scenarios", "exhaustive", label="I5.3B provenance")
    for key in ("result_driven_recipe_tuning", "retry_until_quality_pass", "historical_result_overwritten"):
        _require(provenance, key, False, label="I5.3B provenance")
    if not _valid_sha256(provenance.get("frozen_source_asset_set_sha256")):
        raise ReconstructionValidationError("I5.3C I5.3B source asset-set SHA256 is malformed.")

    bindings = _mapping(payload.get("evidence_bindings"), label="I5.3B evidence_bindings")
    a_binding = _mapping(bindings.get("i5_3a_admission"), label="I5.3B I5.3A binding")
    if not _valid_sha256(a_binding.get("sha256")):
        raise ReconstructionValidationError("I5.3C I5.3B predecessor I5.3A SHA256 binding is malformed.")

    comparison = _validate_i5_3b_image_domain_comparison(
        _mapping(payload.get("image_domain_revalidation"), label="I5.3B image-domain revalidation")
    )
    semantic = {
        key: bool(_mapping(comparison.get(key), label=f"I5.3B {key} image-domain evidence").get("semantic_stress_revalidated"))
        for key in ("blur", "underexposure", "overexposure", "clutter", "symmetry")
    }
    if provenance.get("failed_semantic_revalidation_preserved") is not (not bool(comparison.get("all_semantic_stresses_revalidated"))):
        raise ReconstructionValidationError("I5.3C semantic-revalidation provenance flag is inconsistent.")

    scenarios = _mapping(payload.get("scenarios"), label="I5.3B scenarios")
    for key in ("control", "blur", "underexposure", "overexposure", "clutter", "symmetry"):
        scenario = _mapping(scenarios.get(key), label=f"I5.3B {key} scenario")
        _i53c_registration_string(scenario, label=key)
        status = str(scenario.get("quality_status", ""))
        if status not in {"accepted", "degraded", "rejected"}:
            raise ReconstructionValidationError(f"I5.3C {key} scenario quality status is invalid.")
        sfm_executed = scenario.get("sfm_executed")
        if sfm_executed is not bool(scope.get(f"{key}_sfm_rerun")):
            raise ReconstructionValidationError(f"I5.3C {key} SfM execution flag drifted from I5.3B scope.")
        authority = _mapping(scenario.get("authority"), label=f"I5.3B {key} authority")
        _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.3B {key} authority")
        for flag in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated"):
            _require(authority, flag, False, label=f"I5.3B {key} authority")
        if sfm_executed:
            _require(authority, "per_scenario_gauge", "arbitrary", label=f"I5.3B {key} authority")
        else:
            if status != "rejected":
                raise ReconstructionValidationError(f"I5.3C {key} pre-SfM evidence must retain rejected quality status.")
            _require(scenario, "registration_attempted", False, label=f"I5.3B {key} pre-SfM evidence")
            _require(scenario, "pre_sfm_ingestion_rejected", True, label=f"I5.3B {key} pre-SfM evidence")
            _require(scenario, "rejection_stage", "ingestion", label=f"I5.3B {key} pre-SfM evidence")
            _require(authority, "per_scenario_gauge", "unavailable_no_sfm_model", label=f"I5.3B {key} authority")

    if scenarios["control"].get("quality_status") == "rejected" and quality_status != "rejected":
        raise ReconstructionValidationError("I5.3C rejected control must remain an overall rejected replay.")
    if quality_status == "pass":
        if not comparison.get("all_semantic_stresses_revalidated"):
            raise ReconstructionValidationError("I5.3C PASS replay requires all semantic stresses to be revalidated.")
        if any(_mapping(scenarios[key], label=f"I5.3B {key}").get("quality_status") != "accepted" for key in scenarios):
            raise ReconstructionValidationError("I5.3C PASS replay requires accepted quality for every scenario.")

    authority = _mapping(payload.get("authority"), label="I5.3B authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.3B authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.3B authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented",
        "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.3B authority")

    decision = _mapping(payload.get("decision"), label="I5.3B decision")
    for key in (
        "i4_closed", "i5_authorized", "i5_1_complete", "i5_2_complete", "i5_3_started",
        "i5_3a_complete", "i5_3b_complete", "i5_3c_qualification_closure_authorized",
    ):
        _require(decision, key, True, label="I5.3B decision")
    for key in (
        "i5_3_complete", "i5_4_authorized", "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created", "mp_sfm_authorized",
        "auto_policy_implemented", "hybrid_authorized", "threshold_relaxation_authorized",
        "production_default_modified", "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated",
    ):
        _require(decision, key, False, label="I5.3B decision")
    if payload.get("next") != "I5.3C blur / exposure / clutter / symmetry robustness qualification and closure.":
        raise ReconstructionValidationError("I5.3C predecessor next-boundary contract drifted.")
    return deepcopy(dict(payload))


def validate_i5_3c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != I53C_RECEIPT_PHASE or payload.get("status") != "pass" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.3C requires a passing qualification regression receipt.")
    scope = _mapping(payload.get("scope"), label="I5.3C receipt scope")
    for key in (
        "focused_i5_3c_tests", "predecessor_i5_3_regression", "real_replay_contract_regression",
        "stress_conclusion_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.3C receipt scope")
    for key in (
        "sfm_rerun", "stress_images_rematerialized", "dense_provider_rerun", "quality_replay_rerun",
        "thresholds_modified", "production_default_modified", "mp_sfm_started", "auto_policy_implemented",
        "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.3C receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.3C receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(value, int) and value >= 0 for value in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.3C receipt test counts must be non-negative integers.")
    if focused < 10 or predecessor < 40 or combined < 50 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.3C receipt coverage is below the required closure floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.3C receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.3C receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def _i53c_stress_conclusion(
    *,
    name: str,
    scenario: Mapping[str, Any],
    image_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    semantic = bool(image_evidence.get("semantic_stress_revalidated"))
    registration = _i53c_registration_string(scenario, label=name)
    status = str(scenario.get("quality_status"))
    input_images = int(scenario["input_images"])
    registered_images = int(scenario["registered_images"])
    if not semantic:
        conclusion = "inconclusive_proxy_not_revalidated"
        qualified = False
    elif scenario.get("pre_sfm_ingestion_rejected") is True:
        conclusion = "outside_current_sparse_sfm_ingestion_envelope"
        qualified = False
    elif status == "rejected":
        conclusion = "outside_current_sparse_sfm_reconstruction_envelope"
        qualified = False
    elif registered_images == input_images and status == "accepted":
        conclusion = "survived_admitted_stress_case"
        qualified = True
    elif 0 < registered_images < input_images:
        conclusion = "degraded_partial_registration"
        qualified = False
    else:
        conclusion = "degraded_quality"
        qualified = False
    return {
        "semantic_stress_revalidated": semantic,
        "scenario_quality_status": status,
        "registration": registration,
        "sfm_executed": bool(scenario.get("sfm_executed")),
        "pre_sfm_ingestion_rejected": bool(scenario.get("pre_sfm_ingestion_rejected", False)),
        "robustness_conclusion": conclusion,
        "robustness_qualified": qualified,
        "universal_robustness_claimed": False,
    }


def build_i5_3c_blur_exposure_clutter_symmetry_closure(
    *,
    i5_3a_result: Mapping[str, Any],
    i5_3a_binding: I51EvidenceBinding,
    i5_3b_result: Mapping[str, Any],
    i5_3b_binding: I51EvidenceBinding,
    qualification_receipt: Mapping[str, Any],
    qualification_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    admission = validate_i5_3a_for_i5_3b(i5_3a_result)
    replay = validate_i5_3b_for_i5_3c(i5_3b_result)
    receipt = validate_i5_3c_receipt(qualification_receipt)

    replay_bindings = _mapping(replay.get("evidence_bindings"), label="I5.3B evidence_bindings")
    replay_a_binding = _mapping(replay_bindings.get("i5_3a_admission"), label="I5.3B I5.3A binding")
    if str(replay_a_binding.get("sha256", "")).lower() != str(i5_3a_binding.sha256).lower():
        raise ReconstructionValidationError("I5.3C I5.3B predecessor SHA does not match the independently verified I5.3A artifact.")
    source = _mapping(admission.get("source_identity"), label="I5.3A source_identity")
    provenance = _mapping(replay.get("provenance"), label="I5.3B provenance")
    if provenance.get("frozen_source_asset_set_sha256") != source.get("asset_set_sha256"):
        raise ReconstructionValidationError("I5.3C I5.3A/I5.3B source asset identity drifted.")
    if replay.get("authority") != admission.get("authority"):
        raise ReconstructionValidationError("I5.3C authority drifted between I5.3A and I5.3B.")

    scenarios = _mapping(replay.get("scenarios"), label="I5.3B scenarios")
    comparison = _mapping(replay.get("image_domain_revalidation"), label="I5.3B image-domain revalidation")
    registrations = {
        key: _i53c_registration_string(_mapping(scenarios.get(key), label=f"I5.3B {key}"), label=key)
        for key in ("control", "blur", "underexposure", "overexposure", "clutter", "symmetry")
    }
    conclusions = {
        key: _i53c_stress_conclusion(
            name=key,
            scenario=_mapping(scenarios.get(key), label=f"I5.3B {key}"),
            image_evidence=_mapping(comparison.get(key), label=f"I5.3B image-domain {key}"),
        )
        for key in ("blur", "underexposure", "overexposure", "clutter", "symmetry")
    }
    quality_status = str(replay.get("quality_status"))
    clutter_inconclusive = conclusions["clutter"]["robustness_conclusion"] == "inconclusive_proxy_not_revalidated"
    if quality_status == "degraded" and clutter_inconclusive:
        operating_envelope = "degraded_on_admitted_castle_p19_visual_stress_cases_with_clutter_inconclusive"
    elif quality_status == "pass":
        operating_envelope = "qualified_on_admitted_castle_p19_visual_stress_cases"
    elif quality_status == "degraded":
        operating_envelope = "degraded_on_admitted_castle_p19_visual_stress_cases"
    else:
        operating_envelope = "rejected_on_admitted_castle_p19_visual_stress_cases"

    semantic_confirmed = [key for key, value in conclusions.items() if value["semantic_stress_revalidated"]]
    semantic_inconclusive = [key for key, value in conclusions.items() if not value["semantic_stress_revalidated"]]
    authority = deepcopy(dict(_mapping(replay.get("authority"), label="I5.3B authority")))
    return {
        "schema_version": I53C_SCHEMA_VERSION,
        "phase": I53C_PHASE,
        "status": "blur_exposure_clutter_symmetry_robustness_qualified",
        "pass": True,
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "i5_3a_admission_reused": True,
            "i5_3b_real_replay_reused": True,
            "sfm_rerun": False,
            "stress_images_rematerialized": False,
            "dense_provider_rerun": False,
            "quality_replay_rerun": False,
            "provider_specific_robustness_started": False,
            "mp_sfm_strategy_evaluation_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
        },
        "evidence_bindings": {
            "i5_3a_admission": i5_3a_binding.as_dict(),
            "i5_3b_replay": i5_3b_binding.as_dict(),
            "i5_3c_qualification_receipt": qualification_receipt_binding.as_dict(),
        },
        "qualification": {
            "quality_status": quality_status,
            "operating_envelope": operating_envelope,
            "source_asset_set_sha256": source["asset_set_sha256"],
            "control_registration": registrations["control"],
            "blur_registration": registrations["blur"],
            "underexposure_registration": registrations["underexposure"],
            "overexposure_registration": registrations["overexposure"],
            "clutter_registration": registrations["clutter"],
            "symmetry_registration": registrations["symmetry"],
            "semantic_stresses_confirmed": semantic_confirmed,
            "semantic_stresses_inconclusive": semantic_inconclusive,
            "clutter_proxy_inconclusive": clutter_inconclusive,
            "retry_until_quality_pass": False,
            "result_driven_recipe_tuning": False,
            "universal_visual_stress_robustness_claimed": False,
            "clutter_robustness_claimed": False,
            "dense_reconstruction_robustness_claimed": False,
            "provider_robustness_claimed": False,
        },
        "stress_conclusions": conclusions,
        "image_domain_evidence": deepcopy(dict(comparison)),
        "regression_summary": {
            "focused_i5_3c_tests": receipt["counts"]["focused_tests"],
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
            "i5_3b_complete": True,
            "i5_3c_complete": True,
            "i5_3_complete": True,
            "i5_3_quality_status": quality_status,
            "i5_4_authorized": True,
            "i5_4_started": False,
            "i5_5_authorized": False,
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
            "i5_4_forward_motion_kitti_authorized_not_started": True,
            "i5_5_dynamic_scene_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.4 forward-motion / KITTI robustness.",
    }


__all__ = [
    "I53A_SCHEMA_VERSION",
    "I53A_PHASE",
    "I53A_RECEIPT_PHASE",
    "I53B_SCHEMA_VERSION",
    "I53B_PHASE",
    "I53B_RECEIPT_PHASE",
    "I53C_SCHEMA_VERSION",
    "I53C_PHASE",
    "I53C_RECEIPT_PHASE",
    "I53AStressDesignPolicy",
    "validate_i5_2c_for_i5_3",
    "build_i5_3a_stress_plan",
    "validate_i5_3a_receipt",
    "build_i5_3a_blur_exposure_clutter_symmetry_admission",
    "validate_i5_3a_for_i5_3b",
    "validate_i5_3b_receipt",
    "apply_i5_3b_transform",
    "analyze_i5_3b_image",
    "analyze_i5_3b_image_set",
    "compare_i5_3b_image_domain",
    "analyze_i5_3b_sfm_report",
    "build_i5_3b_blur_exposure_clutter_symmetry_replay",
    "validate_i5_3b_for_i5_3c",
    "validate_i5_3c_receipt",
    "build_i5_3c_blur_exposure_clutter_symmetry_closure",
]
