"""Production manifold-attachment refinement for qualified dense RGB geometry.

I2.6 promotes the conservative ``d12_sparse`` rule qualified by I2.5.3 into
formal dense filtering.  The rule is intentionally scene-relative and only
uses local-manifold evidence derived from the already-qualified F Extended
cloud:

* distance to the strong multi-view manifold, in fused-voxel units; and
* local KNN spacing, also in fused-voxel units.

Peripheral/far labels, extremum-driver labels, source-view blacklists and
metric distances are forbidden inputs.  Those labels were useful for the I2
quality campaign but must not become a scene-specific production heuristic.

This module does not rerun SfM, dense depth or fusion and does not promote
metric scale, measurement or surface-mesh authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

DENSE_MANIFOLD_REFINEMENT_SCHEMA_VERSION = 1
PROMOTED_MANIFOLD_SCENARIO = "d12_sparse"


@dataclass(frozen=True, slots=True)
class DenseManifoldRefinementPolicy:
    """Locked production rule promoted from the qualified I2.5.3 candidate."""

    scenario_name: str = PROMOTED_MANIFOLD_SCENARIO
    anchor_distance_scale: float = 12.0
    knn_spacing_scale: float = 1.0
    reference_source: str = "I2.5.2 central Core non-driver q90"

    def __post_init__(self) -> None:
        if str(self.scenario_name) != PROMOTED_MANIFOLD_SCENARIO:
            raise ReconstructionValidationError(
                f"I2.6 only promotes the manually qualified {PROMOTED_MANIFOLD_SCENARIO!r} rule."
            )
        anchor = float(self.anchor_distance_scale)
        knn = float(self.knn_spacing_scale)
        if not isfinite(anchor) or not np.isclose(anchor, 12.0, rtol=0.0, atol=1e-12):
            raise ReconstructionValidationError("I2.6 production anchor-distance scale is locked to 12.0.")
        if not isfinite(knn) or not np.isclose(knn, 1.0, rtol=0.0, atol=1e-12):
            raise ReconstructionValidationError("I2.6 production KNN-spacing scale is locked to 1.0.")
        if not str(self.reference_source).strip():
            raise ReconstructionValidationError("I2.6 reference source must be non-empty.")
        object.__setattr__(self, "anchor_distance_scale", anchor)
        object.__setattr__(self, "knn_spacing_scale", knn)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DENSE_MANIFOLD_REFINEMENT_SCHEMA_VERSION,
            "scenario_name": self.scenario_name,
            "anchor_distance_scale": self.anchor_distance_scale,
            "knn_spacing_scale": self.knn_spacing_scale,
            "reference_source": self.reference_source,
            "uses_peripheral_or_far_labels": False,
            "uses_extremum_driver_labels": False,
            "uses_source_view_blacklist": False,
            "uses_metric_distance": False,
        }


def _finite_positive(name: str, value: object) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ReconstructionValidationError(f"I2.6 {name} must be finite and positive.")
    return number


def validate_promotion_evidence(
    i252_summary: Mapping[str, Any],
    i253_summary: Mapping[str, Any],
    *,
    scenario_name: str,
    manual_review_pass: bool,
) -> tuple[DenseManifoldRefinementPolicy, Mapping[str, Any], Mapping[str, Any]]:
    """Fail closed unless I2.5.2 + I2.5.3 evidence authorizes this exact rule."""

    if not bool(manual_review_pass):
        raise ReconstructionValidationError("I2.6 requires explicit confirmation that the I2.5.3 manual review passed.")
    if str(i252_summary.get("phase")) != "I2.5.2" or str(i252_summary.get("status")) != "diagnosis_ready":
        raise ReconstructionValidationError("I2.6 requires completed I2.5.2 local-manifold diagnosis evidence.")
    signal = i252_summary.get("diagnostic_signal")
    if not isinstance(signal, Mapping) or str(signal.get("strength")) != "strong" or not bool(signal.get("supports_local_geometry_manifold_hypothesis")):
        raise ReconstructionValidationError("I2.6 requires strong, supported I2.5.2 manifold evidence.")
    if str(i253_summary.get("phase")) != "I2.5.3" or str(i253_summary.get("status")) != "counterfactuals_ready":
        raise ReconstructionValidationError("I2.6 requires completed I2.5.3 counterfactual evidence.")
    if str(i252_summary.get("scene_id")) != str(i253_summary.get("scene_id")):
        raise ReconstructionValidationError("I2.6 I2.5.2/I2.5.3 scene identity mismatch.")
    authority = i253_summary.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I2.6 I2.5.3 authority evidence is missing.")
    forbidden_authority = (
        "source_geometry_changed",
        "production_f_policy_changed",
        "sfm_rerun",
        "da3_rerun",
        "fusion_rerun",
        "persistence_changed",
        "metric_scale_claimed",
        "measurement_authoritative",
        "surface_mesh_generated",
    )
    if any(bool(authority.get(name)) for name in forbidden_authority):
        raise ReconstructionValidationError("I2.6 refuses I2.5.3 evidence that already changed production authority/state.")

    ranking = i253_summary.get("ranking")
    balanced = [] if not isinstance(ranking, Mapping) else list(ranking.get("balanced_candidates_for_manual_review", []))
    if str(scenario_name) not in balanced:
        raise ReconstructionValidationError("I2.6 only promotes a candidate that passed the I2.5.3 balanced gate.")

    scenarios = i253_summary.get("scenarios")
    if not isinstance(scenarios, list):
        raise ReconstructionValidationError("I2.6 I2.5.3 scenario records are missing.")
    matches = [item for item in scenarios if isinstance(item, Mapping) and str(item.get("name")) == str(scenario_name)]
    if len(matches) != 1:
        raise ReconstructionValidationError("I2.6 requires exactly one matching I2.5.3 scenario record.")
    record = matches[0]
    comparison = record.get("comparison")
    resolved = record.get("resolved_rule")
    if not isinstance(comparison, Mapping) or not bool(comparison.get("balanced_for_manual_review")):
        raise ReconstructionValidationError("I2.6 candidate did not pass the I2.5.3 balanced comparison gate.")
    if not isinstance(resolved, Mapping):
        raise ReconstructionValidationError("I2.6 candidate lacks a resolved I2.5.3 rule.")
    if any(bool(resolved.get(name)) for name in (
        "uses_peripheral_or_far_labels",
        "uses_extremum_driver_labels",
        "uses_source_view_blacklist",
        "uses_metric_distance",
    )):
        raise ReconstructionValidationError("I2.6 refuses scene-specific or metric filtering inputs.")

    policy = DenseManifoldRefinementPolicy(
        scenario_name=str(scenario_name),
        anchor_distance_scale=float(resolved.get("anchor_distance_scale", 0.0)),
        knn_spacing_scale=float(resolved.get("knn_spacing_scale", 0.0)),
    )
    return policy, record, signal


def manifold_attachment_rejection_mask(
    nearest_strong_anchor_distance_voxels: object,
    knn_median_distance_voxels: object,
    *,
    anchor_distance_reference_q90_voxels: float,
    knn_spacing_reference_q90_voxels: float,
    policy: DenseManifoldRefinementPolicy,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the locked production rejection mask using only manifold evidence."""

    if not isinstance(policy, DenseManifoldRefinementPolicy):
        raise TypeError("policy must be DenseManifoldRefinementPolicy.")
    distance = np.asarray(nearest_strong_anchor_distance_voxels, dtype=np.float64).reshape(-1)
    knn = np.asarray(knn_median_distance_voxels, dtype=np.float64).reshape(-1)
    if len(distance) == 0 or knn.shape != distance.shape:
        raise ReconstructionValidationError("I2.6 manifold evidence must be non-empty aligned 1D vectors.")
    if np.any(~np.isfinite(distance)) or np.any(distance < 0.0):
        raise ReconstructionValidationError("I2.6 manifold distances must be finite and non-negative.")
    if np.any(~np.isfinite(knn)) or np.any(knn < 0.0):
        raise ReconstructionValidationError("I2.6 KNN spacings must be finite and non-negative.")
    anchor_ref = _finite_positive("anchor-distance reference", anchor_distance_reference_q90_voxels)
    knn_ref = _finite_positive("KNN-spacing reference", knn_spacing_reference_q90_voxels)
    anchor_limit = anchor_ref * policy.anchor_distance_scale
    knn_limit = knn_ref * policy.knn_spacing_scale
    reject = (distance > anchor_limit) & (knn > knn_limit)
    return reject, {
        **policy.as_dict(),
        "anchor_distance_reference_q90_voxels": anchor_ref,
        "knn_spacing_reference_q90_voxels": knn_ref,
        "resolved_anchor_distance_limit_voxels": float(anchor_limit),
        "resolved_knn_spacing_limit_voxels": float(knn_limit),
        "rejected_points": int(np.count_nonzero(reject)),
        "rejected_fraction_of_source_extended": float(np.mean(reject)),
    }


def validate_exact_counterfactual_equivalence(
    production_reject_mask: object,
    counterfactual_reject_mask: object,
    *,
    expected_rejected_points: int,
) -> dict[str, Any]:
    production = np.asarray(production_reject_mask, dtype=bool).reshape(-1)
    counterfactual = np.asarray(counterfactual_reject_mask, dtype=bool).reshape(-1)
    if production.shape != counterfactual.shape:
        raise ReconstructionValidationError("I2.6 production/counterfactual masks are misaligned.")
    mismatch = int(np.count_nonzero(production != counterfactual))
    rejected = int(np.count_nonzero(production))
    if rejected != int(expected_rejected_points) or mismatch:
        raise ReconstructionValidationError(
            "I2.6 promoted rule does not exactly reproduce the manually reviewed I2.5.3 counterfactual."
        )
    return {
        "exact_mask_match": True,
        "mismatched_voxels": 0,
        "rejected_points": rejected,
    }


def promotion_loss_fractions(
    *,
    source_extended_points: int,
    promoted_extended_points: int,
    source_core_points: int,
    promoted_core_points: int,
) -> dict[str, float]:
    values = tuple(int(v) for v in (
        source_extended_points,
        promoted_extended_points,
        source_core_points,
        promoted_core_points,
    ))
    if any(v <= 0 for v in values):
        raise ReconstructionValidationError("I2.6 geometry counts must be positive.")
    source_ext, promoted_ext, source_core, promoted_core = values
    if promoted_ext > source_ext or promoted_core > source_core or source_core > source_ext or promoted_core > promoted_ext:
        raise ReconstructionValidationError("I2.6 geometry count hierarchy is invalid.")
    return {
        "extended_loss_fraction": 1.0 - promoted_ext / float(source_ext),
        "core_loss_fraction": 1.0 - promoted_core / float(source_core),
    }
