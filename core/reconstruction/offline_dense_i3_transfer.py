"""I3.1 second-scene manifold-filter transfer qualification helpers.

I3.1 validates whether the scene-relative manifold-attachment refinement
qualified on I2 transfers safely to a *different* indoor/static sequence.
The helpers deliberately keep the candidate counterfactual: they do not
rewrite formal F/G products and do not promote metric, measurement, or mesh
authority.

The production predicate remains the locked I2.6 ``d12_sparse`` rule.  The
scene-relative evidence is rebuilt from ordinary F/Core geometry using the
same definitions that produced the I2.5.2 reference:

* peripheral masks are diagnostic Core-anchored masks only;
* extremum drivers are robust directional-envelope diagnostics only;
* strong manifold anchors are Core, 3+ view, locally high-supported, and not
  extremum drivers; and
* the reference population is central Core non-driver geometry.

Peripheral/extremum labels never directly decide whether a point is rejected;
they only protect the scene-relative reference used by the manifold predicate.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np

from core.reconstruction.dense_manifold_refinement import (
    DenseManifoldRefinementPolicy,
    manifold_attachment_rejection_mask,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_detached_component_qualification import compare_component_lineage
from core.reconstruction.offline_dense_extremum_local_geometry import (
    I252LocalGeometryPolicy,
    derive_local_geometry_reference,
    local_shape_metrics,
    strong_manifold_metrics,
)
from core.reconstruction.offline_dense_extremum_survivor_diagnosis import (
    I251ExtremumPolicy,
    extremum_driver_mask,
    robust_directional_excess,
)
from core.reconstruction.offline_dense_quality import (
    I2QualityPolicy,
    classify_peripheral_geometry,
    spatial_envelope,
)

I3_TRANSFER_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class I3TransferSafetyPolicy:
    """Conservative transfer gates used before any second-scene promotion."""

    maximum_extended_loss_fraction: float = 0.01
    maximum_core_loss_fraction: float = 0.005
    maximum_robust_diagonal_shrink_fraction: float = 0.03
    maximum_full_to_robust_regression_fraction: float = 0.01
    minimum_dominant_lineage_fraction: float = 1.0
    require_zero_newly_detached: bool = True
    require_zero_removed_from_baseline_dominant: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_extended_loss_fraction",
            "maximum_core_loss_fraction",
            "maximum_robust_diagonal_shrink_fraction",
            "maximum_full_to_robust_regression_fraction",
            "minimum_dominant_lineage_fraction",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ReconstructionValidationError(f"I3.1 {name} must be finite in [0,1].")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _validate_filtered_arrays(payload: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    required = (
        "points_reconstruction",
        "class_code",
        "view_observation_counts",
        "high_support_neighbor_count",
        "colors_rgb",
        "voxel_size_reconstruction_units",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ReconstructionValidationError(f"I3.1 filtered payload is missing fields: {missing!r}")
    points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
    colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
    classes = np.asarray(payload["class_code"], dtype=np.uint8).reshape(-1)
    support = np.asarray(payload["view_observation_counts"], dtype=np.int64).reshape(-1)
    high = np.asarray(payload["high_support_neighbor_count"], dtype=np.int64).reshape(-1)
    voxel = float(np.asarray(payload["voxel_size_reconstruction_units"]).item())
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ReconstructionValidationError("I3.1 Extended points must be finite non-empty Nx3.")
    if colors.shape != points.shape or classes.shape != (len(points),) or support.shape != (len(points),) or high.shape != (len(points),):
        raise ReconstructionValidationError("I3.1 Extended evidence arrays are not aligned.")
    if not isfinite(voxel) or voxel <= 0.0:
        raise ReconstructionValidationError("I3.1 voxel size must be finite and positive.")
    core = classes == 3
    if np.count_nonzero(core) < 128:
        raise ReconstructionValidationError("I3.1 Core population is unexpectedly small.")
    return points, colors, core, support, high, voxel


def build_scene_relative_manifold_evidence(
    filtered_extended: Mapping[str, object],
    *,
    quality_policy: I2QualityPolicy | None = None,
    extremum_policy: I251ExtremumPolicy | None = None,
    local_policy: I252LocalGeometryPolicy | None = None,
    refinement_policy: DenseManifoldRefinementPolicy | None = None,
) -> dict[str, Any]:
    """Rebuild the transferable d12 manifold evidence from one F Extended cloud."""

    points, colors, core, support, high, voxel = _validate_filtered_arrays(filtered_extended)
    q_policy = quality_policy or I2QualityPolicy()
    e_policy = extremum_policy or I251ExtremumPolicy(
        robust_lower_quantile=float(q_policy.robust_lower_quantile),
        robust_upper_quantile=float(q_policy.robust_upper_quantile),
    )
    l_policy = local_policy or I252LocalGeometryPolicy()
    r_policy = refinement_policy or DenseManifoldRefinementPolicy(
        reference_source="scene-relative central Core non-driver q90; I3 transfer reconstruction",
    )

    core_points = points[core]
    peripheral, far, sector, core_envelope = classify_peripheral_geometry(
        points,
        core_points,
        voxel_size=voxel,
        policy=q_policy,
    )
    excess = robust_directional_excess(
        points,
        lower_quantile=float(e_policy.robust_lower_quantile),
        upper_quantile=float(e_policy.robust_upper_quantile),
    )
    drivers, driver_definition = extremum_driver_mask(excess["directional_excess"], policy=e_policy)

    strong_anchor = (
        core
        & (support >= int(l_policy.strong_anchor_min_support))
        & (high >= int(l_policy.strong_anchor_min_high_neighbors))
        & ~drivers
    )
    reference = core & ~peripheral & ~drivers
    if int(np.count_nonzero(strong_anchor)) < 8:
        raise ReconstructionValidationError("I3.1 strong manifold anchor population is unexpectedly small.")
    if int(np.count_nonzero(reference)) < int(l_policy.minimum_reference_points):
        raise ReconstructionValidationError("I3.1 central Core non-driver reference is too small.")

    shape = local_shape_metrics(points, voxel_size=voxel, knn_size=int(l_policy.knn_size))
    manifold = strong_manifold_metrics(
        points,
        strong_anchor_mask=strong_anchor,
        voxel_size=voxel,
        radius_voxels=float(l_policy.manifold_radius_voxels),
    )
    metrics = {
        "nearest_strong_anchor_distance_voxels": np.asarray(manifold["nearest_strong_anchor_distance_voxels"], dtype=np.float64),
        "strong_anchor_neighbors_within_radius": np.asarray(manifold["strong_anchor_neighbors_within_radius"], dtype=np.int32),
        "knn_median_distance_voxels": np.asarray(shape["knn_median_distance_voxels"], dtype=np.float64),
        "knn_p90_distance_voxels": np.asarray(shape["knn_p90_distance_voxels"], dtype=np.float64),
        "local_linearity": np.asarray(shape["local_linearity"], dtype=np.float64),
        "local_planarity": np.asarray(shape["local_planarity"], dtype=np.float64),
        "local_scattering": np.asarray(shape["local_scattering"], dtype=np.float64),
        # derive_local_geometry_reference requires this field. It is irrelevant
        # to the d12 predicate, so a finite neutral vector is sufficient and
        # avoids reconstructing source-view rays in the transfer stage.
        "supporting_ray_alignment": np.zeros(len(points), dtype=np.float64),
    }
    thresholds = derive_local_geometry_reference(metrics, reference_mask=reference, policy=l_policy)
    reject, resolved = manifold_attachment_rejection_mask(
        metrics["nearest_strong_anchor_distance_voxels"],
        metrics["knn_median_distance_voxels"],
        anchor_distance_reference_q90_voxels=float(thresholds["anchor_distance_high"]),
        knn_spacing_reference_q90_voxels=float(thresholds["knn_spacing_high"]),
        policy=r_policy,
    )

    return {
        "points": points,
        "colors": colors,
        "core": core,
        "support": support,
        "high_support_neighbors": high,
        "voxel_size": voxel,
        "peripheral": peripheral,
        "far": far,
        "sector": sector,
        "extremum_drivers": drivers,
        "strong_anchor": strong_anchor,
        "reference": reference,
        "nearest_strong_anchor_distance_voxels": metrics["nearest_strong_anchor_distance_voxels"],
        "knn_median_distance_voxels": metrics["knn_median_distance_voxels"],
        "reject": reject,
        "quality_policy": q_policy.as_dict(),
        "extremum_policy": e_policy.as_dict(),
        "local_policy": l_policy.as_dict(),
        "refinement_policy": r_policy.as_dict(),
        "driver_definition": dict(driver_definition),
        "reference_thresholds": {
            "reference_count": int(round(float(thresholds["reference_count"]))),
            "anchor_distance_q90_voxels": float(thresholds["anchor_distance_high"]),
            "knn_spacing_q90_voxels": float(thresholds["knn_spacing_high"]),
        },
        "resolved_rule": resolved,
        "core_anchor_envelope": core_envelope.as_dict(),
    }


def exact_reference_transfer_preflight(
    rebuilt_evidence: Mapping[str, object],
    *,
    expected_reject_mask: object,
    expected_anchor_reference_q90: float,
    expected_knn_reference_q90: float,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Prove that the generic evidence builder reproduces the closed I2 rule."""

    rebuilt = np.asarray(rebuilt_evidence["reject"], dtype=bool).reshape(-1)
    expected = np.asarray(expected_reject_mask, dtype=bool).reshape(-1)
    if rebuilt.shape != expected.shape:
        raise ReconstructionValidationError("I3.1 I2 reference preflight masks are misaligned.")
    mismatch = int(np.count_nonzero(rebuilt != expected))
    thresholds = rebuilt_evidence["reference_thresholds"]
    if not isinstance(thresholds, Mapping):
        raise ReconstructionValidationError("I3.1 rebuilt reference thresholds are missing.")
    anchor = float(thresholds["anchor_distance_q90_voxels"])
    knn = float(thresholds["knn_spacing_q90_voxels"])
    anchor_match = bool(np.isclose(anchor, float(expected_anchor_reference_q90), rtol=0.0, atol=float(atol)))
    knn_match = bool(np.isclose(knn, float(expected_knn_reference_q90), rtol=0.0, atol=float(atol)))
    return {
        "exact_mask_match": mismatch == 0,
        "mismatched_voxels": mismatch,
        "rebuilt_rejected_points": int(np.count_nonzero(rebuilt)),
        "expected_rejected_points": int(np.count_nonzero(expected)),
        "anchor_reference_match": anchor_match,
        "knn_reference_match": knn_match,
        "rebuilt_anchor_reference_q90_voxels": anchor,
        "expected_anchor_reference_q90_voxels": float(expected_anchor_reference_q90),
        "rebuilt_knn_reference_q90_voxels": knn,
        "expected_knn_reference_q90_voxels": float(expected_knn_reference_q90),
        "pass": bool(mismatch == 0 and anchor_match and knn_match),
    }


def transfer_candidate_assessment(
    evidence: Mapping[str, object],
    *,
    policy: I3TransferSafetyPolicy | None = None,
) -> dict[str, Any]:
    """Assess the second-scene d12 candidate without changing formal F/G."""

    safety = policy or I3TransferSafetyPolicy()
    points = np.asarray(evidence["points"], dtype=np.float64)
    core = np.asarray(evidence["core"], dtype=bool)
    reject = np.asarray(evidence["reject"], dtype=bool)
    voxel = float(evidence["voxel_size"])
    if reject.shape != (len(points),) or core.shape != (len(points),):
        raise ReconstructionValidationError("I3.1 candidate masks are not aligned.")
    keep = ~reject
    if not np.any(keep) or not np.any(core & keep):
        raise ReconstructionValidationError("I3.1 candidate would remove all retained/Core geometry.")

    candidate_points = points[keep]
    baseline_core_points = points[core]
    candidate_core_points = points[core & keep]
    lineage = compare_component_lineage(
        baseline_points=points,
        promoted_points=candidate_points,
        voxel_size=voxel,
    )
    q_policy = I2QualityPolicy(**{key: float(value) for key, value in dict(evidence["quality_policy"]).items()})
    base_env = spatial_envelope(
        points,
        lower_quantile=float(q_policy.robust_lower_quantile),
        upper_quantile=float(q_policy.robust_upper_quantile),
    )
    candidate_env = spatial_envelope(
        candidate_points,
        lower_quantile=float(q_policy.robust_lower_quantile),
        upper_quantile=float(q_policy.robust_upper_quantile),
    )
    base_ratio = float(base_env.full_diagonal / base_env.robust_diagonal)
    candidate_ratio = float(candidate_env.full_diagonal / candidate_env.robust_diagonal)
    full_shrink = 1.0 - candidate_env.full_diagonal / base_env.full_diagonal
    robust_shrink = 1.0 - candidate_env.robust_diagonal / base_env.robust_diagonal
    ratio_improvement = (base_ratio - candidate_ratio) / base_ratio

    ext_loss = float(np.mean(reject))
    core_loss = 1.0 - len(candidate_core_points) / float(len(baseline_core_points))
    gates = {
        "extended_loss_within_gate": ext_loss <= float(safety.maximum_extended_loss_fraction),
        "core_loss_within_gate": core_loss <= float(safety.maximum_core_loss_fraction),
        "robust_diagonal_preserved": robust_shrink <= float(safety.maximum_robust_diagonal_shrink_fraction),
        "full_to_robust_not_materially_worse": ratio_improvement >= -float(safety.maximum_full_to_robust_regression_fraction),
        "dominant_lineage_preserved": float(lineage["promoted_dominant_from_single_baseline_component_fraction"]) >= float(safety.minimum_dominant_lineage_fraction),
        "no_newly_detached_voxels": (int(lineage["newly_detached_voxels"]) == 0) if safety.require_zero_newly_detached else True,
        "no_baseline_dominant_removal": (int(lineage["removed_from_baseline_dominant"]) == 0) if safety.require_zero_removed_from_baseline_dominant else True,
        "largest_component_fraction_not_degraded": float(lineage["promoted"]["largest_component_fraction"]) + 1e-12 >= float(lineage["baseline"]["largest_component_fraction"]),
        "detached_voxels_not_increased": int(lineage["promoted"]["detached_voxels"]) <= int(lineage["baseline"]["detached_voxels"]),
    }
    return {
        "rejected_points": int(np.count_nonzero(reject)),
        "extended_points": {"baseline": int(len(points)), "candidate": int(len(candidate_points)), "loss_fraction": ext_loss},
        "core_points": {"baseline": int(len(baseline_core_points)), "candidate": int(len(candidate_core_points)), "loss_fraction": core_loss},
        "envelope": {
            "baseline": base_env.as_dict(),
            "candidate": candidate_env.as_dict(),
            "full_diagonal_shrink_fraction": float(full_shrink),
            "robust_diagonal_shrink_fraction": float(robust_shrink),
            "full_to_robust_improvement_fraction": float(ratio_improvement),
        },
        "component_lineage": {key: value for key, value in lineage.items() if key != "masks"},
        "gates": gates,
        "safe_for_manual_transfer_review": bool(all(gates.values())),
        "safety_policy": safety.as_dict(),
    }
