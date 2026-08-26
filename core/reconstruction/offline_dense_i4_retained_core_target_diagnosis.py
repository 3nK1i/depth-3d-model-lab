"""I4.1.8A retained-Core object-target diagnosis.

This module is diagnosis-only.  It consumes a completed I4.1.8 bound scene,
its retained Core geometry, and already-saved camera poses to determine whether
the current whole-Core coordinate-wise median is credible as an *object-orbit*
target.  It does not change production binding, thresholds, reconstruction
artifacts, metric scale, measurement authority, or mesh state.

The diagnosis intentionally separates three independent evidence families:

* camera-only trajectory/orbit evidence;
* retained-Core scene-structure evidence, including a dominant-plane probe;
* retained-Core + camera-attention evidence, which produces a non-authoritative
  object-target candidate from actual Core points rather than selecting a
  target merely because it makes an orbit gate pass.

No result from this module is production authority.  A later bounded correction
must still define and validate any promoted target estimator explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import (
    coverage_by_plane_mode,
    optical_axis_consensus_target,
    trajectory_centered_coverage,
    trajectory_circle_target,
)
from core.reconstruction.offline_dense_i4_orbit_qualification import (
    I4OrbitQualificationPolicy,
    orbit_coverage_metrics,
    robust_object_target,
)

I4_1_8A_SCHEMA_VERSION = 1
_EPS = 1e-10
_TARGET_DEPENDENT_FINAL_GATES = frozenset(
    {
        "selected_azimuth_coverage",
        "selected_azimuth_bins",
        "selected_opposed_view_diversity",
        "selected_target_facing",
        "selected_standoff_consistency",
    }
)


@dataclass(frozen=True, slots=True)
class I418ATargetDiagnosisPolicy:
    """Diagnostic heuristics only; these are *not* I4.1 qualification gates."""

    plane_ransac_trials: int = 256
    plane_sample_cap: int = 12_000
    plane_inlier_voxel_multiplier: float = 2.5
    minimum_dominant_plane_fraction: float = 0.35
    attention_power: float = 12.0
    attention_top_fraction: float = 0.15
    minimum_attention_points: int = 128
    minimum_nonplane_attention_points: int = 96
    current_plane_proximity_multiplier: float = 2.5
    alternative_plane_separation_multiplier: float = 2.0
    minimum_target_separation_standoff_fraction: float = 0.04
    median_invariance_probe_warning_voxels: float = 1.0

    def __post_init__(self) -> None:
        if int(self.plane_ransac_trials) < 32:
            raise ReconstructionValidationError("I4.1.8A plane_ransac_trials must be >=32.")
        if int(self.plane_sample_cap) < 512:
            raise ReconstructionValidationError("I4.1.8A plane_sample_cap must be >=512.")
        if float(self.plane_inlier_voxel_multiplier) <= 0.0:
            raise ReconstructionValidationError("I4.1.8A plane inlier multiplier must be positive.")
        if not 0.0 < float(self.minimum_dominant_plane_fraction) < 1.0:
            raise ReconstructionValidationError("I4.1.8A dominant-plane fraction must lie in (0,1).")
        if float(self.attention_power) <= 1.0:
            raise ReconstructionValidationError("I4.1.8A attention power must be >1.")
        if not 0.0 < float(self.attention_top_fraction) < 0.5:
            raise ReconstructionValidationError("I4.1.8A attention top fraction must lie in (0,0.5).")
        if int(self.minimum_attention_points) < 32 or int(self.minimum_nonplane_attention_points) < 32:
            raise ReconstructionValidationError("I4.1.8A attention point minima must be >=32.")
        if float(self.minimum_target_separation_standoff_fraction) <= 0.0:
            raise ReconstructionValidationError("I4.1.8A target-separation fraction must be positive.")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _points(value: object, *, name: str, minimum: int = 64) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum or not np.isfinite(points).all():
        raise ReconstructionValidationError(
            f"I4.1.8A {name} must be finite Nx3 with at least {minimum} points."
        )
    return points


def _poses(value: object, *, name: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 3 or not np.isfinite(poses).all():
        raise ReconstructionValidationError(f"I4.1.8A {name} must be finite Nx4x4 with N>=3.")
    if not np.allclose(poses[:, 3, :], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ReconstructionValidationError(f"I4.1.8A {name} poses have invalid homogeneous rows.")
    return poses


def _target(value: object, *, name: str) -> np.ndarray:
    target = np.asarray(value, dtype=np.float64).reshape(-1)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ReconstructionValidationError(f"I4.1.8A {name} must be a finite 3-vector.")
    return target


def _quantiles(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        raise ReconstructionValidationError("I4.1.8A quantile population is empty.")
    return {
        "p10": float(np.quantile(data, 0.10)),
        "p25": float(np.quantile(data, 0.25)),
        "p50": float(np.quantile(data, 0.50)),
        "p75": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.90)),
    }


def _geometric_median(points: np.ndarray, *, maximum_iterations: int = 80) -> np.ndarray:
    """Rigid-transform-equivariant geometric median for a bounded subset."""

    values = _points(points, name="geometric-median points", minimum=3)
    x = np.mean(values, axis=0)
    scale = max(float(np.median(np.linalg.norm(values - x, axis=1))), _EPS)
    tolerance = max(scale * 1e-9, 1e-12)
    for _ in range(int(maximum_iterations)):
        distances = np.linalg.norm(values - x.reshape(1, 3), axis=1)
        near = distances <= tolerance
        if np.any(near):
            return np.asarray(values[np.flatnonzero(near)[0]], dtype=np.float64)
        weights = 1.0 / np.maximum(distances, tolerance)
        nxt = np.sum(values * weights[:, None], axis=0) / float(np.sum(weights))
        if float(np.linalg.norm(nxt - x)) <= tolerance:
            x = nxt
            break
        x = nxt
    if not np.isfinite(x).all():
        raise ReconstructionValidationError("I4.1.8A geometric median became non-finite.")
    return np.asarray(x, dtype=np.float64)


def _deterministic_sample_indices(count: int, cap: int) -> np.ndarray:
    if count <= cap:
        return np.arange(count, dtype=np.int64)
    return np.rint(np.linspace(0, count - 1, int(cap), dtype=np.float64)).astype(np.int64)


def _dominant_plane(
    points: np.ndarray,
    *,
    voxel_size: float,
    policy: I418ATargetDiagnosisPolicy,
) -> dict[str, Any]:
    """Deterministic RANSAC/PCA dominant-plane probe in arbitrary SfM gauge."""

    if not np.isfinite(float(voxel_size)) or float(voxel_size) <= 0.0:
        raise ReconstructionValidationError("I4.1.8A voxel_size must be finite and positive.")
    sample = points[_deterministic_sample_indices(len(points), int(policy.plane_sample_cap))]
    threshold = float(voxel_size) * float(policy.plane_inlier_voxel_multiplier)
    rng = np.random.default_rng(41801)
    best_count = -1
    best_residual = float("inf")
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    for _ in range(int(policy.plane_ransac_trials)):
        tri = sample[rng.choice(len(sample), size=3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = float(np.linalg.norm(normal))
        if norm <= _EPS:
            continue
        normal = normal / norm
        offset = -float(np.dot(normal, tri[0]))
        distances = np.abs(sample @ normal + offset)
        mask = distances <= threshold
        count = int(np.count_nonzero(mask))
        if count < 3:
            continue
        residual = float(np.median(distances[mask]))
        if count > best_count or (count == best_count and residual < best_residual):
            best_count = count
            best_residual = residual
            best_normal = np.asarray(normal, dtype=np.float64)
            best_offset = offset
    if best_normal is None:
        raise ReconstructionValidationError("I4.1.8A could not estimate a non-degenerate dominant plane.")

    initial_distances = np.abs(sample @ best_normal + best_offset)
    inliers = sample[initial_distances <= threshold]
    if len(inliers) < 3:
        raise ReconstructionValidationError("I4.1.8A dominant-plane refinement has too few inliers.")
    centroid = np.mean(inliers, axis=0)
    centered = inliers - centroid
    covariance = centered.T @ centered / float(len(centered))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    normal = np.asarray(eigenvectors[:, order[0]], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), _EPS)
    offset = -float(np.dot(normal, centroid))

    distances_all = np.abs(points @ normal + offset)
    mask_all = distances_all <= threshold
    plane_fraction = float(np.mean(mask_all))
    inlier_distances = distances_all[mask_all]
    denominator = max(float(eigenvalues[-1]), _EPS)
    return {
        "normal_reconstruction": [float(v) for v in normal],
        "offset_reconstruction": float(offset),
        "inlier_threshold_reconstruction_units": float(threshold),
        "inlier_threshold_voxels": float(threshold / float(voxel_size)),
        "core_inlier_count": int(np.count_nonzero(mask_all)),
        "core_inlier_fraction": plane_fraction,
        "strong_dominant_plane": plane_fraction >= float(policy.minimum_dominant_plane_fraction),
        "inlier_residual_reconstruction_units": _quantiles(inlier_distances if len(inlier_distances) else distances_all),
        "refined_plane_eigenvalues": [float(v) for v in eigenvalues],
        "normal_variance_to_major_variance": float(eigenvalues[0] / denominator),
        "_normal": normal,
        "_offset": float(offset),
        "_distances": distances_all,
        "_inlier_mask": mask_all,
    }


def _attention_core_candidate(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    plane: Mapping[str, Any],
    support_counts: np.ndarray | None,
    filter_confidence: np.ndarray | None,
    policy: I418ATargetDiagnosisPolicy,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Build a diagnosis-only target candidate from camera-attended Core points."""

    plane_mask = np.asarray(plane["_inlier_mask"], dtype=bool)
    nonplane = ~plane_mask if bool(plane["strong_dominant_plane"]) else np.ones(len(points), dtype=bool)
    candidate_indices = np.flatnonzero(nonplane)
    if len(candidate_indices) < int(policy.minimum_nonplane_attention_points):
        return None, {
            "available": False,
            "reason": "insufficient_nonplane_core_support",
            "candidate_pool_count": int(len(candidate_indices)),
        }
    values = points[candidate_indices]
    score = np.zeros(len(values), dtype=np.float64)
    centers = poses[:, :3, 3]
    forward = np.asarray(poses[:, :3, 2], dtype=np.float64)
    forward /= np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), _EPS)
    for center, direction in zip(centers, forward):
        vectors = values - center.reshape(1, 3)
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > _EPS
        cosine = np.zeros(len(values), dtype=np.float64)
        cosine[valid] = (vectors[valid] @ direction) / lengths[valid]
        score += np.power(np.clip(cosine, 0.0, 1.0), float(policy.attention_power))
    score /= float(len(poses))

    if support_counts is not None:
        support = np.asarray(support_counts, dtype=np.float64).reshape(-1)
        if len(support) != len(points) or not np.isfinite(support).all():
            raise ReconstructionValidationError("I4.1.8A Core support-count evidence is misaligned or non-finite.")
        local_support = support[candidate_indices]
        scale = max(float(np.quantile(local_support, 0.90)), 1.0)
        score *= 0.75 + 0.25 * np.clip(local_support / scale, 0.0, 1.0)
    if filter_confidence is not None:
        confidence = np.asarray(filter_confidence, dtype=np.float64).reshape(-1)
        if len(confidence) != len(points) or not np.isfinite(confidence).all():
            raise ReconstructionValidationError("I4.1.8A Core confidence evidence is misaligned or non-finite.")
        local_confidence = confidence[candidate_indices]
        if np.ptp(local_confidence) > _EPS:
            normalized = (local_confidence - float(np.min(local_confidence))) / float(np.ptp(local_confidence))
            score *= 0.80 + 0.20 * normalized

    keep_count = max(
        int(policy.minimum_attention_points),
        int(np.ceil(float(policy.attention_top_fraction) * len(values))),
    )
    keep_count = min(keep_count, len(values))
    if keep_count < int(policy.minimum_attention_points):
        return None, {
            "available": False,
            "reason": "insufficient_attention_points",
            "candidate_pool_count": int(len(values)),
        }
    order = np.argsort(score, kind="stable")
    kept_local = order[-keep_count:]
    kept_global = candidate_indices[kept_local]
    kept_points = points[kept_global]
    target = _geometric_median(kept_points)
    return target, {
        "available": True,
        "source": "retained Core points after dominant-plane exclusion ranked by bound-camera optical attention",
        "candidate_pool_count": int(len(values)),
        "selected_attention_core_count": int(len(kept_points)),
        "selected_attention_core_fraction_of_all_core": float(len(kept_points) / float(len(points))),
        "dominant_plane_exclusion_applied": bool(plane["strong_dominant_plane"]),
        "attention_score": _quantiles(score),
        "production_target_authority": False,
    }


def _target_structure(
    points: np.ndarray,
    target: np.ndarray,
    poses: np.ndarray,
    *,
    plane: Mapping[str, Any],
    voxel_size: float,
) -> dict[str, Any]:
    centers = poses[:, :3, 3]
    standoff = np.linalg.norm(centers - target.reshape(1, 3), axis=1)
    median_standoff = max(float(np.median(standoff)), _EPS)
    local_radius = max(10.0 * float(voxel_size), 0.08 * median_standoff)
    radial = np.linalg.norm(points - target.reshape(1, 3), axis=1)
    local = radial <= local_radius
    local_count = int(np.count_nonzero(local))
    plane_distances = np.asarray(plane["_distances"], dtype=np.float64)
    threshold = float(plane["inlier_threshold_reconstruction_units"])
    if local_count >= 3:
        local_points = points[local]
        covariance = np.cov(local_points.T, bias=True)
        eigenvalues = np.maximum(np.sort(np.linalg.eigvalsh(covariance)), 0.0)
        eig_sum = max(float(np.sum(eigenvalues)), _EPS)
        volumetric_fraction = float(eigenvalues[0] / eig_sum)
        local_plane_fraction = float(np.mean(plane_distances[local] <= threshold))
    else:
        eigenvalues = np.zeros(3, dtype=np.float64)
        volumetric_fraction = 0.0
        local_plane_fraction = 0.0
    normal = np.asarray(plane["_normal"], dtype=np.float64)
    offset = float(plane["_offset"])
    plane_distance = abs(float(np.dot(target, normal) + offset))
    return {
        "target_reconstruction": [float(v) for v in target],
        "nearest_core_distance_reconstruction_units": float(np.min(radial)),
        "nearest_core_distance_voxels": float(np.min(radial) / float(voxel_size)),
        "dominant_plane_distance_reconstruction_units": float(plane_distance),
        "dominant_plane_distance_voxels": float(plane_distance / float(voxel_size)),
        "local_radius_reconstruction_units": float(local_radius),
        "local_core_count": local_count,
        "local_core_fraction": float(local_count / float(len(points))),
        "local_dominant_plane_fraction": local_plane_fraction,
        "local_covariance_eigenvalues": [float(v) for v in eigenvalues],
        "local_volumetric_fraction": volumetric_fraction,
        "camera_standoff_reconstruction_units": _quantiles(standoff),
    }


def _robust_geometry_pass(
    poses: np.ndarray,
    target: np.ndarray,
    *,
    policy: I4OrbitQualificationPolicy,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    current = orbit_coverage_metrics(
        poses,
        target,
        azimuth_bin_count=int(policy.azimuth_bin_count),
        facing_angle_degrees=float(policy.maximum_facing_angle_degrees),
    )
    modes = coverage_by_plane_mode(
        poses,
        target,
        azimuth_bin_count=int(policy.azimuth_bin_count),
        facing_angle_degrees=float(policy.maximum_facing_angle_degrees),
    )
    robust_geometry = all(
        float(item["azimuth_coverage_degrees"]) >= float(policy.minimum_selected_azimuth_coverage_degrees)
        and int(item["occupied_azimuth_bins"]) >= int(policy.minimum_selected_occupied_azimuth_bins)
        and float(item["maximum_opposed_view_angle_degrees"]) >= float(policy.minimum_opposed_view_angle_degrees)
        for item in modes.values()
    )
    current_all = bool(
        float(current["azimuth_coverage_degrees"]) >= float(policy.minimum_selected_azimuth_coverage_degrees)
        and int(current["occupied_azimuth_bins"]) >= int(policy.minimum_selected_occupied_azimuth_bins)
        and float(current["maximum_opposed_view_angle_degrees"]) >= float(policy.minimum_opposed_view_angle_degrees)
        and float(current["target_facing_fraction"]) >= float(policy.minimum_target_facing_fraction)
        and float(current["standoff_p90_to_p10_ratio"]) <= float(policy.maximum_standoff_p90_to_p10_ratio)
    )
    return current, modes, bool(robust_geometry and current_all)


def _coordinate_median_invariance_probe(points: np.ndarray, *, voxel_size: float) -> dict[str, Any]:
    """Measure coordinate-wise median drift under rigid rotations, mapped back."""

    base = robust_object_target(points)
    axes = (
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 1.0]) / np.sqrt(2.0),
        np.asarray([1.0, -2.0, 0.5]) / np.linalg.norm(np.asarray([1.0, -2.0, 0.5])),
    )
    angles = (31.0, 47.0, 73.0)
    drifts: list[float] = []
    for axis, degrees in zip(axes, angles):
        angle = np.deg2rad(degrees)
        k = np.asarray(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        rotation = np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)
        moved = points @ rotation.T
        moved_median = robust_object_target(moved)
        mapped_back = moved_median @ rotation
        drifts.append(float(np.linalg.norm(mapped_back - base)))
    return {
        "probe_rotations_degrees": list(angles),
        "drift_reconstruction_units": _quantiles(np.asarray(drifts, dtype=np.float64)),
        "maximum_drift_reconstruction_units": float(max(drifts)),
        "maximum_drift_voxels": float(max(drifts) / float(voxel_size)),
        "estimator": "coordinate_wise_median",
        "expected_for_rigid_equivariance": "zero drift apart from numerical tolerance",
    }


def build_i4_1_8a_retained_core_target_diagnosis(
    *,
    source_i4_1_8_summary: Mapping[str, Any],
    source_i4_1_7b_summary: Mapping[str, Any],
    production_selected_camera_matrices: object,
    retained_core_points: object,
    retained_core_voxel_size: float,
    current_retained_core_target: object,
    core_support_counts: object | None = None,
    core_filter_confidence: object | None = None,
    qualification_policy: I4OrbitQualificationPolicy | None = None,
    diagnosis_policy: I418ATargetDiagnosisPolicy | None = None,
) -> dict[str, Any]:
    """Diagnose whether the current whole-Core median is object-orbit credible."""

    source = source_i4_1_8_summary
    b_summary = source_i4_1_7b_summary
    if not isinstance(source, Mapping) or str(source.get("phase")) != "I4.1.8":
        raise ReconstructionValidationError("I4.1.8A requires an I4.1.8 summary.")
    if str(source.get("status")) != "degraded":
        raise ReconstructionValidationError("I4.1.8A is intended for a completed DEGRADED I4.1.8 run.")
    if not isinstance(b_summary, Mapping) or str(b_summary.get("phase")) != "I4.1.7B":
        raise ReconstructionValidationError("I4.1.8A requires I4.1.7B evidence.")
    if str(b_summary.get("status")) != "qualified_for_i4_1_8":
        raise ReconstructionValidationError("I4.1.8A requires a qualified I4.1.7B source.")

    final_assessment = source.get("final_assessment")
    stage_gates = source.get("stage_gates")
    authority = source.get("authority")
    if not isinstance(final_assessment, Mapping) or not isinstance(stage_gates, Mapping) or not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.1.8A source summary lacks final/stage/authority evidence.")
    failed = tuple(sorted(str(v) for v in final_assessment.get("failed_gates", [])))
    unrelated_failures = tuple(name for name in failed if name not in _TARGET_DEPENDENT_FINAL_GATES)
    stages_pass = all(bool(stage_gates.get(name)) for name in (
        "production_fusion",
        "formal_filtering",
        "canonical_persistence",
        "ordinary_i2_bound_scene_quality",
    ))
    forbidden_authority = any(bool(authority.get(name)) for name in (
        "metric_scale_claimed",
        "measurement_authoritative",
        "surface_mesh_generated",
        "dense_consistency_threshold_relaxation_authorized",
    ))
    if forbidden_authority:
        raise ReconstructionValidationError("I4.1.8A refuses promoted metric/measurement/mesh/threshold authority.")

    poses = _poses(production_selected_camera_matrices, name="production-selected poses")
    points = _points(retained_core_points, name="retained Core points")
    current = _target(current_retained_core_target, name="current retained-Core target")
    voxel_size = float(retained_core_voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ReconstructionValidationError("I4.1.8A retained Core voxel size must be finite and positive.")
    qp = qualification_policy or I4OrbitQualificationPolicy()
    dp = diagnosis_policy or I418ATargetDiagnosisPolicy()

    recomputed_current = robust_object_target(points)
    current_reproduction_error = float(np.linalg.norm(recomputed_current - current))
    current_reproduction_exact = current_reproduction_error <= max(voxel_size * 1e-7, 1e-10)

    b_target_record = b_summary.get("target_correction")
    if not isinstance(b_target_record, Mapping):
        raise ReconstructionValidationError("I4.1.8A I4.1.7B target-correction evidence is missing.")
    b_target = _target(b_target_record.get("target_reconstruction"), name="I4.1.7B trajectory target")
    production_circle, production_circle_diag = trajectory_circle_target(poses)
    optical_focus, optical_diag = optical_axis_consensus_target(poses)
    trajectory_centered = trajectory_centered_coverage(poses, azimuth_bin_count=int(qp.azimuth_bin_count))

    plane_raw = _dominant_plane(points, voxel_size=voxel_size, policy=dp)
    attention_target, attention_diag = _attention_core_candidate(
        points,
        poses,
        plane=plane_raw,
        support_counts=None if core_support_counts is None else np.asarray(core_support_counts),
        filter_confidence=None if core_filter_confidence is None else np.asarray(core_filter_confidence),
        policy=dp,
    )

    targets: dict[str, np.ndarray] = {
        "current_whole_core_coordinate_median": current,
        "i4_1_7b_registered_trajectory_circle": b_target,
        "production_bound_trajectory_circle": production_circle,
        "production_optical_axis_focus": optical_focus,
    }
    if attention_target is not None:
        targets["retained_core_camera_attention_candidate"] = attention_target

    evaluations: dict[str, Any] = {}
    structures: dict[str, Any] = {}
    for name, target_value in targets.items():
        metric, modes, robust = _robust_geometry_pass(poses, target_value, policy=qp)
        evaluations[name] = {
            "current_metric": metric,
            "plane_modes": modes,
            "robust_locked_orbit_pass": bool(robust),
        }
        structures[name] = _target_structure(
            points,
            target_value,
            poses,
            plane=plane_raw,
            voxel_size=voxel_size,
        )

    current_metric = evaluations["current_whole_core_coordinate_median"]["current_metric"]
    source_selected = final_assessment.get("selected_orbit")
    source_metric_reproduction = False
    if isinstance(source_selected, Mapping):
        source_metric_reproduction = bool(
            np.isclose(float(source_selected.get("azimuth_coverage_degrees", np.nan)), float(current_metric["azimuth_coverage_degrees"]), atol=1e-8, rtol=0.0)
            and int(source_selected.get("occupied_azimuth_bins", -1)) == int(current_metric["occupied_azimuth_bins"])
            and np.isclose(float(source_selected.get("maximum_opposed_view_angle_degrees", np.nan)), float(current_metric["maximum_opposed_view_angle_degrees"]), atol=1e-8, rtol=0.0)
        )

    trajectory_broad = bool(
        float(trajectory_centered["azimuth_coverage_degrees"]) >= float(qp.minimum_selected_azimuth_coverage_degrees)
        and int(trajectory_centered["occupied_azimuth_bins"]) >= int(qp.minimum_selected_occupied_azimuth_bins)
    )
    circle_robust = bool(evaluations["production_bound_trajectory_circle"]["robust_locked_orbit_pass"])
    attention_robust = bool(
        attention_target is not None
        and evaluations["retained_core_camera_attention_candidate"]["robust_locked_orbit_pass"]
    )

    current_structure = structures["current_whole_core_coordinate_median"]
    circle_structure = structures["production_bound_trajectory_circle"]
    plane_threshold = float(plane_raw["inlier_threshold_reconstruction_units"])
    current_near_plane = bool(
        float(current_structure["dominant_plane_distance_reconstruction_units"])
        <= float(dp.current_plane_proximity_multiplier) * plane_threshold
    )
    circle_farther_from_plane = bool(
        float(circle_structure["dominant_plane_distance_reconstruction_units"])
        >= float(current_structure["dominant_plane_distance_reconstruction_units"])
        + float(dp.alternative_plane_separation_multiplier) * plane_threshold
    )
    median_standoff = float(current_structure["camera_standoff_reconstruction_units"]["p50"])
    if attention_target is not None:
        current_attention_separation = float(np.linalg.norm(current - attention_target))
    else:
        current_attention_separation = 0.0
    attention_materially_separated = bool(
        attention_target is not None
        and current_attention_separation
        >= float(dp.minimum_target_separation_standoff_fraction) * max(median_standoff, _EPS)
    )
    current_orbit_pass = bool(evaluations["current_whole_core_coordinate_median"]["robust_locked_orbit_pass"])
    strong_plane = bool(plane_raw["strong_dominant_plane"])
    median_probe = _coordinate_median_invariance_probe(points, voxel_size=voxel_size)
    median_axis_sensitive = bool(
        float(median_probe["maximum_drift_voxels"]) > float(dp.median_invariance_probe_warning_voxels)
    )

    evidence_flags = {
        "source_stage_chain_passed": bool(stages_pass),
        "source_has_only_target_dependent_final_failures": len(unrelated_failures) == 0,
        "current_target_recomputed_exactly_from_core": bool(current_reproduction_exact),
        "source_orbit_metric_reproduced": bool(source_metric_reproduction),
        "target_independent_bound_trajectory_is_broad": bool(trajectory_broad),
        "production_trajectory_circle_is_robust": bool(circle_robust),
        "strong_dominant_core_plane_detected": bool(strong_plane),
        "current_target_is_near_dominant_plane": bool(current_near_plane),
        "production_circle_is_farther_from_plane": bool(circle_farther_from_plane),
        "retained_core_attention_candidate_available": bool(attention_target is not None),
        "retained_core_attention_candidate_is_robust": bool(attention_robust),
        "retained_core_attention_candidate_materially_differs_from_current": bool(attention_materially_separated),
        "coordinatewise_core_median_is_axis_sensitive": bool(median_axis_sensitive),
    }

    if not current_reproduction_exact or not source_metric_reproduction:
        classification = "source_metric_reproduction_mismatch"
    elif not stages_pass or unrelated_failures:
        classification = "unrelated_i4_1_8_failure_present"
    elif current_orbit_pass:
        classification = "current_retained_core_target_already_satisfies_locked_orbit"
    elif attention_robust and attention_materially_separated:
        if strong_plane and current_near_plane and circle_farther_from_plane:
            classification = "whole_core_median_dominant_plane_bias_supported"
        else:
            classification = "whole_core_median_object_target_mismatch_supported"
    elif circle_robust and not attention_robust:
        classification = "trajectory_circle_not_yet_retained_core_supported"
    elif not trajectory_broad:
        classification = "bound_camera_trajectory_itself_is_not_broad"
    else:
        classification = "insufficient_retained_core_evidence_for_target_correction"

    recommend_b = classification in {
        "whole_core_median_dominant_plane_bias_supported",
        "whole_core_median_object_target_mismatch_supported",
    }
    plane_public = {key: value for key, value in plane_raw.items() if not str(key).startswith("_")}
    target_payload = {
        name: {
            "target_reconstruction": [float(v) for v in target_value],
            "structure": structures[name],
            "orbit_evaluation": evaluations[name],
            "production_target_authority": False,
        }
        for name, target_value in targets.items()
    }
    target_payload["production_bound_trajectory_circle"]["diagnostics"] = dict(production_circle_diag)
    target_payload["production_optical_axis_focus"]["diagnostics"] = dict(optical_diag)
    if attention_target is not None:
        target_payload["retained_core_camera_attention_candidate"]["diagnostics"] = dict(attention_diag)

    return {
        "phase": "I4.1.8A",
        "schema_version": I4_1_8A_SCHEMA_VERSION,
        "status": "diagnosis_ready",
        "qualification_policy_reference": qp.as_dict(),
        "diagnosis_policy": dp.as_dict(),
        "source": {
            "i4_1_8_status": str(source.get("status")),
            "i4_1_8_failed_final_gates": list(failed),
            "unrelated_i4_1_8_failures": list(unrelated_failures),
            "i4_1_7b_status": str(b_summary.get("status")),
        },
        "counts": {
            "production_selected_cameras": int(len(poses)),
            "retained_core_points": int(len(points)),
        },
        "retained_core": {
            "voxel_size_reconstruction_units": float(voxel_size),
            "dominant_plane": plane_public,
            "coordinate_median_invariance_probe": median_probe,
        },
        "camera_only_evidence": {
            "trajectory_centered": trajectory_centered,
            "production_bound_trajectory_circle_diagnostics": dict(production_circle_diag),
            "production_optical_axis_focus_diagnostics": dict(optical_diag),
        },
        "targets": target_payload,
        "current_target_reproduction": {
            "saved_target_reconstruction": [float(v) for v in current],
            "recomputed_coordinate_median_reconstruction": [float(v) for v in recomputed_current],
            "difference_reconstruction_units": float(current_reproduction_error),
            "exact_within_numeric_tolerance": bool(current_reproduction_exact),
            "source_orbit_metric_reproduced": bool(source_metric_reproduction),
        },
        "diagnosis": {
            "classification": classification,
            "evidence_flags": evidence_flags,
            "current_to_attention_target_distance_reconstruction_units": float(current_attention_separation),
            "current_to_attention_target_distance_standoff_fraction": float(
                current_attention_separation / max(median_standoff, _EPS)
            ),
            "recommends_i4_1_8b_bounded_target_correction_design": bool(recommend_b),
        },
        "authority": {
            "single_authoritative_sfm_frame": True,
            "sfm_rerun": False,
            "da3_rerun": False,
            "dense_pair_rerun": False,
            "fusion_rerun": False,
            "filtering_rerun": False,
            "persistence_rerun": False,
            "production_binding_rerun": False,
            "production_target_changed": False,
            "production_sampling_integration_authorized": False,
            "orbit_thresholds_changed": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_target_change": False,
            "authorizes_i4_1_final_promotion": False,
            "authorizes_i4_2_detail_preservation": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": (
                "Design I4.1.8B as a bounded retained-Core-supported object-target correction; keep every locked I4.1 gate unchanged."
                if recommend_b
                else "Review the diagnostic evidence before any target/capture change; no production promotion or threshold relaxation is authorized."
            ),
        },
    }


__all__ = [
    "I4_1_8A_SCHEMA_VERSION",
    "I418ATargetDiagnosisPolicy",
    "build_i4_1_8a_retained_core_target_diagnosis",
]
