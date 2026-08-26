"""I4.1.2 SfM-invariant orbit geometry and temporal sampling diagnosis.

This milestone is diagnosis-only.  It was introduced after a visually broad
object orbit produced unexpectedly tiny object-centric coverage values and a
longer source clip still failed dense qualification.  I4.1.2 separates two
independent questions without changing any production policy:

* did the 40-keyframe cap represent the *whole* source clip, or only a prefix;
* are orbit-angle conclusions stable across coordinate-invariant plane/target
  estimators rather than depending on one dense-cloud target.

No SfM, DA3, dense qualification, fusion, filtering, persistence, metric
scale, measurement authority, or mesh state is changed or promoted here.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_qualification import (
    I4OrbitQualificationPolicy,
    orbit_coverage_metrics,
    robust_object_target,
    select_camera_poses,
)

I4_1_2_SCHEMA_VERSION = 1
_EPS = 1e-10


def _finite_matrix_array(value: object, *, minimum: int = 3) -> np.ndarray:
    matrices = np.asarray(value, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or len(matrices) < minimum:
        raise ReconstructionValidationError(f"I4.1.2 requires at least {minimum} finite 4x4 camera poses.")
    if not np.isfinite(matrices).all():
        raise ReconstructionValidationError("I4.1.2 camera poses must be finite.")
    return matrices


def _finite_points(value: object, *, name: str, minimum: int = 3) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum or not np.isfinite(points).all():
        raise ReconstructionValidationError(f"I4.1.2 {name} must be finite Nx3 with at least {minimum} points.")
    return points


def _quantiles(values: object) -> dict[str, float | None]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    return {
        "p10": float(np.quantile(data, 0.10)),
        "p25": float(np.quantile(data, 0.25)),
        "p50": float(np.quantile(data, 0.50)),
        "p75": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.90)),
    }


def _plane_basis(samples: np.ndarray, *, centered: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.float64)
    if centered:
        values = values - np.mean(values, axis=0, keepdims=True)
    covariance = values.T @ values / float(len(values))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = np.asarray(eigenvectors[:, order[:2]], dtype=np.float64)
    normal = np.asarray(eigenvectors[:, order[2]], dtype=np.float64)
    return basis, normal, eigenvalues


def _angular_coverage_from_projected(projected: np.ndarray, *, bin_count: int) -> dict[str, Any]:
    radii = np.linalg.norm(projected, axis=1)
    if np.count_nonzero(radii > _EPS) < 3:
        raise ReconstructionValidationError("I4.1.2 projected camera geometry is degenerate.")
    angles = np.mod(np.arctan2(projected[:, 1], projected[:, 0]), 2.0 * np.pi)
    ordered = np.sort(angles)
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 2.0 * np.pi]))
    largest_gap = float(np.max(gaps))
    coverage = float(np.degrees(2.0 * np.pi - largest_gap))
    width = 2.0 * np.pi / float(bin_count)
    occupied = 0
    for phase in (0.0, 0.25, 0.50, 0.75):
        shifted = np.mod(angles + phase * width, 2.0 * np.pi)
        bins = np.floor(shifted / (2.0 * np.pi) * int(bin_count)).astype(np.int64)
        occupied = max(occupied, int(np.unique(np.clip(bins, 0, int(bin_count) - 1)).size))
    unwrapped = np.unwrap(angles)
    deltas = np.diff(unwrapped)
    active = np.abs(deltas) > np.deg2rad(0.5)
    if np.any(active):
        positive = int(np.count_nonzero(deltas[active] > 0.0))
        negative = int(np.count_nonzero(deltas[active] < 0.0))
        direction_consistency = max(positive, negative) / float(positive + negative)
        travel = float(np.degrees(np.sum(np.abs(deltas[active]))))
    else:
        direction_consistency = 0.0
        travel = 0.0
    return {
        "azimuth_coverage_degrees": coverage,
        "largest_azimuth_gap_degrees": float(np.degrees(largest_gap)),
        "occupied_azimuth_bins": occupied,
        "azimuth_bin_count": int(bin_count),
        "sequence_direction_consistency": float(direction_consistency),
        "sequence_cumulative_angular_travel_degrees": travel,
        "angles_radians": angles,
    }


def _coverage_with_basis(
    matrices: np.ndarray,
    target: np.ndarray,
    basis: np.ndarray,
    normal: np.ndarray,
    *,
    bin_count: int,
    facing_angle_degrees: float,
    plane_mode: str,
) -> dict[str, Any]:
    centers = matrices[:, :3, 3]
    relative = centers - target.reshape(1, 3)
    standoff = np.linalg.norm(relative, axis=1)
    if np.any(standoff <= _EPS):
        raise ReconstructionValidationError("I4.1.2 camera center coincides with target.")
    unit = relative / standoff[:, None]
    projected = relative @ basis
    angular = _angular_coverage_from_projected(projected, bin_count=bin_count)

    dot = np.clip(unit @ unit.T, -1.0, 1.0)
    opposed = float(np.degrees(np.arccos(np.min(dot))))
    forward = np.asarray(matrices[:, :3, 2], dtype=np.float64)
    forward /= np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), _EPS)
    toward = -unit
    facing = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", forward, toward), -1.0, 1.0)))
    standoff_q = _quantiles(standoff)
    p10 = max(float(standoff_q["p10"] or _EPS), _EPS)
    plane_residual = np.abs(relative @ normal) / standoff
    result = {key: value for key, value in angular.items() if key != "angles_radians"}
    result.update({
        "plane_mode": str(plane_mode),
        "target_reconstruction": [float(v) for v in target],
        "maximum_opposed_view_angle_degrees": opposed,
        "target_facing_fraction": float(np.mean(facing <= float(facing_angle_degrees))),
        "facing_angle_degrees": _quantiles(facing),
        "standoff_reconstruction_units": standoff_q,
        "standoff_p90_to_p10_ratio": float(float(standoff_q["p90"] or p10) / p10),
        "orbit_plane_residual_fraction": _quantiles(plane_residual),
    })
    return result


def coverage_by_plane_mode(
    camera_matrices: object,
    target_reconstruction: object,
    *,
    azimuth_bin_count: int = 12,
    facing_angle_degrees: float = 60.0,
) -> dict[str, dict[str, Any]]:
    """Compare three coordinate-invariant orbit-plane constructions."""

    matrices = _finite_matrix_array(camera_matrices)
    target = np.asarray(target_reconstruction, dtype=np.float64).reshape(-1)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ReconstructionValidationError("I4.1.2 target must be a finite 3-vector.")
    centers = matrices[:, :3, 3]
    relative = centers - target.reshape(1, 3)
    norms = np.linalg.norm(relative, axis=1)
    if np.any(norms <= _EPS):
        raise ReconstructionValidationError("I4.1.2 target coincides with a camera center.")
    unit = relative / norms[:, None]

    target_basis, target_normal, _ = _plane_basis(relative, centered=False)
    centered_basis, centered_normal, _ = _plane_basis(centers, centered=True)
    direction_basis, direction_normal, _ = _plane_basis(unit, centered=False)
    return {
        "target_through_pca": _coverage_with_basis(
            matrices, target, target_basis, target_normal,
            bin_count=int(azimuth_bin_count), facing_angle_degrees=float(facing_angle_degrees),
            plane_mode="target_through_pca",
        ),
        "camera_centered_pca": _coverage_with_basis(
            matrices, target, centered_basis, centered_normal,
            bin_count=int(azimuth_bin_count), facing_angle_degrees=float(facing_angle_degrees),
            plane_mode="camera_centered_pca",
        ),
        "direction_pca": _coverage_with_basis(
            matrices, target, direction_basis, direction_normal,
            bin_count=int(azimuth_bin_count), facing_angle_degrees=float(facing_angle_degrees),
            plane_mode="direction_pca",
        ),
    }


def optical_axis_consensus_target(camera_matrices: object) -> tuple[np.ndarray, dict[str, Any]]:
    """Robust least-squares intersection of camera optical axes.

    The target is diagnosis-only.  Iterative angular trimming protects the
    result from cameras whose optical axis is dominated by background content.
    """

    matrices = _finite_matrix_array(camera_matrices)
    centers = matrices[:, :3, 3]
    directions = np.asarray(matrices[:, :3, 2], dtype=np.float64)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), _EPS)
    keep = np.ones(len(matrices), dtype=bool)
    target = np.mean(centers, axis=0)
    condition = float("inf")
    for _ in range(4):
        if np.count_nonzero(keep) < 3:
            raise ReconstructionValidationError("I4.1.2 optical-axis target lost too many cameras.")
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        eye = np.eye(3, dtype=np.float64)
        for center, direction, use in zip(centers, directions, keep):
            if not use:
                continue
            projector = eye - np.outer(direction, direction)
            A += projector
            b += projector @ center
        condition = float(np.linalg.cond(A))
        target = np.linalg.pinv(A, rcond=1e-12) @ b
        vectors = target.reshape(1, 3) - centers
        lengths = np.linalg.norm(vectors, axis=1)
        valid_length = lengths > _EPS
        toward = np.zeros_like(vectors)
        toward[valid_length] = vectors[valid_length] / lengths[valid_length, None]
        cosines = np.clip(np.einsum("ij,ij->i", directions, toward), -1.0, 1.0)
        angles = np.degrees(np.arccos(cosines))
        forward_depth = np.einsum("ij,ij->i", directions, vectors)
        valid = valid_length & (forward_depth > 0.0) & np.isfinite(angles)
        if np.count_nonzero(valid) < 3:
            raise ReconstructionValidationError("I4.1.2 optical-axis target lies behind too many cameras.")
        cutoff = max(10.0, float(np.quantile(angles[valid], 0.80)))
        new_keep = valid & (angles <= cutoff + 1e-9)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    vectors = target.reshape(1, 3) - centers
    lengths = np.linalg.norm(vectors, axis=1)
    toward = vectors / np.maximum(lengths[:, None], _EPS)
    cosines = np.clip(np.einsum("ij,ij->i", directions, toward), -1.0, 1.0)
    angular = np.degrees(np.arccos(cosines))
    forward_depth = np.einsum("ij,ij->i", directions, vectors)
    line_residual = np.linalg.norm(vectors - directions * forward_depth[:, None], axis=1)
    median_standoff = max(float(np.median(lengths)), _EPS)
    diagnostics = {
        "camera_count": int(len(matrices)),
        "inlier_count": int(np.count_nonzero(keep)),
        "inlier_fraction": float(np.mean(keep)),
        "forward_fraction": float(np.mean(forward_depth > 0.0)),
        "angular_residual_degrees": _quantiles(angular),
        "line_residual_over_median_standoff": _quantiles(line_residual / median_standoff),
        "normal_equation_condition_number": condition,
    }
    return np.asarray(target, dtype=np.float64), diagnostics


def trajectory_circle_target(camera_matrices: object) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate an orbit center from a best-fit circle in the camera PCA plane."""

    matrices = _finite_matrix_array(camera_matrices)
    centers = matrices[:, :3, 3]
    centroid = np.mean(centers, axis=0)
    basis, normal, eigenvalues = _plane_basis(centers, centered=True)
    xy = (centers - centroid.reshape(1, 3)) @ basis
    x, y = xy[:, 0], xy[:, 1]
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x), dtype=np.float64)])
    rhs = x * x + y * y
    condition = float(np.linalg.cond(design))
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    cx, cy, q = (float(v) for v in solution)
    radius_sq = q + cx * cx + cy * cy
    if not isfinite(radius_sq) or radius_sq <= _EPS:
        raise ReconstructionValidationError("I4.1.2 trajectory circle fit is degenerate.")
    target = centroid + basis @ np.asarray([cx, cy], dtype=np.float64)
    radii = np.linalg.norm(xy - np.asarray([cx, cy]), axis=1)
    radius = float(np.sqrt(radius_sq))
    residual_fraction = np.abs(radii - radius) / max(radius, _EPS)
    eig_sum = max(float(np.sum(eigenvalues)), _EPS)
    return np.asarray(target, dtype=np.float64), {
        "radius_reconstruction_units": radius,
        "radial_residual_fraction": _quantiles(residual_fraction),
        "plane_variance_fraction": float((eigenvalues[0] + eigenvalues[1]) / eig_sum),
        "normal_equation_condition_number": condition,
        "plane_normal": [float(v) for v in normal],
    }


def trajectory_centered_coverage(camera_matrices: object, *, azimuth_bin_count: int = 12) -> dict[str, Any]:
    """Target-independent trajectory spread diagnostic around camera centroid."""

    matrices = _finite_matrix_array(camera_matrices)
    centers = matrices[:, :3, 3]
    centroid = np.mean(centers, axis=0)
    basis, normal, eigenvalues = _plane_basis(centers, centered=True)
    projected = (centers - centroid.reshape(1, 3)) @ basis
    angular = _angular_coverage_from_projected(projected, bin_count=int(azimuth_bin_count))
    centered = centers - centroid.reshape(1, 3)
    radius = np.linalg.norm(centered, axis=1)
    plane_residual = np.abs(centered @ normal) / np.maximum(radius, _EPS)
    eig_sum = max(float(np.sum(eigenvalues)), _EPS)
    result = {key: value for key, value in angular.items() if key != "angles_radians"}
    result.update({
        "diagnostic_only": True,
        "center_reconstruction": [float(v) for v in centroid],
        "plane_residual_fraction": _quantiles(plane_residual),
        "plane_variance_fraction": float((eigenvalues[0] + eigenvalues[1]) / eig_sum),
    })
    return result


def sampling_coverage_diagnosis(
    sfm_report: Mapping[str, Any],
    *,
    expected_max_sampled_frames: int = 40,
) -> dict[str, Any]:
    """Diagnose whether the sampled SfM workspace represents the full video."""

    if int(expected_max_sampled_frames) < 3:
        raise ReconstructionValidationError("I4.1.2 expected max sampled frames must be >=3.")
    source = sfm_report.get("source")
    result = sfm_report.get("sfm_result")
    if not isinstance(source, Mapping) or not isinstance(result, Mapping):
        raise ReconstructionValidationError("I4.1.2 SfM report is missing source/sfm_result evidence.")
    workspace = result.get("workspace")
    if not isinstance(workspace, Mapping):
        raise ReconstructionValidationError("I4.1.2 SfM report is missing workspace evidence.")
    images = workspace.get("images")
    if not isinstance(images, list) or len(images) < 3:
        raise ReconstructionValidationError("I4.1.2 workspace requires at least three sampled images.")

    fps = float(source.get("expected_frame_rate_hz") or 0.0)
    metadata = source.get("metadata")
    frame_count = int(metadata.get("reported_frame_count", 0)) if isinstance(metadata, Mapping) else 0
    orders = np.asarray([int(item.get("order", -1)) for item in images], dtype=np.int64)
    source_indices = np.asarray([int(item.get("source_sequence_index", -1)) for item in images], dtype=np.int64)
    timestamps = np.asarray([int(item.get("timestamp_ns", -1)) for item in images], dtype=np.int64)
    if np.any(orders < 0) or np.any(source_indices < 0) or np.any(timestamps < 0):
        raise ReconstructionValidationError("I4.1.2 workspace sampling evidence is invalid.")
    if len(np.unique(orders)) != len(orders) or len(np.unique(source_indices)) != len(source_indices):
        raise ReconstructionValidationError("I4.1.2 workspace sampling orders/source indices must be unique.")
    if np.any(np.diff(np.sort(timestamps)) <= 0):
        raise ReconstructionValidationError("I4.1.2 sampled timestamps must be strictly increasing.")

    sampled_count = int(len(images))
    time_order = np.argsort(timestamps)
    ts = timestamps[time_order]
    idx = source_indices[time_order]
    sampled_span_s = float((ts[-1] - ts[0]) / 1e9)
    median_delta_s = float(np.median(np.diff(ts)) / 1e9)
    inferred_fps = 1.0 / median_delta_s if median_delta_s > 0.0 else None
    estimated_duration_s = None
    if isfinite(fps) and fps > 0.0 and frame_count > 1:
        estimated_duration_s = float((frame_count - 1) / fps)
    source_index_fraction = None
    if frame_count > 1:
        source_index_fraction = float(idx[-1] / float(frame_count - 1))
    temporal_span_fraction = None
    if estimated_duration_s is not None and estimated_duration_s > 0.0:
        temporal_span_fraction = float(sampled_span_s / estimated_duration_s)

    cap_reached = sampled_count >= int(expected_max_sampled_frames)
    starts_near_source_start = int(idx[0]) <= max(2, int(round(0.01 * max(frame_count, 1))))
    tail_unrepresented = (
        source_index_fraction is not None and source_index_fraction < 0.80
    ) or (
        temporal_span_fraction is not None and temporal_span_fraction < 0.80
    )
    prefix_cap_likely = bool(cap_reached and starts_near_source_start and tail_unrepresented)
    expected_prefix_span_s = None
    if inferred_fps is not None and inferred_fps > 0.0:
        expected_prefix_span_s = float((expected_max_sampled_frames - 1) / inferred_fps)

    return {
        "sampled_view_count": sampled_count,
        "expected_max_sampled_frames": int(expected_max_sampled_frames),
        "cap_reached": cap_reached,
        "reported_source_frame_count": frame_count,
        "reported_source_fps": fps if fps > 0.0 else None,
        "estimated_source_duration_seconds": estimated_duration_s,
        "sampled_timestamp_span_seconds": sampled_span_s,
        "sampled_temporal_span_fraction": temporal_span_fraction,
        "sampled_first_source_sequence_index": int(idx[0]),
        "sampled_last_source_sequence_index": int(idx[-1]),
        "sampled_last_source_index_fraction": source_index_fraction,
        "inferred_sample_rate_hz": inferred_fps,
        "expected_prefix_span_seconds_at_inferred_rate": expected_prefix_span_s,
        "starts_near_source_start": starts_near_source_start,
        "source_tail_unrepresented": bool(tail_unrepresented),
        "prefix_max_frame_truncation_likely": prefix_cap_likely,
        "full_source_temporal_coverage_proven": bool(not prefix_cap_likely and not tail_unrepresented),
    }


def _target_record(name: str, target: np.ndarray, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(name),
        "target_reconstruction": [float(v) for v in target],
        "diagnostics": dict(diagnostics),
    }


def _evaluate_target(
    matrices: np.ndarray,
    target: np.ndarray,
    *,
    policy: I4OrbitQualificationPolicy,
) -> dict[str, Any]:
    modes = coverage_by_plane_mode(
        matrices,
        target,
        azimuth_bin_count=policy.azimuth_bin_count,
        facing_angle_degrees=policy.maximum_facing_angle_degrees,
    )
    coverages = [float(item["azimuth_coverage_degrees"]) for item in modes.values()]
    return {
        "plane_modes": modes,
        "coverage_min_degrees": float(min(coverages)),
        "coverage_max_degrees": float(max(coverages)),
        "coverage_spread_degrees": float(max(coverages) - min(coverages)),
        "all_plane_modes_meet_150deg": bool(min(coverages) >= policy.minimum_selected_azimuth_coverage_degrees),
    }


def build_i4_1_2_diagnosis(
    *,
    source_i4_summary: Mapping[str, Any],
    sfm_report: Mapping[str, Any],
    registered_camera_ids: Sequence[str],
    registered_camera_matrices: object,
    sparse_landmarks: object,
    aligned_keyframe_ids: Sequence[str] = (),
    selected_keyframe_ids: Sequence[str] = (),
    core_points: object | None = None,
    expected_max_sampled_frames: int = 40,
    policy: I4OrbitQualificationPolicy | None = None,
) -> dict[str, Any]:
    """Build the non-mutating I4.1.2 temporal/orbit diagnosis."""

    if str(source_i4_summary.get("phase")) != "I4.1" or str(source_i4_summary.get("status")) != "degraded":
        raise ReconstructionValidationError("I4.1.2 requires a completed DEGRADED I4.1 campaign.")
    failed_stage = str(source_i4_summary.get("failed_stage", ""))
    if failed_stage not in {"object_orbit_i1_baseline", "object_orbit_baseline_quality"}:
        raise ReconstructionValidationError("I4.1.2 requires an I4.1 I1-baseline or baseline-quality stop.")

    p = policy or I4OrbitQualificationPolicy()
    ids = tuple(str(v) for v in registered_camera_ids)
    poses = _finite_matrix_array(registered_camera_matrices)
    if poses.shape[0] != len(ids) or len(set(ids)) != len(ids):
        raise ReconstructionValidationError("I4.1.2 registered camera IDs/poses are misaligned or duplicated.")
    landmarks = _finite_points(sparse_landmarks, name="sparse landmarks", minimum=64)

    aligned_ids = tuple(str(v) for v in aligned_keyframe_ids if str(v))
    selected_ids = tuple(str(v) for v in selected_keyframe_ids if str(v))
    aligned_poses = None
    selected_poses = None
    if aligned_ids:
        if len(aligned_ids) < 3:
            raise ReconstructionValidationError("I4.1.2 aligned camera set must contain >=3 cameras when supplied.")
        _, aligned_poses = select_camera_poses(ids, poses, aligned_ids)
    if selected_ids:
        if len(selected_ids) < 3:
            raise ReconstructionValidationError("I4.1.2 selected camera set must contain >=3 cameras when supplied.")
        _, selected_poses = select_camera_poses(ids, poses, selected_ids)

    sampling = sampling_coverage_diagnosis(sfm_report, expected_max_sampled_frames=expected_max_sampled_frames)
    sparse_target = np.median(landmarks, axis=0)
    ray_target, ray_diag = optical_axis_consensus_target(poses)
    circle_target, circle_diag = trajectory_circle_target(poses)

    targets: dict[str, dict[str, Any]] = {
        "sparse_median": _target_record("sparse_median", sparse_target, {"source": "SfM sparse landmark coordinate-wise median"}),
        "optical_axis_consensus": _target_record("optical_axis_consensus", ray_target, ray_diag),
        "trajectory_circle_center": _target_record("trajectory_circle_center", circle_target, circle_diag),
    }
    if core_points is not None:
        core = _finite_points(core_points, name="Core geometry", minimum=64)
        core_target = robust_object_target(core)
        targets["core_median_current_i4"] = _target_record(
            "core_median_current_i4", core_target, {"source": "current I4 robust_object_target(Core)"}
        )

    camera_sets: dict[str, np.ndarray] = {"registered": poses}
    if aligned_poses is not None:
        camera_sets["aligned"] = aligned_poses
    if selected_poses is not None:
        camera_sets["selected"] = selected_poses

    evaluations: dict[str, dict[str, Any]] = {}
    for set_name, set_poses in camera_sets.items():
        evaluations[set_name] = {
            "camera_count": int(len(set_poses)),
            "trajectory_centered_pca": trajectory_centered_coverage(set_poses, azimuth_bin_count=p.azimuth_bin_count),
            "targets": {},
        }
        for target_name, target_record in targets.items():
            target = np.asarray(target_record["target_reconstruction"], dtype=np.float64)
            evaluations[set_name]["targets"][target_name] = _evaluate_target(set_poses, target, policy=p)

    current_exact = None
    if "core_median_current_i4" in targets:
        current_target = np.asarray(targets["core_median_current_i4"]["target_reconstruction"], dtype=np.float64)
        legacy = orbit_coverage_metrics(
            poses,
            current_target,
            azimuth_bin_count=p.azimuth_bin_count,
            facing_angle_degrees=p.maximum_facing_angle_degrees,
        )
        recomputed = evaluations["registered"]["targets"]["core_median_current_i4"]["plane_modes"]["target_through_pca"]
        current_exact = bool(
            np.isclose(float(legacy["azimuth_coverage_degrees"]), float(recomputed["azimuth_coverage_degrees"]), atol=1e-9)
            and int(legacy["occupied_azimuth_bins"]) == int(recomputed["occupied_azimuth_bins"])
            and np.isclose(float(legacy["maximum_opposed_view_angle_degrees"]), float(recomputed["maximum_opposed_view_angle_degrees"]), atol=1e-9)
        )

    registered_target_max = {
        name: float(data["coverage_max_degrees"])
        for name, data in evaluations["registered"]["targets"].items()
    }
    best_target_name = max(registered_target_max, key=registered_target_max.get)
    best_registered = float(registered_target_max[best_target_name])
    trajectory_registered = float(evaluations["registered"]["trajectory_centered_pca"]["azimuth_coverage_degrees"])
    robust_registered_broad = best_registered >= float(p.minimum_selected_azimuth_coverage_degrees)

    selected_best = None
    if "selected" in evaluations:
        selected_best = max(
            float(data["coverage_max_degrees"])
            for data in evaluations["selected"]["targets"].values()
        )

    prefix = bool(sampling["prefix_max_frame_truncation_likely"])
    if prefix:
        sampling_classification = "prefix_temporal_sampling_truncation_confirmed"
    elif bool(sampling["source_tail_unrepresented"]):
        sampling_classification = "partial_source_temporal_coverage"
    else:
        sampling_classification = "full_source_temporal_coverage_supported"

    if robust_registered_broad and selected_best is not None and selected_best < float(p.minimum_selected_azimuth_coverage_degrees):
        orbit_classification = "registered_orbit_broad_selected_dense_path_collapsed"
    elif robust_registered_broad:
        orbit_classification = "registered_orbit_broad_under_invariant_target_plane_diagnosis"
    elif trajectory_registered >= float(p.minimum_selected_azimuth_coverage_degrees) and prefix:
        orbit_classification = "trajectory_spread_broad_but_object_target_unresolved_with_prefix_sampling"
    elif prefix:
        orbit_classification = "sampled_prefix_does_not_prove_full_clip_orbit"
    else:
        orbit_classification = "registered_sampled_orbit_geometry_insufficient_or_target_unresolved"

    metric_correction_candidate = bool(
        current_exact is True
        and "core_median_current_i4" in registered_target_max
        and registered_target_max["core_median_current_i4"] < float(p.minimum_selected_azimuth_coverage_degrees)
        and robust_registered_broad
    )
    sampling_policy_correction_candidate = bool(prefix)

    if sampling_policy_correction_candidate:
        next_action = (
            "I4.1.3 should correct max-keyframe sampling to represent the full clip before rerunning expensive orbit qualification; "
            "do not relax orbit/alignment/dense-consistency thresholds."
        )
    elif metric_correction_candidate:
        next_action = (
            "Review invariant target/plane evidence for an I4 orbit-metric correction; do not alter acceptance thresholds yet."
        )
    else:
        next_action = (
            "Use the invariant orbit evidence to decide whether capture geometry or dense-path coverage needs correction; no threshold relaxation."
        )

    return {
        "phase": "I4.1.2",
        "schema_version": I4_1_2_SCHEMA_VERSION,
        "status": "diagnosis_ready",
        "diagnosis_complete": True,
        "source_i4_1": {
            "status": str(source_i4_summary.get("status")),
            "failed_stage": failed_stage,
            "reason": str(source_i4_summary.get("reason", "")),
        },
        "policy_reference": p.as_dict(),
        "counts": {
            "registered_cameras": int(len(poses)),
            "aligned_cameras": None if aligned_poses is None else int(len(aligned_poses)),
            "selected_cameras": None if selected_poses is None else int(len(selected_poses)),
            "sparse_landmarks": int(len(landmarks)),
        },
        "temporal_sampling": sampling,
        "targets": targets,
        "camera_set_evaluations": evaluations,
        "current_i4_metric_reproduction_exact": current_exact,
        "diagnosis": {
            "sampling_classification": sampling_classification,
            "orbit_geometry_classification": orbit_classification,
            "best_registered_target": best_target_name,
            "best_registered_invariant_coverage_degrees": best_registered,
            "registered_trajectory_centered_coverage_degrees": trajectory_registered,
            "best_selected_invariant_coverage_degrees": selected_best,
            "sampling_policy_correction_candidate": sampling_policy_correction_candidate,
            "orbit_metric_correction_candidate": metric_correction_candidate,
        },
        "authority": {
            "sfm_rerun": False,
            "da3_rerun": False,
            "fusion_rerun": False,
            "filter_policy_changed": False,
            "persistence_changed": False,
            "sampling_policy_changed": False,
            "orbit_metric_policy_changed": False,
            "threshold_relaxation": False,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_sampling_policy_change": False,
            "authorizes_orbit_metric_change": False,
            "authorizes_threshold_relaxation": False,
            "authorizes_i4_2_detail_preservation": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": next_action,
        },
    }


__all__ = [
    "I4_1_2_SCHEMA_VERSION",
    "build_i4_1_2_diagnosis",
    "coverage_by_plane_mode",
    "optical_axis_consensus_target",
    "sampling_coverage_diagnosis",
    "trajectory_centered_coverage",
    "trajectory_circle_target",
]
