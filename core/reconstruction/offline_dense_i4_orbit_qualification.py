"""I4.1 object/connector orbit qualification helpers.

I4.1 validates whether the already-qualified offline Dense RGB pipeline can
reconstruct a *small static object* from a camera-orbit capture.  It remains a
quality/coverage milestone only: no metric scale, measurement authority, mesh
or new filtering policy is promoted here.

The orbit diagnostics are intentionally scale invariant.  They use the SfM
camera centres/optical axes and a robust Core-cloud target estimate to measure:

* azimuth coverage around the target;
* opposite-side viewpoint diversity;
* how consistently cameras face the target;
* standoff consistency; and
* selected-view coverage independently from merely registering frames.

This is especially important before I4.2, where thin connector/detail
preservation will stress the manifold filter.  I4.1 must first prove that the
capture itself provides a usable object-centric orbit.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

I4_1_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class I4OrbitQualificationPolicy:
    """Fail-closed object-orbit coverage gates.

    These are deliberately capture/geometry gates rather than quality-tuning
    knobs.  I2's ordinary real-scene gates must already pass independently.
    """

    minimum_registered_fraction: float = 0.80
    minimum_dense_alignment_fraction: float = 0.80
    minimum_selected_views: int = 10
    minimum_selected_azimuth_coverage_degrees: float = 150.0
    minimum_selected_occupied_azimuth_bins: int = 5
    azimuth_bin_count: int = 12
    minimum_opposed_view_angle_degrees: float = 120.0
    maximum_facing_angle_degrees: float = 60.0
    minimum_target_facing_fraction: float = 0.75
    maximum_standoff_p90_to_p10_ratio: float = 2.50

    def __post_init__(self) -> None:
        fractions = (
            "minimum_registered_fraction",
            "minimum_dense_alignment_fraction",
            "minimum_target_facing_fraction",
        )
        for name in fractions:
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ReconstructionValidationError(f"I4.1 {name} must be finite in [0,1].")
        if int(self.minimum_selected_views) < 3:
            raise ReconstructionValidationError("I4.1 minimum_selected_views must be >=3.")
        if int(self.azimuth_bin_count) < 4:
            raise ReconstructionValidationError("I4.1 azimuth_bin_count must be >=4.")
        if not 1 <= int(self.minimum_selected_occupied_azimuth_bins) <= int(self.azimuth_bin_count):
            raise ReconstructionValidationError("I4.1 occupied azimuth-bin requirement is invalid.")
        for name in (
            "minimum_selected_azimuth_coverage_degrees",
            "minimum_opposed_view_angle_degrees",
            "maximum_facing_angle_degrees",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 180.0:
                raise ReconstructionValidationError(f"I4.1 {name} must be finite in [0,180].")
        ratio = float(self.maximum_standoff_p90_to_p10_ratio)
        if not isfinite(ratio) or ratio < 1.0:
            raise ReconstructionValidationError("I4.1 standoff ratio must be finite and >=1.")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def robust_object_target(core_points: object) -> np.ndarray:
    """Return a tail-resistant object/scene target from retained Core geometry."""

    points = np.asarray(core_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 64 or not np.isfinite(points).all():
        raise ReconstructionValidationError("I4.1 Core geometry must be finite Nx3 with at least 64 points.")
    target = np.median(points, axis=0)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ReconstructionValidationError("I4.1 robust object target is invalid.")
    return np.asarray(target, dtype=np.float64)


def camera_poses_from_sfm_report(report: Mapping[str, Any]) -> tuple[tuple[str, ...], np.ndarray]:
    """Extract dependency-neutral camera poses from a real SfM report."""

    sfm = report.get("sfm_result")
    if not isinstance(sfm, Mapping):
        raise ReconstructionValidationError("I4.1 SfM report is missing sfm_result.")
    cameras = sfm.get("cameras")
    if not isinstance(cameras, list) or len(cameras) < 3:
        raise ReconstructionValidationError("I4.1 SfM report requires at least three registered cameras.")
    ids: list[str] = []
    matrices: list[np.ndarray] = []
    for index, item in enumerate(cameras):
        if not isinstance(item, Mapping):
            raise ReconstructionValidationError("I4.1 SfM camera record must be an object.")
        keyframe_id = str(item.get("keyframe_id", "")).strip()
        if not keyframe_id:
            raise ReconstructionValidationError(f"I4.1 SfM camera {index} has no keyframe_id.")
        matrix = np.asarray(item.get("T_reconstruction_from_camera"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ReconstructionValidationError(f"I4.1 camera {keyframe_id!r} pose must be finite 4x4.")
        if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
            raise ReconstructionValidationError(f"I4.1 camera {keyframe_id!r} pose has invalid homogeneous row.")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            float(np.linalg.det(rotation)), 1.0, atol=1e-5
        ):
            raise ReconstructionValidationError(f"I4.1 camera {keyframe_id!r} rotation is not rigid.")
        ids.append(keyframe_id)
        matrices.append(matrix)
    if len(set(ids)) != len(ids):
        raise ReconstructionValidationError("I4.1 SfM camera keyframe IDs must be unique.")
    return tuple(ids), np.asarray(matrices, dtype=np.float64)


def select_camera_poses(
    camera_ids: Sequence[str],
    matrices: object,
    selected_keyframe_ids: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return selected poses in the exact qualified-view order."""

    ids = tuple(str(value).strip() for value in camera_ids)
    selected = tuple(str(value).strip() for value in selected_keyframe_ids)
    poses = np.asarray(matrices, dtype=np.float64)
    if poses.shape != (len(ids), 4, 4):
        raise ReconstructionValidationError("I4.1 registered camera pose array is misaligned.")
    if not selected or any(not value for value in selected):
        raise ReconstructionValidationError("I4.1 selected keyframe IDs must be non-empty.")
    index = {key: i for i, key in enumerate(ids)}
    missing = [key for key in selected if key not in index]
    if missing:
        raise ReconstructionValidationError(f"I4.1 selected cameras are missing from SfM: {missing!r}")
    return selected, np.asarray([poses[index[key]] for key in selected], dtype=np.float64)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0 or not np.isfinite(values).all():
        raise ReconstructionValidationError("I4.1 quantile population must be finite and non-empty.")
    return {
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def orbit_coverage_metrics(
    camera_matrices: object,
    target_reconstruction: object,
    *,
    azimuth_bin_count: int = 12,
    facing_angle_degrees: float = 60.0,
) -> dict[str, Any]:
    """Measure scale-invariant camera-orbit coverage around one target."""

    matrices = np.asarray(camera_matrices, dtype=np.float64)
    target = np.asarray(target_reconstruction, dtype=np.float64).reshape(-1)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or matrices.shape[0] < 3:
        raise ReconstructionValidationError("I4.1 orbit metrics require at least three 4x4 camera poses.")
    if not np.isfinite(matrices).all() or target.shape != (3,) or not np.isfinite(target).all():
        raise ReconstructionValidationError("I4.1 orbit metric inputs must be finite.")
    if int(azimuth_bin_count) < 4:
        raise ReconstructionValidationError("I4.1 azimuth_bin_count must be >=4.")
    if not 0.0 <= float(facing_angle_degrees) <= 180.0:
        raise ReconstructionValidationError("I4.1 facing angle must lie in [0,180].")

    centers = np.asarray(matrices[:, :3, 3], dtype=np.float64)
    relative = centers - target.reshape(1, 3)
    standoff = np.linalg.norm(relative, axis=1)
    eps = 1e-10
    if np.any(standoff <= eps):
        raise ReconstructionValidationError("I4.1 camera centre coincides with robust target.")
    unit = relative / standoff[:, None]

    # Fit an orbit plane *through the target*, avoiding a camera-centroid shift
    # that would artificially inflate coverage on partial arcs.
    covariance = relative.T @ relative / float(len(relative))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = eigenvectors[:, order[:2]]
    normal = eigenvectors[:, order[2]]
    projected = relative @ basis
    projected_radius = np.linalg.norm(projected, axis=1)
    if np.count_nonzero(projected_radius > eps) < 3:
        raise ReconstructionValidationError("I4.1 registered cameras do not define an orbit plane.")

    angles = np.mod(np.arctan2(projected[:, 1], projected[:, 0]), 2.0 * np.pi)
    sorted_angles = np.sort(angles)
    gaps = np.diff(np.concatenate([sorted_angles, sorted_angles[:1] + 2.0 * np.pi]))
    largest_gap = float(np.max(gaps))
    azimuth_coverage = float(np.degrees(2.0 * np.pi - largest_gap))

    bin_count = int(azimuth_bin_count)
    # Bin occupancy must not depend materially on the arbitrary PCA basis
    # phase.  Evaluate a few sub-bin phase offsets and keep the maximum
    # occupancy; this preserves the intended coarse angular-diversity signal
    # without rewarding any particular coordinate-axis orientation.
    bin_width = 2.0 * np.pi / float(bin_count)
    occupied_bins = 0
    for phase_fraction in (0.0, 0.25, 0.50, 0.75):
        shifted = np.mod(angles + phase_fraction * bin_width, 2.0 * np.pi)
        bins = np.floor(shifted / (2.0 * np.pi) * bin_count).astype(np.int64)
        bins = np.clip(bins, 0, bin_count - 1)
        occupied_bins = max(occupied_bins, int(np.unique(bins).size))

    dot = np.clip(unit @ unit.T, -1.0, 1.0)
    maximum_pair_angle = float(np.degrees(np.arccos(np.min(dot))))

    # Internal SfM convention is camera-to-reconstruction with +Z optical
    # forward, matching the positive camera-Z checks used by dense alignment.
    optical_forward = np.asarray(matrices[:, :3, 2], dtype=np.float64)
    optical_forward /= np.maximum(np.linalg.norm(optical_forward, axis=1, keepdims=True), eps)
    toward_target = -unit
    facing_cosine = np.clip(np.einsum("ij,ij->i", optical_forward, toward_target), -1.0, 1.0)
    facing_angles = np.degrees(np.arccos(facing_cosine))
    facing_fraction = float(np.mean(facing_angles <= float(facing_angle_degrees)))

    standoff_q = _quantiles(standoff)
    p10 = max(float(standoff_q["p10"]), eps)
    standoff_ratio = float(standoff_q["p90"] / p10)
    plane_residual_fraction = np.abs(relative @ normal) / standoff
    eigen_sum = max(float(np.sum(eigenvalues)), eps)

    # Sequence-order diagnostic: a healthy handheld orbit usually moves mostly
    # in one angular direction, but this is intentionally not a hard gate.
    unwrapped = np.unwrap(angles)
    deltas = np.diff(unwrapped)
    nonzero = np.abs(deltas) > np.deg2rad(0.5)
    if np.any(nonzero):
        positive = int(np.count_nonzero(deltas[nonzero] > 0.0))
        negative = int(np.count_nonzero(deltas[nonzero] < 0.0))
        direction_consistency = max(positive, negative) / float(positive + negative)
        cumulative_travel = float(np.degrees(np.sum(np.abs(deltas[nonzero]))))
    else:
        direction_consistency = 0.0
        cumulative_travel = 0.0

    return {
        "camera_count": int(len(matrices)),
        "target_reconstruction": [float(v) for v in target],
        "azimuth_coverage_degrees": azimuth_coverage,
        "largest_azimuth_gap_degrees": float(np.degrees(largest_gap)),
        "azimuth_bin_count": bin_count,
        "occupied_azimuth_bins": occupied_bins,
        "occupied_azimuth_fraction": occupied_bins / float(bin_count),
        "maximum_opposed_view_angle_degrees": maximum_pair_angle,
        "target_facing_fraction": facing_fraction,
        "facing_angle_degrees": _quantiles(np.asarray(facing_angles, dtype=np.float64)),
        "standoff_reconstruction_units": standoff_q,
        "standoff_p90_to_p10_ratio": standoff_ratio,
        "orbit_plane_residual_fraction": _quantiles(np.asarray(plane_residual_fraction, dtype=np.float64)),
        "orbit_plane_variance_fraction": float((eigenvalues[0] + eigenvalues[1]) / eigen_sum),
        "sequence_direction_consistency": float(direction_consistency),
        "sequence_cumulative_angular_travel_degrees": cumulative_travel,
    }


def build_i4_1_orbit_assessment(
    *,
    sfm_summary: Mapping[str, Any],
    i1_summary: Mapping[str, Any],
    i2_summary: Mapping[str, Any],
    registered_camera_matrices: object,
    registered_camera_ids: Sequence[str],
    selected_keyframe_ids: Sequence[str],
    core_points: object,
    policy: I4OrbitQualificationPolicy | None = None,
) -> dict[str, Any]:
    """Combine ordinary I2 quality with object-centric orbit coverage gates."""

    p = policy or I4OrbitQualificationPolicy()
    if str(i1_summary.get("phase")) != "I1" or str(i1_summary.get("status")) != "pass":
        raise ReconstructionValidationError("I4.1 requires a passing I1 baseline.")
    if str(i2_summary.get("phase")) != "I2":
        raise ReconstructionValidationError("I4.1 requires an I2 baseline-quality summary.")

    input_views = int(sfm_summary.get("input_keyframes", 0))
    registered_views = int(i1_summary.get("sfm", {}).get("registered_view_count", 0))
    aligned_views = int(i1_summary.get("dense_qualification", {}).get("alignment_accepted_views", 0))
    selected_views = len(tuple(selected_keyframe_ids))
    if input_views <= 0 or registered_views <= 0:
        raise ReconstructionValidationError("I4.1 SfM input/registration counts are invalid.")
    registered_fraction = registered_views / float(input_views)
    alignment_fraction = aligned_views / float(registered_views)

    target = robust_object_target(core_points)
    ids = tuple(str(value) for value in registered_camera_ids)
    poses = np.asarray(registered_camera_matrices, dtype=np.float64)
    selected_ids, selected_poses = select_camera_poses(ids, poses, selected_keyframe_ids)
    all_metrics = orbit_coverage_metrics(
        poses,
        target,
        azimuth_bin_count=int(p.azimuth_bin_count),
        facing_angle_degrees=float(p.maximum_facing_angle_degrees),
    )
    selected_metrics = orbit_coverage_metrics(
        selected_poses,
        target,
        azimuth_bin_count=int(p.azimuth_bin_count),
        facing_angle_degrees=float(p.maximum_facing_angle_degrees),
    )

    i2_pass = bool(i2_summary.get("automated_pass"))
    gates = {
        "ordinary_i2_baseline_quality": i2_pass,
        "registered_fraction": registered_fraction >= float(p.minimum_registered_fraction),
        "dense_alignment_fraction": alignment_fraction >= float(p.minimum_dense_alignment_fraction),
        "selected_view_count": selected_views >= int(p.minimum_selected_views),
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
    passed = all(bool(value) for value in gates.values())
    return {
        "pass": passed,
        "policy": p.as_dict(),
        "counts": {
            "input_views": input_views,
            "registered_views": registered_views,
            "dense_alignment_accepted_views": aligned_views,
            "selected_qualified_views": selected_views,
            "registered_fraction": registered_fraction,
            "dense_alignment_fraction_of_registered": alignment_fraction,
        },
        "target_reconstruction": [float(v) for v in target],
        "registered_orbit": all_metrics,
        "selected_orbit": selected_metrics,
        "selected_keyframe_ids": list(selected_ids),
        "gates": gates,
    }


__all__ = [
    "I4_1_SCHEMA_VERSION",
    "I4OrbitQualificationPolicy",
    "build_i4_1_orbit_assessment",
    "camera_poses_from_sfm_report",
    "orbit_coverage_metrics",
    "robust_object_target",
    "select_camera_poses",
]
