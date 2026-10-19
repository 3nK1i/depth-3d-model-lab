#!/usr/bin/env python3
"""E2 consistency-aware refinement of the passing E1 voxel-fused scene.

E2 does not rerun PyCOLMAP, DA3, or the D4/D4.3 accumulation.  It reuses the
per-view canonical XYZRGB artifacts referenced by E1, reconstructs the exact
E1 voxel/view-observation accounting from the float32-aligned depth that E1 actually fused, and adds three kinds of evidence that E1
intentionally did not use:

* robust per-voxel geometric dispersion across independent view centroids;
* local support/connectivity in the E1 voxel lattice;
* a conservative refinement score / class split for core vs extended geometry.

The result remains validation-only in the arbitrary offline-SfM gauge.  E2 does
not promote metric scale, canonical persistence, measurement authority, surface
mesh authority, or certified metrology.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402

_DEFAULT_E1_DIR = Path("/tmp/depth3d_e1_qualified_ten_view_voxel_fusion")
_DEFAULT_OUTPUT_DIR = Path("/tmp/depth3d_e2_consistency_aware_fusion_refinement")

_NEIGHBOR26 = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)
_NEIGHBOR_R2 = tuple(
    (dx, dy, dz)
    for dx in (-2, -1, 0, 1, 2)
    for dy in (-2, -1, 0, 1, 2)
    for dz in (-2, -1, 0, 1, 2)
    if (dx, dy, dz) != (0, 0, 0)
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-dir", type=Path, default=_DEFAULT_E1_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-core-voxels", type=int, default=1_000)
    parser.add_argument("--minimum-component-size", type=int, default=64)
    parser.add_argument("--core-max-dispersion-voxels", type=float, default=0.70)
    parser.add_argument("--support2-core-max-dispersion-voxels", type=float, default=0.45)
    parser.add_argument("--extended-max-dispersion-voxels", type=float, default=0.95)
    parser.add_argument("--support2-core-min-high-neighbors", type=int, default=2)
    parser.add_argument("--support2-core-min-radius2-neighbors", type=int, default=8)
    parser.add_argument("--extended-min-radius2-neighbors", type=int, default=6)
    parser.add_argument("--fringe-min-extended-neighbors", type=int, default=2)
    args = parser.parse_args()
    if int(args.minimum_core_voxels) <= 0:
        parser.error("--minimum-core-voxels must be positive.")
    if int(args.minimum_component_size) <= 0:
        parser.error("--minimum-component-size must be positive.")
    for name in (
        "core_max_dispersion_voxels",
        "support2_core_max_dispersion_voxels",
        "extended_max_dispersion_voxels",
    ):
        value = float(getattr(args, name))
        if not isfinite(value) or value <= 0.0 or value > 2.0:
            parser.error(f"--{name.replace('_', '-')} must be finite in (0, 2].")
    if float(args.support2_core_max_dispersion_voxels) > float(args.core_max_dispersion_voxels):
        parser.error("support-2 core dispersion limit cannot exceed the general core limit.")
    if float(args.core_max_dispersion_voxels) > float(args.extended_max_dispersion_voxels):
        parser.error("core dispersion limit cannot exceed the extended limit.")
    for name in (
        "support2_core_min_high_neighbors",
        "support2_core_min_radius2_neighbors",
        "extended_min_radius2_neighbors",
        "fringe_min_extended_neighbors",
    ):
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative.")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_e1_summary(summary: dict[str, Any]) -> None:
    if str(summary.get("phase", "")) != "E1":
        raise ReconstructionValidationError("E2 requires an E1 summary.")
    if not bool(summary.get("pass")) or str(summary.get("status", "")) != "pass":
        raise ReconstructionValidationError("E2 requires E1 status=pass.")
    decision = summary.get("decision")
    if not isinstance(decision, dict) or not bool(decision.get("authorizes_next_fusion_refinement_milestone")):
        raise ReconstructionValidationError("E2 requires E1 refinement authorization.")
    fusion = summary.get("fusion")
    if not isinstance(fusion, dict) or not bool(fusion.get("capacity_clean")):
        raise ReconstructionValidationError("E2 requires capacity-clean E1 fusion evidence.")
    frames = summary.get("frames")
    if not isinstance(frames, list) or len(frames) != 10:
        raise ReconstructionValidationError("E2 requires exactly ten E1 source-view records.")
    authority = summary.get("authority")
    if not isinstance(authority, dict):
        raise ReconstructionValidationError("E2 E1 authority evidence is missing.")
    if bool(authority.get("metric_scale_claimed")) or bool(authority.get("measurement_authoritative")):
        raise ReconstructionValidationError("E2 does not accept silently promoted E1 metric/measurement authority.")


def _voxel_indices(points: object, voxel_size: float) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    voxel = float(voxel_size)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ReconstructionValidationError("E2 voxelization requires Nx3 points.")
    if not isfinite(voxel) or voxel <= 0.0:
        raise ReconstructionValidationError("E2 voxel size must be finite and positive.")
    if not np.isfinite(xyz).all():
        raise ReconstructionValidationError("E2 points must be finite.")
    return np.floor(xyz / voxel).astype(np.int64)


def _aggregate_frame_voxels(
    points: object,
    colors: object,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(points, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != (xyz.shape[0], 3):
        raise ReconstructionValidationError("E2 per-view cloud requires matching Nx3 XYZ/RGB arrays.")
    if len(xyz) == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    indices = _voxel_indices(xyz, voxel_size)
    unique, inverse = np.unique(indices, axis=0, return_inverse=True)
    count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    centroids = np.column_stack(
        [np.bincount(inverse, weights=xyz[:, axis], minlength=len(unique)) / count for axis in range(3)]
    )
    mean_colors = np.column_stack(
        [np.bincount(inverse, weights=rgb[:, axis].astype(np.float64), minlength=len(unique)) / count for axis in range(3)]
    )
    mean_colors = np.clip(np.rint(mean_colors), 0, 255).astype(np.uint8)
    return unique, centroids, mean_colors


def _reconstruct_e1_input_points(
    canonical_points: object,
    aligned_depth_reconstruction_units: object,
    T_reconstruction_from_camera: object,
) -> np.ndarray:
    """Reproduce the XYZ that E1 actually voxelized from a saved D4/D4.3 cloud.

    D4/C projected SOURCE_MODEL_UNITS depth after promoting the source float32
    samples to float64 and then applying the alignment scale.  The saved
    ``aligned_depth_reconstruction_units`` array, however, was produced by a
    float32 array multiplied by the Python scale before being written, and E1
    deliberately loaded that saved aligned depth as float32.  A point that lies
    extremely close to a voxel boundary can therefore land in an adjacent
    voxel between the D4 canonical XYZ artifact and E1's real fusion input.

    E2 must reconstruct E1's actual input geometry, not re-voxelize the D4
    validation XYZ verbatim.  The saved canonical point already carries the
    exact undistorted camera ray.  We transform it back to camera coordinates,
    preserve that ray, substitute the float32-quantized aligned depth consumed
    by E1, and transform it back to reconstruction coordinates.
    """

    points = np.asarray(canonical_points, dtype=np.float64)
    depth_saved = np.asarray(aligned_depth_reconstruction_units)
    transform = np.asarray(T_reconstruction_from_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ReconstructionValidationError("E2 canonical source points must be Nx3.")
    if depth_saved.ndim != 2:
        raise ReconstructionValidationError("E2 saved aligned depth must be HxW.")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ReconstructionValidationError("E2 source pose must be a finite 4x4 transform.")
    if not np.allclose(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ReconstructionValidationError("E2 source pose must be homogeneous.")

    # Match E1 exactly: it loads the saved aligned-depth artifact as float32.
    e1_depth = np.asarray(depth_saved, dtype=np.float32).astype(np.float64)
    valid = np.isfinite(e1_depth) & (e1_depth > 0.0)
    target_depth = e1_depth[valid]
    if len(target_depth) != len(points):
        raise ReconstructionValidationError(
            "E2 saved canonical point count does not match the finite-positive E1 aligned-depth samples."
        )
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ReconstructionValidationError("E2 source pose rotation must be orthonormal.")

    points_camera = (points - translation[None, :]) @ rotation
    source_depth = points_camera[:, 2]
    if not np.isfinite(source_depth).all() or np.any(source_depth <= 0.0):
        raise ReconstructionValidationError("E2 canonical source points must remain in front of the camera.")

    ray_scale = target_depth / source_depth
    reconstructed_camera = points_camera * ray_scale[:, None]
    reconstructed = reconstructed_camera @ rotation.T + translation[None, :]
    if not np.isfinite(reconstructed).all():
        raise ReconstructionValidationError("E2 reconstructed E1 input points must be finite.")
    return np.ascontiguousarray(reconstructed, dtype=np.float64)


def _aggregate_saved_e1_source_frame_voxels(
    *,
    canonical_points: object,
    colors: object,
    aligned_depth_reconstruction_units: object,
    T_reconstruction_from_camera: object,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e1_points = _reconstruct_e1_input_points(
        canonical_points,
        aligned_depth_reconstruction_units,
        T_reconstruction_from_camera,
    )
    return _aggregate_frame_voxels(e1_points, colors, voxel_size)


def _robust_observation_statistics(
    observations: object,
    colors: object,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    xyz = np.asarray(observations, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != (xyz.shape[0], 3) or len(xyz) < 2:
        raise ReconstructionValidationError("E2 robust voxel statistics require at least two Nx3 view observations.")
    center = np.median(xyz, axis=0)
    color = np.clip(np.rint(np.median(rgb, axis=0)), 0, 255).astype(np.uint8)
    distances = np.linalg.norm(xyz - center[None, :], axis=1)
    median = float(np.median(distances))
    p90 = float(np.quantile(distances, 0.90))
    maximum = float(np.max(distances))
    voxel = float(voxel_size)
    return center, color, median / voxel, p90 / voxel, maximum / voxel


def _topology_metrics(
    voxel_keys: np.ndarray,
    support_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    keys = np.asarray(voxel_keys, dtype=np.int64)
    counts = np.asarray(support_counts, dtype=np.int64)
    if keys.ndim != 2 or keys.shape[1] != 3 or counts.shape != (len(keys),):
        raise ReconstructionValidationError("E2 topology requires aligned voxel keys/support counts.")
    key_to_index = {tuple(map(int, key)): index for index, key in enumerate(keys)}
    n26 = np.zeros(len(keys), dtype=np.int32)
    nr2 = np.zeros(len(keys), dtype=np.int32)
    high26 = np.zeros(len(keys), dtype=np.int32)
    for index, key_arr in enumerate(keys):
        key = tuple(map(int, key_arr))
        for delta in _NEIGHBOR26:
            neighbor = (key[0] + delta[0], key[1] + delta[1], key[2] + delta[2])
            other = key_to_index.get(neighbor)
            if other is not None:
                n26[index] += 1
                if counts[other] >= 3:
                    high26[index] += 1
        for delta in _NEIGHBOR_R2:
            neighbor = (key[0] + delta[0], key[1] + delta[1], key[2] + delta[2])
            if neighbor in key_to_index:
                nr2[index] += 1

    labels = np.full(len(keys), -1, dtype=np.int32)
    component_sizes: list[int] = []
    component_id = 0
    for start in range(len(keys)):
        if labels[start] >= 0:
            continue
        labels[start] = component_id
        queue: deque[int] = deque([start])
        members = 0
        while queue:
            current = queue.popleft()
            members += 1
            key = tuple(map(int, keys[current]))
            for delta in _NEIGHBOR26:
                neighbor = (key[0] + delta[0], key[1] + delta[1], key[2] + delta[2])
                other = key_to_index.get(neighbor)
                if other is not None and labels[other] < 0:
                    labels[other] = component_id
                    queue.append(other)
        component_sizes.append(members)
        component_id += 1
    sizes = np.asarray(component_sizes, dtype=np.int32)
    per_voxel_size = sizes[labels] if len(keys) else np.empty((0,), dtype=np.int32)
    largest = int(np.max(sizes)) if len(sizes) else 0
    return n26, nr2, high26, per_voxel_size, largest, int(len(component_sizes))


def _refinement_confidence(
    support_counts: object,
    p90_dispersion_voxels: object,
    radius2_neighbors: object,
    high_neighbors: object,
) -> np.ndarray:
    support = np.asarray(support_counts, dtype=np.float64)
    dispersion = np.asarray(p90_dispersion_voxels, dtype=np.float64)
    radius2 = np.asarray(radius2_neighbors, dtype=np.float64)
    high = np.asarray(high_neighbors, dtype=np.float64)
    if not (support.shape == dispersion.shape == radius2.shape == high.shape):
        raise ReconstructionValidationError("E2 confidence inputs must have matching shapes.")
    support_score = np.clip((support - 1.0) / 4.0, 0.0, 1.0)
    dispersion_score = np.clip(1.0 - dispersion / 0.95, 0.0, 1.0)
    density_score = np.clip(radius2 / 24.0, 0.0, 1.0)
    strong_neighbor_score = np.clip(high / 6.0, 0.0, 1.0)
    return np.clip(
        0.40 * support_score
        + 0.35 * dispersion_score
        + 0.15 * density_score
        + 0.10 * strong_neighbor_score,
        0.0,
        1.0,
    ).astype(np.float32)


def _classify_supported_voxels(
    support_counts: object,
    p90_dispersion_voxels: object,
    radius2_neighbors: object,
    high_neighbors: object,
    component_sizes: object,
    *,
    minimum_component_size: int,
    core_max_dispersion_voxels: float,
    support2_core_max_dispersion_voxels: float,
    extended_max_dispersion_voxels: float,
    support2_core_min_high_neighbors: int,
    support2_core_min_radius2_neighbors: int,
    extended_min_radius2_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(support_counts, dtype=np.int64)
    dispersion = np.asarray(p90_dispersion_voxels, dtype=np.float64)
    radius2 = np.asarray(radius2_neighbors, dtype=np.int64)
    high = np.asarray(high_neighbors, dtype=np.int64)
    components = np.asarray(component_sizes, dtype=np.int64)
    if not (support.shape == dispersion.shape == radius2.shape == high.shape == components.shape):
        raise ReconstructionValidationError("E2 classification inputs must have matching shapes.")
    connected = components >= int(minimum_component_size)
    core_high_support = (
        (support >= 3)
        & (dispersion <= float(core_max_dispersion_voxels))
        & connected
        & (radius2 >= max(2, int(extended_min_radius2_neighbors) // 2))
    )
    core_support2 = (
        (support == 2)
        & (dispersion <= float(support2_core_max_dispersion_voxels))
        & connected
        & (high >= int(support2_core_min_high_neighbors))
        & (radius2 >= int(support2_core_min_radius2_neighbors))
    )
    core = core_high_support | core_support2
    extended = (
        (support >= 2)
        & (dispersion <= float(extended_max_dispersion_voxels))
        & connected
        & (radius2 >= int(extended_min_radius2_neighbors))
    ) | core
    return core, extended


def _attached_single_view_fringe(
    all_voxel_keys: np.ndarray,
    all_support_counts: np.ndarray,
    extended_voxel_keys: np.ndarray,
    *,
    minimum_extended_neighbors: int,
) -> np.ndarray:
    all_keys = np.asarray(all_voxel_keys, dtype=np.int64)
    counts = np.asarray(all_support_counts, dtype=np.int64)
    extended_set = {tuple(map(int, key)) for key in np.asarray(extended_voxel_keys, dtype=np.int64)}
    mask = np.zeros(len(all_keys), dtype=bool)
    if int(minimum_extended_neighbors) <= 0:
        return counts == 1
    for index in np.flatnonzero(counts == 1):
        key = tuple(map(int, all_keys[index]))
        neighbors = 0
        for delta in _NEIGHBOR26:
            other = (key[0] + delta[0], key[1] + delta[1], key[2] + delta[2])
            if other in extended_set:
                neighbors += 1
                if neighbors >= int(minimum_extended_neighbors):
                    mask[index] = True
                    break
    return mask


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    xyz = np.asarray(points, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != (xyz.shape[0], 3):
        raise ReconstructionValidationError("E2 PLY requires Nx3 XYZ and Nx3 RGB arrays.")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    vertices = np.empty(len(xyz), dtype=dtype)
    if len(xyz):
        vertices["x"] = xyz[:, 0].astype(np.float32)
        vertices["y"] = xyz[:, 1].astype(np.float32)
        vertices["z"] = xyz[:, 2].astype(np.float32)
        vertices["r"] = rgb[:, 0]
        vertices["g"] = rgb[:, 1]
        vertices["b"] = rgb[:, 2]
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(vertices.tobytes())
    return path


def _save_refined_npz(
    path: Path,
    *,
    points: np.ndarray,
    colors: np.ndarray,
    support: np.ndarray,
    confidence: np.ndarray,
    median_dispersion: np.ndarray,
    p90_dispersion: np.ndarray,
    radius2_neighbors: np.ndarray,
    high_neighbors: np.ndarray,
    component_sizes: np.ndarray,
    class_code: np.ndarray,
    voxel_size: float,
    orders: np.ndarray,
    keyframes: np.ndarray,
    sparse_points: np.ndarray,
    sparse_colors: np.ndarray,
) -> Path:
    np.savez_compressed(
        path,
        points_reconstruction=np.asarray(points, dtype=np.float64),
        colors_rgb=np.asarray(colors, dtype=np.uint8),
        view_observation_counts=np.asarray(support, dtype=np.uint32),
        refinement_confidence=np.asarray(confidence, dtype=np.float32),
        median_dispersion_voxels=np.asarray(median_dispersion, dtype=np.float32),
        p90_dispersion_voxels=np.asarray(p90_dispersion, dtype=np.float32),
        radius2_neighbor_count=np.asarray(radius2_neighbors, dtype=np.int32),
        high_support_neighbor_count=np.asarray(high_neighbors, dtype=np.int32),
        component_size=np.asarray(component_sizes, dtype=np.int32),
        class_code=np.asarray(class_code, dtype=np.uint8),
        voxel_size_reconstruction_units=np.asarray(float(voxel_size), dtype=np.float64),
        selected_workspace_orders=np.asarray(orders, dtype=np.int32),
        selected_keyframe_ids=np.asarray(keyframes, dtype="U160"),
        sparse_landmarks_reconstruction=np.asarray(sparse_points, dtype=np.float64),
        sparse_landmark_colors_rgb=np.asarray(sparse_colors, dtype=np.uint8),
    )
    return path


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"p50": None, "p75": None, "p90": None, "p95": None}
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def main() -> int:
    args = _parse_args()
    e1_dir = args.e1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    e1_summary = _read_json(e1_dir / "e1_summary.json")
    _validate_e1_summary(e1_summary)
    all_path = Path(e1_summary["artifacts"]["all_npz"]).expanduser().resolve()
    with np.load(all_path, allow_pickle=False) as payload:
        e1_points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        e1_colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        e1_counts = np.asarray(payload["view_observation_counts"], dtype=np.uint32)
        voxel_size = float(np.asarray(payload["voxel_size_reconstruction_units"]).item())
        orders = np.asarray(payload["selected_workspace_orders"], dtype=np.int32)
        keyframes = np.asarray(payload["selected_keyframe_ids"], dtype="U160")
        sparse_points = np.asarray(payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64)
        sparse_colors = np.asarray(payload.get("sparse_landmark_colors_rgb", np.empty((0, 3))), dtype=np.uint8)

    all_keys = _voxel_indices(e1_points, voxel_size)
    if len({tuple(map(int, key)) for key in all_keys}) != len(all_keys):
        raise ReconstructionValidationError("E2 expected one unique E1 centroid per voxel key.")
    key_to_all = {tuple(map(int, key)): index for index, key in enumerate(all_keys)}
    recomputed_counts = np.zeros(len(all_keys), dtype=np.uint32)
    supported_global = np.flatnonzero(e1_counts >= 2)
    global_to_supported = {int(global_index): local for local, global_index in enumerate(supported_global)}
    obs_points: list[list[np.ndarray]] = [[] for _ in supported_global]
    obs_colors: list[list[np.ndarray]] = [[] for _ in supported_global]
    source_frame_unique_voxels: list[int] = []

    print("E2: reconstructing per-view voxel observations from saved canonical D4.3/E1 source clouds:", flush=True)
    for frame in e1_summary["frames"]:
        cloud_path = Path(frame["cloud_npz"]).expanduser().resolve()
        with np.load(cloud_path, allow_pickle=False) as payload:
            points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
            if "aligned_depth_reconstruction_units" not in payload or "T_reconstruction_from_camera" not in payload:
                raise ReconstructionValidationError(
                    "E2 source cloud is missing aligned-depth/pose evidence required to reconstruct E1 voxel inputs."
                )
            aligned_depth = np.asarray(payload["aligned_depth_reconstruction_units"])
            pose = np.asarray(payload["T_reconstruction_from_camera"], dtype=np.float64)
        keys, centroids, mean_colors = _aggregate_saved_e1_source_frame_voxels(
            canonical_points=points,
            colors=colors,
            aligned_depth_reconstruction_units=aligned_depth,
            T_reconstruction_from_camera=pose,
            voxel_size=voxel_size,
        )
        source_frame_unique_voxels.append(int(len(keys)))
        matched = 0
        for key_arr, centroid, mean_color in zip(keys, centroids, mean_colors, strict=True):
            global_index = key_to_all.get(tuple(map(int, key_arr)))
            if global_index is None:
                raise ReconstructionValidationError(
                    "E2 source per-view voxel is missing from capacity-clean E1 all-voxel map."
                )
            recomputed_counts[global_index] += 1
            local = global_to_supported.get(global_index)
            if local is not None:
                obs_points[local].append(np.asarray(centroid, dtype=np.float64))
                obs_colors[local].append(np.asarray(mean_color, dtype=np.uint8))
            matched += 1
        print(
            f"  order={int(frame['workspace_order'])} unique_frame_voxels={len(keys)} matched={matched}",
            flush=True,
        )

    accounting_mismatch = recomputed_counts != e1_counts
    mismatch_count = int(np.count_nonzero(accounting_mismatch))
    max_count_delta = int(np.max(np.abs(recomputed_counts.astype(np.int64) - e1_counts.astype(np.int64))))
    if mismatch_count:
        raise ReconstructionValidationError(
            f"E2 exact E1 observation-accounting reconstruction failed for {mismatch_count} voxels."
        )

    supported_keys = all_keys[supported_global]
    supported_counts = e1_counts[supported_global].astype(np.int64)
    refined_points = np.empty((len(supported_global), 3), dtype=np.float64)
    refined_colors = np.empty((len(supported_global), 3), dtype=np.uint8)
    median_dispersion = np.empty(len(supported_global), dtype=np.float64)
    p90_dispersion = np.empty(len(supported_global), dtype=np.float64)
    maximum_dispersion = np.empty(len(supported_global), dtype=np.float64)
    for local, count in enumerate(supported_counts):
        if len(obs_points[local]) != int(count):
            raise ReconstructionValidationError("E2 stored per-view observations do not match E1 support count.")
        center, color, median, p90, maximum = _robust_observation_statistics(
            np.vstack(obs_points[local]), np.vstack(obs_colors[local]), voxel_size
        )
        refined_points[local] = center
        refined_colors[local] = color
        median_dispersion[local] = median
        p90_dispersion[local] = p90
        maximum_dispersion[local] = maximum

    neighbor26, radius2, high26, component_size, largest_component, component_count = _topology_metrics(
        supported_keys, supported_counts
    )
    confidence = _refinement_confidence(supported_counts, p90_dispersion, radius2, high26)
    core_mask, extended_mask = _classify_supported_voxels(
        supported_counts,
        p90_dispersion,
        radius2,
        high26,
        component_size,
        minimum_component_size=int(args.minimum_component_size),
        core_max_dispersion_voxels=float(args.core_max_dispersion_voxels),
        support2_core_max_dispersion_voxels=float(args.support2_core_max_dispersion_voxels),
        extended_max_dispersion_voxels=float(args.extended_max_dispersion_voxels),
        support2_core_min_high_neighbors=int(args.support2_core_min_high_neighbors),
        support2_core_min_radius2_neighbors=int(args.support2_core_min_radius2_neighbors),
        extended_min_radius2_neighbors=int(args.extended_min_radius2_neighbors),
    )
    rejected_mask = ~extended_mask
    fringe_mask_all = _attached_single_view_fringe(
        all_keys,
        e1_counts,
        supported_keys[extended_mask],
        minimum_extended_neighbors=int(args.fringe_min_extended_neighbors),
    )

    class_supported = np.where(core_mask, 3, np.where(extended_mask, 2, 0)).astype(np.uint8)
    core_indices = np.flatnonzero(core_mask)
    extended_indices = np.flatnonzero(extended_mask)
    rejected_indices = np.flatnonzero(rejected_mask)
    fringe_indices = np.flatnonzero(fringe_mask_all)

    core_npz = _save_refined_npz(
        output_dir / "e2_consistency_core.npz",
        points=refined_points[core_indices], colors=refined_colors[core_indices],
        support=supported_counts[core_indices], confidence=confidence[core_indices],
        median_dispersion=median_dispersion[core_indices], p90_dispersion=p90_dispersion[core_indices],
        radius2_neighbors=radius2[core_indices], high_neighbors=high26[core_indices],
        component_sizes=component_size[core_indices], class_code=class_supported[core_indices],
        voxel_size=voxel_size, orders=orders, keyframes=keyframes,
        sparse_points=sparse_points, sparse_colors=sparse_colors,
    )
    extended_npz = _save_refined_npz(
        output_dir / "e2_consistency_extended.npz",
        points=refined_points[extended_indices], colors=refined_colors[extended_indices],
        support=supported_counts[extended_indices], confidence=confidence[extended_indices],
        median_dispersion=median_dispersion[extended_indices], p90_dispersion=p90_dispersion[extended_indices],
        radius2_neighbors=radius2[extended_indices], high_neighbors=high26[extended_indices],
        component_sizes=component_size[extended_indices], class_code=class_supported[extended_indices],
        voxel_size=voxel_size, orders=orders, keyframes=keyframes,
        sparse_points=sparse_points, sparse_colors=sparse_colors,
    )
    rejected_npz = _save_refined_npz(
        output_dir / "e2_rejected_supported_diagnostic.npz",
        points=refined_points[rejected_indices], colors=refined_colors[rejected_indices],
        support=supported_counts[rejected_indices], confidence=confidence[rejected_indices],
        median_dispersion=median_dispersion[rejected_indices], p90_dispersion=p90_dispersion[rejected_indices],
        radius2_neighbors=radius2[rejected_indices], high_neighbors=high26[rejected_indices],
        component_sizes=component_size[rejected_indices], class_code=class_supported[rejected_indices],
        voxel_size=voxel_size, orders=orders, keyframes=keyframes,
        sparse_points=sparse_points, sparse_colors=sparse_colors,
    )

    # Single-view fringe is diagnostic coverage only.  It carries no dispersion estimate.
    fringe_points = e1_points[fringe_indices]
    fringe_colors = e1_colors[fringe_indices]
    fringe_npz = _save_refined_npz(
        output_dir / "e2_attached_single_view_fringe_diagnostic.npz",
        points=fringe_points, colors=fringe_colors,
        support=e1_counts[fringe_indices], confidence=np.full(len(fringe_indices), 0.15, dtype=np.float32),
        median_dispersion=np.full(len(fringe_indices), np.nan, dtype=np.float32),
        p90_dispersion=np.full(len(fringe_indices), np.nan, dtype=np.float32),
        radius2_neighbors=np.zeros(len(fringe_indices), dtype=np.int32),
        high_neighbors=np.zeros(len(fringe_indices), dtype=np.int32),
        component_sizes=np.zeros(len(fringe_indices), dtype=np.int32),
        class_code=np.ones(len(fringe_indices), dtype=np.uint8),
        voxel_size=voxel_size, orders=orders, keyframes=keyframes,
        sparse_points=sparse_points, sparse_colors=sparse_colors,
    )
    coverage_points = np.vstack([refined_points[extended_indices], fringe_points]) if len(fringe_points) else refined_points[extended_indices].copy()
    coverage_colors = np.vstack([refined_colors[extended_indices], fringe_colors]) if len(fringe_colors) else refined_colors[extended_indices].copy()
    coverage_support = np.concatenate([supported_counts[extended_indices], e1_counts[fringe_indices]]) if len(fringe_indices) else supported_counts[extended_indices].copy()
    coverage_conf = np.concatenate([confidence[extended_indices], np.full(len(fringe_indices), 0.15, dtype=np.float32)]) if len(fringe_indices) else confidence[extended_indices].copy()
    coverage_median = np.concatenate([median_dispersion[extended_indices], np.full(len(fringe_indices), np.nan)]) if len(fringe_indices) else median_dispersion[extended_indices].copy()
    coverage_p90 = np.concatenate([p90_dispersion[extended_indices], np.full(len(fringe_indices), np.nan)]) if len(fringe_indices) else p90_dispersion[extended_indices].copy()
    coverage_r2 = np.concatenate([radius2[extended_indices], np.zeros(len(fringe_indices), dtype=np.int32)]) if len(fringe_indices) else radius2[extended_indices].copy()
    coverage_high = np.concatenate([high26[extended_indices], np.zeros(len(fringe_indices), dtype=np.int32)]) if len(fringe_indices) else high26[extended_indices].copy()
    coverage_comp = np.concatenate([component_size[extended_indices], np.zeros(len(fringe_indices), dtype=np.int32)]) if len(fringe_indices) else component_size[extended_indices].copy()
    coverage_class = np.concatenate([class_supported[extended_indices], np.ones(len(fringe_indices), dtype=np.uint8)]) if len(fringe_indices) else class_supported[extended_indices].copy()
    coverage_npz = _save_refined_npz(
        output_dir / "e2_coverage_candidate.npz",
        points=coverage_points, colors=coverage_colors, support=coverage_support,
        confidence=coverage_conf, median_dispersion=coverage_median, p90_dispersion=coverage_p90,
        radius2_neighbors=coverage_r2, high_neighbors=coverage_high,
        component_sizes=coverage_comp, class_code=coverage_class,
        voxel_size=voxel_size, orders=orders, keyframes=keyframes,
        sparse_points=sparse_points, sparse_colors=sparse_colors,
    )

    core_ply = _write_binary_ply(output_dir / "e2_consistency_core.ply", refined_points[core_indices], refined_colors[core_indices])
    extended_ply = _write_binary_ply(output_dir / "e2_consistency_extended.ply", refined_points[extended_indices], refined_colors[extended_indices])

    core_count = int(np.count_nonzero(core_mask))
    extended_count = int(np.count_nonzero(extended_mask))
    rejected_count = int(np.count_nonzero(rejected_mask))
    fringe_count = int(np.count_nonzero(fringe_mask_all))
    source_supported_count = int(len(supported_global))
    structural_pass = (
        mismatch_count == 0
        and max_count_delta == 0
        and core_count >= int(args.minimum_core_voxels)
        and extended_count >= core_count
        and largest_component >= int(args.minimum_component_size)
        and not bool(e1_summary["authority"]["metric_scale_claimed"])
        and not bool(e1_summary["authority"]["measurement_authoritative"])
    )
    status = "pass" if structural_pass else "degraded"

    summary = {
        "phase": "E2",
        "pass": structural_pass,
        "status": status,
        "source_e1_dir": str(e1_dir),
        "source_e1_pass": True,
        "selected_workspace_orders": orders.tolist(),
        "selected_keyframe_ids": keyframes.tolist(),
        "voxel_size_reconstruction_units": voxel_size,
        "observation_accounting": {
            "exact_match": mismatch_count == 0,
            "mismatched_voxel_count": mismatch_count,
            "maximum_count_delta": max_count_delta,
            "frame_unique_voxel_counts": source_frame_unique_voxels,
            "recomputed_frame_voxel_observations": int(sum(source_frame_unique_voxels)),
            "source_e1_frame_voxel_observations": int(e1_summary["fusion"]["result"]["statistics"]["frame_voxel_observations"]),
        },
        "dispersion": {
            "definition": "robust per-view voxel-centroid distance to coordinate-wise median, normalized by E1 voxel edge length",
            "median_dispersion_voxel_quantiles": _quantiles(median_dispersion),
            "p90_dispersion_voxel_quantiles": _quantiles(p90_dispersion),
            "maximum_dispersion_voxel_quantiles": _quantiles(maximum_dispersion),
        },
        "topology": {
            "component_count": component_count,
            "largest_supported_component_voxels": largest_component,
            "neighbor26_quantiles": _quantiles(neighbor26.astype(np.float64)),
            "radius2_neighbor_quantiles": _quantiles(radius2.astype(np.float64)),
            "high_support_neighbor26_quantiles": _quantiles(high26.astype(np.float64)),
        },
        "classification": {
            "source_all_voxels": int(len(e1_counts)),
            "source_supported_voxels": source_supported_count,
            "core_voxels": core_count,
            "extended_voxels": extended_count,
            "rejected_supported_voxels": rejected_count,
            "attached_single_view_fringe_voxels": fringe_count,
            "coverage_candidate_voxels": int(len(coverage_points)),
            "core_fraction_of_supported": float(core_count / source_supported_count) if source_supported_count else 0.0,
            "extended_fraction_of_supported": float(extended_count / source_supported_count) if source_supported_count else 0.0,
        },
        "policy": {
            "minimum_core_voxels": int(args.minimum_core_voxels),
            "minimum_component_size": int(args.minimum_component_size),
            "core_max_dispersion_voxels": float(args.core_max_dispersion_voxels),
            "support2_core_max_dispersion_voxels": float(args.support2_core_max_dispersion_voxels),
            "extended_max_dispersion_voxels": float(args.extended_max_dispersion_voxels),
            "support2_core_min_high_neighbors": int(args.support2_core_min_high_neighbors),
            "support2_core_min_radius2_neighbors": int(args.support2_core_min_radius2_neighbors),
            "extended_min_radius2_neighbors": int(args.extended_min_radius2_neighbors),
            "fringe_min_extended_neighbors": int(args.fringe_min_extended_neighbors),
            "single_view_fringe_is_diagnostic_only": True,
        },
        "artifacts": {
            "core_npz": str(core_npz.resolve()),
            "core_ply": str(core_ply.resolve()),
            "extended_npz": str(extended_npz.resolve()),
            "extended_ply": str(extended_ply.resolve()),
            "rejected_supported_npz": str(rejected_npz.resolve()),
            "attached_single_view_fringe_npz": str(fringe_npz.resolve()),
            "coverage_candidate_npz": str(coverage_npz.resolve()),
        },
        "authority": {
            "spatial_voxel_fusion_source": "E1",
            "consistency_refinement_applied": True,
            "robust_per_view_centroid_estimation_applied": True,
            "local_connectivity_analysis_applied": True,
            "single_view_fringe_promoted_to_primary": False,
            "canonical_geometry_authoritative": False,
            "canonical_persistence_promoted": False,
            "measurement_authoritative": False,
            "metric_scale_claimed": False,
            "sfm_scale_status": "arbitrary",
            "sfm_world_alignment_status": "arbitrary",
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "note": (
                "E2 validation-only consistency refinement. Core/extended classes are reconstruction-confidence "
                "products in the arbitrary SfM gauge, not metric or measurement authority."
            ),
        },
        "decision": {
            "primary_output": "e2_consistency_extended.npz",
            "high_confidence_output": "e2_consistency_core.npz",
            "coverage_candidate_is_diagnostic_only": True,
            "authorizes_next_scene_fusion_expansion_milestone": structural_pass,
        },
    }
    summary_path = _write_json(output_dir / "e2_summary.json", summary)

    print(f"\nE2 consistency-aware fusion refinement: {status.upper()}", flush=True)
    print(
        f"source_all={len(e1_counts)} source_supported={source_supported_count} "
        f"core={core_count} extended={extended_count} rejected_supported={rejected_count} fringe={fringe_count}",
        flush=True,
    )
    print(
        f"accounting_exact={mismatch_count == 0} frame_voxel_observations={sum(source_frame_unique_voxels)} "
        f"largest_supported_component={largest_component}",
        flush=True,
    )
    print(
        f"p90_dispersion_voxels p50={summary['dispersion']['p90_dispersion_voxel_quantiles']['p50']:.4f} "
        f"p90={summary['dispersion']['p90_dispersion_voxel_quantiles']['p90']:.4f}",
        flush=True,
    )
    print("consistency_refinement_applied=True local_connectivity_analysis_applied=True", flush=True)
    print(
        "sfm_scale_status=arbitrary metric_scale_claimed=False canonical_geometry_authoritative=False "
        "measurement_authoritative=False certified_metrology=False surface_mesh_generated=False",
        flush=True,
    )
    print(f"extended_cloud={extended_npz}", flush=True)
    print(f"core_cloud={core_npz}", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0 if structural_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
