#!/usr/bin/env python3
"""Phase F formal confidence / consistency / artifact filtering for E3 geometry.

F consumes a passing E3 expanded fused scene.  It does not rerun PyCOLMAP or
DA3.  It reconstructs the exact per-view voxel observation accounting that E3
fed into DenseVoxelFusion, derives robust per-voxel dispersion and local
connectivity evidence, and applies the reusable production filtering contract
in ``core.reconstruction.dense_confidence_filtering``.

The result remains in the arbitrary offline-SfM gauge.  F qualifies a filtered
dense-geometry product for later canonical persistence (G); it does not itself
promote persistence, metric scale, measurement authority, or mesh authority.
"""
from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.dense_confidence_filtering import (  # noqa: E402
    DENSE_CONFIDENCE_FILTER_SCHEMA_VERSION,
    DenseArtifactReason,
    DenseConfidenceFilterPolicy,
    artifact_reason_histogram,
    filter_dense_supported_geometry,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from scripts.run_e2_consistency_aware_fusion_refinement import (  # noqa: E402
    _aggregate_saved_e1_source_frame_voxels as _aggregate_saved_fusion_source_frame_voxels,
    _attached_single_view_fringe,
    _quantiles,
    _robust_observation_statistics,
    _topology_metrics,
    _voxel_indices,
    _write_binary_ply,
)

_DEFAULT_E3_DIR = Path("/tmp/depth3d_e3_qualified_scene_fusion_expansion")
_DEFAULT_OUTPUT_DIR = Path("/tmp/depth3d_f_confidence_artifact_filtering")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3-dir", type=Path, default=_DEFAULT_E3_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-component-size", type=int, default=64)
    parser.add_argument("--minimum-core-voxels", type=int, default=5_000)
    parser.add_argument("--minimum-core-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-extended-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-largest-component-fraction", type=float, default=0.50)
    parser.add_argument("--core-max-dispersion-voxels", type=float, default=0.70)
    parser.add_argument("--support2-core-max-dispersion-voxels", type=float, default=0.45)
    parser.add_argument("--extended-max-dispersion-voxels", type=float, default=0.95)
    parser.add_argument("--support2-core-min-high-neighbors", type=int, default=2)
    parser.add_argument("--support2-core-min-radius2-neighbors", type=int, default=8)
    parser.add_argument("--extended-min-radius2-neighbors", type=int, default=6)
    parser.add_argument("--support2-extended-min-high-neighbors", type=int, default=1)
    parser.add_argument("--support2-extended-min-radius2-neighbors", type=int, default=8)
    parser.add_argument("--fringe-min-extended-neighbors", type=int, default=2)
    args = parser.parse_args()
    if args.minimum_component_size <= 0 or args.minimum_core_voxels <= 0:
        parser.error("minimum component/core voxel gates must be positive")
    for name in (
        "minimum_core_fraction",
        "minimum_extended_fraction",
        "minimum_largest_component_fraction",
    ):
        value = float(getattr(args, name))
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be finite in [0, 1]")
    if float(args.minimum_core_fraction) > float(args.minimum_extended_fraction):
        parser.error("minimum core fraction cannot exceed minimum extended fraction")
    if args.fringe_min_extended_neighbors < 0:
        parser.error("--fringe-min-extended-neighbors must be non-negative")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_e3_summary(summary: dict[str, Any]) -> None:
    if str(summary.get("phase", "")) != "E3":
        raise ReconstructionValidationError("F requires an E3 summary.")
    if not bool(summary.get("pass")) or str(summary.get("status", "")) != "pass":
        raise ReconstructionValidationError("F requires E3 status=pass.")
    decision = summary.get("decision")
    if not isinstance(decision, dict) or not bool(decision.get("authorizes_next_confidence_filtering_milestone")):
        raise ReconstructionValidationError("F requires explicit E3 authorization for confidence/artifact filtering.")
    fusion = summary.get("fusion")
    if not isinstance(fusion, dict) or not bool(fusion.get("capacity_clean")) or int(fusion.get("capacity_drops", -1)) != 0:
        raise ReconstructionValidationError("F requires capacity-clean E3 fusion evidence.")
    orders = summary.get("selected_workspace_orders")
    keyframes = summary.get("selected_keyframe_ids")
    frames = summary.get("frames")
    selected_count = int(summary.get("selected_view_count", 0))
    if not isinstance(orders, list) or not isinstance(keyframes, list) or not isinstance(frames, list):
        raise ReconstructionValidationError("F E3 selected-view provenance is incomplete.")
    if selected_count < 10 or not (len(orders) == len(keyframes) == len(frames) == selected_count):
        raise ReconstructionValidationError("F requires aligned E3 selected-view provenance.")
    if orders != sorted(orders) or len(set(int(v) for v in orders)) != len(orders):
        raise ReconstructionValidationError("F E3 workspace orders must be strictly increasing and unique.")
    authority = summary.get("authority")
    if not isinstance(authority, dict):
        raise ReconstructionValidationError("F requires E3 authority evidence.")
    if bool(authority.get("metric_scale_claimed")) or bool(authority.get("measurement_authoritative")):
        raise ReconstructionValidationError("F refuses silently promoted E3 metric/measurement authority.")
    if bool(authority.get("canonical_persistence_promoted")) or bool(authority.get("surface_mesh_generated")):
        raise ReconstructionValidationError("F must run before G persistence and J mesh generation.")


def _save_filtered_npz(
    path: Path,
    *,
    points: np.ndarray,
    colors: np.ndarray,
    support: np.ndarray,
    confidence: np.ndarray,
    median_dispersion: np.ndarray,
    p90_dispersion: np.ndarray,
    maximum_dispersion: np.ndarray,
    neighbor26: np.ndarray,
    radius2_neighbors: np.ndarray,
    high_neighbors: np.ndarray,
    component_sizes: np.ndarray,
    class_code: np.ndarray,
    artifact_reason_code: np.ndarray,
    voxel_size: float,
    orders: np.ndarray,
    keyframes: np.ndarray,
    sparse_points: np.ndarray,
    sparse_colors: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points_reconstruction=np.asarray(points, dtype=np.float64),
        colors_rgb=np.asarray(colors, dtype=np.uint8),
        view_observation_counts=np.asarray(support, dtype=np.uint32),
        filter_confidence=np.asarray(confidence, dtype=np.float32),
        median_dispersion_voxels=np.asarray(median_dispersion, dtype=np.float32),
        p90_dispersion_voxels=np.asarray(p90_dispersion, dtype=np.float32),
        maximum_dispersion_voxels=np.asarray(maximum_dispersion, dtype=np.float32),
        neighbor26_count=np.asarray(neighbor26, dtype=np.int32),
        radius2_neighbor_count=np.asarray(radius2_neighbors, dtype=np.int32),
        high_support_neighbor_count=np.asarray(high_neighbors, dtype=np.int32),
        component_size=np.asarray(component_sizes, dtype=np.int32),
        class_code=np.asarray(class_code, dtype=np.uint8),
        artifact_reason_code=np.asarray(artifact_reason_code, dtype=np.uint16),
        voxel_size_reconstruction_units=np.asarray(float(voxel_size), dtype=np.float64),
        selected_workspace_orders=np.asarray(orders, dtype=np.int32),
        selected_keyframe_ids=np.asarray(keyframes, dtype="U160"),
        sparse_landmarks_reconstruction=np.asarray(sparse_points, dtype=np.float64),
        sparse_landmark_colors_rgb=np.asarray(sparse_colors, dtype=np.uint8),
        schema_version=np.asarray(DENSE_CONFIDENCE_FILTER_SCHEMA_VERSION, dtype=np.int32),
    )
    return path


def _single_view_npz(
    path: Path,
    *,
    points: np.ndarray,
    colors: np.ndarray,
    class_code: int,
    artifact_reason: int,
    voxel_size: float,
    orders: np.ndarray,
    keyframes: np.ndarray,
    sparse_points: np.ndarray,
    sparse_colors: np.ndarray,
) -> Path:
    count = len(points)
    return _save_filtered_npz(
        path,
        points=points,
        colors=colors,
        support=np.ones(count, dtype=np.uint32),
        confidence=np.full(count, 0.10 if artifact_reason else 0.15, dtype=np.float32),
        median_dispersion=np.full(count, np.nan, dtype=np.float32),
        p90_dispersion=np.full(count, np.nan, dtype=np.float32),
        maximum_dispersion=np.full(count, np.nan, dtype=np.float32),
        neighbor26=np.zeros(count, dtype=np.int32),
        radius2_neighbors=np.zeros(count, dtype=np.int32),
        high_neighbors=np.zeros(count, dtype=np.int32),
        component_sizes=np.zeros(count, dtype=np.int32),
        class_code=np.full(count, int(class_code), dtype=np.uint8),
        artifact_reason_code=np.full(count, int(artifact_reason), dtype=np.uint16),
        voxel_size=voxel_size,
        orders=orders,
        keyframes=keyframes,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )


def main() -> int:
    args = _parse_args()
    e3_dir = args.e3_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    e3_summary = _read_json(e3_dir / "e3_summary.json")
    _validate_e3_summary(e3_summary)
    all_path = Path(str(e3_summary["artifacts"]["all_npz"])).expanduser().resolve()
    with np.load(all_path, allow_pickle=False) as payload:
        all_points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        all_colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        all_counts = np.asarray(payload["view_observation_counts"], dtype=np.uint32)
        voxel_size = float(np.asarray(payload["voxel_size_reconstruction_units"]).item())
        orders = np.asarray(payload["selected_workspace_orders"], dtype=np.int32)
        keyframes = np.asarray(payload["selected_keyframe_ids"], dtype="U160")
        sparse_points = np.asarray(payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64)
        sparse_colors = np.asarray(payload.get("sparse_landmark_colors_rgb", np.empty((0, 3))), dtype=np.uint8)

    selected_view_count = int(e3_summary["selected_view_count"])
    if len(orders) != selected_view_count or len(keyframes) != selected_view_count:
        raise ReconstructionValidationError("F E3 all-cloud provenance disagrees with E3 summary.")
    if not np.isclose(voxel_size, float(e3_summary["fusion"]["voxel_size_reconstruction_units"]), rtol=0.0, atol=1e-12):
        raise ReconstructionValidationError("F detected E3 voxel-size provenance drift.")
    if len(all_points) != int(e3_summary["fusion"]["all_voxel_count"]):
        raise ReconstructionValidationError("F E3 all-cloud voxel count disagrees with summary.")

    all_keys = _voxel_indices(all_points, voxel_size)
    if len({tuple(map(int, key)) for key in all_keys}) != len(all_keys):
        raise ReconstructionValidationError("F expected one unique E3 centroid per voxel key.")
    key_to_all = {tuple(map(int, key)): index for index, key in enumerate(all_keys)}
    recomputed_counts = np.zeros(len(all_keys), dtype=np.uint32)
    supported_global = np.flatnonzero(all_counts >= 2)
    global_to_supported = {int(global_index): local for local, global_index in enumerate(supported_global)}
    obs_points: list[list[np.ndarray]] = [[] for _ in supported_global]
    obs_colors: list[list[np.ndarray]] = [[] for _ in supported_global]
    frame_unique_counts: list[int] = []

    print("F: reconstructing exact E3 per-view voxel observations from saved selected-view clouds:", flush=True)
    for frame in e3_summary["frames"]:
        cloud_path = Path(str(frame["cloud_npz"])).expanduser().resolve()
        with np.load(cloud_path, allow_pickle=False) as payload:
            points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
            if "aligned_depth_reconstruction_units" not in payload or "T_reconstruction_from_camera" not in payload:
                raise ReconstructionValidationError(
                    "F E3 source cloud is missing aligned-depth/pose evidence required for exact fusion accounting."
                )
            aligned_depth = np.asarray(payload["aligned_depth_reconstruction_units"])
            pose = np.asarray(payload["T_reconstruction_from_camera"], dtype=np.float64)
        keys, centroids, mean_colors = _aggregate_saved_fusion_source_frame_voxels(
            canonical_points=points,
            colors=colors,
            aligned_depth_reconstruction_units=aligned_depth,
            T_reconstruction_from_camera=pose,
            voxel_size=voxel_size,
        )
        frame_unique_counts.append(int(len(keys)))
        matched = 0
        for key_arr, centroid, mean_color in zip(keys, centroids, mean_colors, strict=True):
            global_index = key_to_all.get(tuple(map(int, key_arr)))
            if global_index is None:
                raise ReconstructionValidationError(
                    "F source per-view voxel is missing from capacity-clean E3 all-voxel map."
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

    mismatch = recomputed_counts != all_counts
    mismatch_count = int(np.count_nonzero(mismatch))
    max_count_delta = int(np.max(np.abs(recomputed_counts.astype(np.int64) - all_counts.astype(np.int64))))
    if mismatch_count:
        raise ReconstructionValidationError(
            f"F exact E3 observation-accounting reconstruction failed for {mismatch_count} voxels."
        )

    supported_keys = all_keys[supported_global]
    supported_counts = all_counts[supported_global].astype(np.int64)
    refined_points = np.empty((len(supported_global), 3), dtype=np.float64)
    refined_colors = np.empty((len(supported_global), 3), dtype=np.uint8)
    median_dispersion = np.empty(len(supported_global), dtype=np.float64)
    p90_dispersion = np.empty(len(supported_global), dtype=np.float64)
    maximum_dispersion = np.empty(len(supported_global), dtype=np.float64)
    for local, count in enumerate(supported_counts):
        if len(obs_points[local]) != int(count):
            raise ReconstructionValidationError("F stored per-view observations do not match E3 support count.")
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
    policy = DenseConfidenceFilterPolicy(
        maximum_views=selected_view_count,
        minimum_component_size=int(args.minimum_component_size),
        core_max_dispersion_voxels=float(args.core_max_dispersion_voxels),
        support2_core_max_dispersion_voxels=float(args.support2_core_max_dispersion_voxels),
        extended_max_dispersion_voxels=float(args.extended_max_dispersion_voxels),
        support2_core_min_high_neighbors=int(args.support2_core_min_high_neighbors),
        support2_core_min_radius2_neighbors=int(args.support2_core_min_radius2_neighbors),
        extended_min_radius2_neighbors=int(args.extended_min_radius2_neighbors),
        support2_extended_min_high_neighbors=int(args.support2_extended_min_high_neighbors),
        support2_extended_min_radius2_neighbors=int(args.support2_extended_min_radius2_neighbors),
    )
    filtered = filter_dense_supported_geometry(
        supported_counts,
        p90_dispersion,
        radius2,
        high26,
        component_size,
        policy=policy,
    )

    core_idx = np.flatnonzero(filtered.core_mask)
    extended_idx = np.flatnonzero(filtered.extended_mask)
    rejected_idx = np.flatnonzero(filtered.rejected_mask)
    class_supported = np.where(filtered.core_mask, 3, np.where(filtered.extended_mask, 2, 0)).astype(np.uint8)

    fringe_mask_all = _attached_single_view_fringe(
        all_keys,
        all_counts,
        supported_keys[filtered.extended_mask],
        minimum_extended_neighbors=int(args.fringe_min_extended_neighbors),
    )
    single_view_mask = all_counts == 1
    detached_mask_all = single_view_mask & ~fringe_mask_all
    fringe_idx = np.flatnonzero(fringe_mask_all)
    detached_idx = np.flatnonzero(detached_mask_all)

    zero_reason = np.zeros(len(supported_global), dtype=np.uint16)
    core_npz = _save_filtered_npz(
        output_dir / "f_filtered_core.npz",
        points=refined_points[core_idx], colors=refined_colors[core_idx],
        support=supported_counts[core_idx], confidence=filtered.confidence[core_idx],
        median_dispersion=median_dispersion[core_idx], p90_dispersion=p90_dispersion[core_idx],
        maximum_dispersion=maximum_dispersion[core_idx], neighbor26=neighbor26[core_idx],
        radius2_neighbors=radius2[core_idx], high_neighbors=high26[core_idx],
        component_sizes=component_size[core_idx], class_code=class_supported[core_idx],
        artifact_reason_code=zero_reason[core_idx], voxel_size=voxel_size,
        orders=orders, keyframes=keyframes, sparse_points=sparse_points, sparse_colors=sparse_colors,
    )
    extended_npz = _save_filtered_npz(
        output_dir / "f_filtered_extended.npz",
        points=refined_points[extended_idx], colors=refined_colors[extended_idx],
        support=supported_counts[extended_idx], confidence=filtered.confidence[extended_idx],
        median_dispersion=median_dispersion[extended_idx], p90_dispersion=p90_dispersion[extended_idx],
        maximum_dispersion=maximum_dispersion[extended_idx], neighbor26=neighbor26[extended_idx],
        radius2_neighbors=radius2[extended_idx], high_neighbors=high26[extended_idx],
        component_sizes=component_size[extended_idx], class_code=class_supported[extended_idx],
        artifact_reason_code=zero_reason[extended_idx], voxel_size=voxel_size,
        orders=orders, keyframes=keyframes, sparse_points=sparse_points, sparse_colors=sparse_colors,
    )
    rejected_npz = _save_filtered_npz(
        output_dir / "f_rejected_supported_artifacts.npz",
        points=refined_points[rejected_idx], colors=refined_colors[rejected_idx],
        support=supported_counts[rejected_idx], confidence=filtered.confidence[rejected_idx],
        median_dispersion=median_dispersion[rejected_idx], p90_dispersion=p90_dispersion[rejected_idx],
        maximum_dispersion=maximum_dispersion[rejected_idx], neighbor26=neighbor26[rejected_idx],
        radius2_neighbors=radius2[rejected_idx], high_neighbors=high26[rejected_idx],
        component_sizes=component_size[rejected_idx], class_code=class_supported[rejected_idx],
        artifact_reason_code=filtered.artifact_reason_code[rejected_idx], voxel_size=voxel_size,
        orders=orders, keyframes=keyframes, sparse_points=sparse_points, sparse_colors=sparse_colors,
    )
    fringe_npz = _single_view_npz(
        output_dir / "f_attached_single_view_fringe_diagnostic.npz",
        points=all_points[fringe_idx], colors=all_colors[fringe_idx], class_code=1,
        artifact_reason=0, voxel_size=voxel_size, orders=orders, keyframes=keyframes,
        sparse_points=sparse_points, sparse_colors=sparse_colors,
    )
    detached_npz = _single_view_npz(
        output_dir / "f_detached_single_view_artifacts_diagnostic.npz",
        points=all_points[detached_idx], colors=all_colors[detached_idx], class_code=0,
        artifact_reason=int(DenseArtifactReason.SINGLE_VIEW_DETACHED), voxel_size=voxel_size,
        orders=orders, keyframes=keyframes, sparse_points=sparse_points, sparse_colors=sparse_colors,
    )

    coverage_points = np.vstack([refined_points[extended_idx], all_points[fringe_idx]]) if len(fringe_idx) else refined_points[extended_idx].copy()
    coverage_colors = np.vstack([refined_colors[extended_idx], all_colors[fringe_idx]]) if len(fringe_idx) else refined_colors[extended_idx].copy()
    coverage_support = np.concatenate([supported_counts[extended_idx], np.ones(len(fringe_idx), dtype=np.int64)]) if len(fringe_idx) else supported_counts[extended_idx].copy()
    coverage_conf = np.concatenate([filtered.confidence[extended_idx], np.full(len(fringe_idx), 0.15, dtype=np.float32)]) if len(fringe_idx) else filtered.confidence[extended_idx].copy()
    coverage_nan = np.concatenate([p90_dispersion[extended_idx], np.full(len(fringe_idx), np.nan)]) if len(fringe_idx) else p90_dispersion[extended_idx].copy()
    coverage_class = np.concatenate([class_supported[extended_idx], np.ones(len(fringe_idx), dtype=np.uint8)]) if len(fringe_idx) else class_supported[extended_idx].copy()
    coverage_npz = _save_filtered_npz(
        output_dir / "f_coverage_candidate_diagnostic.npz",
        points=coverage_points, colors=coverage_colors, support=coverage_support,
        confidence=coverage_conf, median_dispersion=coverage_nan, p90_dispersion=coverage_nan,
        maximum_dispersion=coverage_nan, neighbor26=np.zeros(len(coverage_points), dtype=np.int32),
        radius2_neighbors=np.zeros(len(coverage_points), dtype=np.int32),
        high_neighbors=np.zeros(len(coverage_points), dtype=np.int32),
        component_sizes=np.zeros(len(coverage_points), dtype=np.int32), class_code=coverage_class,
        artifact_reason_code=np.zeros(len(coverage_points), dtype=np.uint16), voxel_size=voxel_size,
        orders=orders, keyframes=keyframes, sparse_points=sparse_points, sparse_colors=sparse_colors,
    )

    core_ply = _write_binary_ply(output_dir / "f_filtered_core.ply", refined_points[core_idx], refined_colors[core_idx])
    extended_ply = _write_binary_ply(output_dir / "f_filtered_extended.ply", refined_points[extended_idx], refined_colors[extended_idx])

    source_supported = int(len(supported_global))
    core_count = int(len(core_idx))
    extended_count = int(len(extended_idx))
    rejected_count = int(len(rejected_idx))
    fringe_count = int(len(fringe_idx))
    detached_count = int(len(detached_idx))
    core_fraction = float(core_count / source_supported) if source_supported else 0.0
    extended_fraction = float(extended_count / source_supported) if source_supported else 0.0
    largest_component_fraction = float(largest_component / source_supported) if source_supported else 0.0
    reason_hist = artifact_reason_histogram(filtered.artifact_reason_code[rejected_idx])
    reason_hist["single_view_detached"] = detached_count

    pass_conditions = {
        "exact_observation_accounting": mismatch_count == 0 and max_count_delta == 0,
        "minimum_core_voxels": core_count >= int(args.minimum_core_voxels),
        "minimum_core_fraction": core_fraction >= float(args.minimum_core_fraction),
        "minimum_extended_fraction": extended_fraction >= float(args.minimum_extended_fraction),
        "minimum_largest_component_fraction": largest_component_fraction >= float(args.minimum_largest_component_fraction),
        "core_subset_of_extended": bool(np.all(~filtered.core_mask | filtered.extended_mask)),
        "source_capacity_clean": bool(e3_summary["fusion"]["capacity_clean"]),
        "authority_boundaries_preserved": (
            not bool(e3_summary["authority"]["metric_scale_claimed"])
            and not bool(e3_summary["authority"]["measurement_authoritative"])
        ),
    }
    passed = all(pass_conditions.values())

    summary = {
        "phase": "F",
        "pass": passed,
        "status": "pass" if passed else "degraded",
        "source_e3_dir": str(e3_dir),
        "source_e3_pass": True,
        "source_e3_selected_view_count": selected_view_count,
        "selected_workspace_orders": orders.tolist(),
        "selected_keyframe_ids": keyframes.tolist(),
        "voxel_size_reconstruction_units": voxel_size,
        "observation_accounting": {
            "exact_match": mismatch_count == 0,
            "mismatched_voxel_count": mismatch_count,
            "maximum_count_delta": max_count_delta,
            "frame_unique_voxel_counts": frame_unique_counts,
            "recomputed_frame_voxel_observations": int(sum(frame_unique_counts)),
            "source_e3_total_fused_observations": int(np.sum(all_counts, dtype=np.int64)),
        },
        "dispersion": {
            "definition": "robust per-view voxel-centroid distance to coordinate-wise median, normalized by E3 voxel edge length",
            "median_dispersion_voxel_quantiles": _quantiles(median_dispersion),
            "p90_dispersion_voxel_quantiles": _quantiles(p90_dispersion),
            "maximum_dispersion_voxel_quantiles": _quantiles(maximum_dispersion),
        },
        "topology": {
            "component_count": component_count,
            "largest_supported_component_voxels": largest_component,
            "largest_supported_component_fraction": largest_component_fraction,
            "neighbor26_quantiles": _quantiles(neighbor26.astype(np.float64)),
            "radius2_neighbor_quantiles": _quantiles(radius2.astype(np.float64)),
            "high_support_neighbor26_quantiles": _quantiles(high26.astype(np.float64)),
        },
        "classification": {
            "source_all_voxels": int(len(all_counts)),
            "source_single_view_voxels": int(np.count_nonzero(single_view_mask)),
            "source_supported_voxels": source_supported,
            "core_voxels": core_count,
            "extended_voxels": extended_count,
            "rejected_supported_voxels": rejected_count,
            "attached_single_view_fringe_voxels": fringe_count,
            "detached_single_view_artifact_voxels": detached_count,
            "coverage_candidate_voxels": int(len(coverage_points)),
            "core_fraction_of_supported": core_fraction,
            "extended_fraction_of_supported": extended_fraction,
            "artifact_reason_histogram": reason_hist,
        },
        "confidence": {
            "definition": "view-count-normalized support + robust dispersion + local density + nearby >=3-view support",
            "all_supported_quantiles": _quantiles(filtered.confidence.astype(np.float64)),
            "core_quantiles": _quantiles(filtered.confidence[core_idx].astype(np.float64)),
            "extended_quantiles": _quantiles(filtered.confidence[extended_idx].astype(np.float64)),
            "rejected_quantiles": _quantiles(filtered.confidence[rejected_idx].astype(np.float64)),
        },
        "policy": {
            **policy.as_dict(),
            "minimum_core_voxels_gate": int(args.minimum_core_voxels),
            "minimum_core_fraction_gate": float(args.minimum_core_fraction),
            "minimum_extended_fraction_gate": float(args.minimum_extended_fraction),
            "minimum_largest_component_fraction_gate": float(args.minimum_largest_component_fraction),
            "fringe_min_extended_neighbors": int(args.fringe_min_extended_neighbors),
            "single_view_fringe_is_diagnostic_only": True,
            "detached_single_view_geometry_is_artifact_diagnostic_only": True,
        },
        "pass_conditions": pass_conditions,
        "decision": {
            "authorizes_next_canonical_dense_persistence_milestone": passed,
            "next_milestone": "G - Persistent Canonical Dense RGB Geometry" if passed else None,
            "primary_output": "f_filtered_extended.npz",
            "high_confidence_output": "f_filtered_core.npz",
        },
        "authority": {
            "sfm_scale_status": "arbitrary",
            "sfm_world_alignment_status": "arbitrary",
            "metric_scale_claimed": False,
            "canonical_geometry_authoritative": False,
            "canonical_persistence_promoted": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "confidence_artifact_filtering_formalized": True,
            "filtered_dense_geometry_qualified": passed,
            "note": (
                "F formalizes confidence/consistency/artifact filtering for the expanded E3 dense scene. "
                "Canonical persistence belongs to G; UI integration belongs to H; mesh reconstruction belongs to J."
            ),
        },
        "artifacts": {
            "core_npz": str(core_npz.resolve()),
            "core_ply": str(core_ply.resolve()),
            "extended_npz": str(extended_npz.resolve()),
            "extended_ply": str(extended_ply.resolve()),
            "rejected_supported_npz": str(rejected_npz.resolve()),
            "attached_single_view_fringe_npz": str(fringe_npz.resolve()),
            "detached_single_view_artifacts_npz": str(detached_npz.resolve()),
            "coverage_candidate_npz": str(coverage_npz.resolve()),
        },
    }
    summary_path = _write_json(output_dir / "f_summary.json", summary)

    print(f"\nF confidence / consistency / artifact filtering: {summary['status'].upper()}")
    print(
        f"source_all={len(all_counts)} source_supported={source_supported} "
        f"core={core_count} extended={extended_count} rejected_supported={rejected_count}"
    )
    print(
        f"single_view={int(np.count_nonzero(single_view_mask))} attached_fringe={fringe_count} "
        f"detached_artifact={detached_count}"
    )
    print(
        f"accounting_exact={mismatch_count == 0} frame_voxel_observations={sum(frame_unique_counts)} "
        f"largest_supported_component={largest_component} ({largest_component_fraction:.3f})"
    )
    print(
        f"p90_dispersion_voxels p50={_quantiles(p90_dispersion)['p50']:.4f} "
        f"p90={_quantiles(p90_dispersion)['p90']:.4f}"
    )
    print(f"artifact_reasons={reason_hist}")
    print("confidence_artifact_filtering_formalized=True")
    print(
        "sfm_scale_status=arbitrary metric_scale_claimed=False canonical_persistence_promoted=False "
        "measurement_authoritative=False surface_mesh_generated=False"
    )
    print(f"extended_cloud={extended_npz}")
    print(f"core_cloud={core_npz}")
    print(f"summary={summary_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
