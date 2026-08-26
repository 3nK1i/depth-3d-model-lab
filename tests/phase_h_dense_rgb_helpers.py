from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.reconstruction.dense_scene_persistence import persist_filtered_dense_rgb_scene
from scripts.run_e2_consistency_aware_fusion_refinement import _write_binary_ply

_SESSION_ID = "123e4567-e89b-12d3-a456-426614174000"


def _save_filtered_npz(
    path: Path,
    points: np.ndarray,
    *,
    voxel_size: float,
    orders: list[int],
    keyframes: list[str],
    representation: str,
) -> None:
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    colors = np.empty((count, 3), dtype=np.uint8)
    colors[:, 0] = np.linspace(60, 180, count, dtype=np.uint8)
    colors[:, 1] = 120
    colors[:, 2] = 200
    class_code = np.full(count, 3 if representation == "core" else 2, dtype=np.uint8)
    if representation == "extended" and count:
        class_code[: min(8, count)] = 3
    confidence = np.full(count, 0.82 if representation == "core" else 0.71, dtype=np.float32)
    np.savez_compressed(
        path,
        points_reconstruction=points,
        colors_rgb=colors,
        view_observation_counts=np.full(count, 4, dtype=np.uint32),
        filter_confidence=confidence,
        median_dispersion_voxels=np.full(count, 0.18, dtype=np.float32),
        p90_dispersion_voxels=np.full(count, 0.28, dtype=np.float32),
        maximum_dispersion_voxels=np.full(count, 0.38, dtype=np.float32),
        neighbor26_count=np.full(count, 14, dtype=np.int32),
        radius2_neighbor_count=np.full(count, 50, dtype=np.int32),
        high_support_neighbor_count=np.full(count, 8, dtype=np.int32),
        component_size=np.full(count, count, dtype=np.int32),
        class_code=class_code,
        artifact_reason_code=np.zeros(count, dtype=np.uint16),
        voxel_size_reconstruction_units=np.asarray(voxel_size, dtype=np.float64),
        selected_workspace_orders=np.asarray(orders, dtype=np.int32),
        selected_keyframe_ids=np.asarray(keyframes, dtype="U160"),
        sparse_landmarks_reconstruction=np.empty((0, 3), dtype=np.float64),
        sparse_landmark_colors_rgb=np.empty((0, 3), dtype=np.uint8),
        schema_version=np.asarray(1, dtype=np.int32),
    )


def make_phase_h_dense_rgb_bundle(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray, list[int]]:
    """Create a small integrity-valid G bundle for Phase-H UI tests."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    e3_dir = tmp_path / "e3"
    f_dir = tmp_path / "f"
    e3_dir.mkdir()
    f_dir.mkdir()
    voxel = 0.1
    orders = [0, 1, 2, 3, 4, 5]
    keyframes = [f"{_SESSION_ID}:keyframe:{order:08d}" for order in orders]
    extended_points = np.asarray(
        [
            [(x + 0.2) * voxel, (y + 0.2) * voxel, (10 + z + 0.2) * voxel]
            for x in range(5)
            for y in range(4)
            for z in range(2)
        ],
        dtype=np.float64,
    )
    core_points = extended_points[:24].copy()
    extended_npz = f_dir / "f_filtered_extended.npz"
    core_npz = f_dir / "f_filtered_core.npz"
    _save_filtered_npz(
        extended_npz,
        extended_points,
        voxel_size=voxel,
        orders=orders,
        keyframes=keyframes,
        representation="extended",
    )
    _save_filtered_npz(
        core_npz,
        core_points,
        voxel_size=voxel,
        orders=orders,
        keyframes=keyframes,
        representation="core",
    )
    extended_colors = np.load(extended_npz)["colors_rgb"]
    core_colors = np.load(core_npz)["colors_rgb"]
    extended_ply = _write_binary_ply(f_dir / "f_filtered_extended.ply", extended_points, extended_colors)
    core_ply = _write_binary_ply(f_dir / "f_filtered_core.ply", core_points, core_colors)

    diagnostics: dict[str, str] = {}
    for key, name in (
        ("rejected_supported_npz", "f_rejected_supported_artifacts.npz"),
        ("attached_single_view_fringe_npz", "f_attached_single_view_fringe_diagnostic.npz"),
        ("detached_single_view_artifacts_npz", "f_detached_single_view_artifacts_diagnostic.npz"),
        ("coverage_candidate_npz", "f_coverage_candidate_diagnostic.npz"),
    ):
        path = f_dir / name
        np.savez_compressed(path, points=np.zeros((1, 3), dtype=np.float32))
        diagnostics[key] = str(path)

    e3_summary = {
        "phase": "E3",
        "pass": True,
        "status": "pass",
        "source_sfm_session_id": _SESSION_ID,
        "selected_view_count": len(orders),
        "selected_workspace_orders": orders,
        "selected_keyframe_ids": keyframes,
    }
    (e3_dir / "e3_summary.json").write_text(json.dumps(e3_summary), encoding="utf-8")

    f_summary = {
        "phase": "F",
        "pass": True,
        "status": "pass",
        "source_e3_dir": str(e3_dir),
        "source_e3_selected_view_count": len(orders),
        "selected_workspace_orders": orders,
        "selected_keyframe_ids": keyframes,
        "voxel_size_reconstruction_units": voxel,
        "classification": {
            "source_all_voxels": 64,
            "source_supported_voxels": len(extended_points) + 8,
            "core_voxels": len(core_points),
            "extended_voxels": len(extended_points),
            "rejected_supported_voxels": 8,
            "attached_single_view_fringe_voxels": 4,
            "detached_single_view_artifact_voxels": 12,
            "artifact_reason_histogram": {"weak_high_support_anchor": 8},
        },
        "confidence": {
            "extended_quantiles": {"p50": 0.71},
            "core_quantiles": {"p50": 0.82},
        },
        "dispersion": {"p90_dispersion_voxel_quantiles": {"p50": 0.28}},
        "topology": {"largest_supported_component_fraction": 0.95},
        "policy": {"maximum_views": len(orders), "minimum_component_size": 1},
        "pass_conditions": {"exact_observation_accounting": True},
        "decision": {
            "authorizes_next_canonical_dense_persistence_milestone": True,
            "next_milestone": "G - Persistent Canonical Dense RGB Geometry",
        },
        "authority": {
            "filtered_dense_geometry_qualified": True,
            "metric_scale_claimed": False,
            "canonical_persistence_promoted": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "artifacts": {
            "extended_npz": str(extended_npz),
            "extended_ply": str(extended_ply),
            "core_npz": str(core_npz),
            "core_ply": str(core_ply),
            **diagnostics,
        },
    }
    f_summary_path = f_dir / "f_summary.json"
    f_summary_path.write_text(json.dumps(f_summary), encoding="utf-8")
    bundle = tmp_path / "g_bundle"
    persist_filtered_dense_rgb_scene(
        f_summary_path,
        bundle,
        created_at_utc="2026-08-18T14:00:00+00:00",
    )
    return bundle, extended_points, core_points, orders
