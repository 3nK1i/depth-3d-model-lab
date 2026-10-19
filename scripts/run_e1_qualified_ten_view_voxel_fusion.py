#!/usr/bin/env python3
"""E1 real voxel fusion of the D4.3-qualified ten-view canonical chain.

E1 is the first real multi-view fusion milestone.  It consumes only a passing
D4.3 repaired ten-view chain, reconstructs the already-SfM-aligned RGB-D frames
from the saved validation artifacts, and feeds them through the existing
``DenseVoxelFusion`` production kernel.

The result remains in the arbitrary offline-SfM reconstruction gauge.  E1 does
not establish physical metric scale, canonical persistence authority,
measurement authority, surface-mesh authority, or certified metrology.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseSceneProfile,
    SfmScaleStatus,
    arbitrary_scale_reference,
    camera_geometry_from_estimated_camera,
    fuse_dense_frames,
    reconstruction_camera_from_sfm_model,
    reconstruction_unit_depth_alignment,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from scripts.run_d1_three_view_canonical_accumulation import _sfm_result_from_payload  # noqa: E402

_DEFAULT_D43_DIR = Path("/tmp/depth3d_d4_3_repaired_ten_view_chain")
_DEFAULT_OUTPUT_DIR = Path("/tmp/depth3d_e1_qualified_ten_view_voxel_fusion")
_DEFAULT_VOXEL_SCENE_FRACTION = 0.004
_DEFAULT_MAX_VOXELS = 750_000
_DEFAULT_MIN_SUPPORTED_VIEWS = 2
_DEFAULT_HIGH_SUPPORT_VIEWS = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d4-3-dir", type=Path, default=_DEFAULT_D43_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--voxel-scene-fraction",
        type=float,
        default=_DEFAULT_VOXEL_SCENE_FRACTION,
        help="Voxel edge length as a fraction of the robust sparse-SfM scene diagonal.",
    )
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--max-voxels", type=int, default=_DEFAULT_MAX_VOXELS)
    parser.add_argument("--minimum-supported-views", type=int, default=_DEFAULT_MIN_SUPPORTED_VIEWS)
    parser.add_argument("--high-support-views", type=int, default=_DEFAULT_HIGH_SUPPORT_VIEWS)
    parser.add_argument("--minimum-supported-voxels", type=int, default=1_000)
    parser.add_argument(
        "--sfm-focal-relative-uncertainty",
        type=float,
        default=0.20,
        help=(
            "Conservative contract-only uncertainty for the provisional SfM camera profile. "
            "It is not used to alter fused XYZ geometry."
        ),
    )
    args = parser.parse_args()
    if not 0.0 < float(args.voxel_scene_fraction) < 0.1:
        parser.error("--voxel-scene-fraction must be in (0, 0.1).")
    if int(args.pixel_stride) <= 0:
        parser.error("--pixel-stride must be positive.")
    if int(args.max_voxels) <= 0:
        parser.error("--max-voxels must be positive.")
    if int(args.minimum_supported_views) < 2 or int(args.minimum_supported_views) > 10:
        parser.error("--minimum-supported-views must be in [2, 10].")
    if int(args.high_support_views) < int(args.minimum_supported_views) or int(args.high_support_views) > 10:
        parser.error("--high-support-views must be >= minimum-supported-views and <= 10.")
    if int(args.minimum_supported_voxels) <= 0:
        parser.error("--minimum-supported-voxels must be positive.")
    if not 0.0 <= float(args.sfm_focal_relative_uncertainty) < 1.0:
        parser.error("--sfm-focal-relative-uncertainty must be in [0, 1).")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _derive_voxel_size(scene_scale: float, voxel_scene_fraction: float) -> float:
    scale = float(scene_scale)
    fraction = float(voxel_scene_fraction)
    if not isfinite(scale) or scale <= 0.0:
        raise ReconstructionValidationError("E1 scene scale must be finite and positive.")
    if not isfinite(fraction) or not 0.0 < fraction < 0.1:
        raise ReconstructionValidationError("E1 voxel_scene_fraction must be finite in (0, 0.1).")
    return scale * fraction


def _validate_d4_3_summary(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if str(summary.get("phase", "")) != "D4.3":
        raise ReconstructionValidationError("E1 requires a D4.3 summary.")
    if not bool(summary.get("pass")) or str(summary.get("status", "")) != "pass":
        raise ReconstructionValidationError("E1 requires D4.3 status=pass.")
    selection = summary.get("selection")
    if not isinstance(selection, dict):
        raise ReconstructionValidationError("E1 D4.3 selection evidence is missing.")
    if not bool(selection.get("ten_view_chain_qualified")):
        raise ReconstructionValidationError("E1 requires ten_view_chain_qualified=true.")
    if not bool(selection.get("authorizes_next_fusion_milestone")):
        raise ReconstructionValidationError("E1 requires D4.3 fusion authorization.")
    selected = str(selection.get("selected_candidate", "")).strip()
    candidates = summary.get("candidates")
    if selected not in {"A", "B"} or not isinstance(candidates, dict):
        raise ReconstructionValidationError("E1 D4.3 selected candidate is invalid.")
    candidate = candidates.get(selected)
    if not isinstance(candidate, dict) or not bool(candidate.get("pass")):
        raise ReconstructionValidationError("E1 selected D4.3 candidate is not passing.")
    orders = tuple(int(value) for value in candidate.get("workspace_orders", ()))
    keyframes = tuple(str(value) for value in candidate.get("keyframe_ids", ()))
    if len(orders) != 10 or len(keyframes) != 10 or len(set(keyframes)) != 10:
        raise ReconstructionValidationError("E1 requires exactly ten unique qualified keyframes.")
    return selected, candidate


def _support_masks(
    counts: object,
    *,
    minimum_supported_views: int,
    high_support_views: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
        raise ReconstructionValidationError("E1 view-observation counts must be a non-empty positive vector.")
    minimum = int(minimum_supported_views)
    high = int(high_support_views)
    if not 2 <= minimum <= 10:
        raise ReconstructionValidationError("minimum_supported_views must be in [2, 10].")
    if not minimum <= high <= 10:
        raise ReconstructionValidationError("high_support_views must be in [minimum_supported_views, 10].")
    return values >= minimum, values >= high


def _support_histogram(counts: object) -> dict[str, int]:
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1:
        raise ReconstructionValidationError("E1 support counts must be one-dimensional.")
    counter = Counter(int(value) for value in values.tolist())
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    xyz = np.asarray(points, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != (xyz.shape[0], 3):
        raise ReconstructionValidationError("E1 PLY requires Nx3 XYZ and Nx3 RGB arrays.")
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


def _find_source_view(
    *,
    keyframe_id: str,
    d43_summary: dict[str, Any],
    d4_summary: dict[str, Any],
) -> dict[str, Any]:
    predecessor = d43_summary.get("predecessor_view")
    if isinstance(predecessor, dict) and str(predecessor.get("keyframe_id")) == keyframe_id:
        return {
            "workspace_order": int(predecessor["workspace_order"]),
            "keyframe_id": keyframe_id,
            "cloud_npz": str(predecessor["cloud_npz"]),
            "frame_id": str(predecessor["frame_id"]),
            "camera_id": str(predecessor["camera_id"]),
            "source": "d4_3_predecessor",
        }
    for view in d4_summary.get("views", []):
        if str(view.get("keyframe_id")) == keyframe_id:
            return {
                "workspace_order": int(view["workspace_order"]),
                "keyframe_id": keyframe_id,
                "cloud_npz": str(view["cloud_npz"]),
                "frame_id": str(view["frame_id"]),
                "camera_id": str(view["camera_id"]),
                "source": "d4_existing_view",
            }
    raise ReconstructionValidationError(f"E1 cannot locate saved aligned-depth artifact for {keyframe_id!r}.")


def _save_cloud_npz(
    path: Path,
    *,
    points: np.ndarray,
    colors: np.ndarray,
    counts: np.ndarray,
    mean_confidence: np.ndarray,
    voxel_size: float,
    selected_keyframe_ids: tuple[str, ...],
    selected_workspace_orders: tuple[int, ...],
    sparse_points: np.ndarray,
    sparse_colors: np.ndarray,
) -> Path:
    np.savez_compressed(
        path,
        points_reconstruction=np.asarray(points, dtype=np.float64),
        colors_rgb=np.asarray(colors, dtype=np.uint8),
        view_observation_counts=np.asarray(counts, dtype=np.uint32),
        mean_confidence=np.asarray(mean_confidence, dtype=np.float32),
        voxel_size_reconstruction_units=np.asarray(float(voxel_size), dtype=np.float64),
        selected_keyframe_ids=np.asarray(selected_keyframe_ids, dtype="U160"),
        selected_workspace_orders=np.asarray(selected_workspace_orders, dtype=np.int32),
        sparse_landmarks_reconstruction=np.asarray(sparse_points, dtype=np.float64),
        sparse_landmark_colors_rgb=np.asarray(sparse_colors, dtype=np.uint8),
    )
    return path


def main() -> int:
    args = _parse_args()
    d43_dir = args.d4_3_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    d43_summary = _read_json(d43_dir / "d4_3_summary.json")
    selected_candidate, candidate = _validate_d4_3_summary(d43_summary)
    selected_orders = tuple(int(value) for value in candidate["workspace_orders"])
    selected_keyframes = tuple(str(value) for value in candidate["keyframe_ids"])
    scene_scale = float(candidate["scene_scale"]["robust_diagonal_reconstruction_units"])
    voxel_size = _derive_voxel_size(scene_scale, float(args.voxel_scene_fraction))

    d4_dir = Path(d43_summary["source_d4_dir"]).expanduser().resolve()
    d4_summary = _read_json(d4_dir / "d4_summary.json")
    if not bool(d4_summary.get("pass")):
        raise ReconstructionValidationError("E1 requires the source D4 accumulation to be passing.")
    b2_dir = Path(d4_summary["source_b2_dir"]).expanduser().resolve()
    sfm_result = _sfm_result_from_payload(_read_json(b2_dir / "sfm_result.json"))
    if sfm_result.gauge.scale_status is not SfmScaleStatus.ARBITRARY:
        raise ReconstructionValidationError("E1 expects the validated SfM world to remain arbitrary-scale.")

    camera_estimates = {item.keyframe_id: item for item in sfm_result.cameras}
    camera_models = {item.camera_id: item for item in sfm_result.camera_models}
    workspace_images = {item.keyframe_id: item for item in sfm_result.workspace.images}

    source_views = [
        _find_source_view(keyframe_id=keyframe_id, d43_summary=d43_summary, d4_summary=d4_summary)
        for keyframe_id in selected_keyframes
    ]
    if tuple(item["workspace_order"] for item in source_views) != selected_orders:
        raise ReconstructionValidationError("E1 saved source-view order does not match the selected D4.3 chain.")
    camera_ids = {item["camera_id"] for item in source_views}
    if len(camera_ids) != 1:
        raise ReconstructionValidationError(
            "E1 existing DenseVoxelFusion path currently requires one shared camera model across the qualified chain."
        )
    camera_id = next(iter(camera_ids))
    camera_model = camera_models.get(camera_id)
    if camera_model is None:
        raise ReconstructionValidationError("E1 qualified camera model is unavailable in the SfM result.")
    reconstruction_camera = reconstruction_camera_from_sfm_model(
        camera_model,
        keyframe_id=selected_keyframes[0],
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    camera_profile = camera_geometry_from_estimated_camera(
        profile_id=f"e1-{camera_id}-sfm-estimated",
        camera=reconstruction_camera,
        focal_length_relative_uncertainty=float(args.sfm_focal_relative_uncertainty),
        notes=(
            "E1 validation-only provisional profile built from the globally optimized SfM camera model. "
            "The uncertainty field is contract metadata only and is not used to perturb fused geometry."
        ),
    )
    scale_profile = arbitrary_scale_reference(
        profile_id=f"e1-{sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
        notes="E1 preserves the arbitrary offline-SfM reconstruction gauge; no unit-to-metre claim.",
    )
    policy = DenseFusionPolicy(
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        voxel_size_reconstruction_units=voxel_size,
        pixel_stride=int(args.pixel_stride),
        minimum_view_observations=1,
        max_voxels=int(args.max_voxels),
    )

    frames: list[DenseFusionFrame] = []
    frame_records: list[dict[str, Any]] = []
    for index, source in enumerate(source_views):
        keyframe_id = source["keyframe_id"]
        estimate = camera_estimates.get(keyframe_id)
        workspace = workspace_images.get(keyframe_id)
        if estimate is None or workspace is None:
            raise ReconstructionValidationError(f"E1 keyframe {keyframe_id!r} is missing SfM camera/workspace evidence.")
        rgb_path = Path(sfm_result.workspace.workspace_root).resolve() / workspace.staged_image_path
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ReconstructionValidationError(f"E1 could not decode RGB keyframe: {rgb_path}")
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        cloud_path = Path(source["cloud_npz"]).expanduser().resolve()
        with np.load(cloud_path, allow_pickle=False) as payload:
            aligned_depth = np.asarray(payload["aligned_depth_reconstruction_units"], dtype=np.float32)
            pose = np.asarray(payload["T_reconstruction_from_camera"], dtype=np.float64)
        if aligned_depth.shape != rgb.shape[:2]:
            raise ReconstructionValidationError(
                f"E1 aligned depth/RGB shape mismatch for workspace order {source['workspace_order']}: "
                f"depth={aligned_depth.shape}, rgb={rgb.shape[:2]}."
            )
        frame = DenseFusionFrame(
            frame_id=estimate.frame_id,
            timestamp_ns=int(workspace.timestamp_ns),
            rgb=rgb,
            depth=aligned_depth,
            T_reconstruction_from_camera=pose,
            depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
            depth_alignment=reconstruction_unit_depth_alignment(
                provenance=(
                    f"E1 consumes D4.3-qualified already-aligned depth for workspace order "
                    f"{source['workspace_order']} in arbitrary offline-SfM reconstruction units."
                )
            ),
            pose_layer="offline_sfm_optimized",
        )
        frames.append(frame)
        frame_records.append({
            "fusion_index": index,
            "workspace_order": int(source["workspace_order"]),
            "keyframe_id": keyframe_id,
            "frame_id": estimate.frame_id,
            "timestamp_ns": int(workspace.timestamp_ns),
            "source": source["source"],
            "cloud_npz": str(cloud_path),
        })

    print("E1: qualified D4.3 chain accepted for voxel fusion:", flush=True)
    print(f"  selected_candidate={selected_candidate}", flush=True)
    print(f"  workspace_orders={'->'.join(map(str, selected_orders))}", flush=True)
    print(f"  scene_scale={scene_scale:.6f} reconstruction_units", flush=True)
    print(
        f"  voxel_size={voxel_size:.6f} reconstruction_units "
        f"(scene_fraction={float(args.voxel_scene_fraction):.6f})",
        flush=True,
    )
    print(
        f"  pixel_stride={int(args.pixel_stride)} max_voxels={int(args.max_voxels)} "
        f"support_gate>={int(args.minimum_supported_views)} views",
        flush=True,
    )

    result = fuse_dense_frames(
        frames,
        session_id=f"{sfm_result.session_id}:e1-qualified-fusion",
        camera_profile=camera_profile,
        scale_profile=scale_profile,
        policy=policy,
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    cloud = result.point_cloud
    counts = np.asarray(cloud.view_observation_counts, dtype=np.uint32)
    supported_mask, high_mask = _support_masks(
        counts,
        minimum_supported_views=int(args.minimum_supported_views),
        high_support_views=int(args.high_support_views),
    )

    raw_selected_cloud_path = Path(d43_summary["selected_artifacts"]["combined_npz"]).expanduser().resolve()
    with np.load(raw_selected_cloud_path, allow_pickle=False) as raw_payload:
        raw_point_count = int(np.asarray(raw_payload["points_reconstruction"]).shape[0])
        sparse_points = np.asarray(
            raw_payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64
        )
        sparse_colors = np.asarray(
            raw_payload.get("sparse_landmark_colors_rgb", np.empty((0, 3))), dtype=np.uint8
        )

    all_npz = _save_cloud_npz(
        output_dir / "e1_voxel_fused_all.npz",
        points=cloud.points_reconstruction,
        colors=cloud.colors_rgb,
        counts=counts,
        mean_confidence=cloud.mean_confidence,
        voxel_size=voxel_size,
        selected_keyframe_ids=selected_keyframes,
        selected_workspace_orders=selected_orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    supported_npz = _save_cloud_npz(
        output_dir / "e1_voxel_fused_supported.npz",
        points=cloud.points_reconstruction[supported_mask],
        colors=cloud.colors_rgb[supported_mask],
        counts=counts[supported_mask],
        mean_confidence=cloud.mean_confidence[supported_mask],
        voxel_size=voxel_size,
        selected_keyframe_ids=selected_keyframes,
        selected_workspace_orders=selected_orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    high_npz = _save_cloud_npz(
        output_dir / "e1_voxel_fused_high_support.npz",
        points=cloud.points_reconstruction[high_mask],
        colors=cloud.colors_rgb[high_mask],
        counts=counts[high_mask],
        mean_confidence=cloud.mean_confidence[high_mask],
        voxel_size=voxel_size,
        selected_keyframe_ids=selected_keyframes,
        selected_workspace_orders=selected_orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    all_ply = _write_binary_ply(output_dir / "e1_voxel_fused_all.ply", cloud.points_reconstruction, cloud.colors_rgb)
    supported_ply = _write_binary_ply(
        output_dir / "e1_voxel_fused_supported.ply",
        cloud.points_reconstruction[supported_mask],
        cloud.colors_rgb[supported_mask],
    )

    all_count = int(len(counts))
    supported_count = int(np.count_nonzero(supported_mask))
    high_count = int(np.count_nonzero(high_mask))
    capacity_clean = result.statistics.dropped_new_voxels_capacity == 0
    structural_pass = (
        result.statistics.frames_integrated == 10
        and all_count > 0
        and supported_count >= int(args.minimum_supported_voxels)
        and all_count < raw_point_count
        and capacity_clean
        and result.unit_to_metre is None
        and result.scale_status is SfmScaleStatus.ARBITRARY
        and not result.certified_metrology
    )
    status = "pass" if structural_pass else "degraded"

    summary = {
        "phase": "E1",
        "pass": structural_pass,
        "status": status,
        "source_d4_3_dir": str(d43_dir),
        "source_d4_3_pass": True,
        "source_d4_3_selected_candidate": selected_candidate,
        "source_ten_view_chain_qualified": True,
        "source_fusion_authorized": True,
        "source_sfm_session_id": sfm_result.session_id,
        "selected_workspace_orders": list(selected_orders),
        "selected_keyframe_ids": list(selected_keyframes),
        "scene_scale": {
            "robust_diagonal_reconstruction_units": scene_scale,
            "physical_unit_claimed": False,
        },
        "policy": {
            **policy.as_dict(),
            "voxel_scene_fraction": float(args.voxel_scene_fraction),
            "minimum_supported_views_for_primary_output": int(args.minimum_supported_views),
            "high_support_views": int(args.high_support_views),
            "minimum_supported_voxels_gate": int(args.minimum_supported_voxels),
            "sfm_focal_relative_uncertainty_contract_only": float(args.sfm_focal_relative_uncertainty),
            "uniform_per_pixel_fusion_confidence": True,
        },
        "frames": frame_records,
        "raw_input": {
            "point_count": raw_point_count,
            "qualified_view_count": 10,
        },
        "fusion": {
            "result": result.as_dict(),
            "all_voxel_count": all_count,
            "multi_view_supported_voxel_count": supported_count,
            "high_support_voxel_count": high_count,
            "support_histogram": _support_histogram(counts),
            "multi_view_supported_ratio": float(supported_count / all_count),
            "high_support_ratio": float(high_count / all_count),
            "raw_to_all_voxel_compression_ratio": float(raw_point_count / all_count),
            "raw_to_supported_voxel_compression_ratio": (
                float(raw_point_count / supported_count) if supported_count else None
            ),
            "capacity_clean": capacity_clean,
        },
        "artifacts": {
            "all_npz": str(all_npz.resolve()),
            "all_ply": str(all_ply.resolve()),
            "supported_npz": str(supported_npz.resolve()),
            "supported_ply": str(supported_ply.resolve()),
            "high_support_npz": str(high_npz.resolve()),
        },
        "authority": {
            "spatial_merge_applied": True,
            "voxel_fusion_applied": True,
            "multi_view_support_filter_applied": True,
            "canonical_geometry_authoritative": False,
            "canonical_persistence_promoted": False,
            "measurement_authoritative": False,
            "metric_scale_claimed": False,
            "sfm_scale_status": "arbitrary",
            "sfm_world_alignment_status": "arbitrary",
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "note": (
                "E1 validation-only fused geometry. The primary supported cloud retains voxels observed "
                "by at least the configured number of qualified views; persistence and measurement authority "
                "remain explicitly unpromoted."
            ),
        },
        "decision": {
            "capacity_clean_required": True,
            "minimum_supported_voxels_required": int(args.minimum_supported_voxels),
            "primary_output": "e1_voxel_fused_supported.npz",
            "authorizes_next_fusion_refinement_milestone": structural_pass,
        },
    }
    summary_path = _write_json(output_dir / "e1_summary.json", summary)

    print(f"\nE1 qualified ten-view voxel fusion: {status.upper()}", flush=True)
    print(f"frames_integrated={result.statistics.frames_integrated} raw_points={raw_point_count}", flush=True)
    print(
        f"all_voxels={all_count} supported_voxels={supported_count} high_support_voxels={high_count}",
        flush=True,
    )
    print(
        f"multi_view_supported_ratio={supported_count / all_count:.3f} "
        f"raw_to_all_compression={raw_point_count / all_count:.2f}x",
        flush=True,
    )
    print(
        f"capacity_drops={result.statistics.dropped_new_voxels_capacity} "
        f"voxel_size={voxel_size:.6f} scene_fraction={float(args.voxel_scene_fraction):.6f}",
        flush=True,
    )
    print(
        "spatial_merge_applied=True voxel_fusion_applied=True multi_view_support_filter_applied=True",
        flush=True,
    )
    print(
        "sfm_scale_status=arbitrary metric_scale_claimed=False canonical_geometry_authoritative=False "
        "measurement_authoritative=False certified_metrology=False",
        flush=True,
    )
    print(f"supported_cloud={supported_npz}", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0 if structural_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
