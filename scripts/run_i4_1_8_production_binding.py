#!/usr/bin/env python3
"""I4.1.8 production binding + retained-Core final I4.1 qualification.

Consumes a completed I4.1.7 campaign and a qualified I4.1.7B correction.  The
winning decoupled plan is bounded by the already-recorded final fusion cap,
then the unchanged production fusion/F/G/I2 kernels are executed.  Final orbit
qualification is recomputed from the resulting retained Core geometry.

No PyCOLMAP, DA3, or dense-pair work is rerun.  SfM remains the single global
geometry authority and its gauge remains arbitrary.
"""
from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

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
from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import selected_plan_payload  # noqa: E402
from core.reconstruction.offline_dense_i4_production_binding import (  # noqa: E402
    I4_1_8_SCHEMA_VERSION,
    build_final_i4_1_assessment,
    build_i4_1_8_decision,
    build_production_binding,
)
from scripts.run_d1_three_view_canonical_accumulation import _all_sparse_landmarks, _sfm_result_from_payload  # noqa: E402
from scripts.run_e1_qualified_ten_view_voxel_fusion import _save_cloud_npz, _support_histogram, _write_binary_ply  # noqa: E402
from scripts.run_e3_qualified_scene_fusion_expansion import _robust_diagonal  # noqa: E402
from scripts.run_i1_offline_dense_campaign import _DENSE_THRESHOLDS  # noqa: E402

_DEFAULT_I4_1_7 = Path("/tmp/eia_i4_1_7_box_clean_static_orbit_03_final")
_DEFAULT_I4_1_7B = Path("/tmp/eia_i4_1_7b_box_bounded_target_01")
_DEFAULT_OUTPUT = Path("/tmp/eia_i4_1_8_production_binding")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-1-7-dir", type=Path, default=_DEFAULT_I4_1_7)
    p.add_argument("--i4-1-7b-dir", type=Path, default=_DEFAULT_I4_1_7B)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--voxel-scene-fraction", type=float, default=0.004)
    p.add_argument("--max-voxels", type=int, default=1_500_000)
    p.add_argument("--minimum-supported-views", type=int, default=2)
    p.add_argument("--high-support-views", type=int, default=3)
    p.add_argument("--minimum-supported-voxels", type=int, default=1_000)
    p.add_argument("--sfm-focal-relative-uncertainty", type=float, default=0.20)
    p.add_argument("--skip-i2-view-attribution", action="store_true")
    args = p.parse_args()
    if not 0.0 < float(args.voxel_scene_fraction) < 0.1:
        p.error("--voxel-scene-fraction must be in (0,0.1)")
    if int(args.max_voxels) <= 0 or int(args.minimum_supported_voxels) <= 0:
        p.error("voxel limits must be positive")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.8 cannot read JSON evidence: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.1.8 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _run(command: list[str], *, log_path: Path) -> int:
    print("I4.1.8 command: " + " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(proc.wait())


def _support_masks(counts: np.ndarray, *, minimum: int, high: int, maximum_views: int) -> tuple[np.ndarray, np.ndarray]:
    if not 2 <= int(minimum) <= int(high) <= int(maximum_views):
        raise ReconstructionValidationError("I4.1.8 support gates are incompatible with the bound view count.")
    return counts >= int(minimum), counts >= int(high)


def _selected_rows(plan: Mapping[str, Any], selected_sources: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = plan.get("candidate_rows")
    if not isinstance(rows, list):
        raise ReconstructionValidationError("I4.1.8 selected plan lacks candidate_rows.")
    by_source: dict[int, dict[str, Any]] = {}
    for item in rows:
        if isinstance(item, Mapping) and item.get("source_sequence_index") is not None:
            source = int(item["source_sequence_index"])
            if source in by_source:
                raise ReconstructionValidationError("I4.1.8 plan has duplicate candidate source indices.")
            by_source[source] = dict(item)
    result: list[dict[str, Any]] = []
    for source in selected_sources:
        row = by_source.get(int(source))
        if row is None:
            raise ReconstructionValidationError(f"I4.1.8 cannot find bound source {source} in the selected plan.")
        if not bool(row.get("alignment_accepted")):
            raise ReconstructionValidationError(f"I4.1.8 bound source {source} is not alignment-qualified.")
        cloud = Path(str(row.get("cloud_npz", ""))).expanduser().resolve()
        if not cloud.is_file():
            raise ReconstructionValidationError(f"I4.1.8 bound source cloud is missing: {cloud}")
        row["cloud_npz"] = str(cloud)
        result.append(row)
    return result


def _workspace_maps(sfm_result: Any) -> tuple[dict[int, Any], dict[int, Any]]:
    by_source = {int(item.source_sequence_index): item for item in sfm_result.workspace.images}
    by_order = {int(item.order): item for item in sfm_result.workspace.images}
    return by_source, by_order


def _frame_source(
    *,
    row: Mapping[str, Any],
    i4_dir: Path,
    sfm_result: Any,
    workspace_by_source: Mapping[int, Any],
    workspace_by_order: Mapping[int, Any],
) -> tuple[Path, int, str]:
    source_index = int(row["source_sequence_index"])
    if bool(row.get("is_global_anchor")):
        order = row.get("global_workspace_order")
        workspace = workspace_by_order.get(int(order)) if order is not None else workspace_by_source.get(source_index)
        if workspace is None:
            raise ReconstructionValidationError(f"I4.1.8 cannot resolve global workspace source {source_index}.")
        image = Path(sfm_result.workspace.workspace_root).resolve() / workspace.staged_image_path
        return image, int(workspace.timestamp_ns), str(workspace.frame_id)

    image = i4_dir / "02_decoupled_localization" / "candidate_frames" / f"candidate_{source_index:08d}.jpg"
    localization_path = (
        i4_dir / "02_decoupled_localization" / "localization_evidence" / f"candidate_{source_index:08d}.json"
    )
    localization = _read_json(localization_path)
    timestamp_ns = int(localization.get("timestamp_ns", 0))
    if timestamp_ns < 0:
        raise ReconstructionValidationError(f"I4.1.8 localized source {source_index} has invalid timestamp.")
    frame_id = str(localization.get("frame_id") or f"i4-localized-source-{source_index}")
    return image, timestamp_ns, frame_id


def _load_bound_frames(
    *,
    rows: list[dict[str, Any]],
    i4_dir: Path,
    sfm_result: Any,
) -> tuple[list[DenseFusionFrame], list[dict[str, Any]], np.ndarray]:
    workspace_by_source, workspace_by_order = _workspace_maps(sfm_result)
    frames: list[DenseFusionFrame] = []
    frame_rows: list[dict[str, Any]] = []
    poses: list[np.ndarray] = []
    for fusion_index, row in enumerate(rows):
        cloud_path = Path(str(row["cloud_npz"])).resolve()
        try:
            with np.load(cloud_path, allow_pickle=False) as payload:
                depth = np.asarray(payload["aligned_depth_reconstruction_units"], dtype=np.float32)
                pose = np.asarray(payload["T_reconstruction_from_camera"], dtype=np.float64)
                keyframe_raw = payload.get("keyframe_id")
                keyframe_from_cloud = None if keyframe_raw is None else str(np.asarray(keyframe_raw).item())
        except (OSError, ValueError, KeyError) as exc:
            raise ReconstructionValidationError(f"I4.1.8 cannot read bound source cloud {cloud_path}: {exc}") from exc
        if depth.ndim != 2 or not np.isfinite(depth[depth > 0]).all():
            raise ReconstructionValidationError(f"I4.1.8 bound source {row['source_sequence_index']} depth is invalid.")
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ReconstructionValidationError(f"I4.1.8 bound source {row['source_sequence_index']} pose is invalid.")
        image_path, timestamp_ns, frame_id = _frame_source(
            row=row,
            i4_dir=i4_dir,
            sfm_result=sfm_result,
            workspace_by_source=workspace_by_source,
            workspace_by_order=workspace_by_order,
        )
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ReconstructionValidationError(f"I4.1.8 cannot decode bound RGB source: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != depth.shape:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        keyframe_id = str(row.get("keyframe_id") or keyframe_from_cloud or f"i4-bound-{row['source_sequence_index']}")
        frames.append(
            DenseFusionFrame(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                rgb=rgb,
                depth=depth,
                T_reconstruction_from_camera=pose,
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=reconstruction_unit_depth_alignment(
                    provenance=(
                        "I4.1.8 consumes previously qualified I4.1.6B aligned depth in the same authoritative "
                        "offline SfM reconstruction frame; arbitrary gauge."
                    )
                ),
                pose_layer="offline_sfm_optimized_or_localized_same_frame",
            )
        )
        poses.append(pose)
        frame_rows.append(
            {
                "fusion_index": int(fusion_index),
                "workspace_order": int(fusion_index),
                "source_sequence_index": int(row["source_sequence_index"]),
                "keyframe_id": keyframe_id,
                "frame_id": frame_id,
                "timestamp_ns": int(timestamp_ns),
                "source": "global_sfm_anchor" if bool(row.get("is_global_anchor")) else "localized_extra_observation",
                "cloud_npz": str(cloud_path),
                "rgb_source": str(image_path.resolve()),
            }
        )
    if len({item["keyframe_id"] for item in frame_rows}) != len(frame_rows):
        raise ReconstructionValidationError("I4.1.8 bound keyframe IDs must be unique.")
    return frames, frame_rows, np.asarray(poses, dtype=np.float64)


def _build_e3_compatibility(
    *,
    output_dir: Path,
    sfm_result: Any,
    frame_rows: list[dict[str, Any]],
    fusion_result: Any,
    voxel_size: float,
    sparse_points: np.ndarray,
    sparse_colors: np.ndarray,
    minimum_supported_views: int,
    high_support_views: int,
    minimum_supported_voxels: int,
    plan_name: str,
    qualified_view_count: int,
) -> dict[str, Any]:
    cloud = fusion_result.point_cloud
    counts = np.asarray(cloud.view_observation_counts, dtype=np.uint32)
    selected_orders = tuple(int(item["workspace_order"]) for item in frame_rows)
    selected_keyframes = tuple(str(item["keyframe_id"]) for item in frame_rows)
    supported_mask, high_mask = _support_masks(
        counts,
        minimum=int(minimum_supported_views),
        high=int(high_support_views),
        maximum_views=len(frame_rows),
    )
    all_npz = _save_cloud_npz(
        output_dir / "e3_expanded_voxel_fused_all.npz",
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
        output_dir / "e3_expanded_voxel_fused_supported.npz",
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
        output_dir / "e3_expanded_voxel_fused_high_support.npz",
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
    all_ply = _write_binary_ply(
        output_dir / "e3_expanded_voxel_fused_all.ply", cloud.points_reconstruction, cloud.colors_rgb
    )
    supported_ply = _write_binary_ply(
        output_dir / "e3_expanded_voxel_fused_supported.ply",
        cloud.points_reconstruction[supported_mask],
        cloud.colors_rgb[supported_mask],
    )
    supported_count = int(np.count_nonzero(supported_mask))
    high_count = int(np.count_nonzero(high_mask))
    capacity_clean = int(fusion_result.statistics.dropped_new_voxels_capacity) == 0
    passed = bool(
        len(frame_rows) >= 10
        and capacity_clean
        and int(fusion_result.statistics.frames_integrated) == len(frame_rows)
        and supported_count >= int(minimum_supported_voxels)
        and fusion_result.scale_status is SfmScaleStatus.ARBITRARY
        and fusion_result.unit_to_metre is None
    )
    summary = {
        "phase": "E3",
        "pass": passed,
        "status": "pass" if passed else "degraded",
        "campaign_adapter": {
            "source_phase": "I4.1.8",
            "purpose": "bounded production binding of qualified decoupled dense observations",
            "selection_policy": "deterministic uniform cap over already-qualified I4.1.6B path",
            "plan_name": plan_name,
            "qualified_decoupled_view_count": int(qualified_view_count),
            "threshold_relaxation": False,
        },
        "source_sfm_session_id": str(sfm_result.session_id),
        "selected_workspace_orders": list(selected_orders),
        "selected_keyframe_ids": list(selected_keyframes),
        "selected_view_count": len(frame_rows),
        "new_qualified_view_count": len(frame_rows),
        "frames": frame_rows,
        "candidate_policy": {
            "candidate_orders": list(selected_orders),
            "dense_thresholds_reused_unchanged_from_i1_i4_1_6b": dict(_DENSE_THRESHOLDS),
            "final_fusion_cap_applied": len(frame_rows),
        },
        "candidate_views": frame_rows,
        "edge_diagnostics": [],
        "campaign_path": {
            "orders": list(selected_orders),
            "view_count": len(frame_rows),
            "span": len(frame_rows) - 1,
            "source": "I4.1.6B pass-only selected path; bounded without re-running pair tests",
        },
        "fusion": {
            "voxel_size_reconstruction_units": float(voxel_size),
            "all_voxel_count": int(len(counts)),
            "supported_voxel_count": supported_count,
            "high_support_voxel_count": high_count,
            "support_histogram": _support_histogram(counts),
            "capacity_clean": capacity_clean,
            "capacity_drops": int(fusion_result.statistics.dropped_new_voxels_capacity),
            "frames_integrated": int(fusion_result.statistics.frames_integrated),
            "sampled_pixels": int(fusion_result.statistics.sampled_pixels),
        },
        "decision": {
            "authorizes_next_confidence_filtering_milestone": passed,
            "next_milestone": "F - Confidence / Consistency / Artifact Filtering" if passed else None,
            "primary_output": "e3_expanded_voxel_fused_supported.npz",
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
            "confidence_artifact_filtering_formalized": False,
        },
        "artifacts": {
            "all_npz": str(all_npz.resolve()),
            "all_ply": str(all_ply.resolve()),
            "supported_npz": str(supported_npz.resolve()),
            "supported_ply": str(supported_ply.resolve()),
            "high_support_npz": str(high_npz.resolve()),
        },
    }
    _write_json(output_dir / "e3_summary.json", summary)
    return summary


def _write_i1_adapter(
    *,
    adapter_dir: Path,
    source_i1: Mapping[str, Any],
    sfm_result: Any,
    binding: Any,
    e3_summary: Mapping[str, Any],
    f_summary: Mapping[str, Any],
    g_summary: Mapping[str, Any],
) -> Path:
    selected_count = int(binding.selected_view_count)
    orders = list(range(selected_count))
    final = {
        "phase": "I1",
        "schema_version": int(source_i1.get("schema_version", 1)),
        "status": "pass",
        "pipeline_execution_complete": True,
        "quality_campaign_status": "pending_I2_real_scene_visual_and_coverage_review",
        "campaign_adapter": {
            "source_phase": "I4.1.8",
            "production_binding": True,
            "plan_name": binding.plan_name,
            "qualified_source_count": int(binding.qualified_source_count),
            "bound_view_count": selected_count,
            "final_fusion_cap": int(binding.final_fusion_cap),
            "pycolmap_rerun": False,
            "da3_rerun": False,
            "dense_pair_rerun": False,
        },
        "source": dict(source_i1.get("source", {})),
        "sfm": {
            "session_id": str(sfm_result.session_id),
            "registered_view_count": len(sfm_result.cameras),
            "registered_orders": [int(item.order) for item in sfm_result.workspace.images if item.keyframe_id in {c.keyframe_id for c in sfm_result.cameras}],
            "candidate_dense_orders": orders,
            "scale_status": "arbitrary",
        },
        "dense_qualification": {
            "alignment_accepted_views": selected_count,
            "selected_pass_path": {"orders": orders, "view_count": selected_count, "span": selected_count - 1},
            "thresholds": dict(_DENSE_THRESHOLDS),
            "threshold_relaxation": False,
        },
        "fusion": dict(e3_summary["fusion"]),
        "filtering": {
            "core_voxels": int(f_summary["classification"]["core_voxels"]),
            "extended_voxels": int(f_summary["classification"]["extended_voxels"]),
            "rejected_supported_voxels": int(f_summary["classification"]["rejected_supported_voxels"]),
        },
        "persistent_scene": {
            "scene_id": str(g_summary["scene_id"]),
            "fingerprint": str(g_summary["scene_fingerprint_sha256"]),
            "scene_dir": str((adapter_dir / "06_persistent_scene").resolve()),
            "extended_points": int(g_summary["geometry"]["primary_point_count"]),
            "core_points": int(g_summary["geometry"]["core_point_count"]),
            "self_contained": bool(g_summary["persistence"]["bundle_self_contained"]),
        },
        "stages": {
            "sfm": {"reused": True, "source_dir": str((adapter_dir / "01_sfm").resolve())},
            "qualified_voxel_fusion": {"status": str(e3_summary["status"])},
            "formal_filtering": {"status": "pass" if bool(f_summary.get("pass")) else "degraded"},
            "canonical_persistence": {"status": "pass" if bool(g_summary.get("pass")) else "degraded"},
        },
        "authority": {
            "canonical_geometry_authoritative": True,
            "canonical_persistence_promoted": True,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_i2_current_full_video_validation": True,
            "next_milestone": "I4.1.8 bound-scene I2 quality qualification",
            "note": "I4.1.8 adapter reuses unchanged I2 quality kernels over the bound production scene.",
        },
    }
    return _write_json(adapter_dir / "i1_campaign_summary.json", final)


def _load_core_points(f_summary: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(f_summary.get("artifacts", {}).get("core_npz", ""))).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"I4.1.8 retained Core artifact missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.8 cannot read retained Core geometry: {path}: {exc}") from exc
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 64 or not np.isfinite(points).all():
        raise ReconstructionValidationError("I4.1.8 retained Core points are invalid.")
    return points


def main() -> int:
    args = _args()
    i4_dir = args.i4_1_7_dir.expanduser().resolve()
    b_dir = args.i4_1_7b_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.8 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_summary = _read_json(i4_dir / "i4_1_7_summary.json")
    i417b = _read_json(b_dir / "i4_1_7b_summary.json")
    dense_ab = _read_json(i4_dir / "03_real_dense_ab" / "i4_1_6b_summary.json")
    source_i1_dir = i4_dir / "01_i1_baseline"
    source_i1 = _read_json(source_i1_dir / "i1_campaign_summary.json")
    sfm_report = _read_json(source_i1_dir / "01_sfm" / "real_sfm_report.json")
    sfm_result = _sfm_result_from_payload(sfm_report["sfm_result"])
    if sfm_result.gauge.scale_status is not SfmScaleStatus.ARBITRARY or sfm_result.gauge.unit_to_metre is not None:
        raise ReconstructionValidationError("I4.1.8 requires arbitrary-scale SfM with no unit-to-metre promotion.")
    if len(sfm_result.camera_models) != 1:
        raise ReconstructionValidationError("I4.1.8 currently requires one shared SfM camera model.")

    plan_name = str(i417b.get("source", {}).get("qualification_plan") or source_summary.get("qualification_plan", ""))
    plan = selected_plan_payload(dense_ab, plan_name)
    qualified_sources = tuple(int(v) for v in plan.get("selected_source_sequence_indices", []))
    dense_decision = dense_ab.get("decision")
    if not isinstance(dense_decision, Mapping):
        raise ReconstructionValidationError("I4.1.8 dense A/B decision is missing.")
    recorded_cap = int(dense_decision.get("final_fusion_cap_preserved", 0))
    if recorded_cap < 10:
        raise ReconstructionValidationError("I4.1.8 dense A/B evidence has no valid preserved final fusion cap.")
    binding = build_production_binding(
        i4_1_7b_summary=i417b,
        plan_name=plan_name,
        qualified_source_sequence_indices=qualified_sources,
        final_fusion_cap=recorded_cap,
    )
    rows = _selected_rows(plan, binding.selected_source_sequence_indices)
    frames, frame_rows, selected_poses = _load_bound_frames(rows=rows, i4_dir=i4_dir, sfm_result=sfm_result)

    binding_manifest = {
        "phase": "I4.1.8",
        "schema_version": I4_1_8_SCHEMA_VERSION,
        "binding": binding.as_dict(),
        "views": frame_rows,
        "source_i4_1_7": str(i4_dir),
        "source_i4_1_7b": str(b_dir),
        "single_authoritative_sfm_frame": True,
        "production_default_changed": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "surface_mesh_generated": False,
    }
    _write_json(output / "00_production_binding" / "i4_1_8_binding_manifest.json", binding_manifest)
    np.savez_compressed(
        output / "00_production_binding" / "bound_camera_poses.npz",
        selected_camera_matrices=selected_poses,
        selected_source_sequence_indices=np.asarray(binding.selected_source_sequence_indices, dtype=np.int64),
        schema_version=np.asarray(I4_1_8_SCHEMA_VERSION, dtype=np.int32),
    )

    sparse_points, sparse_colors = _all_sparse_landmarks(sfm_result)
    scene_scale = _robust_diagonal(sparse_points)
    if not isfinite(scene_scale) or scene_scale <= 0.0:
        raise ReconstructionValidationError("I4.1.8 robust sparse scene scale is invalid.")
    voxel_size = float(scene_scale * float(args.voxel_scene_fraction))
    model = sfm_result.camera_models[0]
    reconstruction_camera = reconstruction_camera_from_sfm_model(
        model,
        keyframe_id=str(frame_rows[0]["keyframe_id"]),
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    camera_profile = camera_geometry_from_estimated_camera(
        profile_id=f"i4-1-8-{model.camera_id}-sfm-estimated",
        camera=reconstruction_camera,
        focal_length_relative_uncertainty=float(args.sfm_focal_relative_uncertainty),
        notes="I4.1.8 bound production camera profile from authoritative global SfM; arbitrary gauge only.",
    )
    scale_profile = arbitrary_scale_reference(
        profile_id=f"i4-1-8-{sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
        notes="I4.1.8 preserves arbitrary offline-SfM reconstruction gauge; no physical unit claim.",
    )
    policy = DenseFusionPolicy(
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        voxel_size_reconstruction_units=voxel_size,
        pixel_stride=1,
        minimum_view_observations=1,
        max_voxels=int(args.max_voxels),
    )
    print(
        f"I4.1.8 production binding: plan={binding.plan_name} qualified={binding.qualified_source_count} "
        f"bound={binding.selected_view_count} final_fusion_cap={binding.final_fusion_cap} voxel={voxel_size:.6f}",
        flush=True,
    )
    fusion_dir = output / "01_bound_production_i1" / "04_qualified_fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fusion_result = fuse_dense_frames(
        frames,
        session_id=f"{sfm_result.session_id}:i4-1-8-bound-production-fusion",
        camera_profile=camera_profile,
        scale_profile=scale_profile,
        policy=policy,
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    fusion_duration = time.perf_counter() - started
    e3_summary = _build_e3_compatibility(
        output_dir=fusion_dir,
        sfm_result=sfm_result,
        frame_rows=frame_rows,
        fusion_result=fusion_result,
        voxel_size=voxel_size,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
        minimum_supported_views=int(args.minimum_supported_views),
        high_support_views=int(args.high_support_views),
        minimum_supported_voxels=int(args.minimum_supported_voxels),
        plan_name=binding.plan_name,
        qualified_view_count=binding.qualified_source_count,
    )
    if not bool(e3_summary.get("pass")):
        raise ReconstructionValidationError("I4.1.8 bound production fusion did not pass unchanged E3/F prerequisites.")

    adapter = output / "01_bound_production_i1"
    sfm_adapter_dir = adapter / "01_sfm"
    sfm_adapter_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_i1_dir / "01_sfm" / "real_sfm_summary.json", sfm_adapter_dir / "real_sfm_summary.json")
    shutil.copy2(source_i1_dir / "01_sfm" / "real_sfm_report.json", sfm_adapter_dir / "real_sfm_report.json")

    filter_dir = adapter / "05_filtering"
    code = _run(
        [sys.executable, str(ROOT / "scripts/run_f_confidence_artifact_filtering.py"), "--e3-dir", str(fusion_dir), "--output-dir", str(filter_dir)],
        log_path=output / "logs" / "02_formal_filtering.log",
    )
    if code != 0:
        raise ReconstructionValidationError("I4.1.8 formal F filtering failed.")
    f_summary = _read_json(filter_dir / "f_summary.json")
    if not bool(f_summary.get("pass")):
        raise ReconstructionValidationError("I4.1.8 formal F filtering did not PASS.")

    persistent_dir = adapter / "06_persistent_scene"
    code = _run(
        [sys.executable, str(ROOT / "scripts/run_g_persistent_dense_rgb_geometry.py"), "--f-dir", str(filter_dir), "--output-dir", str(persistent_dir)],
        log_path=output / "logs" / "03_canonical_persistence.log",
    )
    if code != 0:
        raise ReconstructionValidationError("I4.1.8 canonical G persistence failed.")
    g_summary = _read_json(persistent_dir / "g_summary.json")
    if not bool(g_summary.get("pass")):
        raise ReconstructionValidationError("I4.1.8 canonical G persistence did not PASS.")

    i1_adapter_path = _write_i1_adapter(
        adapter_dir=adapter,
        source_i1=source_i1,
        sfm_result=sfm_result,
        binding=binding,
        e3_summary=e3_summary,
        f_summary=f_summary,
        g_summary=g_summary,
    )

    i2_dir = output / "02_bound_scene_i2_quality"
    i2_command = [
        sys.executable,
        str(ROOT / "scripts/run_i2_full_video_quality_qualification.py"),
        "--i1-dir",
        str(adapter),
        "--output-dir",
        str(i2_dir),
    ]
    if args.skip_i2_view_attribution:
        i2_command.append("--skip-view-attribution")
    code = _run(i2_command, log_path=output / "logs" / "04_i2_bound_scene_quality.log")
    i2_summary_path = i2_dir / "i2_summary.json"
    i2_summary = _read_json(i2_summary_path) if i2_summary_path.is_file() else {"phase": "I2", "automated_pass": False}
    if code != 0 or not bool(i2_summary.get("automated_pass")):
        # Continue to emit a fail-closed final summary rather than losing stage evidence.
        i2_pass = False
    else:
        i2_pass = True

    retained_core = _load_core_points(f_summary)
    registered_matrices = np.asarray([item.T_reconstruction_from_camera for item in sfm_result.cameras], dtype=np.float64)
    final_assessment = build_final_i4_1_assessment(
        input_global_views=len(sfm_result.workspace.images),
        registered_global_views=len(sfm_result.cameras),
        production_alignment_accepted_views=binding.selected_view_count,
        registered_camera_matrices=registered_matrices,
        production_selected_camera_matrices=selected_poses,
        retained_core_points=retained_core,
        i2_summary=i2_summary,
        binding=binding,
    )
    result = build_i4_1_8_decision(
        binding=binding,
        final_assessment=final_assessment,
        fusion_pass=bool(e3_summary.get("pass")),
        filtering_pass=bool(f_summary.get("pass")),
        persistence_pass=bool(g_summary.get("pass")),
        i2_automated_pass=i2_pass,
    )
    result["source"] = {
        "i4_1_7_dir": str(i4_dir),
        "i4_1_7b_dir": str(b_dir),
        "i4_1_7_status": str(source_summary.get("status")),
        "i4_1_7b_status": str(i417b.get("status")),
        "qualified_plan": binding.plan_name,
    }
    result["execution"] = {
        "pycolmap_rerun": False,
        "da3_rerun": False,
        "dense_pair_rerun": False,
        "production_binding_executed": True,
        "bounded_fusion_executed": True,
        "formal_filtering_executed": True,
        "canonical_persistence_executed": True,
        "i2_bound_scene_quality_executed": True,
        "fusion_duration_s": float(fusion_duration),
    }
    result["artifacts"] = {
        "binding_manifest": str((output / "00_production_binding" / "i4_1_8_binding_manifest.json").resolve()),
        "bound_camera_poses": str((output / "00_production_binding" / "bound_camera_poses.npz").resolve()),
        "i1_adapter_summary": str(i1_adapter_path.resolve()),
        "e3_summary": str((fusion_dir / "e3_summary.json").resolve()),
        "f_summary": str((filter_dir / "f_summary.json").resolve()),
        "g_summary": str((persistent_dir / "g_summary.json").resolve()),
        "i2_summary": str(i2_summary_path.resolve()),
        "persistent_scene": str(persistent_dir.resolve()),
    }
    final_diag = output / "i4_1_8_final_orbit_diagnostics.npz"
    np.savez_compressed(
        final_diag,
        retained_core_target_reconstruction=np.asarray(final_assessment["target"]["target_reconstruction"], dtype=np.float64),
        registered_camera_matrices=registered_matrices,
        production_selected_camera_matrices=selected_poses,
        production_selected_source_sequence_indices=np.asarray(binding.selected_source_sequence_indices, dtype=np.int64),
        schema_version=np.asarray(I4_1_8_SCHEMA_VERSION, dtype=np.int32),
    )
    result["artifacts"]["final_orbit_diagnostics_npz"] = str(final_diag.resolve())
    summary_path = _write_json(output / "i4_1_8_summary.json", result)

    selected_orbit = final_assessment["selected_orbit"]
    print("\nI4.1.8 Production Binding + Final I4.1 Qualification: " + ("READY_FOR_MANUAL_ORBIT_REVIEW" if result["automated_i4_1_final_qualification_pass"] else "DEGRADED"), flush=True)
    print(
        f"plan={binding.plan_name} qualified_sources={binding.qualified_source_count} bound_views={binding.selected_view_count}/{binding.final_fusion_cap} "
        f"source_span_preserved={binding.source_span_preserved}",
        flush=True,
    )
    print(
        f"bound_scene fusion_pass={bool(e3_summary.get('pass'))} filtering_pass={bool(f_summary.get('pass'))} "
        f"persistence_pass={bool(g_summary.get('pass'))} i2_automated_pass={i2_pass}",
        flush=True,
    )
    print(
        f"retained_core_target selected_azimuth={selected_orbit['azimuth_coverage_degrees']:.1f}deg "
        f"bins={selected_orbit['occupied_azimuth_bins']}/12 opposed={selected_orbit['maximum_opposed_view_angle_degrees']:.1f}deg "
        f"facing={selected_orbit['target_facing_fraction']:.3f} standoff_p90_p10={selected_orbit['standoff_p90_to_p10_ratio']:.3f}",
        flush=True,
    )
    print(
        f"final_i4_1_automated_qualification_pass={result['automated_i4_1_final_qualification_pass']} "
        f"production_sampling_integration_authorized={result['authority']['production_sampling_integration_authorized']} "
        "manual_orbit_review_required=True i4_2_detail_preservation_authorized=False",
        flush=True,
    )
    print("pycolmap_rerun=False da3_rerun=False dense_pair_rerun=False", flush=True)
    print("orbit_thresholds_changed=False dense_consistency_threshold_relaxation_authorized=False", flush=True)
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False", flush=True)
    if final_assessment.get("failed_gates"):
        print("failed_final_gates=" + ",".join(final_assessment["failed_gates"]), flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0 if result["automated_i4_1_final_qualification_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
