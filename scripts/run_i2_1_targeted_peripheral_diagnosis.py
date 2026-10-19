#!/usr/bin/env python3
"""I2.1 targeted peripheral-artifact diagnosis for the current full-video scene.

Consumes the already-generated I1/I2 outputs. No SfM, DA3, voxel fusion,
formal F filtering, G persistence, or UI state is modified or rerun.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_peripheral_diagnosis import (  # noqa: E402
    I21_DIAGNOSIS_SCHEMA_VERSION,
    I21DiagnosisPolicy,
    automatic_temporal_suspect_segments,
    pair_cosupport_records,
    removal_sensitivity,
    scale_profile,
    validate_observation_matrix,
)
from scripts.run_e2_consistency_aware_fusion_refinement import (  # noqa: E402
    _aggregate_saved_e1_source_frame_voxels,
    _voxel_indices,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1-dir", type=Path, default=Path("/tmp/depth3d_i1_offline_dense_campaign_recovered"))
    parser.add_argument("--i2-dir", type=Path, default=Path("/tmp/depth3d_i2_full_video_quality"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/depth3d_i2_1_targeted_peripheral_diagnosis"))
    parser.add_argument("--minimum-support-views", type=int, default=2)
    parser.add_argument("--top-views-per-temporal-half", type=int, default=6)
    parser.add_argument(
        "--segment", action="append", default=[], metavar="NAME=ORDER,ORDER,...",
        help="Optional additional support-removal counterfactual segment. May be repeated.",
    )
    parser.add_argument("--top-pairs", type=int, default=20)
    args = parser.parse_args()
    if args.minimum_support_views < 2 or args.top_views_per_temporal_half <= 0 or args.top_pairs <= 0:
        parser.error("support floor/top view count/top pair count must be positive and support >=2")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read required I2.1 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I2.1 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _parse_segment(value: str) -> tuple[str, tuple[int, ...]]:
    if "=" not in value:
        raise ReconstructionValidationError("I2.1 --segment must use NAME=ORDER,ORDER,...")
    name, raw = value.split("=", 1)
    name = name.strip()
    if not name or not raw.strip():
        raise ReconstructionValidationError("I2.1 --segment name/orders cannot be empty")
    try:
        orders = tuple(sorted({int(token.strip()) for token in raw.split(",") if token.strip()}))
    except ValueError as exc:
        raise ReconstructionValidationError(f"Invalid I2.1 --segment {value!r}") from exc
    if not orders:
        raise ReconstructionValidationError("I2.1 --segment requires at least one order")
    return name, orders


def _load_i2_diagnostics(i2_summary: Mapping[str, Any]) -> dict[str, np.ndarray | float]:
    path = Path(str(i2_summary["artifacts"]["spatial_diagnostics_npz"])).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"I2.1 missing I2 spatial diagnostics: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {
            "points": np.asarray(payload["points_reconstruction"], dtype=np.float64),
            "colors": np.asarray(payload["colors_rgb"], dtype=np.uint8),
            "support": np.asarray(payload["view_observation_counts"], dtype=np.int64),
            "is_core": np.asarray(payload["is_core"], dtype=bool),
            "peripheral": np.asarray(payload["is_peripheral"], dtype=bool),
            "far": np.asarray(payload["is_far_peripheral"], dtype=bool),
            "weak": np.asarray(payload["is_weak_peripheral"], dtype=bool),
            "sector": np.asarray(payload["directional_sector_code"], dtype=np.uint8),
            "voxel_size": float(np.asarray(payload["voxel_size_reconstruction_units"]).item()),
        }


def _membership_matrix(
    *,
    e3_summary: Mapping[str, Any],
    points: np.ndarray,
    voxel_size: float,
) -> tuple[tuple[int, ...], np.ndarray, dict[int, int]]:
    frames = e3_summary.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ReconstructionValidationError("I2.1 requires selected source-frame provenance from I1 fusion")
    keys = [tuple(map(int, key)) for key in _voxel_indices(points, voxel_size)]
    if len(set(keys)) != len(keys):
        raise ReconstructionValidationError("I2.1 expected one retained Extended point per voxel key")
    key_to_index = {key: index for index, key in enumerate(keys)}
    orders = tuple(int(frame["workspace_order"]) for frame in frames)
    matrix = np.zeros((len(frames), len(points)), dtype=bool)
    unique_counts: dict[int, int] = {}
    for row, frame in enumerate(frames):
        cloud_path = Path(str(frame["cloud_npz"])).expanduser().resolve()
        if not cloud_path.is_file():
            raise ReconstructionValidationError(f"I2.1 selected source cloud is unavailable: {cloud_path}")
        with np.load(cloud_path, allow_pickle=False) as payload:
            canonical = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
            aligned_depth = np.asarray(payload["aligned_depth_reconstruction_units"])
            pose = np.asarray(payload["T_reconstruction_from_camera"], dtype=np.float64)
        frame_keys, _, _ = _aggregate_saved_e1_source_frame_voxels(
            canonical_points=canonical,
            colors=colors,
            aligned_depth_reconstruction_units=aligned_depth,
            T_reconstruction_from_camera=pose,
            voxel_size=voxel_size,
        )
        unique_counts[orders[row]] = int(len(frame_keys))
        for key in frame_keys:
            index = key_to_index.get(tuple(map(int, key)))
            if index is not None:
                matrix[row, index] = True
    return orders, matrix, unique_counts


def _scale_map(e3_summary: Mapping[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in e3_summary.get("candidate_views", []):
        if item.get("scale_to_reconstruction_units") is not None:
            result[int(item["workspace_order"])] = float(item["scale_to_reconstruction_units"])
    return result


def _edges_touching(e3_summary: Mapping[str, Any], suspect_orders: set[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for edge in e3_summary.get("edge_diagnostics", []):
        raw_orders = edge.get("workspace_orders", [])
        if not isinstance(raw_orders, list) or len(raw_orders) != 2:
            continue
        a, b = map(int, raw_orders)
        if a not in suspect_orders and b not in suspect_orders:
            continue
        item = {
            "workspace_orders": [a, b],
            "status": str(edge.get("status", "unknown")),
            "order_gap": int(edge.get("order_gap", b - a)),
            "scale_ratio_max_to_min": edge.get("scale_ratio_max_to_min"),
        }
        for name in ("a_to_b", "b_to_a"):
            direction = edge.get(name)
            if isinstance(direction, Mapping):
                item[name] = {
                    "status": direction.get("status"),
                    "consistent_ratio_of_comparable": direction.get("consistent_ratio_of_comparable"),
                    "median_absolute_relative_depth_residual": direction.get("median_absolute_relative_depth_residual"),
                    "p90_absolute_relative_depth_residual": direction.get("p90_absolute_relative_depth_residual"),
                    "front_surface_conflict_ratio_of_comparable": direction.get("front_surface_conflict_ratio_of_comparable"),
                }
        records.append(item)
    records.sort(key=lambda item: (item["workspace_orders"][0], item["workspace_orders"][1]))
    return records


def main() -> int:
    args = _args()
    i1_dir = args.i1_dir.expanduser().resolve()
    i2_dir = args.i2_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I2.1 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i1 = _read_json(i1_dir / "i1_campaign_summary.json")
    i2 = _read_json(i2_dir / "i2_summary.json")
    e3 = _read_json(i1_dir / "04_qualified_fusion" / "e3_summary.json")
    if str(i1.get("status")) != "pass" or not bool(i1.get("pipeline_execution_complete")):
        raise ReconstructionValidationError("I2.1 requires a completed passing I1 campaign")
    if str(i2.get("phase")) != "I2" or not bool(i2.get("automated_pass")):
        raise ReconstructionValidationError("I2.1 requires a passing automated I2 qualification")
    if not bool(i2.get("manual_visual_review_required")):
        raise ReconstructionValidationError("I2.1 expects I2 to still require manual quality review")

    data = _load_i2_diagnostics(i2)
    points = np.asarray(data["points"], dtype=np.float64)
    support = np.asarray(data["support"], dtype=np.int64)
    peripheral = np.asarray(data["peripheral"], dtype=bool)
    far = np.asarray(data["far"], dtype=bool)
    weak = np.asarray(data["weak"], dtype=bool)
    is_core = np.asarray(data["is_core"], dtype=bool)
    sector = np.asarray(data["sector"], dtype=np.uint8)
    voxel_size = float(data["voxel_size"])

    orders, matrix, frame_unique_counts = _membership_matrix(e3_summary=e3, points=points, voxel_size=voxel_size)
    policy = I21DiagnosisPolicy(
        minimum_support_views=int(args.minimum_support_views),
        top_views_per_temporal_half=int(args.top_views_per_temporal_half),
    )
    validate_observation_matrix(matrix, support, minimum_support_views=policy.minimum_support_views)
    order_to_row = {order: index for index, order in enumerate(orders)}

    attribution = i2.get("view_attribution", {})
    if not bool(attribution.get("available")):
        raise ReconstructionValidationError("I2.1 requires I2 per-view attribution")
    segments = automatic_temporal_suspect_segments(
        orders,
        attribution.get("per_view", []),
        top_n=policy.top_views_per_temporal_half,
    )
    for raw in args.segment:
        name, custom_orders = _parse_segment(raw)
        unknown = [order for order in custom_orders if order not in order_to_row]
        if unknown:
            raise ReconstructionValidationError(f"I2.1 segment {name!r} contains unselected orders: {unknown}")
        segments[name] = custom_orders

    single_view_results: list[dict[str, Any]] = []
    view_masks = np.zeros((len(orders), len(points)), dtype=np.uint8)
    for row, order in enumerate(orders):
        metrics, collapse, _ = removal_sensitivity(
            points=points, observation_matrix=matrix, baseline_support=support,
            removed_view_indices=(row,), peripheral_mask=peripheral, far_mask=far, weak_mask=weak,
            core_mask=is_core, sector_codes=sector, policy=policy,
        )
        metrics["workspace_order"] = order
        metrics["unique_frame_voxels"] = frame_unique_counts[order]
        single_view_results.append(metrics)
        view_masks[row] = collapse.astype(np.uint8)
    single_view_results.sort(
        key=lambda item: (
            item["collapsed_peripheral"]["collapsed_count"],
            item["collapsed_far"]["collapsed_count"],
            item["collapsed_extended"]["collapsed_count"],
            -item["workspace_order"],
        ),
        reverse=True,
    )

    segment_names = list(segments)
    segment_masks = np.zeros((len(segment_names), len(points)), dtype=np.uint8)
    segment_results: dict[str, Any] = {}
    scales = _scale_map(e3)
    for index, name in enumerate(segment_names):
        segment_orders = segments[name]
        rows = tuple(order_to_row[order] for order in segment_orders)
        metrics, collapse, _ = removal_sensitivity(
            points=points, observation_matrix=matrix, baseline_support=support,
            removed_view_indices=rows, peripheral_mask=peripheral, far_mask=far, weak_mask=weak,
            core_mask=is_core, sector_codes=sector, policy=policy,
        )
        metrics["workspace_orders"] = list(segment_orders)
        metrics["scale_profile"] = scale_profile(scales, segment_orders)
        segment_results[name] = metrics
        segment_masks[index] = collapse.astype(np.uint8)

    pairs = pair_cosupport_records(
        matrix, support, orders, peripheral_mask=peripheral, far_mask=far, weak_mask=weak
    )
    top_pairs = pairs[: int(args.top_pairs)]
    suspect_order_union = {order for values in segments.values() for order in values}
    touching_edges = _edges_touching(e3, suspect_order_union)

    sensitivity_npz = output / "i2_1_support_sensitivity_diagnostics.npz"
    np.savez_compressed(
        sensitivity_npz,
        points_reconstruction=points,
        colors_rgb=np.asarray(data["colors"], dtype=np.uint8),
        baseline_support=support.astype(np.int32),
        is_core=is_core.astype(np.uint8),
        is_peripheral=peripheral.astype(np.uint8),
        is_far_peripheral=far.astype(np.uint8),
        is_weak_peripheral=weak.astype(np.uint8),
        directional_sector_code=sector,
        selected_workspace_orders=np.asarray(orders, dtype=np.int32),
        single_view_collapse_masks=view_masks,
        segment_names=np.asarray(segment_names, dtype="U64"),
        segment_collapse_masks=segment_masks,
        voxel_size_reconstruction_units=np.asarray(voxel_size, dtype=np.float64),
        schema_version=np.asarray(I21_DIAGNOSIS_SCHEMA_VERSION, dtype=np.int32),
    )

    ranked_views = [
        {
            "workspace_order": int(item["workspace_order"]),
            "collapsed_peripheral": int(item["collapsed_peripheral"]["collapsed_count"]),
            "collapsed_far": int(item["collapsed_far"]["collapsed_count"]),
            "collapsed_core": int(item["collapsed_core"]["collapsed_count"]),
            "collapsed_extended": int(item["collapsed_extended"]["collapsed_count"]),
        }
        for item in single_view_results[:10]
    ]
    strongest_segment = max(
        segment_results.items(),
        key=lambda item: (
            item[1]["collapsed_peripheral"]["collapsed_count"],
            item[1]["collapsed_far"]["collapsed_count"],
        ),
    )
    summary = {
        "phase": "I2.1",
        "schema_version": I21_DIAGNOSIS_SCHEMA_VERSION,
        "status": "diagnosis_ready",
        "source_i1_dir": str(i1_dir),
        "source_i2_dir": str(i2_dir),
        "scene_id": str(i2["scene"]["scene_id"]),
        "selected_view_count": len(orders),
        "selected_workspace_orders": list(orders),
        "baseline": {
            "extended_points": len(points),
            "core_points": int(np.count_nonzero(is_core)),
            "peripheral_points": int(np.count_nonzero(peripheral)),
            "far_peripheral_points": int(np.count_nonzero(far)),
            "weak_peripheral_points": int(np.count_nonzero(weak)),
            "voxel_size_reconstruction_units": voxel_size,
        },
        "exact_support_accounting": {
            "pass": True,
            "recomputed_observations": int(matrix.sum(dtype=np.int64)),
            "baseline_observations": int(support.sum(dtype=np.int64)),
            "mismatched_voxels": 0,
        },
        "automatic_segments": {name: list(values) for name, values in segments.items()},
        "single_view_sensitivity": single_view_results,
        "segment_sensitivity": segment_results,
        "top_support2_cosupport_pairs": top_pairs,
        "suspect_edges": touching_edges,
        "diagnostic_findings": {
            "top_single_view_dependencies": ranked_views,
            "strongest_segment_by_peripheral_support_collapse": strongest_segment[0],
            "strongest_segment_orders": list(segments[strongest_segment[0]]),
            "peripheral_geometry_is_mostly_low_support": bool(
                float(i2["spatial"]["peripheral_retained"]["support_quantiles"]["p50"]) <= 2.0
            ),
            "far_geometry_present": int(np.count_nonzero(far)) > 0,
            "interpretation_scope": (
                "diagnostic support dependency only; exact refusion/re-filtering is intentionally deferred until this evidence "
                "identifies a targeted hypothesis"
            ),
        },
        "authority": {
            "geometry_changed": False,
            "filter_policy_changed": False,
            "sfm_rerun": False,
            "da3_rerun": False,
            "fusion_rerun": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_i2_closure": False,
            "authorizes_i3": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": (
                "Inspect support-collapse sensitivity for automatic early/late suspect segments and top support-2 co-support pairs; "
                "then decide whether a narrowly targeted exact counterfactual refusion is justified."
            ),
        },
        "artifacts": {
            "support_sensitivity_npz": str(sensitivity_npz.resolve()),
        },
        "policy": policy.as_dict(),
    }
    summary_path = _write_json(output / "i2_1_summary.json", summary)

    print("\nI2.1 targeted peripheral artifact diagnosis: DIAGNOSIS_READY")
    print(
        f"views={len(orders)} extended={len(points)} core={int(np.count_nonzero(is_core))} "
        f"peripheral={int(np.count_nonzero(peripheral))} far={int(np.count_nonzero(far))} weak={int(np.count_nonzero(weak))}"
    )
    print(
        f"exact_support_accounting=True observations={int(support.sum(dtype=np.int64))} "
        "mismatched_voxels=0"
    )
    for name in segment_names:
        item = segment_results[name]
        print(
            f"segment {name} orders={','.join(map(str, segments[name]))} "
            f"peripheral_collapse={item['collapsed_peripheral']['collapsed_count']} "
            f"({item['collapsed_peripheral']['collapsed_fraction']:.3f}) "
            f"far_collapse={item['collapsed_far']['collapsed_count']} "
            f"core_collapse={item['collapsed_core']['collapsed_count']}"
        )
    top = single_view_results[:5]
    print("top_single_view_dependencies=" + ",".join(str(item["workspace_order"]) for item in top))
    if top_pairs:
        print(
            "top_support2_pairs="
            + ",".join(
                f"{item['order_a']}-{item['order_b']}:{item['support2_peripheral_pair']}"
                for item in top_pairs[:8]
            )
        )
    print("geometry_changed=False filter_policy_changed=False sfm_rerun=False da3_rerun=False fusion_rerun=False")
    print("i2_closure_authorized=False mesh_authorization=False")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
