#!/usr/bin/env python3
"""D3.1 five-view dense-chain qualification diagnostics.

D3.1 consumes an unfused D3 five-view accumulation and applies the already
validated D2 bidirectional reprojection/depth-consistency diagnostic to every
adjacent pair in the chain: 0<->1, 1<->2, 2<->3, and 3<->4.

The five-view chain passes only when all four adjacent pairs pass in both
directions under the unchanged D2 thresholds.  This is qualification only:
no points are fused, filtered, persisted, or promoted to geometry/measurement
authority, and the arbitrary SfM gauge remains arbitrary.
"""

from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import reconstruction_camera_from_sfm_model  # noqa: E402
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from scripts.run_d1_three_view_canonical_accumulation import _sfm_result_from_payload  # noqa: E402
from scripts.run_d2_three_view_overlap_diagnostics import (  # noqa: E402
    _assess_direction,
    _directed_reprojection_diagnostic,
    _format_optional,
    _load_aligned_depth,
    _load_view_cloud,
    _pair_status,
    _read_json,
    _robust_scene_scale,
    _spatial_nn_diagnostic,
    _write_json,
)

_D31_VIEW_COUNT = 5
_D31_ADJACENT_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 4))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d3-dir",
        type=Path,
        default=Path("/tmp/eia_d3_1_five_view_candidate"),
        help="Successful unfused D3 five-view output directory to qualify.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/eia_d3_1_five_view_qualification"),
    )
    parser.add_argument("--max-points-per-direction", type=int, default=120_000)
    parser.add_argument("--nn-max-points-per-view", type=int, default=60_000)
    parser.add_argument("--consistency-relative-tolerance", type=float, default=0.12)
    parser.add_argument("--occlusion-relative-margin", type=float, default=0.20)
    parser.add_argument("--minimum-comparable-samples", type=int, default=1_000)
    parser.add_argument("--minimum-consistent-ratio", type=float, default=0.50)
    parser.add_argument("--maximum-median-relative-residual", type=float, default=0.10)
    parser.add_argument("--maximum-p90-relative-residual", type=float, default=0.25)
    parser.add_argument("--maximum-front-conflict-ratio", type=float, default=0.25)
    parser.add_argument("--nn-support-scene-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.max_points_per_direction <= 0 or args.nn_max_points_per_view <= 0:
        parser.error("D3.1 point budgets must be positive.")
    if args.minimum_comparable_samples <= 0:
        parser.error("--minimum-comparable-samples must be positive.")
    for name in (
        "consistency_relative_tolerance",
        "occlusion_relative_margin",
        "minimum_consistent_ratio",
        "maximum_median_relative_residual",
        "maximum_p90_relative_residual",
        "maximum_front_conflict_ratio",
        "nn_support_scene_fraction",
    ):
        value = float(getattr(args, name))
        if not isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if args.consistency_relative_tolerance >= args.occlusion_relative_margin:
        parser.error("--consistency-relative-tolerance must be smaller than --occlusion-relative-margin.")
    if args.minimum_consistent_ratio > 1.0 or args.maximum_front_conflict_ratio > 1.0:
        parser.error("Ratio thresholds must not exceed 1.0.")
    return args


def _validate_d3_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(payload.get("pass")) or str(payload.get("phase")) != "D3":
        raise ReconstructionValidationError("D3.1 requires a successful D3 result.")
    if int(payload.get("view_count", 0)) != _D31_VIEW_COUNT:
        raise ReconstructionValidationError("D3.1 requires exactly five D3 views.")
    raw = payload.get("raw_accumulation")
    if not isinstance(raw, dict):
        raise ReconstructionValidationError("D3.1 requires D3 raw-accumulation metadata.")
    if bool(raw.get("spatial_merge_applied")) or bool(raw.get("voxel_fusion_applied")) or bool(raw.get("multi_view_filtering_applied")):
        raise ReconstructionValidationError("D3.1 must qualify unfused/unfiltered D3 geometry.")
    views = payload.get("views")
    if not isinstance(views, list) or len(views) != _D31_VIEW_COUNT:
        raise ReconstructionValidationError("D3.1 requires exactly five D3 view summaries.")
    for index, item in enumerate(views):
        if not isinstance(item, dict) or int(item.get("view_index", -1)) != index:
            raise ReconstructionValidationError("D3.1 requires D3 view indices 0..4 in order.")
    return views


def _overall_chain_status(pair_statuses: list[str]) -> tuple[str, bool, int]:
    values = set(pair_statuses)
    if "degraded" in values:
        return "degraded", False, 2
    if "inconclusive" in values:
        return "inconclusive", False, 3
    if pair_statuses and values == {"pass"}:
        return "pass", True, 0
    raise ReconstructionValidationError("D3.1 received invalid adjacent-pair statuses.")


def main() -> int:
    args = _parse_args()
    d3_dir = args.d3_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    d3_summary = _read_json(d3_dir / "d3_summary.json")
    views_payload = _validate_d3_summary(d3_summary)

    source_b2_dir = Path(str(d3_summary["source_b2_dir"])).expanduser().resolve()
    sfm_result = _sfm_result_from_payload(_read_json(source_b2_dir / "sfm_result.json"))
    cameras_by_keyframe = {item.keyframe_id: item for item in sfm_result.cameras}
    camera_models_by_id = {item.camera_id: item for item in sfm_result.camera_models}

    combined_npz = Path(str(d3_summary["artifacts"]["combined_npz"])).expanduser().resolve()
    try:
        with np.load(combined_npz, allow_pickle=False) as payload:
            sparse_landmarks = np.asarray(
                payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64
            )
    except (OSError, ValueError) as exc:
        raise ReconstructionValidationError(f"Unable to read D3 combined cloud: {exc}") from exc

    view_records: list[dict[str, Any]] = []
    view_points: list[np.ndarray] = []
    for index, item in enumerate(views_payload):
        keyframe_id = str(item["keyframe_id"])
        camera_estimate = cameras_by_keyframe.get(keyframe_id)
        if camera_estimate is None:
            raise ReconstructionValidationError(f"D3.1 cannot find SfM camera estimate for {keyframe_id}.")
        camera_model = camera_models_by_id.get(camera_estimate.camera_id)
        if camera_model is None:
            raise ReconstructionValidationError(f"D3.1 cannot find SfM camera model {camera_estimate.camera_id}.")
        camera = reconstruction_camera_from_sfm_model(
            camera_model,
            keyframe_id=keyframe_id,
            reconstruction_frame_id="offline-sfm-reconstruction",
        )
        points, colors, transform = _load_view_cloud(Path(str(item["cloud_npz"])).expanduser().resolve())
        if not np.allclose(transform, camera_estimate.T_reconstruction_from_camera, rtol=0.0, atol=1e-9):
            raise ReconstructionValidationError("D3.1 detected pose drift between D3 cloud and SfM result.")
        aligned_depth, pixel_adapter = _load_aligned_depth(
            item,
            sfm_width=camera_model.image_size.width,
            sfm_height=camera_model.image_size.height,
        )
        view_points.append(points)
        view_records.append(
            {
                "view_index": index,
                "keyframe_id": keyframe_id,
                "workspace_order": int(item["workspace_order"]),
                "scale_to_reconstruction_units": float(item["scale_to_reconstruction_units"]),
                "relative_scale_uncertainty": float(item["relative_scale_uncertainty"]),
                "camera_center_reconstruction": np.asarray(item["camera_center_reconstruction"], dtype=np.float64),
                "camera_estimate": camera_estimate,
                "camera": camera,
                "aligned_depth": aligned_depth,
                "points": points,
                "colors": colors,
                "pixel_adapter": pixel_adapter,
            }
        )

    scene_scale, scene_scale_source, scene_low, scene_high = _robust_scene_scale(sparse_landmarks, view_points)
    thresholds = {
        "consistency_relative_tolerance": float(args.consistency_relative_tolerance),
        "occlusion_relative_margin": float(args.occlusion_relative_margin),
        "minimum_comparable_samples": int(args.minimum_comparable_samples),
        "minimum_consistent_ratio": float(args.minimum_consistent_ratio),
        "maximum_median_relative_residual": float(args.maximum_median_relative_residual),
        "maximum_p90_relative_residual": float(args.maximum_p90_relative_residual),
        "maximum_front_conflict_ratio": float(args.maximum_front_conflict_ratio),
        "nn_support_scene_fraction": float(args.nn_support_scene_fraction),
    }

    pair_summaries: list[dict[str, Any]] = []
    for a_index, b_index in _D31_ADJACENT_PAIRS:
        a = view_records[a_index]
        b = view_records[b_index]
        a_to_b, diag_a_to_b = _directed_reprojection_diagnostic(
            a["points"],
            target_camera=b["camera"],
            target_T_reconstruction_from_camera=b["camera_estimate"].T_reconstruction_from_camera,
            target_aligned_depth=b["aligned_depth"],
            maximum_points=int(args.max_points_per_direction),
            consistency_relative_tolerance=float(args.consistency_relative_tolerance),
            occlusion_relative_margin=float(args.occlusion_relative_margin),
        )
        b_to_a, diag_b_to_a = _directed_reprojection_diagnostic(
            b["points"],
            target_camera=a["camera"],
            target_T_reconstruction_from_camera=a["camera_estimate"].T_reconstruction_from_camera,
            target_aligned_depth=a["aligned_depth"],
            maximum_points=int(args.max_points_per_direction),
            consistency_relative_tolerance=float(args.consistency_relative_tolerance),
            occlusion_relative_margin=float(args.occlusion_relative_margin),
        )
        status_ab, failures_ab = _assess_direction(
            a_to_b,
            minimum_comparable_samples=int(args.minimum_comparable_samples),
            minimum_consistent_ratio=float(args.minimum_consistent_ratio),
            maximum_median_relative_residual=float(args.maximum_median_relative_residual),
            maximum_p90_relative_residual=float(args.maximum_p90_relative_residual),
            maximum_front_conflict_ratio=float(args.maximum_front_conflict_ratio),
        )
        status_ba, failures_ba = _assess_direction(
            b_to_a,
            minimum_comparable_samples=int(args.minimum_comparable_samples),
            minimum_consistent_ratio=float(args.minimum_consistent_ratio),
            maximum_median_relative_residual=float(args.maximum_median_relative_residual),
            maximum_p90_relative_residual=float(args.maximum_p90_relative_residual),
            maximum_front_conflict_ratio=float(args.maximum_front_conflict_ratio),
        )
        pair_status = _pair_status(status_ab, status_ba)
        baseline = float(np.linalg.norm(a["camera_center_reconstruction"] - b["camera_center_reconstruction"]))
        nn = _spatial_nn_diagnostic(
            a["points"],
            b["points"],
            scene_scale=scene_scale,
            maximum_points=int(args.nn_max_points_per_view),
            support_scene_fraction=float(args.nn_support_scene_fraction),
        )
        diagnostic_path = output_dir / f"d3_1_pair_{a_index}{b_index}_diagnostic.npz"
        np.savez_compressed(
            diagnostic_path,
            pair=np.asarray([a_index, b_index], dtype=np.uint8),
            keyframe_ids=np.asarray([a["keyframe_id"], b["keyframe_id"]], dtype="U160"),
            a_to_b_points_reconstruction=diag_a_to_b["points_reconstruction"],
            a_to_b_classification=diag_a_to_b["classification"],
            a_to_b_signed_relative_depth_residual=diag_a_to_b["signed_relative_depth_residual"],
            a_to_b_absolute_relative_depth_residual=diag_a_to_b["absolute_relative_depth_residual"],
            b_to_a_points_reconstruction=diag_b_to_a["points_reconstruction"],
            b_to_a_classification=diag_b_to_a["classification"],
            b_to_a_signed_relative_depth_residual=diag_b_to_a["signed_relative_depth_residual"],
            b_to_a_absolute_relative_depth_residual=diag_b_to_a["absolute_relative_depth_residual"],
            sparse_landmarks_reconstruction=sparse_landmarks,
        )
        pair_summaries.append(
            {
                "pair": [a_index, b_index],
                "pair_label": f"{a_index}-{b_index}",
                "workspace_orders": [a["workspace_order"], b["workspace_order"]],
                "keyframe_ids": [a["keyframe_id"], b["keyframe_id"]],
                "status": pair_status,
                "camera_baseline_reconstruction_units": baseline,
                "camera_baseline_scene_fraction": float(baseline / scene_scale),
                "scale_ratio_max_to_min": float(
                    max(a["scale_to_reconstruction_units"], b["scale_to_reconstruction_units"])
                    / min(a["scale_to_reconstruction_units"], b["scale_to_reconstruction_units"])
                ),
                "a_to_b": {**a_to_b, "status": status_ab, "failure_reasons": failures_ab},
                "b_to_a": {**b_to_a, "status": status_ba, "failure_reasons": failures_ba},
                "spatial_nearest_neighbour_secondary": nn,
                "diagnostic_npz": str(diagnostic_path.resolve()),
            }
        )

    overall_status, passed, exit_code = _overall_chain_status([str(item["status"]) for item in pair_summaries])
    summary = {
        "phase": "D3.1",
        "pass": passed,
        "status": overall_status,
        "source_d3_dir": str(d3_dir),
        "source_d3_pass": True,
        "source_sfm_session_id": str(d3_summary["source_sfm_session_id"]),
        "view_count": _D31_VIEW_COUNT,
        "adjacent_pair_count": len(pair_summaries),
        "selected_workspace_orders": [int(item["workspace_order"]) for item in views_payload],
        "selected_keyframe_ids": [str(item["keyframe_id"]) for item in views_payload],
        "scene_scale": {
            "robust_diagonal_reconstruction_units": scene_scale,
            "source": scene_scale_source,
            "quantile_bounds_min": scene_low,
            "quantile_bounds_max": scene_high,
            "physical_unit_claimed": False,
        },
        "thresholds": thresholds,
        "adjacent_pairs": pair_summaries,
        "decision": {
            "required_pairs": ["0-1", "1-2", "2-3", "3-4"],
            "rule": "All four adjacent five-view pairs must pass bidirectional reprojection-depth consistency under unchanged D2 thresholds.",
            "status": overall_status,
        },
        "authority": {
            "sfm_scale_status": str(d3_summary["authority"]["sfm_scale_status"]),
            "sfm_world_alignment_status": str(d3_summary["authority"]["sfm_world_alignment_status"]),
            "metric_scale_claimed": False,
            "canonical_geometry_authoritative": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "spatial_merge_applied": False,
            "voxel_fusion_applied": False,
            "multi_view_filtering_applied": False,
            "note": "D3.1 validation-only five-view chain qualification; D3 raw geometry remains unchanged.",
        },
    }
    summary_path = _write_json(output_dir / "d3_1_summary.json", summary)

    print(f"\nD3.1 five-view chain qualification: {overall_status.upper()}")
    print(
        f"workspace_orders={'->'.join(str(item['workspace_order']) for item in views_payload)} "
        f"scene_scale={scene_scale:.6f} reconstruction_units physical_unit_claimed=False"
    )
    for item in pair_summaries:
        a_to_b = item["a_to_b"]
        b_to_a = item["b_to_a"]
        print(
            f"pair[{item['pair_label']}] orders={item['workspace_orders'][0]}<->{item['workspace_orders'][1]} "
            f"status={item['status']} baseline_scene_fraction={item['camera_baseline_scene_fraction']:.4f} "
            f"scale_ratio={item['scale_ratio_max_to_min']:.4f}"
        )
        print(
            f"  {item['pair'][0]}->{item['pair'][1]} status={a_to_b['status']} "
            f"comparable={a_to_b['comparable_samples']} consistent={a_to_b['consistent_ratio_of_comparable']:.3f} "
            f"front_conflict={a_to_b['front_conflict_ratio_of_comparable']:.3f} "
            f"median={_format_optional(a_to_b['median_absolute_relative_depth_residual'])} "
            f"p90={_format_optional(a_to_b['p90_absolute_relative_depth_residual'])} "
            f"occluded={a_to_b['occluded_ratio_of_valid_target']:.3f}"
        )
        print(
            f"  {item['pair'][1]}->{item['pair'][0]} status={b_to_a['status']} "
            f"comparable={b_to_a['comparable_samples']} consistent={b_to_a['consistent_ratio_of_comparable']:.3f} "
            f"front_conflict={b_to_a['front_conflict_ratio_of_comparable']:.3f} "
            f"median={_format_optional(b_to_a['median_absolute_relative_depth_residual'])} "
            f"p90={_format_optional(b_to_a['p90_absolute_relative_depth_residual'])} "
            f"occluded={b_to_a['occluded_ratio_of_valid_target']:.3f}"
        )
        print(f"  diagnostic={item['diagnostic_npz']}")
    print("spatial_merge_applied=False voxel_fusion_applied=False multi_view_filtering_applied=False")
    print("sfm_scale_status=arbitrary metric_scale_claimed=False")
    print("canonical_geometry_authoritative=False measurement_authoritative=False certified_metrology=False")
    print(f"summary={summary_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
