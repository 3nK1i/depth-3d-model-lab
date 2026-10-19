#!/usr/bin/env python3
"""D4.2 weak-link bridge diagnosis for a degraded D4.1 ten-view chain.

D4.2 does not change geometry.  It consumes the existing D4 raw ten-view
clouds plus the D4.1 qualification result, finds the single degraded adjacent
link, and evaluates the two skip-one bridge pairs around it using the exact
D4.1/D2 acceptance thresholds:

    left neighbour <-> right member of weak pair
    left member of weak pair <-> right neighbour

For the current 6..15 chain with weak 7<->8 this is 6<->8 and 7<->9.
The result is diagnostic evidence only; no view is removed automatically and
no fusion, filtering, persistence, metric claim, or authority promotion occurs.
"""

from __future__ import annotations

import argparse
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
from scripts.run_d4_1_ten_view_chain_diagnostics import _validate_d4_summary  # noqa: E402

_D42_EXPECTED_PHASE = "D4.1"
_D42_VIEW_COUNT = 10
_D42_MAX_POINTS_PER_DIRECTION = 120_000
_D42_NN_MAX_POINTS_PER_VIEW = 60_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d4-1-dir",
        type=Path,
        default=Path("/tmp/depth3d_d4_1_ten_view_qualification"),
        help="D4.1 output containing one degraded adjacent pair.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/depth3d_d4_2_weak_link_bridge_diagnosis"),
    )
    parser.add_argument("--max-points-per-direction", type=int, default=_D42_MAX_POINTS_PER_DIRECTION)
    parser.add_argument("--nn-max-points-per-view", type=int, default=_D42_NN_MAX_POINTS_PER_VIEW)
    args = parser.parse_args()
    if args.max_points_per_direction <= 0 or args.nn_max_points_per_view <= 0:
        parser.error("D4.2 point budgets must be positive.")
    return args


def _validate_d4_1_summary(payload: dict[str, Any]) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    if str(payload.get("phase")) != _D42_EXPECTED_PHASE:
        raise ReconstructionValidationError("D4.2 requires a D4.1 qualification result.")
    if bool(payload.get("pass")) or str(payload.get("status")) != "degraded":
        raise ReconstructionValidationError("D4.2 requires a degraded (not passed) D4.1 chain.")
    if int(payload.get("view_count", 0)) != _D42_VIEW_COUNT:
        raise ReconstructionValidationError("D4.2 requires the ten-view D4.1 chain.")
    pairs = payload.get("adjacent_pairs")
    if not isinstance(pairs, list) or len(pairs) != _D42_VIEW_COUNT - 1:
        raise ReconstructionValidationError("D4.2 requires all nine D4.1 adjacent-pair results.")
    degraded = [item for item in pairs if isinstance(item, dict) and str(item.get("status")) == "degraded"]
    if len(degraded) != 1:
        raise ReconstructionValidationError("D4.2 requires exactly one degraded D4.1 adjacent pair.")
    pair = degraded[0].get("pair")
    if not isinstance(pair, list) or len(pair) != 2:
        raise ReconstructionValidationError("D4.2 degraded pair metadata is invalid.")
    a_index, b_index = int(pair[0]), int(pair[1])
    if b_index != a_index + 1:
        raise ReconstructionValidationError("D4.2 requires an adjacent degraded pair.")
    if a_index <= 0 or b_index >= _D42_VIEW_COUNT - 1:
        raise ReconstructionValidationError("D4.2 weak pair must have one neighbour on each side for bridge diagnosis.")
    return (a_index, b_index), pairs


def _derive_bridge_pairs(weak_pair: tuple[int, int], *, view_count: int = _D42_VIEW_COUNT) -> dict[str, tuple[int, int]]:
    a_index, b_index = weak_pair
    if b_index != a_index + 1 or a_index <= 0 or b_index >= view_count - 1:
        raise ReconstructionValidationError("D4.2 cannot derive two skip-one bridges for this weak pair.")
    return {
        "left": (a_index - 1, b_index),
        "right": (a_index, b_index + 1),
    }


def _bridge_diagnosis(
    *,
    weak_pair: tuple[int, int],
    left_status: str,
    right_status: str,
) -> dict[str, Any]:
    allowed = {"pass", "degraded", "inconclusive"}
    if left_status not in allowed or right_status not in allowed:
        raise ReconstructionValidationError("D4.2 bridge status is invalid.")
    a_index, b_index = weak_pair
    if left_status == "pass" and right_status == "pass":
        return {
            "classification": "pair_specific_incompatibility",
            "preferred_single_view_exclusion_index": None,
            "next_action": "test_candidate_chain_without_one_weak-pair_member_before_fusion",
            "note": "Both skip-one bridges pass; neither weak-pair member is individually disqualified by bridge evidence.",
        }
    if left_status == "pass" and right_status == "degraded":
        return {
            "classification": "left_member_suspect",
            "preferred_single_view_exclusion_index": a_index,
            "next_action": "qualify_candidate_chain_excluding_left_member",
            "note": "Right weak-pair member bridges left successfully while left member fails to bridge right; diagnostic suggestion only.",
        }
    if left_status == "degraded" and right_status == "pass":
        return {
            "classification": "right_member_suspect",
            "preferred_single_view_exclusion_index": b_index,
            "next_action": "qualify_candidate_chain_excluding_right_member",
            "note": "Left weak-pair member bridges right successfully while right member fails to bridge left; diagnostic suggestion only.",
        }
    if "inconclusive" in {left_status, right_status}:
        return {
            "classification": "inconclusive_bridge_evidence",
            "preferred_single_view_exclusion_index": None,
            "next_action": "collect_or_select_better_overlap_evidence",
            "note": "At least one skip-one bridge lacks sufficient comparable evidence; do not infer a culprit.",
        }
    return {
        "classification": "span_instability",
        "preferred_single_view_exclusion_index": None,
        "next_action": "diagnose_local_depth_shape_or_select_alternative_views",
        "note": "Both skip-one bridges degrade; a single-view exclusion is not justified by bridge evidence alone.",
    }


def _thresholds_from_d4_1(payload: dict[str, Any]) -> dict[str, float | int]:
    raw = payload.get("thresholds")
    required = (
        "consistency_relative_tolerance",
        "occlusion_relative_margin",
        "minimum_comparable_samples",
        "minimum_consistent_ratio",
        "maximum_median_relative_residual",
        "maximum_p90_relative_residual",
        "maximum_front_conflict_ratio",
        "nn_support_scene_fraction",
    )
    if not isinstance(raw, dict) or any(name not in raw for name in required):
        raise ReconstructionValidationError("D4.2 requires the complete D4.1 threshold record.")
    return {
        "consistency_relative_tolerance": float(raw["consistency_relative_tolerance"]),
        "occlusion_relative_margin": float(raw["occlusion_relative_margin"]),
        "minimum_comparable_samples": int(raw["minimum_comparable_samples"]),
        "minimum_consistent_ratio": float(raw["minimum_consistent_ratio"]),
        "maximum_median_relative_residual": float(raw["maximum_median_relative_residual"]),
        "maximum_p90_relative_residual": float(raw["maximum_p90_relative_residual"]),
        "maximum_front_conflict_ratio": float(raw["maximum_front_conflict_ratio"]),
        "nn_support_scene_fraction": float(raw["nn_support_scene_fraction"]),
    }


def main() -> int:
    args = _parse_args()
    d4_1_dir = args.d4_1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    d4_1_summary = _read_json(d4_1_dir / "d4_1_summary.json")
    weak_pair, adjacent_pairs = _validate_d4_1_summary(d4_1_summary)
    bridges = _derive_bridge_pairs(weak_pair)
    thresholds = _thresholds_from_d4_1(d4_1_summary)

    d4_dir = Path(str(d4_1_summary["source_d4_dir"])).expanduser().resolve()
    d4_summary = _read_json(d4_dir / "d4_summary.json")
    views_payload = _validate_d4_summary(d4_summary)
    if [int(item["workspace_order"]) for item in views_payload] != [int(v) for v in d4_1_summary["selected_workspace_orders"]]:
        raise ReconstructionValidationError("D4.2 detected a D4/D4.1 selected-view mismatch.")

    source_b2_dir = Path(str(d4_summary["source_b2_dir"])).expanduser().resolve()
    sfm_result = _sfm_result_from_payload(_read_json(source_b2_dir / "sfm_result.json"))
    cameras_by_keyframe = {item.keyframe_id: item for item in sfm_result.cameras}
    camera_models_by_id = {item.camera_id: item for item in sfm_result.camera_models}

    combined_npz = Path(str(d4_summary["artifacts"]["combined_npz"])).expanduser().resolve()
    try:
        with np.load(combined_npz, allow_pickle=False) as payload:
            sparse_landmarks = np.asarray(
                payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64
            )
    except (OSError, ValueError) as exc:
        raise ReconstructionValidationError(f"Unable to read D4 combined cloud: {exc}") from exc

    needed_indices = sorted({index for pair in bridges.values() for index in pair})
    records: dict[int, dict[str, Any]] = {}
    view_points: list[np.ndarray] = []
    for index in needed_indices:
        item = views_payload[index]
        keyframe_id = str(item["keyframe_id"])
        camera_estimate = cameras_by_keyframe.get(keyframe_id)
        if camera_estimate is None:
            raise ReconstructionValidationError(f"D4.2 cannot find SfM camera estimate for {keyframe_id}.")
        camera_model = camera_models_by_id.get(camera_estimate.camera_id)
        if camera_model is None:
            raise ReconstructionValidationError(f"D4.2 cannot find SfM camera model {camera_estimate.camera_id}.")
        camera = reconstruction_camera_from_sfm_model(
            camera_model,
            keyframe_id=keyframe_id,
            reconstruction_frame_id="offline-sfm-reconstruction",
        )
        points, colors, transform = _load_view_cloud(Path(str(item["cloud_npz"])).expanduser().resolve())
        if not np.allclose(transform, camera_estimate.T_reconstruction_from_camera, rtol=0.0, atol=1e-9):
            raise ReconstructionValidationError("D4.2 detected pose drift between D4 cloud and SfM result.")
        aligned_depth, pixel_adapter = _load_aligned_depth(
            item,
            sfm_width=camera_model.image_size.width,
            sfm_height=camera_model.image_size.height,
        )
        view_points.append(points)
        records[index] = {
            "view_index": index,
            "keyframe_id": keyframe_id,
            "workspace_order": int(item["workspace_order"]),
            "scale_to_reconstruction_units": float(item["scale_to_reconstruction_units"]),
            "camera_center_reconstruction": np.asarray(item["camera_center_reconstruction"], dtype=np.float64),
            "camera_estimate": camera_estimate,
            "camera": camera,
            "aligned_depth": aligned_depth,
            "points": points,
            "colors": colors,
            "pixel_adapter": pixel_adapter,
        }

    scene_scale, scene_scale_source, scene_low, scene_high = _robust_scene_scale(sparse_landmarks, view_points)
    bridge_summaries: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for role in ("left", "right"):
        a_index, b_index = bridges[role]
        a = records[a_index]
        b = records[b_index]
        a_to_b, diag_a_to_b = _directed_reprojection_diagnostic(
            a["points"],
            target_camera=b["camera"],
            target_T_reconstruction_from_camera=b["camera_estimate"].T_reconstruction_from_camera,
            target_aligned_depth=b["aligned_depth"],
            maximum_points=int(args.max_points_per_direction),
            consistency_relative_tolerance=float(thresholds["consistency_relative_tolerance"]),
            occlusion_relative_margin=float(thresholds["occlusion_relative_margin"]),
        )
        b_to_a, diag_b_to_a = _directed_reprojection_diagnostic(
            b["points"],
            target_camera=a["camera"],
            target_T_reconstruction_from_camera=a["camera_estimate"].T_reconstruction_from_camera,
            target_aligned_depth=a["aligned_depth"],
            maximum_points=int(args.max_points_per_direction),
            consistency_relative_tolerance=float(thresholds["consistency_relative_tolerance"]),
            occlusion_relative_margin=float(thresholds["occlusion_relative_margin"]),
        )
        status_ab, failures_ab = _assess_direction(
            a_to_b,
            minimum_comparable_samples=int(thresholds["minimum_comparable_samples"]),
            minimum_consistent_ratio=float(thresholds["minimum_consistent_ratio"]),
            maximum_median_relative_residual=float(thresholds["maximum_median_relative_residual"]),
            maximum_p90_relative_residual=float(thresholds["maximum_p90_relative_residual"]),
            maximum_front_conflict_ratio=float(thresholds["maximum_front_conflict_ratio"]),
        )
        status_ba, failures_ba = _assess_direction(
            b_to_a,
            minimum_comparable_samples=int(thresholds["minimum_comparable_samples"]),
            minimum_consistent_ratio=float(thresholds["minimum_consistent_ratio"]),
            maximum_median_relative_residual=float(thresholds["maximum_median_relative_residual"]),
            maximum_p90_relative_residual=float(thresholds["maximum_p90_relative_residual"]),
            maximum_front_conflict_ratio=float(thresholds["maximum_front_conflict_ratio"]),
        )
        status = _pair_status(status_ab, status_ba)
        statuses[role] = status
        baseline = float(np.linalg.norm(a["camera_center_reconstruction"] - b["camera_center_reconstruction"]))
        nn = _spatial_nn_diagnostic(
            a["points"],
            b["points"],
            scene_scale=scene_scale,
            maximum_points=int(args.nn_max_points_per_view),
            support_scene_fraction=float(thresholds["nn_support_scene_fraction"]),
        )
        diagnostic_path = output_dir / f"d4_2_bridge_{role}_diagnostic.npz"
        np.savez_compressed(
            diagnostic_path,
            bridge_role=np.asarray(role),
            pair=np.asarray([a_index, b_index], dtype=np.uint8),
            keyframe_ids=np.asarray([a["keyframe_id"], b["keyframe_id"]], dtype="U160"),
            workspace_orders=np.asarray([a["workspace_order"], b["workspace_order"]], dtype=np.int32),
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
        bridge_summaries.append(
            {
                "role": role,
                "pair": [a_index, b_index],
                "pair_label": f"{a_index}-{b_index}",
                "workspace_orders": [a["workspace_order"], b["workspace_order"]],
                "keyframe_ids": [a["keyframe_id"], b["keyframe_id"]],
                "status": status,
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

    weak_summary = next(
        item for item in adjacent_pairs if [int(v) for v in item["pair"]] == [weak_pair[0], weak_pair[1]]
    )
    diagnosis = _bridge_diagnosis(
        weak_pair=weak_pair,
        left_status=statuses["left"],
        right_status=statuses["right"],
    )
    preferred_index = diagnosis["preferred_single_view_exclusion_index"]
    if preferred_index is not None:
        diagnosis["preferred_single_view_exclusion_workspace_order"] = int(views_payload[int(preferred_index)]["workspace_order"])
        diagnosis["preferred_single_view_exclusion_keyframe_id"] = str(views_payload[int(preferred_index)]["keyframe_id"])
    else:
        diagnosis["preferred_single_view_exclusion_workspace_order"] = None
        diagnosis["preferred_single_view_exclusion_keyframe_id"] = None

    summary = {
        "phase": "D4.2",
        "pass": True,
        "status": "diagnostic_complete",
        "diagnostic_complete": True,
        "ten_view_chain_qualified": False,
        "authorizes_fusion": False,
        "source_d4_1_dir": str(d4_1_dir),
        "source_d4_dir": str(d4_dir),
        "source_sfm_session_id": str(d4_summary["source_sfm_session_id"]),
        "selected_workspace_orders": [int(item["workspace_order"]) for item in views_payload],
        "weak_adjacent_pair": {
            "pair": [weak_pair[0], weak_pair[1]],
            "workspace_orders": [int(v) for v in weak_summary["workspace_orders"]],
            "keyframe_ids": [str(v) for v in weak_summary["keyframe_ids"]],
            "status": str(weak_summary["status"]),
            "a_to_b": weak_summary["a_to_b"],
            "b_to_a": weak_summary["b_to_a"],
        },
        "thresholds_reused_unchanged_from_d4_1": thresholds,
        "bridge_pairs": bridge_summaries,
        "diagnosis": diagnosis,
        "scene_scale": {
            "robust_diagonal_reconstruction_units": scene_scale,
            "source": scene_scale_source,
            "quantile_bounds_min": scene_low,
            "quantile_bounds_max": scene_high,
            "physical_unit_claimed": False,
        },
        "authority": {
            "sfm_scale_status": "arbitrary",
            "sfm_world_alignment_status": "arbitrary",
            "metric_scale_claimed": False,
            "spatial_merge_applied": False,
            "voxel_fusion_applied": False,
            "multi_view_filtering_applied": False,
            "canonical_geometry_authoritative": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "note": "D4.2 bridge diagnosis only; no geometry or source-view selection is mutated automatically.",
        },
    }
    summary_path = output_dir / "d4_2_summary.json"
    _write_json(summary_path, summary)

    print("\nD4.2 weak-link bridge diagnosis: COMPLETE")
    print(
        f"weak_pair indices={weak_pair[0]}<->{weak_pair[1]} "
        f"orders={weak_summary['workspace_orders'][0]}<->{weak_summary['workspace_orders'][1]} status=degraded"
    )
    for item in bridge_summaries:
        a_to_b = item["a_to_b"]
        b_to_a = item["b_to_a"]
        print(
            f"bridge[{item['role']}] indices={item['pair'][0]}<->{item['pair'][1]} "
            f"orders={item['workspace_orders'][0]}<->{item['workspace_orders'][1]} status={item['status']} "
            f"baseline_scene_fraction={item['camera_baseline_scene_fraction']:.4f} "
            f"scale_ratio={item['scale_ratio_max_to_min']:.4f}"
        )
        print(
            f"  A->B status={a_to_b['status']} comparable={a_to_b['comparable_samples']} "
            f"consistent={a_to_b['consistent_ratio_of_comparable']:.3f} "
            f"front_conflict={a_to_b['front_conflict_ratio_of_comparable']:.3f} "
            f"median={_format_optional(a_to_b['median_absolute_relative_depth_residual'])} "
            f"p90={_format_optional(a_to_b['p90_absolute_relative_depth_residual'])}"
        )
        print(
            f"  B->A status={b_to_a['status']} comparable={b_to_a['comparable_samples']} "
            f"consistent={b_to_a['consistent_ratio_of_comparable']:.3f} "
            f"front_conflict={b_to_a['front_conflict_ratio_of_comparable']:.3f} "
            f"median={_format_optional(b_to_a['median_absolute_relative_depth_residual'])} "
            f"p90={_format_optional(b_to_a['p90_absolute_relative_depth_residual'])}"
        )
        print(f"  diagnostic={item['diagnostic_npz']}")
    print(
        f"diagnosis={diagnosis['classification']} "
        f"preferred_exclusion_order={diagnosis['preferred_single_view_exclusion_workspace_order']}"
    )
    print("No view removed; no fusion/filtering/persistence/authority promotion performed.")
    print(f"summary={summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
