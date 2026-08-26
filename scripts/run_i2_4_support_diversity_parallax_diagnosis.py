#!/usr/bin/env python3
"""I2.4 support-diversity / parallax diagnosis for retained full-video geometry.

Consumes the already-generated I1/I2/I2.1 evidence.  It reconstructs exact
per-view retained-voxel membership, loads the saved SfM camera poses, and
measures whether retained peripheral/far geometry is supported by diverse
viewpoints or only by narrow-baseline / low-parallax local camera groups.

No geometry, F policy, SfM, DA3, fusion, persistence, UI, metric authority, or
mesh authority is changed by this diagnostic.
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
from core.reconstruction.offline_dense_peripheral_diagnosis import validate_observation_matrix  # noqa: E402
from core.reconstruction.offline_dense_support_diversity import (  # noqa: E402
    I24_SUPPORT_DIVERSITY_SCHEMA_VERSION,
    I24SupportDiversityPolicy,
    classify_low_diversity,
    derive_reference_thresholds,
    diagnostic_signal,
    diversity_region_statistics,
    support2_pair_diversity_records,
    support_diversity_metrics,
)
from scripts.run_i2_1_targeted_peripheral_diagnosis import (  # noqa: E402
    _load_i2_diagnostics,
    _membership_matrix,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1-dir", type=Path, default=Path("/tmp/eia_i1_offline_dense_campaign_recovered"))
    parser.add_argument("--i2-dir", type=Path, default=Path("/tmp/eia_i2_full_video_quality"))
    parser.add_argument("--i2-1-dir", type=Path, default=Path("/tmp/eia_i2_1_targeted_peripheral_diagnosis"))
    parser.add_argument("--i2-3-dir", type=Path, default=Path("/tmp/eia_i2_3_balanced_cross_segment_counterfactuals"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/eia_i2_4_support_diversity_parallax_diagnosis"))
    parser.add_argument("--reference-quantile", type=float, default=0.25)
    parser.add_argument("--maximum-support-for-low-diversity", type=int, default=3)
    parser.add_argument("--minimum-reference-points", type=int, default=128)
    parser.add_argument("--top-pairs", type=int, default=20)
    args = parser.parse_args()
    if args.top_pairs <= 0:
        parser.error("--top-pairs must be positive")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read required I2.4 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I2.4 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _camera_centers_from_selected_clouds(e3_summary: Mapping[str, Any], orders: tuple[int, ...]) -> np.ndarray:
    frames = e3_summary.get("frames")
    if not isinstance(frames, list) or len(frames) != len(orders):
        raise ReconstructionValidationError("I2.4 requires selected I1 fusion frame provenance for every view.")
    centers: list[np.ndarray] = []
    seen_orders: list[int] = []
    for frame in frames:
        order = int(frame["workspace_order"])
        cloud_path = Path(str(frame["cloud_npz"])).expanduser().resolve()
        if not cloud_path.is_file():
            raise ReconstructionValidationError(f"I2.4 selected source cloud is unavailable: {cloud_path}")
        with np.load(cloud_path, allow_pickle=False) as payload:
            if "T_reconstruction_from_camera" not in payload:
                raise ReconstructionValidationError(f"I2.4 source cloud lacks camera pose: {cloud_path}")
            pose = np.asarray(payload["T_reconstruction_from_camera"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ReconstructionValidationError(f"I2.4 invalid T_reconstruction_from_camera for order {order}")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-8):
            raise ReconstructionValidationError(f"I2.4 non-rigid homogeneous camera pose for order {order}")
        centers.append(pose[:3, 3].copy())
        seen_orders.append(order)
    if tuple(seen_orders) != tuple(orders):
        raise ReconstructionValidationError("I2.4 selected camera-pose order does not match exact observation membership order.")
    result = np.asarray(centers, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ReconstructionValidationError("I2.4 camera centers must be finite.")
    return result


def _region_comparison(reference: Mapping[str, Any], region: Mapping[str, Any]) -> dict[str, Any]:
    ref_angle = reference["maximum_pair_parallax_degrees_quantiles"].get("p50")
    reg_angle = region["maximum_pair_parallax_degrees_quantiles"].get("p50")
    ref_ratio = reference["maximum_baseline_to_range_ratio_quantiles"].get("p50")
    reg_ratio = region["maximum_baseline_to_range_ratio_quantiles"].get("p50")
    return {
        "median_parallax_ratio_to_reference": (
            float(reg_angle / ref_angle) if ref_angle not in (None, 0.0) and reg_angle is not None else None
        ),
        "median_baseline_to_range_ratio_to_reference": (
            float(reg_ratio / ref_ratio) if ref_ratio not in (None, 0.0) and reg_ratio is not None else None
        ),
        "low_diversity_fraction_delta_vs_reference": float(
            region["low_diversity_fraction"] - reference["low_diversity_fraction"]
        ),
    }


def main() -> int:
    args = _args()
    i1_dir = args.i1_dir.expanduser().resolve()
    i2_dir = args.i2_dir.expanduser().resolve()
    i21_dir = args.i2_1_dir.expanduser().resolve()
    i23_dir = args.i2_3_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I2.4 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i1 = _read_json(i1_dir / "i1_campaign_summary.json")
    i2 = _read_json(i2_dir / "i2_summary.json")
    i21 = _read_json(i21_dir / "i2_1_summary.json")
    i23 = _read_json(i23_dir / "i2_3_summary.json")
    e3 = _read_json(i1_dir / "04_qualified_fusion" / "e3_summary.json")
    if str(i1.get("status")) != "pass" or not bool(i1.get("pipeline_execution_complete")):
        raise ReconstructionValidationError("I2.4 requires a completed passing I1 campaign.")
    if str(i2.get("phase")) != "I2" or not bool(i2.get("automated_pass")):
        raise ReconstructionValidationError("I2.4 requires passing automated I2 evidence.")
    if str(i21.get("status")) != "diagnosis_ready" or not bool(i21.get("exact_support_accounting", {}).get("pass")):
        raise ReconstructionValidationError("I2.4 requires completed exact-support I2.1 diagnosis.")
    if str(i21.get("scene_id")) != str(i2.get("scene", {}).get("scene_id")):
        raise ReconstructionValidationError("I2.4 I2/I2.1 scene identity mismatch.")
    if str(i23.get("status")) != "balanced_counterfactuals_ready" or str(i23.get("scene_id")) != str(i2["scene"]["scene_id"]):
        raise ReconstructionValidationError("I2.4 requires completed same-scene I2.3 evidence.")
    balanced = i23.get("ranking", {}).get("balanced_candidates_for_manual_review", [])
    if balanced:
        raise ReconstructionValidationError("I2.4 is intended after I2.3 found no balanced whole-view exclusion candidate.")

    data = _load_i2_diagnostics(i2)
    points = np.asarray(data["points"], dtype=np.float64)
    support = np.asarray(data["support"], dtype=np.int64)
    is_core = np.asarray(data["is_core"], dtype=bool)
    peripheral = np.asarray(data["peripheral"], dtype=bool)
    far = np.asarray(data["far"], dtype=bool)
    weak = np.asarray(data["weak"], dtype=bool)
    voxel_size = float(data["voxel_size"])

    orders, matrix, unique_counts = _membership_matrix(e3_summary=e3, points=points, voxel_size=voxel_size)
    matrix, exact_support = validate_observation_matrix(matrix, support, minimum_support_views=2)
    if int(matrix.sum(dtype=np.int64)) != int(i21["exact_support_accounting"]["recomputed_observations"]):
        raise ReconstructionValidationError("I2.4 exact support observation total disagrees with I2.1.")
    centers = _camera_centers_from_selected_clouds(e3, orders)

    policy = I24SupportDiversityPolicy(
        reference_quantile=float(args.reference_quantile),
        maximum_support_for_low_diversity=int(args.maximum_support_for_low_diversity),
        minimum_reference_points=int(args.minimum_reference_points),
    )
    metrics = support_diversity_metrics(points, matrix, centers, orders)
    if not np.array_equal(np.asarray(metrics["support"], dtype=np.int64), exact_support):
        raise ReconstructionValidationError("I2.4 metric support disagrees with exact retained support.")

    thresholds, reference_mask = derive_reference_thresholds(
        metrics, core_mask=is_core, peripheral_mask=peripheral, policy=policy
    )
    low_diversity = classify_low_diversity(metrics, thresholds, policy=policy)
    masks = {
        "central_core_reference": reference_mask,
        "core": is_core,
        "central_extended": ~peripheral,
        "peripheral": peripheral,
        "far_peripheral": far,
        "weak_peripheral": weak,
        "extended": np.ones(len(points), dtype=bool),
    }
    regions = {
        name: diversity_region_statistics(mask, metrics=metrics, low_diversity_mask=low_diversity)
        for name, mask in masks.items()
    }
    reference_stats = regions["central_core_reference"]
    for name in ("peripheral", "far_peripheral", "weak_peripheral"):
        regions[name]["comparison_to_central_core_reference"] = _region_comparison(reference_stats, regions[name])

    signal = diagnostic_signal(
        reference_stats=reference_stats,
        peripheral_stats=regions["peripheral"],
        far_stats=regions["far_peripheral"],
        policy=policy,
    )

    pairs = support2_pair_diversity_records(
        points=points,
        observation_matrix=matrix,
        camera_centers=centers,
        view_orders=orders,
        peripheral_mask=peripheral,
        far_mask=far,
        weak_mask=weak,
        core_mask=is_core,
        low_diversity_mask=low_diversity,
    )
    top_pairs = pairs[: int(args.top_pairs)]
    i21_pairs = {
        (int(item["order_a"]), int(item["order_b"]))
        for item in i21.get("top_support2_cosupport_pairs", [])[: int(args.top_pairs)]
    }
    i24_pairs = {(int(item["order_a"]), int(item["order_b"])) for item in top_pairs}

    diagnostics_npz = output / "i2_4_support_diversity_diagnostics.npz"
    np.savez_compressed(
        diagnostics_npz,
        points_reconstruction=points,
        colors_rgb=np.asarray(data["colors"], dtype=np.uint8),
        baseline_support=exact_support.astype(np.int32),
        is_core=is_core.astype(np.uint8),
        is_peripheral=peripheral.astype(np.uint8),
        is_far_peripheral=far.astype(np.uint8),
        is_weak_peripheral=weak.astype(np.uint8),
        is_reference_central_core=reference_mask.astype(np.uint8),
        is_low_support_diversity=low_diversity.astype(np.uint8),
        maximum_pair_parallax_degrees=np.asarray(metrics["maximum_pair_parallax_degrees"], dtype=np.float32),
        maximum_baseline_to_range_ratio=np.asarray(metrics["maximum_baseline_to_range_ratio"], dtype=np.float32),
        maximum_camera_baseline_reconstruction_units=np.asarray(
            metrics["maximum_camera_baseline_reconstruction_units"], dtype=np.float32
        ),
        supporting_rank_span=np.asarray(metrics["supporting_rank_span"], dtype=np.int16),
        supporting_order_span=np.asarray(metrics["supporting_order_span"], dtype=np.int16),
        selected_workspace_orders=np.asarray(orders, dtype=np.int32),
        camera_centers_reconstruction=np.asarray(centers, dtype=np.float64),
        voxel_size_reconstruction_units=np.asarray(voxel_size, dtype=np.float64),
        parallax_threshold_degrees=np.asarray(
            thresholds["maximum_pair_parallax_degrees_threshold"], dtype=np.float64
        ),
        baseline_to_range_threshold=np.asarray(
            thresholds["maximum_baseline_to_range_ratio_threshold"], dtype=np.float64
        ),
        schema_version=np.asarray(I24_SUPPORT_DIVERSITY_SCHEMA_VERSION, dtype=np.int32),
    )

    summary = {
        "phase": "I2.4",
        "schema_version": I24_SUPPORT_DIVERSITY_SCHEMA_VERSION,
        "status": "diagnosis_ready",
        "source_i1_dir": str(i1_dir),
        "source_i2_dir": str(i2_dir),
        "source_i2_1_dir": str(i21_dir),
        "source_i2_3_dir": str(i23_dir),
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
            "baseline_observations": int(exact_support.sum(dtype=np.int64)),
            "mismatched_voxels": 0,
            "per_view_unique_voxel_counts": {str(k): int(v) for k, v in sorted(unique_counts.items())},
        },
        "camera_geometry": {
            "pose_convention": "T_reconstruction_from_camera",
            "camera_center_definition": "translation column of rigid T_reconstruction_from_camera",
            "camera_count": len(centers),
            "arbitrary_sfm_gauge": True,
            "baseline_to_range_is_scale_invariant": True,
            "parallax_is_scale_invariant": True,
        },
        "reference_thresholds": thresholds,
        "regions": regions,
        "diagnostic_signal": signal,
        "top_support2_pair_diversity": top_pairs,
        "crosscheck_with_i2_1": {
            "top_pair_overlap_count": len(i21_pairs & i24_pairs),
            "top_pair_overlap": [list(pair) for pair in sorted(i21_pairs & i24_pairs)],
            "i2_1_top_single_view_dependencies": [
                int(item["workspace_order"])
                for item in i21.get("diagnostic_findings", {}).get("top_single_view_dependencies", [])
            ],
        },
        "authority": {
            "geometry_changed": False,
            "filter_policy_changed": False,
            "sfm_rerun": False,
            "da3_rerun": False,
            "fusion_rerun": False,
            "persistence_changed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_i2_closure": False,
            "authorizes_i3": False,
            "authorizes_mesh_reconstruction": False,
            "support_diversity_filtering_authorized": False,
            "next_action": (
                "Review I2.4 parallax/support-diversity enrichment and top exact support-2 pair geometry. "
                "Only if suspicious regions are demonstrably enriched for weak geometric diversity should a targeted "
                "diversity-aware filtering counterfactual be implemented."
            ),
        },
        "artifacts": {"support_diversity_npz": str(diagnostics_npz.resolve())},
        "policy": policy.as_dict(),
    }
    summary_path = _write_json(output / "i2_4_summary.json", summary)

    print("\nI2.4 support-diversity / parallax artifact diagnosis: DIAGNOSIS_READY")
    print(
        f"views={len(orders)} extended={len(points)} core={int(np.count_nonzero(is_core))} "
        f"peripheral={int(np.count_nonzero(peripheral))} far={int(np.count_nonzero(far))} weak={int(np.count_nonzero(weak))}"
    )
    print(
        f"exact_support_accounting=True observations={int(exact_support.sum(dtype=np.int64))} mismatched_voxels=0"
    )
    print(
        "reference=" + str(thresholds["source"])
        + f" count={thresholds['reference_count']}"
        + f" parallax_q{int(round(float(policy.reference_quantile)*100)):02d}="
        + f"{thresholds['maximum_pair_parallax_degrees_threshold']:.4f}deg"
        + f" baseline/range_q{int(round(float(policy.reference_quantile)*100)):02d}="
        + f"{thresholds['maximum_baseline_to_range_ratio_threshold']:.6f}"
    )
    def _fmt(value: Any, digits: int) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    for name in ("central_core_reference", "peripheral", "far_peripheral", "weak_peripheral"):
        item = regions[name]
        angle_p50 = item["maximum_pair_parallax_degrees_quantiles"]["p50"]
        ratio_p50 = item["maximum_baseline_to_range_ratio_quantiles"]["p50"]
        print(
            f"{name}: count={item['count']} low_diversity={item['low_diversity_count']} "
            f"({item['low_diversity_fraction']:.3f}) support2={item['support2_fraction']:.3f} "
            f"parallax_p50={_fmt(angle_p50, 4)}deg baseline/range_p50={_fmt(ratio_p50, 6)}"
        )
    print(
        "diagnostic_signal=" + str(signal["strength"])
        + f" hypothesis_supported={signal['supports_support_diversity_hypothesis']}"
        + f" peripheral_enrichment={signal['peripheral_enrichment_vs_reference']}"
        + f" far_enrichment={signal['far_enrichment_vs_reference']}"
    )
    if top_pairs:
        print(
            "top_support2_diversity_pairs="
            + ",".join(
                f"{item['order_a']}-{item['order_b']}:"
                f"periph={item['support2_peripheral_count']}:"
                f"low={item['support2_low_diversity_peripheral_count']}:"
                f"far={item['support2_far_count']}:"
                f"angle_p50={float(item['parallax_degrees_quantiles']['p50']):.3f}"
                for item in top_pairs[:8]
            )
        )
    print(
        "geometry_changed=False filter_policy_changed=False sfm_rerun=False da3_rerun=False fusion_rerun=False"
    )
    print("i2_closure_authorized=False support_diversity_filtering_authorized=False mesh_authorization=False")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
