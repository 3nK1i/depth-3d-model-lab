#!/usr/bin/env python3
"""I2.5.1 diagnosis of spatial-extremum geometry surviving I2.5 policies."""
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
from core.reconstruction.offline_dense_extremum_survivor_diagnosis import (  # noqa: E402
    I251_EXTREMUM_SURVIVOR_SCHEMA_VERSION,
    I251ExtremumPolicy,
    aligned_region_statistics,
    extremum_driver_mask,
    policy_envelope_decomposition,
    rejection_consensus,
    robust_directional_excess,
    source_view_attribution,
)
from core.reconstruction.offline_dense_peripheral_diagnosis import validate_observation_matrix  # noqa: E402
from core.reconstruction.offline_dense_quality import I2QualityPolicy  # noqa: E402
from scripts.run_i2_1_targeted_peripheral_diagnosis import _membership_matrix  # noqa: E402


_DEFAULT_I1 = Path("/tmp/depth3d_i1_offline_dense_campaign_recovered")
_DEFAULT_I2 = Path("/tmp/depth3d_i2_full_video_quality")
_DEFAULT_I24 = Path("/tmp/depth3d_i2_4_support_diversity_parallax_diagnosis")
_DEFAULT_I25 = Path("/tmp/depth3d_i2_5_conservative_diversity_filter_counterfactual")
_DEFAULT_OUT = Path("/tmp/depth3d_i2_5_1_spatial_extremum_survivor_diagnosis")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1-dir", type=Path, default=_DEFAULT_I1)
    parser.add_argument("--i2-dir", type=Path, default=_DEFAULT_I2)
    parser.add_argument("--i2-4-dir", type=Path, default=_DEFAULT_I24)
    parser.add_argument("--i2-5-dir", type=Path, default=_DEFAULT_I25)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--directional-tail-quantile", type=float, default=0.90)
    parser.add_argument("--minimum-fraction-of-directional-max", type=float, default=0.50)
    parser.add_argument("--top-source-views", type=int, default=12)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read required I2.5.1 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I2.5.1 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_f_extended(i1_dir: Path) -> tuple[dict[str, np.ndarray | float], dict[str, Any]]:
    f_summary = _read_json(i1_dir / "05_filtering" / "f_summary.json")
    path = Path(str(f_summary["artifacts"]["extended_npz"])).expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        data: dict[str, np.ndarray | float] = {
            "points": np.asarray(payload["points_reconstruction"], dtype=np.float64),
            "colors": np.asarray(payload["colors_rgb"], dtype=np.uint8),
            "support": np.asarray(payload["view_observation_counts"], dtype=np.int64),
            "confidence": np.asarray(payload["filter_confidence"], dtype=np.float64),
            "dispersion": np.asarray(payload["p90_dispersion_voxels"], dtype=np.float64),
            "high": np.asarray(payload["high_support_neighbor_count"], dtype=np.int64),
            "radius2": np.asarray(payload["radius2_neighbor_count"], dtype=np.int64),
            "component_size": np.asarray(payload["component_size"], dtype=np.int64),
            "voxel_size": float(np.asarray(payload["voxel_size_reconstruction_units"]).item()),
        }
    return data, f_summary


def _load_i24(summary: Mapping[str, Any]) -> dict[str, np.ndarray | float]:
    path = Path(str(summary["artifacts"]["support_diversity_npz"])).expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        return {
            "points": np.asarray(payload["points_reconstruction"], dtype=np.float64),
            "is_core": np.asarray(payload["is_core"], dtype=bool),
            "peripheral": np.asarray(payload["is_peripheral"], dtype=bool),
            "far": np.asarray(payload["is_far_peripheral"], dtype=bool),
            "weak": np.asarray(payload["is_weak_peripheral"], dtype=bool),
            "low_diversity": np.asarray(payload["is_low_support_diversity"], dtype=bool),
            "angle": np.asarray(payload["maximum_pair_parallax_degrees"], dtype=np.float64),
            "baseline_ratio": np.asarray(payload["maximum_baseline_to_range_ratio"], dtype=np.float64),
            "voxel_size": float(np.asarray(payload["voxel_size_reconstruction_units"]).item()),
        }


def _scenario_rejection_masks(i25_dir: Path, summary: Mapping[str, Any], points: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    names: list[str] = []
    masks: list[np.ndarray] = []
    for record in summary.get("scenarios", []):
        name = str(record["name"])
        artifact = Path(str(record["artifacts"]["diagnostics_npz"])).expanduser().resolve()
        if not artifact.is_file():
            artifact = i25_dir / "scenarios" / name / "i2_5_policy_diagnostics.npz"
        with np.load(artifact, allow_pickle=False) as payload:
            scenario_points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            reject = np.asarray(payload["is_rejected_by_policy"], dtype=bool)
        if scenario_points.shape != points.shape or not np.array_equal(scenario_points, points):
            raise ReconstructionValidationError(f"I2.5.1 scenario {name} does not share the exact baseline point order.")
        if reject.shape != (len(points),):
            raise ReconstructionValidationError(f"I2.5.1 scenario {name} rejection mask is misaligned.")
        names.append(name)
        masks.append(reject)
    if not names:
        raise ReconstructionValidationError("I2.5.1 requires completed I2.5 scenario diagnostics.")
    return names, masks


def main() -> int:
    args = _args()
    i1_dir = args.i1_dir.expanduser().resolve()
    i2_dir = args.i2_dir.expanduser().resolve()
    i24_dir = args.i2_4_dir.expanduser().resolve()
    i25_dir = args.i2_5_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I2.5.1 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i1 = _read_json(i1_dir / "i1_campaign_summary.json")
    i2 = _read_json(i2_dir / "i2_summary.json")
    i24 = _read_json(i24_dir / "i2_4_summary.json")
    i25 = _read_json(i25_dir / "i2_5_summary.json")
    e3 = _read_json(i1_dir / "04_qualified_fusion" / "e3_summary.json")
    if str(i1.get("status")) != "pass" or not bool(i1.get("pipeline_execution_complete")):
        raise ReconstructionValidationError("I2.5.1 requires completed passing I1 evidence.")
    if str(i2.get("phase")) != "I2" or not bool(i2.get("automated_pass")):
        raise ReconstructionValidationError("I2.5.1 requires passing automated I2 evidence.")
    if str(i24.get("status")) != "diagnosis_ready" or not bool(i24.get("diagnostic_signal", {}).get("supports_support_diversity_hypothesis")):
        raise ReconstructionValidationError("I2.5.1 requires completed strong-signal I2.4 evidence.")
    if str(i25.get("status")) != "counterfactuals_ready":
        raise ReconstructionValidationError("I2.5.1 requires completed I2.5 counterfactual evidence.")
    if i25.get("ranking", {}).get("balanced_candidates_for_manual_review") or i25.get("ranking", {}).get("preferred_candidates_for_manual_review"):
        raise ReconstructionValidationError("I2.5.1 is intended only after I2.5 found no balanced/preferred production candidate.")
    scene_id = str(i2["scene"]["scene_id"])
    if str(i24.get("scene_id")) != scene_id or str(i25.get("scene_id")) != scene_id:
        raise ReconstructionValidationError("I2.5.1 scene identity mismatch across I2/I2.4/I2.5.")

    f, f_summary = _load_f_extended(i1_dir)
    d = _load_i24(i24)
    points = np.asarray(f["points"], dtype=np.float64)
    colors = np.asarray(f["colors"], dtype=np.uint8)
    if points.shape != np.asarray(d["points"]).shape or not np.array_equal(points, np.asarray(d["points"])):
        raise ReconstructionValidationError("I2.5.1 F Extended and I2.4 point order disagree.")
    if not np.isclose(float(f["voxel_size"]), float(d["voxel_size"]), rtol=0.0, atol=1e-12):
        raise ReconstructionValidationError("I2.5.1 voxel size mismatch.")

    quality_policy = I2QualityPolicy(**{key: float(value) for key, value in i2["policy"].items()})
    policy = I251ExtremumPolicy(
        robust_lower_quantile=float(quality_policy.robust_lower_quantile),
        robust_upper_quantile=float(quality_policy.robust_upper_quantile),
        directional_tail_quantile=float(args.directional_tail_quantile),
        minimum_fraction_of_directional_max=float(args.minimum_fraction_of_directional_max),
    )
    excess = robust_directional_excess(
        points,
        lower_quantile=policy.robust_lower_quantile,
        upper_quantile=policy.robust_upper_quantile,
    )
    drivers, driver_definition = extremum_driver_mask(excess["directional_excess"], policy=policy)
    scenario_names, reject_masks = _scenario_rejection_masks(i25_dir, i25, points)
    consensus = rejection_consensus(reject_masks, length=len(points))
    stubborn = drivers & np.asarray(consensus["survives_all_policies"], dtype=bool)
    any_rejected_driver = drivers & np.asarray(consensus["rejected_by_any_policy"], dtype=bool)

    support = np.asarray(f["support"], dtype=np.int64)
    confidence = np.asarray(f["confidence"], dtype=np.float64)
    dispersion = np.asarray(f["dispersion"], dtype=np.float64)
    high = np.asarray(f["high"], dtype=np.int64)
    radius2 = np.asarray(f["radius2"], dtype=np.int64)
    is_core = np.asarray(d["is_core"], dtype=bool)
    peripheral = np.asarray(d["peripheral"], dtype=bool)
    far = np.asarray(d["far"], dtype=bool)
    low_div = np.asarray(d["low_diversity"], dtype=bool)
    angle = np.asarray(d["angle"], dtype=np.float64)
    baseline_ratio = np.asarray(d["baseline_ratio"], dtype=np.float64)

    regions = {
        "extremum_drivers": drivers,
        "stubborn_extremum_survivors": stubborn,
        "drivers_rejected_by_any_i2_5_policy": any_rejected_driver,
        "ordinary_peripheral_non_drivers": peripheral & ~drivers,
        "central_core_non_drivers": is_core & ~peripheral & ~drivers,
    }
    region_stats = {
        name: aligned_region_statistics(
            mask,
            support=support,
            confidence=confidence,
            dispersion=dispersion,
            high_support_neighbors=high,
            radius2_neighbors=radius2,
            parallax_degrees=angle,
            baseline_to_range=baseline_ratio,
            low_diversity=low_div,
            is_core=is_core,
            is_peripheral=peripheral,
            is_far=far,
        )
        for name, mask in regions.items()
    }

    decompositions: list[dict[str, Any]] = []
    for name, reject in zip(scenario_names, reject_masks):
        item = policy_envelope_decomposition(
            points,
            reject,
            lower_quantile=policy.robust_lower_quantile,
            upper_quantile=policy.robust_upper_quantile,
            extremum_driver_mask_fixed=drivers,
        )
        item["name"] = name
        decompositions.append(item)

    orders, matrix, _ = _membership_matrix(e3_summary=e3, points=points, voxel_size=float(f["voxel_size"]))
    matrix, exact_support = validate_observation_matrix(matrix, support, minimum_support_views=2)
    if not np.array_equal(exact_support, support):
        raise ReconstructionValidationError("I2.5.1 exact source-view membership does not reproduce F support counts.")
    top_driver_views = source_view_attribution(matrix, orders, drivers, top_k=int(args.top_source_views))
    top_stubborn_views = source_view_attribution(matrix, orders, stubborn, top_k=int(args.top_source_views))

    stubborn_count = int(np.count_nonzero(stubborn))
    driver_count = int(np.count_nonzero(drivers))
    stubborn_fraction = stubborn_count / float(driver_count)
    stubborn_lowdiv_fraction = float(np.mean(low_div[stubborn])) if stubborn_count else 0.0
    stubborn_far_fraction = float(np.mean(far[stubborn])) if stubborn_count else 0.0
    stubborn_core_fraction = float(np.mean(is_core[stubborn])) if stubborn_count else 0.0
    ratio_worsening_count = sum(bool(item["ratio_worsening_explained_by_robust_shrink"]) for item in decompositions)

    diagnostics_npz = output / "i2_5_1_extremum_survivor_diagnostics.npz"
    np.savez_compressed(
        diagnostics_npz,
        points_reconstruction=points,
        colors_rgb=colors,
        is_extremum_driver=drivers.astype(np.uint8),
        is_stubborn_extremum_survivor=stubborn.astype(np.uint8),
        is_driver_rejected_by_any_i2_5_policy=any_rejected_driver.astype(np.uint8),
        i2_5_rejection_count=np.asarray(consensus["rejection_count"], dtype=np.int16),
        maximum_normalized_robust_box_excess=np.asarray(excess["maximum_normalized_excess"], dtype=np.float32),
        directional_normalized_excess=np.asarray(excess["directional_excess"], dtype=np.float32),
        dominant_extremum_direction_code=np.asarray(excess["dominant_direction_code"], dtype=np.uint8),
        baseline_support=support.astype(np.int16),
        filter_confidence=confidence.astype(np.float32),
        p90_dispersion_voxels=dispersion.astype(np.float32),
        high_support_neighbor_count=high.astype(np.int16),
        radius2_neighbor_count=radius2.astype(np.int16),
        maximum_pair_parallax_degrees=angle.astype(np.float32),
        maximum_baseline_to_range_ratio=baseline_ratio.astype(np.float32),
        is_low_support_diversity=low_div.astype(np.uint8),
        is_core=is_core.astype(np.uint8),
        is_peripheral=peripheral.astype(np.uint8),
        is_far_peripheral=far.astype(np.uint8),
        selected_workspace_orders=np.asarray(orders, dtype=np.int32),
        robust_lower=np.asarray(excess["lower"], dtype=np.float64),
        robust_upper=np.asarray(excess["upper"], dtype=np.float64),
        schema_version=np.asarray(I251_EXTREMUM_SURVIVOR_SCHEMA_VERSION, dtype=np.int32),
    )

    summary = {
        "phase": "I2.5.1",
        "schema_version": I251_EXTREMUM_SURVIVOR_SCHEMA_VERSION,
        "status": "diagnosis_ready",
        "scene_id": scene_id,
        "source_i1_dir": str(i1_dir),
        "source_i2_dir": str(i2_dir),
        "source_i2_4_dir": str(i24_dir),
        "source_i2_5_dir": str(i25_dir),
        "baseline": {
            "extended_points": len(points),
            "core_points": int(np.count_nonzero(is_core)),
            "peripheral_points": int(np.count_nonzero(peripheral)),
            "far_peripheral_points": int(np.count_nonzero(far)),
            "full_to_robust_ratio": float(i2["spatial"]["extended_envelope"]["full_to_robust_diagonal_ratio"]),
            "voxel_size_reconstruction_units": float(f["voxel_size"]),
        },
        "extremum_definition": {**policy.as_dict(), **driver_definition},
        "extremum_findings": {
            "driver_count": driver_count,
            "stubborn_survivor_count": stubborn_count,
            "stubborn_survivor_fraction_of_drivers": stubborn_fraction,
            "stubborn_low_diversity_fraction": stubborn_lowdiv_fraction,
            "stubborn_far_fraction": stubborn_far_fraction,
            "stubborn_core_fraction": stubborn_core_fraction,
            "i2_5_policy_count": len(reject_masks),
            "policies_where_ratio_worsening_is_explained_by_robust_shrink": ratio_worsening_count,
            "all_i2_5_policies_miss_some_extremum_drivers": bool(stubborn_count > 0),
        },
        "regions": region_stats,
        "i2_5_policy_envelope_decomposition": decompositions,
        "source_view_attribution": {
            "top_extremum_driver_views": top_driver_views,
            "top_stubborn_extremum_views": top_stubborn_views,
        },
        "exact_support_accounting": {
            "pass": True,
            "observations": int(matrix.sum(dtype=np.int64)),
            "mismatched_voxels": 0,
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
            "production_filter_authorized": False,
            "next_action": (
                "Use the stubborn extremum population and envelope decomposition to identify the missing local-manifold/"
                "anchoring discriminator before any production filtering promotion."
            ),
        },
        "artifacts": {"extremum_survivor_npz": str(diagnostics_npz.resolve())},
    }
    path = _write_json(output / "i2_5_1_summary.json", summary)

    print("I2.5.1 spatial-extremum survivor diagnosis: DIAGNOSIS_READY")
    print(
        f"extended={len(points)} drivers={driver_count} ({driver_count/float(len(points)):.3f}) "
        f"stubborn={stubborn_count} ({stubborn_fraction:.3f} of drivers)"
    )
    print(
        f"stubborn_low_diversity={stubborn_lowdiv_fraction:.3f} stubborn_far={stubborn_far_fraction:.3f} "
        f"stubborn_core={stubborn_core_fraction:.3f}"
    )
    print(f"exact_support_accounting=True observations={int(matrix.sum(dtype=np.int64))} mismatched_voxels=0")
    for item in decompositions:
        print(
            f"policy {item['name']}: driver_reject={item['extremum_driver_rejection_fraction']:.3f} "
            f"full_shrink={item['full_diagonal_shrink_fraction']:.3f} "
            f"robust_shrink={item['robust_diagonal_shrink_fraction']:.3f} "
            f"full/robust={item['candidate_full_to_robust_ratio']:.3f} "
            f"robust_denominator_effect={item['ratio_worsening_explained_by_robust_shrink']}"
        )
    print("top_extremum_orders=" + ",".join(str(item["workspace_order"]) for item in top_driver_views[:8]))
    print("top_stubborn_orders=" + ",".join(str(item["workspace_order"]) for item in top_stubborn_views[:8]))
    print(
        f"ratio_worsening_from_robust_shrink_policies={ratio_worsening_count}/{len(decompositions)} "
        "geometry_changed=False filter_policy_changed=False sfm_rerun=False da3_rerun=False fusion_rerun=False"
    )
    print("i2_closure_authorized=False production_filter_authorized=False mesh_authorization=False")
    print(f"summary={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
