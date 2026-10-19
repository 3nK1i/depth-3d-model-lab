#!/usr/bin/env python3
"""I2.5.3 conservative manifold-attachment filtering counterfactual campaign.

Consumes the unchanged full-video I1/I2/I2.5.1/I2.5.2 evidence and evaluates a
small, fixed family of scene-relative local-manifold rejection policies.  The
filter rules use only I2.5.2 anchor distance, local spacing and optional
linearity evidence.  Peripheral/far/extremum labels are evaluation-only.

No source geometry, formal F/G policy, SfM, DA3, fusion, persistence,
metric/measurement authority or mesh authority is changed.
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
from core.reconstruction.offline_dense_counterfactual_refusion import scenario_quality_profile  # noqa: E402
from core.reconstruction.offline_dense_manifold_filter_counterfactual import (  # noqa: E402
    I253_MANIFOLD_FILTER_COUNTERFACTUAL_SCHEMA_VERSION,
    default_manifold_filter_scenarios,
    manifold_counterfactual_comparison,
    manifold_effect_counts,
    manifold_pareto_front,
    manifold_rejection_mask,
)
from core.reconstruction.offline_dense_quality import I2QualityPolicy  # noqa: E402
from scripts.run_i2_5_1_spatial_extremum_survivor_diagnosis import _load_f_extended  # noqa: E402
from scripts.run_i2_5_conservative_diversity_filter_counterfactual import _candidate_topology  # noqa: E402


_DEFAULT_I1 = Path("/tmp/depth3d_i1_offline_dense_campaign_recovered")
_DEFAULT_I2 = Path("/tmp/depth3d_i2_full_video_quality")
_DEFAULT_I251 = Path("/tmp/depth3d_i2_5_1_spatial_extremum_survivor_diagnosis")
_DEFAULT_I252 = Path("/tmp/depth3d_i2_5_2_extremum_local_geometry_manifold_diagnosis")
_DEFAULT_OUT = Path("/tmp/depth3d_i2_5_3_conservative_manifold_filter_counterfactual")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1-dir", type=Path, default=_DEFAULT_I1)
    parser.add_argument("--i2-dir", type=Path, default=_DEFAULT_I2)
    parser.add_argument("--i2-5-1-dir", type=Path, default=_DEFAULT_I251)
    parser.add_argument("--i2-5-2-dir", type=Path, default=_DEFAULT_I252)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read required I2.5.3 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I2.5.3 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_i251(summary: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(str(summary["artifacts"]["extremum_survivor_npz"])).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"I2.5.3 I2.5.1 artifact is unavailable: {path}")
    with np.load(path, allow_pickle=False) as p:
        return {
            "points": np.asarray(p["points_reconstruction"], dtype=np.float64),
            "peripheral": np.asarray(p["is_peripheral"], dtype=bool),
            "far": np.asarray(p["is_far_peripheral"], dtype=bool),
            "drivers": np.asarray(p["is_extremum_driver"], dtype=bool),
            "stubborn": np.asarray(p["is_stubborn_extremum_survivor"], dtype=bool),
        }


def _load_i252(summary: Mapping[str, Any]) -> dict[str, np.ndarray | float]:
    path = Path(str(summary["artifacts"]["local_geometry_manifold_npz"])).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"I2.5.3 I2.5.2 artifact is unavailable: {path}")
    with np.load(path, allow_pickle=False) as p:
        return {
            "points": np.asarray(p["points_reconstruction"], dtype=np.float64),
            "colors": np.asarray(p["colors_rgb"], dtype=np.uint8),
            "core": np.asarray(p["is_core"], dtype=bool),
            "drivers": np.asarray(p["is_extremum_driver"], dtype=bool),
            "stubborn": np.asarray(p["is_stubborn_extremum_survivor"], dtype=bool),
            "distance": np.asarray(p["nearest_strong_anchor_distance_voxels"], dtype=np.float64),
            "knn": np.asarray(p["knn_median_distance_voxels"], dtype=np.float64),
            "linearity": np.asarray(p["local_linearity"], dtype=np.float64),
            "support": np.asarray(p["baseline_support"], dtype=np.int64),
            "confidence": np.asarray(p["filter_confidence"], dtype=np.float64),
            "dispersion": np.asarray(p["p90_dispersion_voxels"], dtype=np.float64),
            "voxel_size": float(np.asarray(p["voxel_size_reconstruction_units"]).item()),
        }


def _save_scenario(
    path: Path,
    *,
    points: np.ndarray,
    colors: np.ndarray,
    reject: np.ndarray,
    core: np.ndarray,
    peripheral: np.ndarray,
    far: np.ndarray,
    weak: np.ndarray,
    drivers: np.ndarray,
    stubborn: np.ndarray,
    distance: np.ndarray,
    knn: np.ndarray,
    linearity: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points_reconstruction=np.asarray(points, dtype=np.float64),
        colors_rgb=np.asarray(colors, dtype=np.uint8),
        is_rejected_by_policy=np.asarray(reject, dtype=np.uint8),
        is_kept=np.asarray(~reject, dtype=np.uint8),
        is_core_before=np.asarray(core, dtype=np.uint8),
        is_peripheral_before=np.asarray(peripheral, dtype=np.uint8),
        is_far_before=np.asarray(far, dtype=np.uint8),
        is_weak_peripheral_before=np.asarray(weak, dtype=np.uint8),
        is_extremum_driver=np.asarray(drivers, dtype=np.uint8),
        is_stubborn_extremum_survivor=np.asarray(stubborn, dtype=np.uint8),
        nearest_strong_anchor_distance_voxels=np.asarray(distance, dtype=np.float32),
        knn_median_distance_voxels=np.asarray(knn, dtype=np.float32),
        local_linearity=np.asarray(linearity, dtype=np.float32),
        schema_version=np.asarray(I253_MANIFOLD_FILTER_COUNTERFACTUAL_SCHEMA_VERSION, dtype=np.int32),
    )
    return path


def main() -> int:
    args = _args()
    i1_dir = args.i1_dir.expanduser().resolve()
    i2_dir = args.i2_dir.expanduser().resolve()
    i251_dir = args.i2_5_1_dir.expanduser().resolve()
    i252_dir = args.i2_5_2_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I2.5.3 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i1 = _read_json(i1_dir / "i1_campaign_summary.json")
    i2 = _read_json(i2_dir / "i2_summary.json")
    i251 = _read_json(i251_dir / "i2_5_1_summary.json")
    i252 = _read_json(i252_dir / "i2_5_2_summary.json")
    if str(i1.get("status")) != "pass" or not bool(i1.get("pipeline_execution_complete")):
        raise ReconstructionValidationError("I2.5.3 requires completed passing I1 evidence.")
    if str(i2.get("phase")) != "I2" or not bool(i2.get("automated_pass")):
        raise ReconstructionValidationError("I2.5.3 requires passing automated I2 evidence.")
    if str(i251.get("status")) != "diagnosis_ready":
        raise ReconstructionValidationError("I2.5.3 requires completed I2.5.1 evidence.")
    if str(i252.get("status")) != "diagnosis_ready":
        raise ReconstructionValidationError("I2.5.3 requires completed I2.5.2 evidence.")
    if not bool(i252.get("diagnostic_signal", {}).get("supports_local_geometry_manifold_hypothesis")):
        raise ReconstructionValidationError("I2.5.3 requires strong I2.5.2 local-manifold evidence.")
    scene_id = str(i2["scene"]["scene_id"])
    if str(i251.get("scene_id")) != scene_id or str(i252.get("scene_id")) != scene_id:
        raise ReconstructionValidationError("I2.5.3 scene identity mismatch.")

    f, _ = _load_f_extended(i1_dir)
    d251 = _load_i251(i251)
    d252 = _load_i252(i252)
    points = np.asarray(f["points"], dtype=np.float64)
    colors = np.asarray(f["colors"], dtype=np.uint8)
    support = np.asarray(f["support"], dtype=np.int64)
    confidence = np.asarray(f["confidence"], dtype=np.float64)
    dispersion = np.asarray(f["dispersion"], dtype=np.float64)
    voxel_size = float(f["voxel_size"])

    for source_name, source_points in (("I2.5.1", d251["points"]), ("I2.5.2", d252["points"])):
        source_points = np.asarray(source_points, dtype=np.float64)
        if source_points.shape != points.shape or not np.array_equal(source_points, points):
            raise ReconstructionValidationError(f"I2.5.3 {source_name} does not share exact F Extended point order.")
    if not np.array_equal(np.asarray(d252["colors"], dtype=np.uint8), colors):
        raise ReconstructionValidationError("I2.5.3 I2.5.2/F colors disagree.")
    if not np.array_equal(np.asarray(d252["support"], dtype=np.int64), support):
        raise ReconstructionValidationError("I2.5.3 I2.5.2/F support counts disagree.")
    if not np.isclose(float(d252["voxel_size"]), voxel_size, rtol=0.0, atol=1e-12):
        raise ReconstructionValidationError("I2.5.3 I2.5.2/F voxel size differs.")

    core = np.asarray(d252["core"], dtype=bool)
    peripheral = np.asarray(d251["peripheral"], dtype=bool)
    far = np.asarray(d251["far"], dtype=bool)
    drivers = np.asarray(d252["drivers"], dtype=bool)
    stubborn = np.asarray(d252["stubborn"], dtype=bool)
    if not np.array_equal(drivers, np.asarray(d251["drivers"], dtype=bool)) or not np.array_equal(stubborn, np.asarray(d251["stubborn"], dtype=bool)):
        raise ReconstructionValidationError("I2.5.3 I2.5.1/I2.5.2 extremum masks disagree.")

    central_core = core & ~peripheral
    core_conf_p25 = float(np.quantile(confidence[core], 0.25))
    core_disp_p90 = float(np.quantile(dispersion[core], 0.90))
    weak = peripheral & ((support <= 2) | (confidence < core_conf_p25) | (dispersion > core_disp_p90))
    core_points = points[core]
    core_confidence = confidence[core]
    core_dispersion = dispersion[core]

    policy = I2QualityPolicy(**{key: float(value) for key, value in i2["policy"].items()})
    baseline_full = float(i2["spatial"]["extended_envelope"]["full_diagonal"])
    baseline_robust = float(i2["spatial"]["extended_envelope"]["robust_diagonal"])
    baseline_ratio = float(i2["spatial"]["extended_envelope"]["full_to_robust_diagonal_ratio"])
    if not np.isclose(baseline_full / baseline_robust, baseline_ratio, rtol=1e-6, atol=1e-9):
        raise ReconstructionValidationError("I2.5.3 I2 spatial envelope ratio is internally inconsistent.")

    thresholds = i252["reference"]["thresholds"]
    anchor_q90 = float(thresholds["anchor_distance_high"])
    knn_q90 = float(thresholds["knn_spacing_high"])
    linearity_q90 = float(thresholds["linearity_high"])
    distance = np.asarray(d252["distance"], dtype=np.float64)
    knn = np.asarray(d252["knn"], dtype=np.float64)
    linearity = np.asarray(d252["linearity"], dtype=np.float64)

    scenarios = default_manifold_filter_scenarios()
    print("I2.5.3 conservative manifold-attachment filtering counterfactuals:")
    print(
        f"baseline extended={len(points)} core={int(np.count_nonzero(core))} drivers={int(np.count_nonzero(drivers))} "
        f"stubborn={int(np.count_nonzero(stubborn))} peripheral={int(np.count_nonzero(peripheral))} "
        f"far={int(np.count_nonzero(far))} full/robust={baseline_ratio:.3f}"
    )
    print(
        f"reference anchor_dist_q90={anchor_q90:.3f}vox knn_q90={knn_q90:.3f}vox "
        f"linearity_q90={linearity_q90:.3f} scenarios={len(scenarios)}"
    )
    print("filter_inputs=local_manifold_only peripheral_labels=False extremum_labels=False source_view_blacklist=False metric_distance=False")

    records: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, 1):
        reject, resolved = manifold_rejection_mask(
            scenario=scenario,
            nearest_strong_anchor_distance_voxels=distance,
            knn_median_distance_voxels=knn,
            local_linearity=linearity,
            anchor_distance_reference_q90_voxels=anchor_q90,
            knn_spacing_reference_q90_voxels=knn_q90,
            linearity_reference_q90=linearity_q90,
        )
        kept = ~reject
        candidate_core_mask = core & kept
        if int(np.count_nonzero(kept)) < 1000 or int(np.count_nonzero(candidate_core_mask)) < 1000:
            raise ReconstructionValidationError(f"I2.5.3 scenario {scenario.name} is unexpectedly destructive.")

        candidate_points = points[kept]
        candidate_support = support[kept]
        candidate_confidence = confidence[kept]
        candidate_dispersion = dispersion[kept]
        candidate_core_points = points[candidate_core_mask]
        component_size = _candidate_topology(candidate_points, candidate_support, voxel_size)
        profile, _ = scenario_quality_profile(
            extended_points=candidate_points,
            core_points=candidate_core_points,
            support=candidate_support,
            confidence=candidate_confidence,
            dispersion=candidate_dispersion,
            component_size=component_size,
            voxel_size=voxel_size,
            baseline_core_points=core_points,
            baseline_core_confidence=core_confidence,
            baseline_core_dispersion=core_dispersion,
            policy=policy,
        )
        effects = manifold_effect_counts(
            reject,
            is_core=core,
            is_peripheral=peripheral,
            is_far_peripheral=far,
            is_weak_peripheral=weak,
            is_extremum_driver=drivers,
            is_stubborn_extremum_survivor=stubborn,
        )
        envelope = profile["extended_envelope"]
        comparison = manifold_counterfactual_comparison(
            baseline_extended_points=len(points),
            baseline_core_points=int(np.count_nonzero(core)),
            baseline_full_diagonal=baseline_full,
            baseline_robust_diagonal=baseline_robust,
            candidate_extended_points=len(candidate_points),
            candidate_core_points=len(candidate_core_points),
            candidate_full_diagonal=float(envelope["full_diagonal"]),
            candidate_robust_diagonal=float(envelope["robust_diagonal"]),
            candidate_dominant_component_fraction=float(profile["dominant_component_fraction_of_extended"]),
            effects=effects,
        )
        scenario_dir = output / "scenarios" / scenario.name
        artifact = _save_scenario(
            scenario_dir / "i2_5_3_policy_diagnostics.npz",
            points=points,
            colors=colors,
            reject=reject,
            core=core,
            peripheral=peripheral,
            far=far,
            weak=weak,
            drivers=drivers,
            stubborn=stubborn,
            distance=distance,
            knn=knn,
            linearity=linearity,
        )
        record = {
            "name": scenario.name,
            "scenario": scenario.as_dict(),
            "resolved_rule": resolved,
            "effects_against_baseline": effects,
            "candidate_profile": profile,
            "comparison": comparison,
            "artifacts": {"diagnostics_npz": str(artifact.resolve())},
        }
        _write_json(scenario_dir / "scenario_summary.json", record)
        records.append(record)
        print(
            f"[{index}/{len(scenarios)}] {scenario.name}: reject={effects['rejected_points']} "
            f"core_loss={comparison['core_loss_fraction']:.3f} central_core_loss={comparison['central_core_loss_fraction']:.3f} "
            f"driver_reject={comparison['extremum_driver_rejection_fraction']:.3f} "
            f"stubborn_reject={comparison['stubborn_extremum_rejection_fraction']:.3f} "
            f"far_reduction={comparison['far_peripheral_reduction_fraction']:.3f} "
            f"full_shrink={comparison['full_diagonal_shrink_fraction']:.3f} "
            f"robust_shrink={comparison['robust_diagonal_shrink_fraction']:.3f} "
            f"full/robust={comparison['candidate_full_to_robust_ratio']:.3f} "
            f"ratio_improvement={comparison['full_to_robust_ratio_improvement_fraction']:.3f} "
            f"balanced={comparison['balanced_for_manual_review']} preferred={comparison['preferred_for_manual_review']}"
        )

    ranked = sorted(records, key=lambda item: float(item["comparison"]["diagnostic_score"]), reverse=True)
    pareto = manifold_pareto_front(records)
    balanced = [item["name"] for item in ranked if bool(item["comparison"]["balanced_for_manual_review"])]
    preferred = [item["name"] for item in ranked if bool(item["comparison"]["preferred_for_manual_review"])]
    manual_priority = preferred[:3] or balanced[:3] or [item["name"] for item in ranked[:3]]

    summary = {
        "phase": "I2.5.3",
        "schema_version": I253_MANIFOLD_FILTER_COUNTERFACTUAL_SCHEMA_VERSION,
        "status": "counterfactuals_ready",
        "scene_id": scene_id,
        "source_i1_dir": str(i1_dir),
        "source_i2_dir": str(i2_dir),
        "source_i2_5_1_dir": str(i251_dir),
        "source_i2_5_2_dir": str(i252_dir),
        "baseline": {
            "extended_points": len(points),
            "core_points": int(np.count_nonzero(core)),
            "central_core_points": int(np.count_nonzero(central_core)),
            "peripheral_points": int(np.count_nonzero(peripheral)),
            "far_peripheral_points": int(np.count_nonzero(far)),
            "weak_peripheral_points": int(np.count_nonzero(weak)),
            "extremum_driver_count": int(np.count_nonzero(drivers)),
            "stubborn_extremum_survivor_count": int(np.count_nonzero(stubborn)),
            "full_diagonal": baseline_full,
            "robust_diagonal": baseline_robust,
            "full_to_robust_ratio": baseline_ratio,
            "voxel_size_reconstruction_units": voxel_size,
        },
        "reference_thresholds": {
            "source": "I2.5.2 central Core non-driver q90 diagnostics",
            "anchor_distance_q90_voxels": anchor_q90,
            "knn_spacing_q90_voxels": knn_q90,
            "linearity_q90": linearity_q90,
            "filter_rules_do_not_use_peripheral_or_far_labels": True,
            "filter_rules_do_not_use_extremum_labels": True,
            "filter_rules_do_not_use_source_view_blacklists": True,
            "filter_rules_do_not_use_metric_distance": True,
        },
        "scenarios": records,
        "ranking": {
            "by_diagnostic_score": [item["name"] for item in ranked],
            "pareto_front": list(pareto),
            "balanced_candidates_for_manual_review": balanced,
            "preferred_candidates_for_manual_review": preferred,
            "manual_review_priority": manual_priority,
            "diagnostic_only": True,
        },
        "authority": {
            "source_geometry_changed": False,
            "production_f_policy_changed": False,
            "counterfactual_geometry_generated": True,
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
            "manifold_aware_production_filtering_authorized": False,
            "next_action": (
                "Manually inspect only preferred/balanced I2.5.3 candidates. Production F promotion belongs to I2.6 "
                "only if a candidate visibly preserves the main scene while materially shrinking the pathological "
                "full envelope and retaining the robust denominator."
            ),
        },
    }
    path = _write_json(output / "i2_5_3_summary.json", summary)
    print("\nI2.5.3 conservative manifold-attachment filtering counterfactuals: COUNTERFACTUALS_READY")
    print("ranking=" + ",".join(item["name"] for item in ranked))
    print("pareto_front=" + (",".join(pareto) if pareto else "none"))
    print("balanced_candidates=" + (",".join(balanced) if balanced else "none"))
    print("preferred_candidates=" + (",".join(preferred) if preferred else "none"))
    print("manual_review_priority=" + ",".join(manual_priority))
    print("source_geometry_changed=False production_F_policy_changed=False sfm_rerun=False da3_rerun=False fusion_rerun=False")
    print("i2_closure_authorized=False production_filter_authorized=False mesh_authorization=False")
    print(f"summary={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
