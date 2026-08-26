#!/usr/bin/env python3
"""I2.5 conservative diversity-aware filtering counterfactual campaign.

Consumes the unchanged full-video F/I2/I2.4 evidence and evaluates a small,
fixed ladder of post-F diversity-aware rejection policies.  The policies use
only support/parallax/baseline-range/topology/confidence evidence; I2
peripheral/far labels are evaluation-only and never filtering inputs.

No source geometry or F/G policy is changed.  No SfM, DA3, or fusion stage is
rerun, and no metric/measurement/mesh authority is promoted.
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
from core.reconstruction.offline_dense_diversity_filter_counterfactual import (  # noqa: E402
    I25_DIVERSITY_FILTER_COUNTERFACTUAL_SCHEMA_VERSION,
    counterfactual_quality_comparison,
    default_diversity_filter_scenarios,
    diversity_filter_rejection_mask,
    pareto_front,
    policy_effect_counts,
)
from core.reconstruction.offline_dense_quality import I2QualityPolicy, classify_peripheral_geometry  # noqa: E402
from scripts.run_e2_consistency_aware_fusion_refinement import _topology_metrics, _voxel_indices  # noqa: E402


_DEFAULT_I1 = Path("/tmp/eia_i1_offline_dense_campaign_recovered")
_DEFAULT_I2 = Path("/tmp/eia_i2_full_video_quality")
_DEFAULT_I24 = Path("/tmp/eia_i2_4_support_diversity_parallax_diagnosis")
_DEFAULT_OUT = Path("/tmp/eia_i2_5_conservative_diversity_filter_counterfactual")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1-dir", type=Path, default=_DEFAULT_I1)
    parser.add_argument("--i2-dir", type=Path, default=_DEFAULT_I2)
    parser.add_argument("--i2-4-dir", type=Path, default=_DEFAULT_I24)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read required I2.5 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I2.5 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _resolve_f_summary(i1_dir: Path) -> Path:
    path = i1_dir / "05_filtering" / "f_summary.json"
    if path.is_file():
        return path
    raise ReconstructionValidationError("I2.5 requires the completed I1 formal-F output directory.")


def _load_extended(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as payload:
        required = (
            "points_reconstruction",
            "colors_rgb",
            "view_observation_counts",
            "filter_confidence",
            "p90_dispersion_voxels",
            "radius2_neighbor_count",
            "high_support_neighbor_count",
            "component_size",
            "voxel_size_reconstruction_units",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ReconstructionValidationError(f"I2.5 F Extended NPZ is missing fields: {missing}")
        result: dict[str, np.ndarray | float] = {
            "points": np.asarray(payload["points_reconstruction"], dtype=np.float64),
            "colors": np.asarray(payload["colors_rgb"], dtype=np.uint8),
            "support": np.asarray(payload["view_observation_counts"], dtype=np.uint32),
            "confidence": np.asarray(payload["filter_confidence"], dtype=np.float32),
            "dispersion": np.asarray(payload["p90_dispersion_voxels"], dtype=np.float32),
            "radius2": np.asarray(payload["radius2_neighbor_count"], dtype=np.int32),
            "high": np.asarray(payload["high_support_neighbor_count"], dtype=np.int32),
            "component_size": np.asarray(payload["component_size"], dtype=np.int32),
            "voxel_size": float(np.asarray(payload["voxel_size_reconstruction_units"]).item()),
        }
    n = len(np.asarray(result["points"]))
    if n == 0:
        raise ReconstructionValidationError("I2.5 requires non-empty F Extended geometry.")
    for name in ("colors", "support", "confidence", "dispersion", "radius2", "high", "component_size"):
        if len(np.asarray(result[name])) != n:
            raise ReconstructionValidationError(f"I2.5 Extended field {name} is not aligned.")
    return result


def _load_core(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "points": np.asarray(payload["points_reconstruction"], dtype=np.float64),
            "confidence": np.asarray(payload["filter_confidence"], dtype=np.float32),
            "dispersion": np.asarray(payload["p90_dispersion_voxels"], dtype=np.float32),
            "voxel_size": float(np.asarray(payload["voxel_size_reconstruction_units"]).item()),
        }


def _load_i24(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "points": np.asarray(payload["points_reconstruction"], dtype=np.float64),
            "colors": np.asarray(payload["colors_rgb"], dtype=np.uint8),
            "support": np.asarray(payload["baseline_support"], dtype=np.uint32),
            "is_core": np.asarray(payload["is_core"], dtype=bool),
            "peripheral": np.asarray(payload["is_peripheral"], dtype=bool),
            "far": np.asarray(payload["is_far_peripheral"], dtype=bool),
            "weak": np.asarray(payload["is_weak_peripheral"], dtype=bool),
            "reference": np.asarray(payload["is_reference_central_core"], dtype=bool),
            "low_diversity": np.asarray(payload["is_low_support_diversity"], dtype=bool),
            "angle": np.asarray(payload["maximum_pair_parallax_degrees"], dtype=np.float64),
            "baseline_ratio": np.asarray(payload["maximum_baseline_to_range_ratio"], dtype=np.float64),
            "voxel_size": float(np.asarray(payload["voxel_size_reconstruction_units"]).item()),
        }


def _assert_same_baseline(extended: Mapping[str, Any], diversity: Mapping[str, Any]) -> None:
    points = np.asarray(extended["points"], dtype=np.float64)
    dpoints = np.asarray(diversity["points"], dtype=np.float64)
    if points.shape != dpoints.shape or not np.array_equal(points, dpoints):
        raise ReconstructionValidationError("I2.5 F Extended and I2.4 point order/values are not exactly identical.")
    if not np.array_equal(np.asarray(extended["colors"], dtype=np.uint8), np.asarray(diversity["colors"], dtype=np.uint8)):
        raise ReconstructionValidationError("I2.5 F Extended and I2.4 colors are not exactly identical.")
    if not np.array_equal(np.asarray(extended["support"], dtype=np.uint32), np.asarray(diversity["support"], dtype=np.uint32)):
        raise ReconstructionValidationError("I2.5 F Extended and I2.4 support evidence disagrees.")
    if not np.isclose(float(extended["voxel_size"]), float(diversity["voxel_size"]), rtol=0.0, atol=1e-12):
        raise ReconstructionValidationError("I2.5 F/I2.4 voxel size differs.")


def _candidate_topology(points: np.ndarray, support: np.ndarray, voxel_size: float) -> np.ndarray:
    keys = _voxel_indices(points, voxel_size)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ReconstructionValidationError("I2.5 candidate retained points are not one point per voxel.")
    _, _, _, component_size, _, _ = _topology_metrics(keys, support)
    return np.asarray(component_size, dtype=np.int32)


def _save_scenario_npz(
    path: Path,
    *,
    baseline_points: np.ndarray,
    colors: np.ndarray,
    reject: np.ndarray,
    baseline_core: np.ndarray,
    baseline_peripheral: np.ndarray,
    baseline_far: np.ndarray,
    baseline_low_diversity: np.ndarray,
    candidate_peripheral: np.ndarray,
    candidate_far: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points_reconstruction=np.asarray(baseline_points, dtype=np.float64),
        colors_rgb=np.asarray(colors, dtype=np.uint8),
        is_rejected_by_policy=np.asarray(reject, dtype=np.uint8),
        is_kept=np.asarray(~reject, dtype=np.uint8),
        is_core_before=np.asarray(baseline_core, dtype=np.uint8),
        is_core_after=np.asarray(baseline_core & ~reject, dtype=np.uint8),
        is_peripheral_before=np.asarray(baseline_peripheral, dtype=np.uint8),
        is_far_before=np.asarray(baseline_far, dtype=np.uint8),
        is_low_diversity=np.asarray(baseline_low_diversity, dtype=np.uint8),
        is_candidate_peripheral=np.asarray(candidate_peripheral, dtype=np.uint8),
        is_candidate_far=np.asarray(candidate_far, dtype=np.uint8),
        schema_version=np.asarray(I25_DIVERSITY_FILTER_COUNTERFACTUAL_SCHEMA_VERSION, dtype=np.int32),
    )
    return path


def main() -> int:
    args = _args()
    i1_dir = args.i1_dir.expanduser().resolve()
    i2_dir = args.i2_dir.expanduser().resolve()
    i24_dir = args.i2_4_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I2.5 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i1 = _read_json(i1_dir / "i1_campaign_summary.json")
    i2 = _read_json(i2_dir / "i2_summary.json")
    i24 = _read_json(i24_dir / "i2_4_summary.json")
    if str(i1.get("status")) != "pass" or not bool(i1.get("pipeline_execution_complete")):
        raise ReconstructionValidationError("I2.5 requires completed passing I1 evidence.")
    if str(i2.get("phase")) != "I2" or not bool(i2.get("automated_pass")):
        raise ReconstructionValidationError("I2.5 requires passing automated I2 evidence.")
    if str(i24.get("status")) != "diagnosis_ready" or str(i24.get("scene_id")) != str(i2["scene"]["scene_id"]):
        raise ReconstructionValidationError("I2.5 requires completed same-scene I2.4 evidence.")
    if not bool(i24.get("diagnostic_signal", {}).get("supports_support_diversity_hypothesis")):
        raise ReconstructionValidationError("I2.5 is only appropriate after a strong I2.4 support-diversity signal.")

    f_summary_path = _resolve_f_summary(i1_dir)
    f_summary = _read_json(f_summary_path)
    if not bool(f_summary.get("pass")):
        raise ReconstructionValidationError("I2.5 requires a passing formal F baseline.")
    extended = _load_extended(Path(str(f_summary["artifacts"]["extended_npz"])).expanduser().resolve())
    core = _load_core(Path(str(f_summary["artifacts"]["core_npz"])).expanduser().resolve())
    diversity = _load_i24(Path(str(i24["artifacts"]["support_diversity_npz"])).expanduser().resolve())
    _assert_same_baseline(extended, diversity)

    points = np.asarray(extended["points"], dtype=np.float64)
    colors = np.asarray(extended["colors"], dtype=np.uint8)
    support = np.asarray(extended["support"], dtype=np.uint32)
    confidence = np.asarray(extended["confidence"], dtype=np.float32)
    dispersion = np.asarray(extended["dispersion"], dtype=np.float32)
    radius2 = np.asarray(extended["radius2"], dtype=np.int32)
    high = np.asarray(extended["high"], dtype=np.int32)
    is_core = np.asarray(diversity["is_core"], dtype=bool)
    peripheral = np.asarray(diversity["peripheral"], dtype=bool)
    far = np.asarray(diversity["far"], dtype=bool)
    weak = np.asarray(diversity["weak"], dtype=bool)
    reference = np.asarray(diversity["reference"], dtype=bool)
    low_diversity = np.asarray(diversity["low_diversity"], dtype=bool)
    angle = np.asarray(diversity["angle"], dtype=np.float64)
    baseline_ratio = np.asarray(diversity["baseline_ratio"], dtype=np.float64)
    voxel_size = float(extended["voxel_size"])
    core_points = np.asarray(core["points"], dtype=np.float64)

    if int(np.count_nonzero(is_core)) != len(core_points):
        raise ReconstructionValidationError("I2.5 I2.4 Core membership count disagrees with formal F Core cloud.")
    if int(i2["retained_geometry"]["extended_points"]) != len(points):
        raise ReconstructionValidationError("I2.5 I2/F Extended count disagreement.")
    if int(i2["retained_geometry"]["core_points"]) != int(np.count_nonzero(is_core)):
        raise ReconstructionValidationError("I2.5 I2/F Core count disagreement.")

    quality_policy = I2QualityPolicy(**{key: float(value) for key, value in i2["policy"].items()})
    baseline_full_to_robust = float(i2["spatial"]["extended_envelope"]["full_to_robust_diagonal_ratio"])
    baseline_robust = float(i2["spatial"]["extended_envelope"]["robust_diagonal"])
    reference_thresholds = i24["reference_thresholds"]
    parallax_ref = float(reference_thresholds["maximum_pair_parallax_degrees_threshold"])
    ratio_ref = float(reference_thresholds["maximum_baseline_to_range_ratio_threshold"])

    baseline_profile, _ = scenario_quality_profile(
        extended_points=points,
        core_points=core_points,
        support=support,
        confidence=confidence,
        dispersion=dispersion,
        component_size=np.asarray(extended["component_size"], dtype=np.int32),
        voxel_size=voxel_size,
        baseline_core_points=core_points,
        baseline_core_confidence=np.asarray(core["confidence"], dtype=np.float32),
        baseline_core_dispersion=np.asarray(core["dispersion"], dtype=np.float32),
        policy=quality_policy,
    )
    baseline_profile["baseline_anchor"]["peripheral_points"] = int(np.count_nonzero(peripheral))
    baseline_profile["baseline_anchor"]["far_peripheral_points"] = int(np.count_nonzero(far))
    baseline_profile["baseline_anchor"]["weak_peripheral_points"] = int(np.count_nonzero(weak))

    scenarios = default_diversity_filter_scenarios()
    print("I2.5 conservative diversity-aware filtering counterfactuals:")
    print(
        f"baseline extended={len(points)} core={int(np.count_nonzero(is_core))} peripheral={int(np.count_nonzero(peripheral))} "
        f"far={int(np.count_nonzero(far))} full/robust={baseline_full_to_robust:.3f}"
    )
    print(
        f"reference parallax={parallax_ref:.4f}deg baseline/range={ratio_ref:.6f} scenarios={len(scenarios)} "
        "source_F_changed=False production_F_policy_changed=False"
    )

    records: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, 1):
        reject, resolved = diversity_filter_rejection_mask(
            scenario=scenario,
            support=support,
            maximum_pair_parallax_degrees=angle,
            maximum_baseline_to_range_ratio=baseline_ratio,
            high_support_neighbors=high,
            radius2_neighbors=radius2,
            confidence=confidence,
            reference_mask=reference,
            parallax_reference_threshold_degrees=parallax_ref,
            baseline_to_range_reference_threshold=ratio_ref,
        )
        kept = ~reject
        candidate_core_mask = is_core & kept
        if int(np.count_nonzero(kept)) < 1000 or int(np.count_nonzero(candidate_core_mask)) < 1000:
            raise ReconstructionValidationError(f"I2.5 scenario {scenario.name} is unexpectedly destructive.")
        candidate_points = points[kept]
        candidate_colors = colors[kept]
        candidate_support = support[kept]
        candidate_confidence = confidence[kept]
        candidate_dispersion = dispersion[kept]
        component_size = _candidate_topology(candidate_points, candidate_support, voxel_size)
        candidate_core_points = points[candidate_core_mask]

        profile, _ = scenario_quality_profile(
            extended_points=candidate_points,
            core_points=candidate_core_points,
            support=candidate_support,
            confidence=candidate_confidence,
            dispersion=candidate_dispersion,
            component_size=component_size,
            voxel_size=voxel_size,
            baseline_core_points=core_points,
            baseline_core_confidence=np.asarray(core["confidence"], dtype=np.float32),
            baseline_core_dispersion=np.asarray(core["dispersion"], dtype=np.float32),
            policy=quality_policy,
        )
        effects = policy_effect_counts(
            reject,
            is_core=is_core,
            is_peripheral=peripheral,
            is_far_peripheral=far,
            is_weak_peripheral=weak,
        )
        candidate_ratio = float(profile["extended_envelope"]["full_to_robust_diagonal_ratio"])
        candidate_robust = float(profile["extended_envelope"]["robust_diagonal"])
        comparison = counterfactual_quality_comparison(
            baseline_extended_points=len(points),
            baseline_core_points=int(np.count_nonzero(is_core)),
            baseline_full_to_robust_ratio=baseline_full_to_robust,
            baseline_robust_diagonal=baseline_robust,
            candidate_extended_points=len(candidate_points),
            candidate_core_points=len(candidate_core_points),
            candidate_full_to_robust_ratio=candidate_ratio,
            candidate_robust_diagonal=candidate_robust,
            candidate_dominant_component_fraction=float(profile["dominant_component_fraction_of_extended"]),
            effect_counts=effects,
        )

        # Evaluation-only spatial labels for visual review. They are NOT inputs
        # to the candidate filter rule.
        cand_peripheral_short, cand_far_short, _, _ = classify_peripheral_geometry(
            candidate_points,
            core_points,
            voxel_size=voxel_size,
            policy=quality_policy,
        )
        cand_peripheral = np.zeros(len(points), dtype=bool)
        cand_far = np.zeros(len(points), dtype=bool)
        cand_peripheral[kept] = cand_peripheral_short
        cand_far[kept] = cand_far_short

        scenario_dir = output / "scenarios" / scenario.name
        artifact = _save_scenario_npz(
            scenario_dir / "i2_5_policy_diagnostics.npz",
            baseline_points=points,
            colors=colors,
            reject=reject,
            baseline_core=is_core,
            baseline_peripheral=peripheral,
            baseline_far=far,
            baseline_low_diversity=low_diversity,
            candidate_peripheral=cand_peripheral,
            candidate_far=cand_far,
        )
        record = {
            "name": scenario.name,
            "scenario": scenario.as_dict(),
            "resolved_rule": resolved,
            "effects_against_baseline_regions": effects,
            "candidate_profile": profile,
            "comparison": comparison,
            "artifacts": {"diagnostics_npz": str(artifact.resolve())},
        }
        _write_json(scenario_dir / "scenario_summary.json", record)
        records.append(record)
        print(
            f"[{index}/{len(scenarios)}] {scenario.name}: reject={effects['rejected_points']} "
            f"core_loss={comparison['core_loss_fraction']:.3f} central_core_loss={comparison['central_core_loss_fraction']:.3f} "
            f"peripheral_reduction={comparison['peripheral_reduction_fraction']:.3f} "
            f"far_reduction={comparison['far_peripheral_reduction_fraction']:.3f} "
            f"full/robust={candidate_ratio:.3f} ratio_improvement={comparison['full_to_robust_ratio_improvement_fraction']:.3f} "
            f"balanced={comparison['balanced_for_manual_review']} preferred={comparison['preferred_for_manual_review']}"
        )

    ranked = sorted(records, key=lambda item: float(item["comparison"]["diagnostic_score"]), reverse=True)
    pareto = pareto_front(records)
    balanced = [item["name"] for item in ranked if bool(item["comparison"]["balanced_for_manual_review"])]
    preferred = [item["name"] for item in ranked if bool(item["comparison"]["preferred_for_manual_review"])]
    manual_priority = preferred[:3] or balanced[:3] or [item["name"] for item in ranked[:3]]

    summary = {
        "phase": "I2.5",
        "schema_version": I25_DIVERSITY_FILTER_COUNTERFACTUAL_SCHEMA_VERSION,
        "status": "counterfactuals_ready",
        "source_i1_dir": str(i1_dir),
        "source_i2_dir": str(i2_dir),
        "source_i2_4_dir": str(i24_dir),
        "scene_id": str(i2["scene"]["scene_id"]),
        "baseline": {
            "extended_points": len(points),
            "core_points": int(np.count_nonzero(is_core)),
            "peripheral_points": int(np.count_nonzero(peripheral)),
            "far_peripheral_points": int(np.count_nonzero(far)),
            "weak_peripheral_points": int(np.count_nonzero(weak)),
            "full_to_robust_diagonal_ratio": baseline_full_to_robust,
            "extended_robust_diagonal": baseline_robust,
            "voxel_size_reconstruction_units": voxel_size,
        },
        "reference_thresholds": {
            "source": str(reference_thresholds["source"]),
            "maximum_pair_parallax_degrees": parallax_ref,
            "maximum_baseline_to_range_ratio": ratio_ref,
            "filter_rules_do_not_use_peripheral_labels": True,
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
            "diversity_aware_production_filtering_authorized": False,
            "next_action": (
                "Review only the highest-ranked balanced/preferred I2.5 policy counterfactuals. "
                "Promotion into formal F belongs to I2.6 only if one policy preserves central/Core geometry while "
                "materially reducing peripheral/far artifacts and improving the spatial envelope."
            ),
        },
    }
    path = _write_json(output / "i2_5_summary.json", summary)
    print("\nI2.5 conservative diversity-aware filtering counterfactuals: COUNTERFACTUALS_READY")
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
