#!/usr/bin/env python3
"""I2.2 narrow exact counterfactual refusion.

Consumes the passing I1 full-video campaign plus I2/I2.1 diagnostics. It reuses
saved aligned dense views, performs real DenseVoxelFusion for narrowly excluded
single/pair scenarios, reruns unchanged formal F filtering for each scenario,
and compares exact spatial quality against the baseline. No SfM/DA3 rerun, no
source G-bundle mutation, no metric/measurement/mesh authority promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    DenseFusionPolicy,
    DenseSceneProfile,
    arbitrary_scale_reference,
    camera_geometry_from_estimated_camera,
    fuse_dense_frames,
    reconstruction_camera_from_sfm_model,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_counterfactual_refusion import (  # noqa: E402
    I22_COUNTERFACTUAL_SCHEMA_VERSION,
    CounterfactualScenario,
    comparison_against_baseline,
    pareto_front,
    scenario_quality_profile,
    scenarios_from_i21,
    validate_scenario_against_baseline,
)
from core.reconstruction.offline_dense_quality import I2QualityPolicy  # noqa: E402
from scripts.run_d1_three_view_canonical_accumulation import _sfm_result_from_payload  # noqa: E402
from scripts.run_e1_qualified_ten_view_voxel_fusion import (  # noqa: E402
    _save_cloud_npz,
    _support_histogram,
    _write_binary_ply,
)
from scripts.run_e3_qualified_scene_fusion_expansion import (  # noqa: E402
    _build_dense_fusion_frames,
    _load_cloud_record,
)


_DEFAULT_I1 = Path("/tmp/depth3d_i1_offline_dense_campaign_recovered")
_DEFAULT_I2 = Path("/tmp/depth3d_i2_full_video_quality")
_DEFAULT_I21 = Path("/tmp/depth3d_i2_1_targeted_peripheral_diagnosis")
_DEFAULT_OUT = Path("/tmp/depth3d_i2_2_narrow_exact_counterfactual_refusion")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1-dir", type=Path, default=_DEFAULT_I1)
    parser.add_argument("--i2-dir", type=Path, default=_DEFAULT_I2)
    parser.add_argument("--i2-1-dir", type=Path, default=_DEFAULT_I21)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--top-single-views", type=int, default=5)
    parser.add_argument("--top-pairs", type=int, default=5)
    parser.add_argument("--max-voxels", type=int, default=1_500_000)
    parser.add_argument("--sfm-focal-relative-uncertainty", type=float, default=0.20)
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        metavar="NAME=ORDER[,ORDER]",
        help="Optional explicit narrow scenario(s). If omitted, use I2.1 top five singles + top five support-2 pairs.",
    )
    args = parser.parse_args()
    if args.top_single_views < 0 or args.top_pairs < 0 or args.max_voxels <= 0:
        parser.error("scenario counts must be non-negative and --max-voxels positive")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read I2.2 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I2.2 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _parse_explicit(text: str) -> CounterfactualScenario:
    if "=" not in text:
        raise ReconstructionValidationError("I2.2 --scenario must be NAME=ORDER[,ORDER].")
    name, values = text.split("=", 1)
    orders = tuple(int(part.strip()) for part in values.split(",") if part.strip())
    if not orders or len(orders) > 2:
        raise ReconstructionValidationError("I2.2 narrow scenarios must exclude one or two views.")
    return CounterfactualScenario(name.strip(), orders, "explicit_user_scenario")


def _resolve_sfm_report(i1_dir: Path, i1: Mapping[str, Any]) -> Path:
    local = i1_dir / "01_sfm" / "real_sfm_report.json"
    if local.is_file():
        return local
    sfm_stage = i1.get("stages", {}).get("sfm", {}) if isinstance(i1.get("stages"), Mapping) else {}
    source = sfm_stage.get("source_dir") if isinstance(sfm_stage, Mapping) else None
    if source:
        candidate = Path(str(source)).expanduser().resolve() / "real_sfm_report.json"
        if candidate.is_file():
            return candidate
    raise ReconstructionValidationError("I2.2 cannot resolve the reused I1 SfM report.")


def _load_filtered(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "points": np.asarray(payload["points_reconstruction"], dtype=np.float64),
            "colors": np.asarray(payload["colors_rgb"], dtype=np.uint8),
            "support": np.asarray(payload["view_observation_counts"], dtype=np.uint32),
            "confidence": np.asarray(payload["filter_confidence"], dtype=np.float32),
            "dispersion": np.asarray(payload["p90_dispersion_voxels"], dtype=np.float32),
            "component_size": np.asarray(payload["component_size"], dtype=np.int32),
            "voxel_size": float(np.asarray(payload["voxel_size_reconstruction_units"]).item()),
        }


def _run_f(e3_dir: Path, out: Path, log: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_f_confidence_artifact_filtering.py"),
        "--e3-dir", str(e3_dir),
        "--output-dir", str(out),
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print("    " + line, end="", flush=True)
            handle.write(line)
        code = process.wait()
    if code != 0:
        raise ReconstructionValidationError(f"I2.2 formal F filtering failed for {e3_dir.parent.name}; see {log}")


def _write_f_compatibility(
    *,
    directory: Path,
    fusion_result: Any,
    records: list[dict[str, Any]],
    voxel_size: float,
    sparse_points: np.ndarray,
    sparse_colors: np.ndarray,
    excluded_orders: tuple[int, ...],
) -> dict[str, Any]:
    cloud = fusion_result.point_cloud
    counts = np.asarray(cloud.view_observation_counts, dtype=np.uint32)
    orders = tuple(int(r["workspace_order"]) for r in records)
    keyframes = tuple(str(r["keyframe_id"]) for r in records)
    supported = counts >= 2
    high = counts >= 3
    all_npz = _save_cloud_npz(
        directory / "e3_expanded_voxel_fused_all.npz",
        points=cloud.points_reconstruction,
        colors=cloud.colors_rgb,
        counts=counts,
        mean_confidence=cloud.mean_confidence,
        voxel_size=voxel_size,
        selected_keyframe_ids=keyframes,
        selected_workspace_orders=orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    supported_npz = _save_cloud_npz(
        directory / "e3_expanded_voxel_fused_supported.npz",
        points=cloud.points_reconstruction[supported],
        colors=cloud.colors_rgb[supported],
        counts=counts[supported],
        mean_confidence=cloud.mean_confidence[supported],
        voxel_size=voxel_size,
        selected_keyframe_ids=keyframes,
        selected_workspace_orders=orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    high_npz = _save_cloud_npz(
        directory / "e3_expanded_voxel_fused_high_support.npz",
        points=cloud.points_reconstruction[high],
        colors=cloud.colors_rgb[high],
        counts=counts[high],
        mean_confidence=cloud.mean_confidence[high],
        voxel_size=voxel_size,
        selected_keyframe_ids=keyframes,
        selected_workspace_orders=orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    all_ply = _write_binary_ply(directory / "e3_expanded_voxel_fused_all.ply", cloud.points_reconstruction, cloud.colors_rgb)
    supported_ply = _write_binary_ply(
        directory / "e3_expanded_voxel_fused_supported.ply", cloud.points_reconstruction[supported], cloud.colors_rgb[supported]
    )
    frames = [
        {
            "fusion_index": index,
            "workspace_order": int(record["workspace_order"]),
            "keyframe_id": str(record["keyframe_id"]),
            "frame_id": str(record["frame_id"]),
            "timestamp_ns": int(record["workspace"].timestamp_ns),
            "source": "i2.2_exact_counterfactual_source_view",
            "cloud_npz": str(Path(record["cloud_npz"]).resolve()),
        }
        for index, record in enumerate(records)
    ]
    capacity_drops = int(fusion_result.statistics.dropped_new_voxels_capacity)
    passed = len(records) >= 10 and capacity_drops == 0 and int(np.count_nonzero(supported)) >= 1000
    summary = {
        "phase": "E3",
        "pass": bool(passed),
        "status": "pass" if passed else "degraded",
        "campaign_adapter": {
            "source_phase": "I2.2",
            "purpose": "exact narrow counterfactual refusion for diagnostic comparison only",
            "counterfactual": True,
            "excluded_workspace_orders": list(excluded_orders),
        },
        "selected_workspace_orders": list(orders),
        "selected_keyframe_ids": list(keyframes),
        "selected_view_count": len(orders),
        "frames": frames,
        "fusion": {
            "voxel_size_reconstruction_units": float(voxel_size),
            "all_voxel_count": int(len(counts)),
            "supported_voxel_count": int(np.count_nonzero(supported)),
            "high_support_voxel_count": int(np.count_nonzero(high)),
            "support_histogram": _support_histogram(counts),
            "capacity_clean": capacity_drops == 0,
            "capacity_drops": capacity_drops,
            "frames_integrated": int(fusion_result.statistics.frames_integrated),
            "sampled_pixels": int(fusion_result.statistics.sampled_pixels),
        },
        "decision": {
            "authorizes_next_confidence_filtering_milestone": bool(passed),
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
            "all_npz": str(all_npz.resolve()), "all_ply": str(all_ply.resolve()),
            "supported_npz": str(supported_npz.resolve()), "supported_ply": str(supported_ply.resolve()),
            "high_support_npz": str(high_npz.resolve()),
        },
    }
    _write_json(directory / "e3_summary.json", summary)
    return summary


def _save_diagnostic(path: Path, cloud: Mapping[str, np.ndarray | float], masks: Mapping[str, np.ndarray], core_points: np.ndarray) -> Path:
    # Mark Core approximately by exact voxel keys. F uses one point per voxel.
    voxel = float(cloud["voxel_size"])
    def keys(points: np.ndarray) -> set[tuple[int, int, int]]:
        return {tuple(map(int, row)) for row in np.floor(points / voxel).astype(np.int64)}
    core_keys = keys(core_points)
    ext_keys = [tuple(map(int, row)) for row in np.floor(np.asarray(cloud["points"]) / voxel).astype(np.int64)]
    is_core = np.asarray([key in core_keys for key in ext_keys], dtype=np.uint8)
    np.savez_compressed(
        path,
        points_reconstruction=np.asarray(cloud["points"], dtype=np.float64),
        colors_rgb=np.asarray(cloud["colors"], dtype=np.uint8),
        view_observation_counts=np.asarray(cloud["support"], dtype=np.uint32),
        filter_confidence=np.asarray(cloud["confidence"], dtype=np.float32),
        p90_dispersion_voxels=np.asarray(cloud["dispersion"], dtype=np.float32),
        component_size=np.asarray(cloud["component_size"], dtype=np.int32),
        is_core=is_core,
        is_peripheral=np.asarray(masks["peripheral"], dtype=np.uint8),
        is_far_peripheral=np.asarray(masks["far"], dtype=np.uint8),
        is_weak_peripheral=np.asarray(masks["weak"], dtype=np.uint8),
        directional_sector_code=np.asarray(masks["sector"], dtype=np.uint8),
        voxel_size_reconstruction_units=np.asarray(voxel, dtype=np.float64),
        schema_version=np.asarray(I22_COUNTERFACTUAL_SCHEMA_VERSION, dtype=np.int32),
    )
    return path


def main() -> int:
    args = _args()
    i1_dir = args.i1_dir.expanduser().resolve()
    i2_dir = args.i2_dir.expanduser().resolve()
    i21_dir = args.i2_1_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I2.2 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i1 = _read_json(i1_dir / "i1_campaign_summary.json")
    i2 = _read_json(i2_dir / "i2_summary.json")
    i21 = _read_json(i21_dir / "i2_1_summary.json")
    if str(i1.get("status")) != "pass" or not bool(i1.get("pipeline_execution_complete")):
        raise ReconstructionValidationError("I2.2 requires a completed passing I1 campaign.")
    if not bool(i2.get("automated_pass")) or str(i21.get("status")) != "diagnosis_ready":
        raise ReconstructionValidationError("I2.2 requires passing I2 automated gates and ready I2.1 diagnosis.")
    if str(i2.get("scene", {}).get("scene_id")) != str(i21.get("scene_id")):
        raise ReconstructionValidationError("I2.2 detected I2/I2.1 scene identity drift.")

    fusion_dir = i1_dir / "04_qualified_fusion"
    filter_dir = i1_dir / "05_filtering"
    baseline_e3 = _read_json(fusion_dir / "e3_summary.json")
    baseline_f = _read_json(filter_dir / "f_summary.json")
    selected_orders = tuple(int(v) for v in baseline_e3["selected_workspace_orders"])
    if args.scenario:
        scenarios = tuple(_parse_explicit(text) for text in args.scenario)
    else:
        scenarios = scenarios_from_i21(
            i21, selected_orders, top_single_views=int(args.top_single_views), top_pairs=int(args.top_pairs)
        )
    for scenario in scenarios:
        validate_scenario_against_baseline(scenario, selected_orders)

    sfm_report = _read_json(_resolve_sfm_report(i1_dir, i1))
    sfm_result = _sfm_result_from_payload(sfm_report["sfm_result"])
    frame_by_order = {int(frame["workspace_order"]): frame for frame in baseline_e3["frames"]}
    records: dict[int, dict[str, Any]] = {}
    for order in selected_orders:
        frame = frame_by_order[order]
        records[order] = _load_cloud_record(
            order=order,
            keyframe_id=str(frame["keyframe_id"]),
            cloud_npz=Path(str(frame["cloud_npz"])),
            sfm_result=sfm_result,
            scale_to_reconstruction_units=None,
            source="i2.2_reused_i1_aligned_source",
        )

    first = records[selected_orders[0]]
    models = {item.camera_id: item for item in sfm_result.camera_models}
    model = models[str(first["camera_id"])]
    camera = reconstruction_camera_from_sfm_model(
        model, keyframe_id=str(first["keyframe_id"]), reconstruction_frame_id="offline-sfm-reconstruction"
    )
    camera_profile = camera_geometry_from_estimated_camera(
        profile_id=f"i2-2-{first['camera_id']}-sfm-estimated",
        camera=camera,
        focal_length_relative_uncertainty=float(args.sfm_focal_relative_uncertainty),
        notes="I2.2 counterfactual camera profile; arbitrary SfM gauge only.",
    )
    scale_profile = arbitrary_scale_reference(
        profile_id=f"i2-2-{sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
        notes="I2.2 exact counterfactual preserves arbitrary SfM gauge; no physical unit claim.",
    )
    voxel_size = float(baseline_e3["fusion"]["voxel_size_reconstruction_units"])
    sparse_path = Path(str(baseline_e3["artifacts"]["all_npz"])).expanduser().resolve()
    with np.load(sparse_path, allow_pickle=False) as payload:
        sparse_points = np.asarray(payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64)
        sparse_colors = np.asarray(payload.get("sparse_landmark_colors_rgb", np.empty((0, 3))), dtype=np.uint8)

    baseline_extended = _load_filtered(Path(str(baseline_f["artifacts"]["extended_npz"])).expanduser().resolve())
    baseline_core = _load_filtered(Path(str(baseline_f["artifacts"]["core_npz"])).expanduser().resolve())
    policy = I2QualityPolicy(**{k: float(v) for k, v in i2["policy"].items()})
    baseline_profile, baseline_masks = scenario_quality_profile(
        extended_points=baseline_extended["points"], core_points=baseline_core["points"],
        support=baseline_extended["support"], confidence=baseline_extended["confidence"],
        dispersion=baseline_extended["dispersion"], component_size=baseline_extended["component_size"],
        voxel_size=float(baseline_extended["voxel_size"]), baseline_core_points=baseline_core["points"],
        baseline_core_confidence=baseline_core["confidence"], baseline_core_dispersion=baseline_core["dispersion"],
        policy=policy,
    )
    baseline_profile["baseline_anchor"]["peripheral_points"] = int(i2["spatial"]["peripheral_retained"]["count"])
    baseline_profile["baseline_anchor"]["far_peripheral_points"] = int(i2["spatial"]["far_peripheral_retained"]["count"])
    baseline_profile["baseline_anchor"]["weak_peripheral_points"] = int(i2["spatial"]["weak_peripheral_retained"]["count"])

    print("I2.2 narrow exact counterfactual refusion:")
    print(
        f"baseline views={len(selected_orders)} extended={baseline_profile['extended_points']} core={baseline_profile['core_points']} "
        f"peripheral={baseline_profile['baseline_anchor']['peripheral_points']} "
        f"far={baseline_profile['baseline_anchor']['far_peripheral_points']}"
    )
    print(f"scenarios={len(scenarios)} voxel={voxel_size:.6f} sfm_rerun=False da3_rerun=False")

    scenario_results: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, 1):
        remaining_orders = validate_scenario_against_baseline(scenario, selected_orders)
        scenario_dir = output / "scenarios" / scenario.name
        fusion_out = scenario_dir / "04_exact_refusion"
        filter_out = scenario_dir / "05_filtering"
        fusion_out.mkdir(parents=True, exist_ok=True)
        selected_records = [records[order] for order in remaining_orders]
        frames, _, _ = _build_dense_fusion_frames(selected_records, sfm_result=sfm_result)
        fusion_policy = DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=voxel_size,
            pixel_stride=1,
            minimum_view_observations=1,
            max_voxels=int(args.max_voxels),
        )
        print(
            f"\n[{index}/{len(scenarios)}] {scenario.name} exclude={','.join(map(str, scenario.excluded_orders))} "
            f"views={len(frames)}",
            flush=True,
        )
        fusion_result = fuse_dense_frames(
            frames,
            session_id=f"{sfm_result.session_id}:i2.2:{scenario.name}",
            camera_profile=camera_profile,
            scale_profile=scale_profile,
            policy=fusion_policy,
            reconstruction_frame_id="offline-sfm-reconstruction",
        )
        compatibility = _write_f_compatibility(
            directory=fusion_out, fusion_result=fusion_result, records=selected_records, voxel_size=voxel_size,
            sparse_points=sparse_points, sparse_colors=sparse_colors, excluded_orders=scenario.excluded_orders,
        )
        if not compatibility["pass"]:
            raise ReconstructionValidationError(f"I2.2 exact refusion failed the F compatibility gate: {scenario.name}")
        _run_f(fusion_out, filter_out, scenario_dir / "f_filtering.log")
        f_summary = _read_json(filter_out / "f_summary.json")
        if not bool(f_summary.get("pass")):
            raise ReconstructionValidationError(f"I2.2 formal F result is not PASS: {scenario.name}")
        extended = _load_filtered(Path(str(f_summary["artifacts"]["extended_npz"])).expanduser().resolve())
        core = _load_filtered(Path(str(f_summary["artifacts"]["core_npz"])).expanduser().resolve())
        profile, masks = scenario_quality_profile(
            extended_points=extended["points"], core_points=core["points"], support=extended["support"],
            confidence=extended["confidence"], dispersion=extended["dispersion"], component_size=extended["component_size"],
            voxel_size=float(extended["voxel_size"]), baseline_core_points=baseline_core["points"],
            baseline_core_confidence=baseline_core["confidence"], baseline_core_dispersion=baseline_core["dispersion"],
            policy=policy,
        )
        comparison = comparison_against_baseline(baseline_profile, profile)
        diagnostic = _save_diagnostic(
            scenario_dir / "i2_2_spatial_diagnostics.npz", extended, masks, np.asarray(core["points"], dtype=np.float64)
        )
        result = {
            **scenario.as_dict(),
            "remaining_workspace_orders": list(remaining_orders),
            "remaining_view_count": len(remaining_orders),
            "fusion": compatibility["fusion"],
            "filtering": {
                "source_supported_voxels": int(f_summary["classification"]["source_supported_voxels"]),
                "core_voxels": int(f_summary["classification"]["core_voxels"]),
                "extended_voxels": int(f_summary["classification"]["extended_voxels"]),
                "rejected_supported_voxels": int(f_summary["classification"]["rejected_supported_voxels"]),
                "exact_observation_accounting": bool(f_summary["observation_accounting"]["exact_match"]),
            },
            "quality_profile": profile,
            "comparison": comparison,
            "artifacts": {
                "fusion_dir": str(fusion_out.resolve()),
                "filter_dir": str(filter_out.resolve()),
                "diagnostics_npz": str(diagnostic.resolve()),
            },
        }
        _write_json(scenario_dir / "scenario_summary.json", result)
        scenario_results.append(result)
        print(
            f"  core={profile['core_points']} loss={comparison['core_loss_fraction']:.3f} "
            f"extended={profile['extended_points']} loss={comparison['extended_loss_fraction']:.3f} "
            f"peripheral={profile['baseline_anchor']['peripheral_points']} reduction={comparison['peripheral_reduction_fraction']:.3f} "
            f"far={profile['baseline_anchor']['far_peripheral_points']} reduction={comparison['far_peripheral_reduction_fraction']:.3f} "
            f"full/robust={profile['extended_envelope']['full_to_robust_diagonal_ratio']:.3f} "
            f"score={comparison['diagnostic_score']:.3f}"
        )

    ranked = sorted(scenario_results, key=lambda item: float(item["comparison"]["diagnostic_score"]), reverse=True)
    pareto = pareto_front(scenario_results)
    promising = [item["name"] for item in ranked if bool(item["comparison"]["promising_for_manual_exact_review"])]
    summary = {
        "phase": "I2.2",
        "schema_version": I22_COUNTERFACTUAL_SCHEMA_VERSION,
        "status": "exact_counterfactuals_ready",
        "source_i1_dir": str(i1_dir),
        "source_i2_dir": str(i2_dir),
        "source_i2_1_dir": str(i21_dir),
        "scene_id": str(i2["scene"]["scene_id"]),
        "baseline": baseline_profile,
        "scenarios": scenario_results,
        "ranking": {
            "by_diagnostic_score": [item["name"] for item in ranked],
            "pareto_front": list(pareto),
            "promising_for_manual_exact_review": promising,
            "diagnostic_only": True,
        },
        "authority": {
            "source_geometry_changed": False,
            "source_persistent_scene_changed": False,
            "counterfactual_geometry_generated": True,
            "sfm_rerun": False,
            "da3_rerun": False,
            "fusion_rerun_per_scenario": True,
            "formal_f_filtering_rerun_per_scenario": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_i2_closure": False,
            "authorizes_i3": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": "Inspect the top Pareto/promising exact counterfactuals visually before choosing any production correction.",
        },
    }
    path = _write_json(output / "i2_2_summary.json", summary)
    print("\nI2.2 narrow exact counterfactual refusion: EXACT_COUNTERFACTUALS_READY")
    print("ranking=" + ",".join(item["name"] for item in ranked))
    print("pareto_front=" + (",".join(pareto) if pareto else "none"))
    print("promising=" + (",".join(promising) if promising else "none"))
    print("source_geometry_changed=False sfm_rerun=False da3_rerun=False mesh_authorization=False")
    print(f"summary={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
