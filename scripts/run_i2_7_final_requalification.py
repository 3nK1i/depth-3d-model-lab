#!/usr/bin/env python3
"""I2.7 final full-video requalification before manual closure.

This is validation-only. It consumes the original I2 baseline plus the promoted
I2.6 formal F/G scene, recomputes spatial/component diagnostics, and emits a
fail-closed READY_FOR_FINAL_MANUAL_REVIEW summary when all automated gates pass.
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

from core.reconstruction.dense_scene_persistence import (  # noqa: E402
    read_persistent_dense_rgb_scene,
    restore_persistent_dense_geometry,
    verify_persistent_dense_scene_files,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_i2_closure import (  # noqa: E402
    I2_CLOSURE_SCHEMA_VERSION,
    I2ClosurePolicy,
    baseline_aware_component_topology_gates,
    closure_quality_gates,
    voxel_component_profile,
)
from core.reconstruction.offline_dense_quality import (  # noqa: E402
    I2QualityPolicy,
    automated_quality_gates,
    classify_peripheral_geometry,
    directional_sector_summary,
    region_statistics,
    spatial_envelope,
)
from scripts.run_e2_consistency_aware_fusion_refinement import _voxel_indices  # noqa: E402
from scripts.run_i2_full_video_quality_qualification import _resolve_sfm_summary  # noqa: E402

_DEFAULT_I1 = Path("/tmp/eia_i1_offline_dense_campaign_recovered")
_DEFAULT_I2 = Path("/tmp/eia_i2_full_video_quality")
_DEFAULT_I26 = Path("/tmp/eia_i2_6_manifold_filter_promotion")
_DEFAULT_OUT = Path("/tmp/eia_i2_7_final_requalification")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i1-dir", type=Path, default=_DEFAULT_I1)
    p.add_argument("--i2-baseline-dir", type=Path, default=_DEFAULT_I2)
    p.add_argument("--i2-6-dir", type=Path, default=_DEFAULT_I26)
    p.add_argument("--i2-7-1-dir", type=Path, default=None, help="Optional I2.7.1 detached-component qualification used for baseline-aware topology gating.")
    p.add_argument("--confirm-i2-7-1-lineage-review-pass", action="store_true", help="Explicit confirmation that the I2.7.1 lineage visualization was manually reviewed and passed.")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read I2.7 JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I2.7 JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_filtered(path: Path) -> dict[str, np.ndarray | float]:
    if not path.is_file():
        raise ReconstructionValidationError(f"I2.7 promoted filtered cloud is missing: {path}")
    with np.load(path, allow_pickle=False) as p:
        out: dict[str, np.ndarray | float] = {
            "points": np.asarray(p["points_reconstruction"], dtype=np.float64),
            "colors": np.asarray(p["colors_rgb"], dtype=np.uint8),
            "support": np.asarray(p["view_observation_counts"], dtype=np.uint32),
            "confidence": np.asarray(p["filter_confidence"], dtype=np.float32),
            "dispersion": np.asarray(p["p90_dispersion_voxels"], dtype=np.float32),
            "voxel_size": float(np.asarray(p["voxel_size_reconstruction_units"]).item()),
        }
    n = len(np.asarray(out["points"]))
    if n == 0 or np.asarray(out["colors"]).shape != (n, 3):
        raise ReconstructionValidationError("I2.7 promoted filtered cloud is invalid.")
    for name in ("support", "confidence", "dispersion"):
        if np.asarray(out[name]).shape != (n,):
            raise ReconstructionValidationError(f"I2.7 promoted field {name} is misaligned.")
    return out


def _safe_reduction(baseline: int, current: int) -> float:
    return 0.0 if baseline <= 0 else float((baseline - current) / baseline)


def main() -> int:
    a = _args()
    i1_dir = a.i1_dir.expanduser().resolve()
    i2_dir = a.i2_baseline_dir.expanduser().resolve()
    i26_dir = a.i2_6_dir.expanduser().resolve()
    i271_dir = a.i2_7_1_dir.expanduser().resolve() if a.i2_7_1_dir is not None else None
    out = a.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I2.7 output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    i1 = _read_json(i1_dir / "i1_campaign_summary.json")
    baseline = _read_json(i2_dir / "i2_summary.json")
    i26 = _read_json(i26_dir / "i2_6_summary.json")
    if str(baseline.get("phase")) != "I2" or not bool(baseline.get("automated_pass")):
        raise ReconstructionValidationError("I2.7 requires the original automated-passing I2 baseline summary.")
    if str(i26.get("phase")) != "I2.6" or not bool(i26.get("pass")) or not bool(i26.get("decision", {}).get("authorizes_i2_7_final_requalification")):
        raise ReconstructionValidationError("I2.7 requires a passing I2.6 promotion with explicit authorization.")

    i271 = None
    if i271_dir is not None:
        i271 = _read_json(i271_dir / "i2_7_1_summary.json")
        if not bool(a.confirm_i2_7_1_lineage_review_pass):
            raise ReconstructionValidationError(
                "Baseline-aware I2.7 requalification requires --confirm-i2-7-1-lineage-review-pass after the lineage visualization has been reviewed."
            )

    f_summary = _read_json(Path(str(i26["artifacts"]["promoted_f_summary"])).expanduser().resolve())
    ext = _load_filtered(Path(str(i26["artifacts"]["promoted_extended_npz"])).expanduser().resolve())
    core = _load_filtered(Path(str(i26["artifacts"]["promoted_core_npz"])).expanduser().resolve())
    points = np.asarray(ext["points"], dtype=np.float64)
    colors = np.asarray(ext["colors"], dtype=np.uint8)
    core_points = np.asarray(core["points"], dtype=np.float64)
    voxel = float(ext["voxel_size"])
    if not np.isclose(voxel, float(core["voxel_size"]), atol=1e-12, rtol=0.0):
        raise ReconstructionValidationError("I2.7 promoted Core/Extended voxel sizes disagree.")

    manifest = Path(str(i26["artifacts"]["persistent_scene_manifest"])).expanduser().resolve()
    scene = read_persistent_dense_rgb_scene(manifest, verify_files=True)
    refs = verify_persistent_dense_scene_files(scene, manifest)
    restored_ext = restore_persistent_dense_geometry(manifest, representation="extended")
    restored_core = restore_persistent_dense_geometry(manifest, representation="core")
    if restored_ext.point_count != len(points) or restored_core.point_count != len(core_points):
        raise ReconstructionValidationError("I2.7 persistent F/G counts disagree.")
    if not np.array_equal(np.asarray(restored_ext.points_reconstruction, dtype=np.float64), points):
        raise ReconstructionValidationError("I2.7 persistent Extended geometry does not exactly match promoted F.")
    if not np.array_equal(np.asarray(restored_core.points_reconstruction, dtype=np.float64), core_points):
        raise ReconstructionValidationError("I2.7 persistent Core geometry does not exactly match promoted F.")

    quality_policy = I2QualityPolicy()
    peripheral, far, sector, core_env = classify_peripheral_geometry(points, core_points, voxel_size=voxel, policy=quality_policy)
    ext_env = spatial_envelope(points, lower_quantile=quality_policy.robust_lower_quantile, upper_quantile=quality_policy.robust_upper_quantile)
    support = np.asarray(ext["support"], dtype=np.uint32)
    confidence = np.asarray(ext["confidence"], dtype=np.float32)
    dispersion = np.asarray(ext["dispersion"], dtype=np.float32)

    ext_keys = [tuple(map(int, row)) for row in _voxel_indices(points, voxel)]
    core_keys = {tuple(map(int, row)) for row in _voxel_indices(core_points, voxel)}
    if not core_keys.issubset(set(ext_keys)):
        raise ReconstructionValidationError("I2.7 promoted Core is not a voxel subset of Extended.")
    is_core = np.asarray([key in core_keys for key in ext_keys], dtype=bool)
    core_conf_p25 = float(np.quantile(np.asarray(core["confidence"], dtype=np.float32), 0.25))
    core_disp_p90 = float(np.quantile(np.asarray(core["dispersion"], dtype=np.float32), 0.90))
    weak = peripheral & ((support <= 2) | (confidence < core_conf_p25) | (dispersion > core_disp_p90))

    components = voxel_component_profile(points, voxel)
    largest_fraction = float(components["largest_component_fraction_of_extended"])
    peripheral_stats = region_statistics(peripheral, support=support, confidence=confidence, dispersion=dispersion, component_size=np.ones(len(points), dtype=np.int32))
    far_stats = region_statistics(far, support=support, confidence=confidence, dispersion=dispersion, component_size=np.ones(len(points), dtype=np.int32))
    weak_stats = region_statistics(weak, support=support, confidence=confidence, dispersion=dispersion, component_size=np.ones(len(points), dtype=np.int32))
    peripheral_stats["fraction_of_extended"] = float(np.mean(peripheral))
    far_stats["fraction_of_extended"] = float(np.mean(far))
    weak_stats["fraction_of_extended"] = float(np.mean(weak))
    weak_stats["fraction_of_peripheral"] = float(np.count_nonzero(weak) / max(np.count_nonzero(peripheral), 1))
    sectors = directional_sector_summary(sector, support=support, confidence=confidence, dispersion=dispersion)

    baseline_ext_env = baseline["spatial"]["extended_envelope"]
    baseline_ratio = float(baseline_ext_env["full_diagonal"]) / float(baseline_ext_env["robust_diagonal"])
    current_ratio = float(ext_env.full_diagonal / ext_env.robust_diagonal)
    ratio_improvement = float((baseline_ratio - current_ratio) / baseline_ratio)
    full_shrink = float(1.0 - ext_env.full_diagonal / float(baseline_ext_env["full_diagonal"]))
    robust_shrink = float(1.0 - ext_env.robust_diagonal / float(baseline_ext_env["robust_diagonal"]))
    baseline_far = int(baseline["spatial"]["far_peripheral_retained"]["count"])
    baseline_peripheral = int(baseline["spatial"]["peripheral_retained"]["count"])
    baseline_weak = int(baseline["spatial"]["weak_peripheral_retained"]["count"])
    far_reduction = _safe_reduction(baseline_far, int(np.count_nonzero(far)))

    source_supported = int(f_summary["classification"]["source_supported_voxels"])
    rejected_supported = int(f_summary["classification"]["rejected_supported_voxels"])
    expected_ext = int(f_summary["classification"]["extended_voxels"])
    expected_core = int(f_summary["classification"]["core_voxels"])
    if expected_ext != len(points) or expected_core != len(core_points) or expected_ext + rejected_supported != source_supported:
        raise ReconstructionValidationError("I2.7 formal supported-voxel accounting is not exact.")

    sfm = _resolve_sfm_summary(i1_dir, i1)
    input_keyframes = int(sfm.get("input_keyframes", i1["sfm"]["registered_view_count"]))
    registered = int(i1["sfm"]["registered_view_count"])
    candidates = len(i1["sfm"]["candidate_dense_orders"])
    aligned = int(i1["dense_qualification"]["alignment_accepted_views"])
    selected = int(i1["dense_qualification"]["selected_pass_path"]["view_count"])
    upstream_gates = automated_quality_gates(
        input_sfm_keyframes=input_keyframes,
        registered_views=registered,
        candidate_dense_views=candidates,
        alignment_accepted_views=aligned,
        selected_views=selected,
        source_supported=source_supported,
        core_points=expected_core,
        extended_points=expected_ext,
        rejected_supported=rejected_supported,
        largest_component_fraction=largest_fraction,
        exact_observation_accounting=bool(f_summary["observation_accounting"]["exact_match"]),
        persistence_self_contained=bool(i26["persistence"]["bundle_self_contained"]),
        authority_boundaries_preserved=(
            bool(scene.canonical_geometry_authoritative)
            and bool(scene.canonical_persistence_promoted)
            and not bool(scene.metric_scale_claimed)
            and not bool(scene.measurement_authoritative)
            and not bool(scene.surface_mesh_generated)
        ),
        policy=quality_policy,
    )
    upstream_pass = all(upstream_gates.values())

    promotion = i26["promotion"]
    comparison = promotion["qualified_counterfactual_comparison"]
    losses_ext = float(promotion["extended_loss_fraction"])
    losses_core = float(promotion["core_loss_fraction"])
    driver_reject = float(comparison["extremum_driver_rejection_fraction"])
    stubborn_reject = float(comparison["stubborn_extremum_rejection_fraction"])
    expected_rejections = int(promotion["policy"]["rejected_points"])
    actual_reason_count = int(f_summary["classification"]["artifact_reason_histogram"]["weak_manifold_attachment"])
    exact_cf = bool(promotion["exact_counterfactual_equivalence"]["exact_mask_match"])
    persistent_ok = (
        len(refs) == 10
        and bool(i26["persistence"]["all_references_relative"])
        and bool(i26["persistence"]["bundle_self_contained"])
        and bool(i26["persistence"]["relocation_restore_pass"])
    )
    authority_ok = (
        bool(scene.canonical_geometry_authoritative)
        and bool(scene.canonical_persistence_promoted)
        and not bool(scene.metric_scale_claimed)
        and not bool(scene.measurement_authoritative)
        and not bool(scene.certified_metrology)
        and not bool(scene.surface_mesh_generated)
    )
    closure_policy = I2ClosurePolicy()
    component_topology_gates = None
    component_topology_pass = None
    if i271 is not None:
        component_topology_gates = baseline_aware_component_topology_gates(
            i271,
            expected_scene_id=scene.scene_id,
            expected_scene_fingerprint=scene.scene_fingerprint_sha256,
            manual_lineage_review_pass=bool(a.confirm_i2_7_1_lineage_review_pass),
        )
        # Cross-check the promoted component profile against this fresh I2.7 run.
        promoted_profile = i271.get("component_comparison", {}).get("promoted", {})
        if int(promoted_profile.get("largest_component_voxels", -1)) != int(components["largest_component_voxels"]):
            raise ReconstructionValidationError("I2.7.2 promoted dominant-component size disagrees with fresh I2.7 recomputation.")
        if not np.isclose(
            float(promoted_profile.get("largest_component_fraction", -1.0)),
            largest_fraction,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ReconstructionValidationError("I2.7.2 promoted dominant-component fraction disagrees with fresh I2.7 recomputation.")
        component_topology_pass = all(component_topology_gates.values())

    gates = closure_quality_gates(
        policy=closure_policy,
        i2_6_pass=bool(i26["pass"]),
        exact_counterfactual_equivalence=exact_cf,
        weak_manifold_attachment_rejections=actual_reason_count,
        expected_manifold_rejections=expected_rejections,
        extended_loss_fraction=losses_ext,
        core_loss_fraction=losses_core,
        full_to_robust_improvement_fraction=ratio_improvement,
        full_diagonal_shrink_fraction=full_shrink,
        robust_diagonal_shrink_fraction=robust_shrink,
        far_peripheral_reduction_fraction=far_reduction,
        largest_component_fraction_of_extended=largest_fraction,
        driver_rejection_fraction=driver_reject,
        stubborn_rejection_fraction=stubborn_reject,
        upstream_quality_gates_pass=upstream_pass,
        persistent_integrity_pass=persistent_ok,
        authority_boundaries_preserved=authority_ok,
        baseline_component_topology_preserved=component_topology_pass,
    )
    automated_pass = all(gates.values())

    diagnostics = out / "i2_7_spatial_diagnostics.npz"
    np.savez_compressed(
        diagnostics,
        points_reconstruction=points,
        colors_rgb=colors,
        view_observation_counts=support,
        filter_confidence=confidence,
        p90_dispersion_voxels=dispersion,
        is_core=is_core.astype(np.uint8),
        is_peripheral=peripheral.astype(np.uint8),
        is_far_peripheral=far.astype(np.uint8),
        is_weak_peripheral=weak.astype(np.uint8),
        directional_sector_code=sector,
        voxel_size_reconstruction_units=np.asarray(voxel, dtype=np.float64),
        schema_version=np.asarray(I2_CLOSURE_SCHEMA_VERSION, dtype=np.int32),
    )

    summary = {
        "phase": "I2.7",
        "schema_version": I2_CLOSURE_SCHEMA_VERSION,
        "status": "ready_for_final_manual_review" if automated_pass else "degraded",
        "automated_pass": bool(automated_pass),
        "scene": {
            "scene_id": scene.scene_id,
            "fingerprint": scene.scene_fingerprint_sha256,
            "manifest": str(manifest),
            "registered_views": registered,
            "input_keyframes": input_keyframes,
            "dense_alignment_accepted_views": aligned,
            "candidate_dense_views": candidates,
            "selected_qualified_views": selected,
            "voxel_size_reconstruction_units": voxel,
            "sfm_scale_status": "arbitrary",
        },
        "geometry": {
            "source_supported_voxels": source_supported,
            "extended_points": expected_ext,
            "core_points": expected_core,
            "rejected_supported_voxels": rejected_supported,
            "extended_loss_fraction_vs_i2_baseline": losses_ext,
            "core_loss_fraction_vs_i2_baseline": losses_core,
            "weak_manifold_attachment_rejections": actual_reason_count,
            "component_profile": components,
        },
        "spatial": {
            "baseline_full_to_robust_ratio": baseline_ratio,
            "promoted_full_to_robust_ratio": current_ratio,
            "full_to_robust_improvement_fraction": ratio_improvement,
            "full_diagonal_shrink_fraction": full_shrink,
            "robust_diagonal_shrink_fraction": robust_shrink,
            "core_anchor_envelope": core_env.as_dict(),
            "extended_envelope": ext_env.as_dict(),
            "baseline_peripheral_count": baseline_peripheral,
            "promoted_peripheral": peripheral_stats,
            "peripheral_reduction_fraction": _safe_reduction(baseline_peripheral, int(np.count_nonzero(peripheral))),
            "baseline_far_peripheral_count": baseline_far,
            "promoted_far_peripheral": far_stats,
            "far_peripheral_reduction_fraction": far_reduction,
            "baseline_weak_peripheral_count": baseline_weak,
            "promoted_weak_peripheral": weak_stats,
            "weak_peripheral_reduction_fraction": _safe_reduction(baseline_weak, int(np.count_nonzero(weak))),
            "directional_sectors": sectors,
        },
        "promotion_evidence": {
            "scenario": str(promotion["scenario"]),
            "exact_counterfactual_equivalence": exact_cf,
            "extremum_driver_rejection_fraction": driver_reject,
            "stubborn_driver_rejection_fraction": stubborn_reject,
            "manual_i2_5_3_review": str(promotion["manual_visual_review"]),
        },
        "persistence": {
            "file_reference_count": len(refs),
            "all_references_relative": bool(i26["persistence"]["all_references_relative"]),
            "bundle_self_contained": bool(i26["persistence"]["bundle_self_contained"]),
            "relocation_restore_pass": bool(i26["persistence"]["relocation_restore_pass"]),
        },
        "upstream_quality_gates": upstream_gates,
        "component_gate": {
            "mode": "baseline_aware_lineage" if component_topology_gates is not None else "legacy_absolute_fraction",
            "legacy_absolute_fraction_threshold": closure_policy.minimum_largest_component_fraction_of_extended,
            "legacy_absolute_fraction_is_diagnostic_only": bool(component_topology_gates is not None),
            "baseline_aware_gates": component_topology_gates,
            "baseline_aware_pass": component_topology_pass,
        },
        "closure_policy": closure_policy.as_dict(),
        "automated_closure_gates": gates,
        "authority": {
            "canonical_geometry_authoritative": True,
            "canonical_persistence_promoted": True,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "sfm_rerun": False,
            "da3_rerun": False,
            "fusion_rerun": False,
        },
        "manual_review": {
            "fixed_view_visual_review_required": True,
            "reconstruction_ui_sanity_required": True,
            "fixed_view_visual_review": "pending",
            "reconstruction_ui_sanity": "pending",
        },
        "decision": {
            "authorizes_final_manual_review": bool(automated_pass),
            "authorizes_i2_closure": False,
            "authorizes_i3_second_scene_campaign": False,
            "authorizes_surface_mesh_reconstruction": False,
            "next_action": "Run fixed-view previews and promoted-scene Reconstruction UI sanity, then finalize I2 closure." if automated_pass else "Investigate failed automated I2.7 gate(s).",
        },
        "artifacts": {"spatial_diagnostics_npz": str(diagnostics.resolve())},
    }
    summary_path = _write_json(out / "i2_7_summary.json", summary)

    print("I2.7 final full-video requalification: " + summary["status"].upper())
    print(f"scene_id={scene.scene_id} fingerprint={scene.scene_fingerprint_sha256}")
    print(f"views registered={registered}/{input_keyframes} aligned={aligned}/{candidates} selected={selected}")
    print(f"supported={source_supported} extended={expected_ext} core={expected_core} rejected={rejected_supported} manifold_rejections={actual_reason_count}")
    print(f"largest_component_fraction={largest_fraction:.3f} components={components['component_count']}")
    if component_topology_gates is not None:
        comparison = i271["component_comparison"]
        print(
            "component_gate=baseline_aware_lineage "
            f"baseline_fraction={comparison['baseline']['largest_component_fraction']:.6f} "
            f"promoted_fraction={comparison['promoted']['largest_component_fraction']:.6f} "
            f"dominant={comparison['baseline']['largest_component_voxels']}->{comparison['promoted']['largest_component_voxels']} "
            f"newly_detached={comparison['newly_detached_voxels']} topology_pass={component_topology_pass}"
        )
    print(f"peripheral={int(np.count_nonzero(peripheral))} baseline={baseline_peripheral} far={int(np.count_nonzero(far))} baseline_far={baseline_far} weak={int(np.count_nonzero(weak))} baseline_weak={baseline_weak}")
    print(f"full/robust={baseline_ratio:.3f}->{current_ratio:.3f} improvement={ratio_improvement:.3f} full_shrink={full_shrink:.3f} robust_shrink={robust_shrink:.3f} far_reduction={far_reduction:.3f}")
    print(f"driver_reject={driver_reject:.3f} stubborn_reject={stubborn_reject:.3f} exact_counterfactual={exact_cf}")
    print(f"persistence references={len(refs)} relative={i26['persistence']['all_references_relative']} self_contained={i26['persistence']['bundle_self_contained']} relocation_restore={i26['persistence']['relocation_restore_pass']}")
    print(f"automated_gates_pass={automated_pass}")
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print("final_fixed_view_review_required=True reconstruction_ui_sanity_required=True i2_closure_authorized=False")
    print(f"summary={summary_path}")
    return 0 if automated_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
