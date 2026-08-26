#!/usr/bin/env python3
"""I2.5.2 local-geometry / manifold diagnosis of stubborn extremum survivors."""
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
from core.reconstruction.offline_dense_extremum_local_geometry import (  # noqa: E402
    I252_LOCAL_GEOMETRY_SCHEMA_VERSION,
    I252LocalGeometryPolicy,
    derive_local_geometry_reference,
    diagnostic_signal,
    local_geometry_region_statistics,
    local_geometry_signatures,
    local_shape_metrics,
    strong_manifold_metrics,
    supporting_ray_alignment,
)
from core.reconstruction.offline_dense_peripheral_diagnosis import validate_observation_matrix  # noqa: E402
from scripts.run_i2_1_targeted_peripheral_diagnosis import _membership_matrix  # noqa: E402
from scripts.run_i2_4_support_diversity_parallax_diagnosis import _camera_centers_from_selected_clouds  # noqa: E402
from scripts.run_i2_5_1_spatial_extremum_survivor_diagnosis import _load_f_extended  # noqa: E402


_DEFAULT_I1 = Path("/tmp/eia_i1_offline_dense_campaign_recovered")
_DEFAULT_I2 = Path("/tmp/eia_i2_full_video_quality")
_DEFAULT_I251 = Path("/tmp/eia_i2_5_1_spatial_extremum_survivor_diagnosis")
_DEFAULT_OUT = Path("/tmp/eia_i2_5_2_extremum_local_geometry_manifold_diagnosis")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1-dir", type=Path, default=_DEFAULT_I1)
    parser.add_argument("--i2-dir", type=Path, default=_DEFAULT_I2)
    parser.add_argument("--i2-5-1-dir", type=Path, default=_DEFAULT_I251)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--knn-size", type=int, default=16)
    parser.add_argument("--manifold-radius-voxels", type=float, default=3.0)
    parser.add_argument("--reference-quantile", type=float, default=0.90)
    parser.add_argument("--minimum-reference-points", type=int, default=128)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read required I2.5.2 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I2.5.2 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_i251(summary: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(str(summary["artifacts"]["extremum_survivor_npz"])).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"I2.5.2 I2.5.1 diagnostics are unavailable: {path}")
    with np.load(path, allow_pickle=False) as payload:
        keys = {
            "points": "points_reconstruction",
            "colors": "colors_rgb",
            "drivers": "is_extremum_driver",
            "stubborn": "is_stubborn_extremum_survivor",
            "support": "baseline_support",
            "confidence": "filter_confidence",
            "dispersion": "p90_dispersion_voxels",
            "high": "high_support_neighbor_count",
            "radius2": "radius2_neighbor_count",
            "core": "is_core",
            "peripheral": "is_peripheral",
            "far": "is_far_peripheral",
        }
        data: dict[str, np.ndarray] = {}
        for name, key in keys.items():
            if key not in payload:
                raise ReconstructionValidationError(f"I2.5.2 I2.5.1 diagnostics are missing {key}.")
            data[name] = np.asarray(payload[key])
        data["robust_lower"] = np.asarray(payload["robust_lower"], dtype=np.float64)
        data["robust_upper"] = np.asarray(payload["robust_upper"], dtype=np.float64)
    data["points"] = np.asarray(data["points"], dtype=np.float64)
    data["colors"] = np.asarray(data["colors"], dtype=np.uint8)
    for name in ("drivers", "stubborn", "core", "peripheral", "far"):
        data[name] = np.asarray(data[name], dtype=bool)
    for name in ("support", "high", "radius2"):
        data[name] = np.asarray(data[name], dtype=np.int64)
    data["confidence"] = np.asarray(data["confidence"], dtype=np.float64)
    data["dispersion"] = np.asarray(data["dispersion"], dtype=np.float64)
    return data


def _median(stats: Mapping[str, Any], name: str) -> float | None:
    value = stats.get(name + "_quantiles", {}).get("p50")
    return None if value is None else float(value)


def main() -> int:
    args = _args()
    i1_dir = args.i1_dir.expanduser().resolve()
    i2_dir = args.i2_dir.expanduser().resolve()
    i251_dir = args.i2_5_1_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I2.5.2 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i1 = _read_json(i1_dir / "i1_campaign_summary.json")
    i2 = _read_json(i2_dir / "i2_summary.json")
    i251 = _read_json(i251_dir / "i2_5_1_summary.json")
    e3 = _read_json(i1_dir / "04_qualified_fusion" / "e3_summary.json")
    if str(i1.get("status")) != "pass" or not bool(i1.get("pipeline_execution_complete")):
        raise ReconstructionValidationError("I2.5.2 requires completed passing I1 evidence.")
    if str(i2.get("phase")) != "I2" or not bool(i2.get("automated_pass")):
        raise ReconstructionValidationError("I2.5.2 requires passing automated I2 evidence.")
    if str(i251.get("status")) != "diagnosis_ready":
        raise ReconstructionValidationError("I2.5.2 requires completed I2.5.1 extremum diagnosis.")
    if not bool(i251.get("extremum_findings", {}).get("all_i2_5_policies_miss_some_extremum_drivers")):
        raise ReconstructionValidationError("I2.5.2 is intended only when stubborn extremum survivors exist.")
    scene_id = str(i2["scene"]["scene_id"])
    if str(i251.get("scene_id")) != scene_id:
        raise ReconstructionValidationError("I2.5.2 I2/I2.5.1 scene identity mismatch.")

    data = _load_i251(i251)
    f, _ = _load_f_extended(i1_dir)
    points = np.asarray(data["points"], dtype=np.float64)
    if points.shape != np.asarray(f["points"]).shape or not np.array_equal(points, np.asarray(f["points"], dtype=np.float64)):
        raise ReconstructionValidationError("I2.5.2 I2.5.1 and F Extended point order disagree.")
    voxel_size = float(f["voxel_size"])
    support = np.asarray(data["support"], dtype=np.int64)
    high = np.asarray(data["high"], dtype=np.int64)
    core = np.asarray(data["core"], dtype=bool)
    peripheral = np.asarray(data["peripheral"], dtype=bool)
    far = np.asarray(data["far"], dtype=bool)
    drivers = np.asarray(data["drivers"], dtype=bool)
    stubborn = np.asarray(data["stubborn"], dtype=bool)

    policy = I252LocalGeometryPolicy(
        knn_size=int(args.knn_size),
        manifold_radius_voxels=float(args.manifold_radius_voxels),
        reference_quantile=float(args.reference_quantile),
        minimum_reference_points=int(args.minimum_reference_points),
    )

    # Strong manifold anchors deliberately use existing F/Core evidence only;
    # no I2 peripheral/far label participates in anchor selection.
    strong_anchor = (
        core
        & (support >= int(policy.strong_anchor_min_support))
        & (high >= int(policy.strong_anchor_min_high_neighbors))
        & ~drivers
    )
    reference = core & ~peripheral & ~drivers
    if int(np.count_nonzero(strong_anchor)) < 8:
        raise ReconstructionValidationError("I2.5.2 strong manifold anchor population is unexpectedly small.")

    shape = local_shape_metrics(points, voxel_size=voxel_size, knn_size=int(policy.knn_size))
    manifold = strong_manifold_metrics(
        points,
        strong_anchor_mask=strong_anchor,
        voxel_size=voxel_size,
        radius_voxels=float(policy.manifold_radius_voxels),
    )

    orders, matrix, _ = _membership_matrix(e3_summary=e3, points=points, voxel_size=voxel_size)
    matrix, exact_support = validate_observation_matrix(matrix, support, minimum_support_views=2)
    if not np.array_equal(exact_support, support):
        raise ReconstructionValidationError("I2.5.2 exact source-view membership does not reproduce F support counts.")
    centers = _camera_centers_from_selected_clouds(e3, orders)
    ray_alignment = supporting_ray_alignment(
        points,
        principal_axis=np.asarray(shape["principal_axis"], dtype=np.float64),
        observation_matrix=matrix,
        camera_centers=centers,
    )

    metrics = {
        **{name: value for name, value in shape.items() if name not in ("principal_axis", "knn_indices")},
        **manifold,
        "supporting_ray_alignment": ray_alignment,
    }
    thresholds = derive_local_geometry_reference(metrics, reference_mask=reference, policy=policy)
    signatures = local_geometry_signatures(metrics, thresholds)

    masks = {
        "central_core_reference": reference,
        "strong_manifold_anchors": strong_anchor,
        "extremum_drivers": drivers,
        "stubborn_extremum_survivors": stubborn,
        "stubborn_far_extremum_survivors": stubborn & far,
        "ordinary_peripheral_non_drivers": peripheral & ~drivers,
        "extended": np.ones(len(points), dtype=bool),
    }
    regions = {
        name: local_geometry_region_statistics(mask, metrics=metrics, signatures=signatures)
        for name, mask in masks.items()
    }
    signal = diagnostic_signal(
        reference_stats=regions["central_core_reference"],
        stubborn_stats=regions["stubborn_extremum_survivors"],
        driver_stats=regions["extremum_drivers"],
        policy=policy,
    )

    diagnostics_npz = output / "i2_5_2_local_geometry_manifold_diagnostics.npz"
    np.savez_compressed(
        diagnostics_npz,
        points_reconstruction=points,
        colors_rgb=np.asarray(data["colors"], dtype=np.uint8),
        is_extremum_driver=drivers.astype(np.uint8),
        is_stubborn_extremum_survivor=stubborn.astype(np.uint8),
        is_far_peripheral=far.astype(np.uint8),
        is_core=core.astype(np.uint8),
        is_strong_manifold_anchor=strong_anchor.astype(np.uint8),
        is_reference_central_core=reference.astype(np.uint8),
        is_manifold_separated=np.asarray(signatures["manifold_separated"], dtype=np.uint8),
        is_locally_sparse=np.asarray(signatures["locally_sparse"], dtype=np.uint8),
        is_ray_like=np.asarray(signatures["ray_like"], dtype=np.uint8),
        has_combined_extremum_signature=np.asarray(signatures["combined_extremum_signature"], dtype=np.uint8),
        nearest_strong_anchor_distance_voxels=np.asarray(metrics["nearest_strong_anchor_distance_voxels"], dtype=np.float32),
        strong_anchor_neighbors_within_radius=np.asarray(metrics["strong_anchor_neighbors_within_radius"], dtype=np.int16),
        knn_median_distance_voxels=np.asarray(metrics["knn_median_distance_voxels"], dtype=np.float32),
        knn_p90_distance_voxels=np.asarray(metrics["knn_p90_distance_voxels"], dtype=np.float32),
        local_linearity=np.asarray(metrics["local_linearity"], dtype=np.float32),
        local_planarity=np.asarray(metrics["local_planarity"], dtype=np.float32),
        local_scattering=np.asarray(metrics["local_scattering"], dtype=np.float32),
        supporting_ray_alignment=np.asarray(metrics["supporting_ray_alignment"], dtype=np.float32),
        principal_axis=np.asarray(shape["principal_axis"], dtype=np.float32),
        selected_workspace_orders=np.asarray(orders, dtype=np.int32),
        camera_centers_reconstruction=np.asarray(centers, dtype=np.float64),
        baseline_support=support.astype(np.int16),
        filter_confidence=np.asarray(data["confidence"], dtype=np.float32),
        p90_dispersion_voxels=np.asarray(data["dispersion"], dtype=np.float32),
        robust_lower=np.asarray(data["robust_lower"], dtype=np.float64),
        robust_upper=np.asarray(data["robust_upper"], dtype=np.float64),
        voxel_size_reconstruction_units=np.asarray(voxel_size, dtype=np.float64),
        schema_version=np.asarray(I252_LOCAL_GEOMETRY_SCHEMA_VERSION, dtype=np.int32),
    )

    summary = {
        "phase": "I2.5.2",
        "schema_version": I252_LOCAL_GEOMETRY_SCHEMA_VERSION,
        "status": "diagnosis_ready",
        "scene_id": scene_id,
        "source_i1_dir": str(i1_dir),
        "source_i2_dir": str(i2_dir),
        "source_i2_5_1_dir": str(i251_dir),
        "baseline": {
            "extended_points": int(len(points)),
            "core_points": int(np.count_nonzero(core)),
            "extremum_driver_count": int(np.count_nonzero(drivers)),
            "stubborn_extremum_survivor_count": int(np.count_nonzero(stubborn)),
            "voxel_size_reconstruction_units": voxel_size,
        },
        "policy": policy.as_dict(),
        "strong_manifold": {
            "definition": "Core AND support>=3 AND high-support-neighbors>=2 AND not an I2.5.1 extremum driver",
            "anchor_count": int(np.count_nonzero(strong_anchor)),
            "radius_voxels": float(policy.manifold_radius_voxels),
        },
        "reference": {
            "definition": "central Core non-extremum-driver geometry; used only to derive diagnostic local-geometry quantiles",
            "count": int(np.count_nonzero(reference)),
            "thresholds": thresholds,
        },
        "regions": regions,
        "diagnostic_signal": signal,
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
                "If stubborn extremum survivors show a strong local-manifold/ray-like enrichment, evaluate a narrow "
                "local-geometry-aware counterfactual before any F promotion. Otherwise continue diagnosis without "
                "using scene-radius or whole-view deletion heuristics."
            ),
        },
        "artifacts": {"local_geometry_manifold_npz": str(diagnostics_npz.resolve())},
    }
    path = _write_json(output / "i2_5_2_summary.json", summary)

    ref_stats = regions["central_core_reference"]
    driver_stats = regions["extremum_drivers"]
    stubborn_stats = regions["stubborn_extremum_survivors"]
    print("I2.5.2 extremum local-geometry / manifold diagnosis: DIAGNOSIS_READY")
    print(
        f"views={len(orders)} extended={len(points)} drivers={int(np.count_nonzero(drivers))} "
        f"stubborn={int(np.count_nonzero(stubborn))} anchors={int(np.count_nonzero(strong_anchor))}"
    )
    print(f"exact_support_accounting=True observations={int(matrix.sum(dtype=np.int64))} mismatched_voxels=0")
    print(
        "reference=central_core_non_driver "
        f"count={int(np.count_nonzero(reference))} "
        f"anchor_dist_q90={float(thresholds['anchor_distance_high']):.3f}vox "
        f"knn_q90={float(thresholds['knn_spacing_high']):.3f}vox "
        f"linearity_q90={float(thresholds['linearity_high']):.3f} "
        f"ray_align_q90={float(thresholds['ray_alignment_high']):.3f}"
    )
    for name, stats in (("central_core", ref_stats), ("drivers", driver_stats), ("stubborn", stubborn_stats)):
        print(
            f"{name}: signature={float(stats['combined_extremum_signature_fraction']):.3f} "
            f"manifold_sep={float(stats['manifold_separated_fraction']):.3f} "
            f"sparse={float(stats['locally_sparse_fraction']):.3f} "
            f"ray_like={float(stats['ray_like_fraction']):.3f} "
            f"anchor_dist_p50={_median(stats,'nearest_strong_anchor_distance_voxels'):.3f} "
            f"knn_p50={_median(stats,'knn_median_distance_voxels'):.3f} "
            f"linearity_p50={_median(stats,'local_linearity'):.3f} "
            f"ray_align_p50={_median(stats,'supporting_ray_alignment'):.3f}"
        )
    print(
        f"diagnostic_signal={signal['strength']} hypothesis_supported={signal['supports_local_geometry_manifold_hypothesis']} "
        f"driver_enrichment={signal['driver_enrichment_vs_reference']} "
        f"stubborn_enrichment={signal['stubborn_enrichment_vs_reference']}"
    )
    print("geometry_changed=False filter_policy_changed=False sfm_rerun=False da3_rerun=False fusion_rerun=False")
    print("i2_closure_authorized=False production_filter_authorized=False mesh_authorization=False")
    print(f"summary={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
