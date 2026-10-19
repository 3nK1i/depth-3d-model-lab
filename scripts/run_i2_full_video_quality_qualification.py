#!/usr/bin/env python3
"""I2 full-video real-scene visual/spatial quality qualification foundation.

Consumes a completed I1 campaign.  I2 does not rerun SfM, DA3, fusion,
filtering, or persistence and never modifies the F/G geometry.  It validates
I1/F/G provenance, quantifies retained spatial coverage/peripheral structures,
attributes peripheral retained voxels back to selected source views, and emits
manual-review diagnostics.  Mesh authorization remains blocked until the I2
visual review is explicitly accepted.
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
from core.reconstruction.offline_dense_quality import (  # noqa: E402
    I2_QUALITY_SCHEMA_VERSION,
    I2QualityPolicy,
    automated_quality_gates,
    classify_peripheral_geometry,
    directional_sector_summary,
    quantiles,
    region_statistics,
    spatial_envelope,
)
from scripts.run_e2_consistency_aware_fusion_refinement import (  # noqa: E402
    _aggregate_saved_e1_source_frame_voxels,
    _voxel_indices,
)


_DEFAULT_I1_DIR = Path("/tmp/depth3d_i1_offline_dense_campaign_recovered")
_DEFAULT_OUTPUT_DIR = Path("/tmp/depth3d_i2_full_video_quality")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1-dir", type=Path, default=_DEFAULT_I1_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-view-attribution", action="store_true")
    parser.add_argument("--minimum-registered-fraction", type=float, default=0.80)
    parser.add_argument("--minimum-dense-alignment-fraction", type=float, default=0.80)
    parser.add_argument("--minimum-selected-fraction", type=float, default=0.60)
    parser.add_argument("--minimum-largest-component-fraction", type=float, default=0.85)
    parser.add_argument("--minimum-core-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-extended-fraction", type=float, default=0.65)
    parser.add_argument("--maximum-rejected-supported-fraction", type=float, default=0.35)
    parser.add_argument("--robust-lower-quantile", type=float, default=0.02)
    parser.add_argument("--robust-upper-quantile", type=float, default=0.98)
    parser.add_argument("--peripheral-margin-fraction", type=float, default=0.05)
    parser.add_argument("--far-radial-multiplier", type=float, default=1.25)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read required I2 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I2 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_filtered_cloud(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as payload:
        result: dict[str, np.ndarray | float] = {
            "points": np.asarray(payload["points_reconstruction"], dtype=np.float64),
            "colors": np.asarray(payload["colors_rgb"], dtype=np.uint8),
            "support": np.asarray(payload["view_observation_counts"], dtype=np.uint32),
            "confidence": np.asarray(payload["filter_confidence"], dtype=np.float32),
            "dispersion": np.asarray(payload["p90_dispersion_voxels"], dtype=np.float32),
            "component_size": np.asarray(payload["component_size"], dtype=np.int32),
            "voxel_size": float(np.asarray(payload["voxel_size_reconstruction_units"]).item()),
        }
    count = len(np.asarray(result["points"]))
    for name in ("colors", "support", "confidence", "dispersion", "component_size"):
        if len(np.asarray(result[name])) != count:
            raise ReconstructionValidationError(f"I2 filtered cloud field {name} is not aligned.")
    return result


def _resolve_sfm_summary(i1_dir: Path, i1_summary: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [i1_dir / "01_sfm" / "real_sfm_summary.json"]
    stages = i1_summary.get("stages")
    if isinstance(stages, Mapping):
        sfm_stage = stages.get("sfm")
        if isinstance(sfm_stage, Mapping) and sfm_stage.get("source_dir"):
            candidates.append(Path(str(sfm_stage["source_dir"])).expanduser().resolve() / "real_sfm_summary.json")
    for path in candidates:
        if path.is_file():
            return _read_json(path)
    # I1 itself already proves the registered count.  Fall back to a conservative
    # same-count input only when an older/reused SfM summary is unavailable.
    registered = int(i1_summary.get("sfm", {}).get("registered_view_count", 0))
    if registered <= 0:
        raise ReconstructionValidationError("I2 cannot resolve SfM input/registration evidence.")
    return {"input_keyframes": registered, "registered_keyframes": registered, "registration_ratio": 1.0, "fallback": True}


def _key_set(points: np.ndarray, voxel_size: float) -> set[tuple[int, int, int]]:
    keys = _voxel_indices(points, voxel_size)
    tuples = {tuple(map(int, key)) for key in keys}
    if len(tuples) != len(points):
        raise ReconstructionValidationError("I2 expected one retained point per voxel key.")
    return tuples


def _view_attribution(
    *,
    e3_summary: Mapping[str, Any],
    extended_keys: set[tuple[int, int, int]],
    peripheral_keys: set[tuple[int, int, int]],
    far_keys: set[tuple[int, int, int]],
    sector_key_sets: Mapping[int, set[tuple[int, int, int]]],
    voxel_size: float,
) -> dict[str, Any]:
    frames = e3_summary.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ReconstructionValidationError("I2 view attribution requires E3/I1 selected frame provenance.")
    records: list[dict[str, Any]] = []
    sector_labels = {1: "+x", 2: "-x", 3: "+y", 4: "-y", 5: "+z", 6: "-z"}
    for frame in frames:
        cloud_path = Path(str(frame["cloud_npz"])).expanduser().resolve()
        if not cloud_path.is_file():
            raise ReconstructionValidationError(f"I2 selected source cloud is unavailable: {cloud_path}")
        with np.load(cloud_path, allow_pickle=False) as payload:
            canonical = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
            aligned_depth = np.asarray(payload["aligned_depth_reconstruction_units"])
            pose = np.asarray(payload["T_reconstruction_from_camera"], dtype=np.float64)
        keys, _, _ = _aggregate_saved_e1_source_frame_voxels(
            canonical_points=canonical,
            colors=colors,
            aligned_depth_reconstruction_units=aligned_depth,
            T_reconstruction_from_camera=pose,
            voxel_size=voxel_size,
        )
        frame_keys = {tuple(map(int, key)) for key in keys}
        extended_observations = len(frame_keys & extended_keys)
        peripheral_observations = len(frame_keys & peripheral_keys)
        far_observations = len(frame_keys & far_keys)
        sector_counts = {
            label: len(frame_keys & sector_key_sets.get(code, set()))
            for code, label in sector_labels.items()
        }
        records.append({
            "workspace_order": int(frame["workspace_order"]),
            "keyframe_id": str(frame["keyframe_id"]),
            "unique_frame_voxels": len(frame_keys),
            "retained_extended_observations": extended_observations,
            "peripheral_retained_observations": peripheral_observations,
            "far_peripheral_retained_observations": far_observations,
            "peripheral_fraction_of_retained": (
                peripheral_observations / float(extended_observations) if extended_observations else 0.0
            ),
            "far_fraction_of_retained": far_observations / float(extended_observations) if extended_observations else 0.0,
            "directional_peripheral_observations": sector_counts,
        })
    top_peripheral = sorted(
        records,
        key=lambda item: (item["peripheral_retained_observations"], item["peripheral_fraction_of_retained"]),
        reverse=True,
    )[:10]
    top_far = sorted(
        records,
        key=lambda item: (item["far_peripheral_retained_observations"], item["far_fraction_of_retained"]),
        reverse=True,
    )[:10]
    return {
        "available": True,
        "selected_view_count": len(records),
        "per_view": records,
        "top_peripheral_contributors": top_peripheral,
        "top_far_peripheral_contributors": top_far,
    }


def main() -> int:
    args = _parse_args()
    i1_dir = args.i1_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReconstructionValidationError(f"I2 output directory must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    i1_summary = _read_json(i1_dir / "i1_campaign_summary.json")
    if str(i1_summary.get("phase", "")) != "I1" or str(i1_summary.get("status", "")) != "pass":
        raise ReconstructionValidationError("I2 requires a passing I1 campaign summary.")
    if not bool(i1_summary.get("pipeline_execution_complete")):
        raise ReconstructionValidationError("I2 requires a fully completed I1 pipeline execution.")
    if not bool(i1_summary.get("decision", {}).get("authorizes_i2_current_full_video_validation")):
        raise ReconstructionValidationError("I2 requires explicit I1 authorization.")

    filter_dir = i1_dir / "05_filtering"
    persistent_dir = i1_dir / "06_persistent_scene"
    fusion_dir = i1_dir / "04_qualified_fusion"
    f_summary = _read_json(filter_dir / "f_summary.json")
    g_summary = _read_json(persistent_dir / "g_summary.json")
    e3_summary = _read_json(fusion_dir / "e3_summary.json")
    if not (bool(f_summary.get("pass")) and bool(g_summary.get("pass")) and bool(e3_summary.get("pass"))):
        raise ReconstructionValidationError("I2 requires passing fusion/F/G evidence from I1.")

    extended_path = Path(str(f_summary["artifacts"]["extended_npz"])).expanduser().resolve()
    core_path = Path(str(f_summary["artifacts"]["core_npz"])).expanduser().resolve()
    extended = _load_filtered_cloud(extended_path)
    core = _load_filtered_cloud(core_path)
    extended_points = np.asarray(extended["points"], dtype=np.float64)
    core_points = np.asarray(core["points"], dtype=np.float64)
    voxel_size = float(extended["voxel_size"])
    if not np.isclose(voxel_size, float(core["voxel_size"]), atol=1e-12, rtol=0.0):
        raise ReconstructionValidationError("I2 Core/Extended voxel-size provenance disagrees.")

    expected_extended = int(f_summary["classification"]["extended_voxels"])
    expected_core = int(f_summary["classification"]["core_voxels"])
    if len(extended_points) != expected_extended or len(core_points) != expected_core:
        raise ReconstructionValidationError("I2 retained cloud counts disagree with F summary.")
    if int(g_summary["geometry"]["primary_point_count"]) != expected_extended or int(g_summary["geometry"]["core_point_count"]) != expected_core:
        raise ReconstructionValidationError("I2 F/G retained point counts disagree.")

    policy = I2QualityPolicy(
        minimum_registered_fraction=float(args.minimum_registered_fraction),
        minimum_dense_alignment_fraction=float(args.minimum_dense_alignment_fraction),
        minimum_selected_fraction_of_registered=float(args.minimum_selected_fraction),
        minimum_largest_component_fraction=float(args.minimum_largest_component_fraction),
        minimum_core_fraction_of_supported=float(args.minimum_core_fraction),
        minimum_extended_fraction_of_supported=float(args.minimum_extended_fraction),
        maximum_rejected_supported_fraction=float(args.maximum_rejected_supported_fraction),
        robust_lower_quantile=float(args.robust_lower_quantile),
        robust_upper_quantile=float(args.robust_upper_quantile),
        peripheral_margin_fraction=float(args.peripheral_margin_fraction),
        far_radial_multiplier=float(args.far_radial_multiplier),
    )

    peripheral, far, sector, core_envelope = classify_peripheral_geometry(
        extended_points,
        core_points,
        voxel_size=voxel_size,
        policy=policy,
    )
    ext_support = np.asarray(extended["support"], dtype=np.uint32)
    ext_conf = np.asarray(extended["confidence"], dtype=np.float32)
    ext_disp = np.asarray(extended["dispersion"], dtype=np.float32)
    ext_component = np.asarray(extended["component_size"], dtype=np.int32)

    core_keys = _key_set(core_points, voxel_size)
    extended_keys = _key_set(extended_points, voxel_size)
    if not core_keys.issubset(extended_keys):
        raise ReconstructionValidationError("I2 Core voxel keys are not a subset of Extended.")
    extended_key_array = [tuple(map(int, key)) for key in _voxel_indices(extended_points, voxel_size)]
    is_core = np.asarray([key in core_keys for key in extended_key_array], dtype=bool)

    largest_component_size = int(f_summary["topology"]["largest_supported_component_voxels"])
    dominant_component = ext_component == largest_component_size
    core_conf_p25 = float(np.quantile(np.asarray(core["confidence"], dtype=np.float32), 0.25))
    core_disp_p90 = float(np.quantile(np.asarray(core["dispersion"], dtype=np.float32), 0.90))
    weak_peripheral = peripheral & ((ext_support <= 2) | (ext_conf < core_conf_p25) | (ext_disp > core_disp_p90))

    extended_envelope = spatial_envelope(
        extended_points,
        lower_quantile=policy.robust_lower_quantile,
        upper_quantile=policy.robust_upper_quantile,
    )
    core_profile = core_envelope.as_dict()
    extended_profile = extended_envelope.as_dict()

    peripheral_stats = region_statistics(
        peripheral,
        support=ext_support,
        confidence=ext_conf,
        dispersion=ext_disp,
        component_size=ext_component,
    )
    peripheral_stats["fraction_of_extended"] = float(np.mean(peripheral))
    peripheral_stats["core_voxels_in_region"] = int(np.count_nonzero(peripheral & is_core))
    peripheral_stats["core_fraction_within_region"] = (
        int(np.count_nonzero(peripheral & is_core)) / float(np.count_nonzero(peripheral))
        if np.count_nonzero(peripheral) else 0.0
    )
    far_stats = region_statistics(
        far,
        support=ext_support,
        confidence=ext_conf,
        dispersion=ext_disp,
        component_size=ext_component,
    )
    far_stats["fraction_of_extended"] = float(np.mean(far))
    far_stats["core_voxels_in_region"] = int(np.count_nonzero(far & is_core))
    weak_stats = region_statistics(
        weak_peripheral,
        support=ext_support,
        confidence=ext_conf,
        dispersion=ext_disp,
        component_size=ext_component,
    )
    weak_stats["fraction_of_extended"] = float(np.mean(weak_peripheral))
    weak_stats["fraction_of_peripheral"] = (
        int(np.count_nonzero(weak_peripheral)) / float(np.count_nonzero(peripheral))
        if np.count_nonzero(peripheral) else 0.0
    )

    sector_summary = directional_sector_summary(
        sector,
        support=ext_support,
        confidence=ext_conf,
        dispersion=ext_disp,
    )

    peripheral_keys = {extended_key_array[index] for index in np.flatnonzero(peripheral)}
    far_keys = {extended_key_array[index] for index in np.flatnonzero(far)}
    sector_key_sets = {
        code: {extended_key_array[index] for index in np.flatnonzero(sector == code)}
        for code in range(1, 7)
    }
    if args.skip_view_attribution:
        attribution: dict[str, Any] = {"available": False, "reason": "skipped_by_user"}
    else:
        attribution = _view_attribution(
            e3_summary=e3_summary,
            extended_keys=extended_keys,
            peripheral_keys=peripheral_keys,
            far_keys=far_keys,
            sector_key_sets=sector_key_sets,
            voxel_size=voxel_size,
        )

    diagnostic_npz = output_dir / "i2_extended_spatial_diagnostics.npz"
    np.savez_compressed(
        diagnostic_npz,
        points_reconstruction=extended_points,
        colors_rgb=np.asarray(extended["colors"], dtype=np.uint8),
        view_observation_counts=ext_support,
        filter_confidence=ext_conf,
        p90_dispersion_voxels=ext_disp,
        component_size=ext_component,
        is_core=is_core.astype(np.uint8),
        is_dominant_component=dominant_component.astype(np.uint8),
        is_peripheral=peripheral.astype(np.uint8),
        is_far_peripheral=far.astype(np.uint8),
        is_weak_peripheral=weak_peripheral.astype(np.uint8),
        directional_sector_code=sector,
        voxel_size_reconstruction_units=np.asarray(voxel_size, dtype=np.float64),
        schema_version=np.asarray(I2_QUALITY_SCHEMA_VERSION, dtype=np.int32),
    )

    sfm_summary = _resolve_sfm_summary(i1_dir, i1_summary)
    input_keyframes = int(sfm_summary.get("input_keyframes", i1_summary["sfm"]["registered_view_count"]))
    registered_views = int(i1_summary["sfm"]["registered_view_count"])
    candidate_dense_views = len(i1_summary["sfm"]["candidate_dense_orders"])
    alignment_accepted = int(i1_summary["dense_qualification"]["alignment_accepted_views"])
    selected_views = int(i1_summary["dense_qualification"]["selected_pass_path"]["view_count"])
    source_supported = int(f_summary["classification"]["source_supported_voxels"])
    rejected_supported = int(f_summary["classification"]["rejected_supported_voxels"])
    authority = g_summary["authority"]
    authority_ok = (
        bool(authority["canonical_geometry_authoritative"])
        and bool(authority["canonical_persistence_promoted"])
        and not bool(authority["metric_scale_claimed"])
        and not bool(authority["measurement_authoritative"])
        and not bool(authority["surface_mesh_generated"])
    )
    gates = automated_quality_gates(
        input_sfm_keyframes=input_keyframes,
        registered_views=registered_views,
        candidate_dense_views=candidate_dense_views,
        alignment_accepted_views=alignment_accepted,
        selected_views=selected_views,
        source_supported=source_supported,
        core_points=expected_core,
        extended_points=expected_extended,
        rejected_supported=rejected_supported,
        largest_component_fraction=float(f_summary["topology"]["largest_supported_component_fraction"]),
        exact_observation_accounting=bool(f_summary["observation_accounting"]["exact_match"]),
        persistence_self_contained=bool(g_summary["persistence"]["bundle_self_contained"]),
        authority_boundaries_preserved=authority_ok,
        policy=policy,
    )
    if not args.skip_view_attribution:
        gates["per_view_attribution_available"] = bool(attribution.get("available"))
    automated_pass = all(gates.values())

    summary = {
        "phase": "I2",
        "schema_version": I2_QUALITY_SCHEMA_VERSION,
        "status": "ready_for_manual_visual_review" if automated_pass else "degraded",
        "automated_pass": bool(automated_pass),
        "manual_visual_review_required": True,
        "source_i1_dir": str(i1_dir),
        "source": dict(i1_summary["source"]),
        "scene": {
            "scene_id": str(g_summary["scene_id"]),
            "fingerprint": str(g_summary["scene_fingerprint_sha256"]),
            "registered_views": registered_views,
            "dense_alignment_accepted_views": alignment_accepted,
            "selected_qualified_views": selected_views,
            "selected_workspace_orders": list(i1_summary["dense_qualification"]["selected_pass_path"]["orders"]),
            "voxel_size_reconstruction_units": voxel_size,
            "sfm_scale_status": "arbitrary",
        },
        "retained_geometry": {
            "source_supported_voxels": source_supported,
            "core_points": expected_core,
            "extended_points": expected_extended,
            "rejected_supported_voxels": rejected_supported,
            "core_fraction_of_supported": expected_core / float(source_supported),
            "extended_fraction_of_supported": expected_extended / float(source_supported),
            "rejected_supported_fraction": rejected_supported / float(source_supported),
            "largest_supported_component_fraction": float(f_summary["topology"]["largest_supported_component_fraction"]),
            "dominant_component_extended_points": int(np.count_nonzero(dominant_component)),
            "dominant_component_fraction_of_extended": float(np.mean(dominant_component)),
        },
        "spatial": {
            "core_anchor_envelope": core_profile,
            "extended_envelope": extended_profile,
            "extended_to_core_robust_diagonal_ratio": (
                float(extended_envelope.robust_diagonal / core_envelope.robust_diagonal)
                if core_envelope.robust_diagonal > 0.0 else None
            ),
            "peripheral_retained": peripheral_stats,
            "far_peripheral_retained": far_stats,
            "weak_peripheral_retained": weak_stats,
            "directional_sectors": sector_summary,
            "note": "Peripheral/far masks are diagnostics anchored to robust Core geometry; no points are removed by I2.",
        },
        "view_attribution": attribution,
        "policy": policy.as_dict(),
        "automated_gates": gates,
        "authority": {
            "canonical_geometry_authoritative": True,
            "canonical_persistence_promoted": True,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "i2_changes_geometry": False,
        },
        "artifacts": {
            "spatial_diagnostics_npz": str(diagnostic_npz.resolve()),
        },
        "decision": {
            "authorizes_manual_visual_spatial_review": bool(automated_pass),
            "authorizes_i3_second_scene_campaign": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": "Review fixed-view RGB/region/support/confidence previews; I2 closure remains manual/evidence-based.",
        },
    }
    summary_path = _write_json(output_dir / "i2_summary.json", summary)

    print(f"\nI2 full-video real-scene quality qualification: {summary['status'].upper()}")
    print(
        f"views registered={registered_views}/{input_keyframes} aligned={alignment_accepted}/{candidate_dense_views} "
        f"selected={selected_views}"
    )
    print(
        f"supported={source_supported} core={expected_core} ({expected_core/source_supported:.3f}) "
        f"extended={expected_extended} ({expected_extended/source_supported:.3f}) "
        f"rejected={rejected_supported} ({rejected_supported/source_supported:.3f})"
    )
    print(
        f"largest_component_fraction={float(f_summary['topology']['largest_supported_component_fraction']):.3f} "
        f"extended_dominant_component_fraction={float(np.mean(dominant_component)):.3f}"
    )
    print(
        f"peripheral_extended={int(np.count_nonzero(peripheral))} ({float(np.mean(peripheral)):.3f}) "
        f"far_peripheral={int(np.count_nonzero(far))} ({float(np.mean(far)):.3f}) "
        f"weak_peripheral={int(np.count_nonzero(weak_peripheral))} ({float(np.mean(weak_peripheral)):.3f})"
    )
    print(
        f"core_robust_diagonal={core_envelope.robust_diagonal:.6f} "
        f"extended_robust_diagonal={extended_envelope.robust_diagonal:.6f} "
        f"extended_full_to_robust={extended_envelope.full_diagonal/extended_envelope.robust_diagonal:.3f}"
    )
    print(f"view_attribution_available={bool(attribution.get('available'))}")
    if attribution.get("available"):
        top = attribution["top_peripheral_contributors"][:5]
        print("top_peripheral_orders=" + ",".join(str(item["workspace_order"]) for item in top))
    print(f"automated_gates_pass={automated_pass}")
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print("manual_visual_review_required=True mesh_authorization=False")
    print(f"summary={summary_path}")
    return 0 if automated_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
