#!/usr/bin/env python3
"""I4.1.8B isolate object-support Core from saved production evidence.

This step does not rerun reconstruction.  It reconstructs exact per-Core-voxel
view attribution from the already-saved 40 bound source clouds, verifies that
attribution against F's saved support counts, and performs target-independent
retained-Core object-support isolation.
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
from core.reconstruction.offline_dense_i4_retained_core_object_support import (  # noqa: E402
    I4_1_8B_SCHEMA_VERSION,
    build_i4_1_8b_retained_core_object_support_isolation,
)
from scripts.run_e2_consistency_aware_fusion_refinement import (  # noqa: E402
    _aggregate_saved_e1_source_frame_voxels,
    _voxel_indices,
)

_DEFAULT_I4_1_8 = Path("/tmp/depth3d_i4_1_8_box_production_binding_01")
_DEFAULT_I4_1_8A = Path("/tmp/depth3d_i4_1_8a_box_target_diag_01")
_DEFAULT_OUTPUT = Path("/tmp/depth3d_i4_1_8b_retained_core_object_support")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-1-8-dir", type=Path, default=_DEFAULT_I4_1_8)
    p.add_argument("--i4-1-8a-dir", type=Path, default=_DEFAULT_I4_1_8A)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.8B cannot read JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.1.8B JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_orbit_diagnostics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as p:
            poses = np.asarray(p["production_selected_camera_matrices"], dtype=np.float64)
            sources = np.asarray(p["production_selected_source_sequence_indices"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.8B cannot read production orbit diagnostics: {path}") from exc
    return poses, sources


def _load_core(path: Path) -> dict[str, np.ndarray | float]:
    try:
        with np.load(path, allow_pickle=False) as p:
            result: dict[str, np.ndarray | float] = {
                "points": np.asarray(p["points_reconstruction"], dtype=np.float64),
                "colors": np.asarray(p["colors_rgb"], dtype=np.uint8),
                "support": np.asarray(p["view_observation_counts"], dtype=np.uint32),
                "confidence": np.asarray(p["filter_confidence"], dtype=np.float32),
                "voxel_size": float(np.asarray(p["voxel_size_reconstruction_units"], dtype=np.float64).reshape(())),
                "orders": np.asarray(p["selected_workspace_orders"], dtype=np.int32),
            }
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.8B cannot read retained Core NPZ: {path}") from exc
    return result


def _source_frame_voxel_keys(path: Path, voxel_size: float) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as p:
            canonical = np.asarray(p["points_reconstruction"], dtype=np.float64)
            colors = np.asarray(p["colors_rgb"], dtype=np.uint8)
            depth = np.asarray(p["aligned_depth_reconstruction_units"])
            pose = np.asarray(p["T_reconstruction_from_camera"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.8B cannot read bound source cloud: {path}") from exc
    keys, _, _ = _aggregate_saved_e1_source_frame_voxels(
        canonical_points=canonical,
        colors=colors,
        aligned_depth_reconstruction_units=depth,
        T_reconstruction_from_camera=pose,
        voxel_size=float(voxel_size),
    )
    return np.asarray(keys, dtype=np.int64)


def _exact_core_view_attribution(
    *,
    core_points: np.ndarray,
    voxel_size: float,
    frames: list[Mapping[str, Any]],
    expected_sources: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    core_keys = _voxel_indices(core_points, voxel_size)
    if len(np.unique(core_keys, axis=0)) != len(core_keys):
        raise ReconstructionValidationError("I4.1.8B retained Core must contain exactly one point per voxel key.")
    key_to_core = {tuple(map(int, key)): index for index, key in enumerate(core_keys)}
    matrix = np.zeros((len(frames), len(core_points)), dtype=bool)
    records: list[dict[str, Any]] = []
    frame_sources = np.asarray([int(frame["source_sequence_index"]) for frame in frames], dtype=np.int64)
    if not np.array_equal(frame_sources, expected_sources):
        raise ReconstructionValidationError(
            "I4.1.8B E3 frame/source order disagrees with final production orbit diagnostics."
        )
    for rank, frame in enumerate(frames):
        cloud_path = Path(str(frame.get("cloud_npz", ""))).expanduser().resolve()
        if not cloud_path.is_file():
            raise ReconstructionValidationError(f"I4.1.8B bound source cloud is unavailable: {cloud_path}")
        frame_keys = _source_frame_voxel_keys(cloud_path, voxel_size)
        matched: list[int] = []
        for key in frame_keys:
            core_index = key_to_core.get(tuple(map(int, key)))
            if core_index is not None:
                matched.append(int(core_index))
        if matched:
            matrix[rank, np.asarray(matched, dtype=np.int64)] = True
        records.append({
            "production_rank": int(rank),
            "workspace_order": int(frame.get("workspace_order", rank)),
            "source_sequence_index": int(frame_sources[rank]),
            "unique_frame_voxels": int(len(frame_keys)),
            "retained_core_observations": int(len(matched)),
            "cloud_npz": str(cloud_path),
        })
        print(
            f"I4.1.8B attribution {rank+1}/{len(frames)} source={frame_sources[rank]} "
            f"frame_voxels={len(frame_keys)} core_matches={len(matched)}",
            flush=True,
        )
    return matrix, records


def main() -> int:
    args = _args()
    i418_dir = args.i4_1_8_dir.expanduser().resolve()
    i418a_dir = args.i4_1_8a_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.8B output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i418_summary_path = i418_dir / "i4_1_8_summary.json"
    i418a_summary_path = i418a_dir / "i4_1_8a_summary.json"
    i418 = _read_json(i418_summary_path)
    i418a = _read_json(i418a_summary_path)
    artifacts = i418.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ReconstructionValidationError("I4.1.8B I4.1.8 artifact map is missing.")
    diag_path = Path(str(artifacts.get("final_orbit_diagnostics_npz", ""))).expanduser().resolve()
    e3_path = Path(str(artifacts.get("e3_summary", ""))).expanduser().resolve()
    f_path = Path(str(artifacts.get("f_summary", ""))).expanduser().resolve()
    e3 = _read_json(e3_path)
    f_summary = _read_json(f_path)
    f_artifacts = f_summary.get("artifacts")
    if not isinstance(f_artifacts, Mapping):
        raise ReconstructionValidationError("I4.1.8B F artifact map is missing.")
    core_path = Path(str(f_artifacts.get("core_npz", ""))).expanduser().resolve()

    poses, sources = _load_orbit_diagnostics(diag_path)
    core = _load_core(core_path)
    points = np.asarray(core["points"], dtype=np.float64)
    colors = np.asarray(core["colors"], dtype=np.uint8)
    support = np.asarray(core["support"], dtype=np.uint32)
    confidence = np.asarray(core["confidence"], dtype=np.float32)
    voxel_size = float(core["voxel_size"])
    orders = np.asarray(core["orders"], dtype=np.int32)
    frames = e3.get("frames")
    if not isinstance(frames, list) or len(frames) != len(poses):
        raise ReconstructionValidationError("I4.1.8B E3 frame provenance is missing or misaligned.")
    if len(orders) != len(poses):
        raise ReconstructionValidationError("I4.1.8B retained Core selected-order provenance is misaligned.")

    matrix, attribution_records = _exact_core_view_attribution(
        core_points=points,
        voxel_size=voxel_size,
        frames=frames,
        expected_sources=sources,
    )
    reconstructed_support = matrix.sum(axis=0, dtype=np.uint32)
    if not np.array_equal(reconstructed_support, support):
        mismatch = int(np.count_nonzero(reconstructed_support != support))
        raise ReconstructionValidationError(
            f"I4.1.8B exact Core attribution does not reproduce F support counts ({mismatch} mismatches)."
        )
    print("I4.1.8B exact retained-Core view attribution: PASS", flush=True)

    result = build_i4_1_8b_retained_core_object_support_isolation(
        source_i4_1_8_summary=i418,
        source_i4_1_8a_summary=i418a,
        production_selected_camera_matrices=poses,
        production_selected_source_sequence_indices=sources,
        retained_core_points=points,
        retained_core_voxel_size=voxel_size,
        exact_view_observation_matrix=matrix,
        core_support_counts=support,
        core_filter_confidence=confidence,
    )
    selected_indices = np.asarray(result.pop("_selected_core_indices"), dtype=np.int64)
    result["source_artifacts"] = {
        "i4_1_8_dir": str(i418_dir),
        "i4_1_8_summary": str(i418_summary_path),
        "i4_1_8a_dir": str(i418a_dir),
        "i4_1_8a_summary": str(i418a_summary_path),
        "final_orbit_diagnostics_npz": str(diag_path),
        "e3_summary": str(e3_path),
        "f_summary": str(f_path),
        "retained_core_npz": str(core_path),
    }
    result["attribution_reconstruction"] = {
        "exact_support_count_reproduction": True,
        "per_view": attribution_records,
    }

    diagnostic_npz = output / "i4_1_8b_object_support_diagnostics.npz"
    np.savez_compressed(
        diagnostic_npz,
        production_selected_camera_matrices=np.asarray(poses, dtype=np.float64),
        production_selected_source_sequence_indices=np.asarray(sources, dtype=np.int64),
        isolated_core_indices=np.asarray(selected_indices, dtype=np.int64),
        isolated_core_points_reconstruction=np.asarray(points[selected_indices], dtype=np.float64),
        isolated_core_colors_rgb=np.asarray(colors[selected_indices], dtype=np.uint8),
        isolated_core_support_counts=np.asarray(support[selected_indices], dtype=np.uint32),
        isolated_core_filter_confidence=np.asarray(confidence[selected_indices], dtype=np.float32),
        isolated_core_view_observation_matrix=np.asarray(matrix[:, selected_indices], dtype=np.uint8),
        voxel_size_reconstruction_units=np.asarray(voxel_size, dtype=np.float64),
        schema_version=np.asarray(I4_1_8B_SCHEMA_VERSION, dtype=np.int32),
    )
    result["artifacts"] = {"object_support_diagnostics_npz": str(diagnostic_npz.resolve())}
    summary_path = _write_json(output / "i4_1_8b_summary.json", result)

    diagnosis = result["diagnosis"]
    selected = result.get("selected_support")
    print(
        "I4.1.8B Retained-Core Object Support Isolation: "
        + ("OBJECT_SUPPORT_ISOLATED" if diagnosis["object_support_isolation_ready"] else "DEGRADED"),
        flush=True,
    )
    print(
        f"classification={diagnosis['classification']} "
        f"recommends_i4_1_8c={diagnosis['recommends_i4_1_8c_final_target_binding_design']}",
        flush=True,
    )
    print(
        f"core={len(points)} eligible={result['counts']['attention_eligible_core_points']} "
        f"seeds={result['counts']['seed_core_points']} components={result['counts']['seed_components']} "
        f"isolated={result['counts']['isolated_support_points']}",
        flush=True,
    )
    if isinstance(selected, Mapping):
        print(
            f"isolated_support views={selected['active_view_count']}/{len(poses)} "
            f"source_span={selected['active_source_span_fraction']:.3f} quartiles={selected['active_view_quartiles']}/4 "
            f"core_fraction={selected['core_fraction']:.4f} uniqueness={selected['best_to_second_score_ratio']}",
            flush=True,
        )
        stability = selected["target_bootstrap_stability"]
        metric = selected["posthoc_orbit_diagnosis"]["current_metric"]
        print(
            f"isolated_target bootstrap_max_drift={stability['maximum_drift_voxels']:.2f}vox "
            f"posthoc_orbit={metric['azimuth_coverage_degrees']:.1f}deg bins={metric['occupied_azimuth_bins']}/12 "
            f"opposed={metric['maximum_opposed_view_angle_degrees']:.1f}deg "
            f"posthoc_used_for_selection=False",
            flush=True,
        )
    print(
        "sfm_rerun=False da3_rerun=False dense_pair_rerun=False fusion_rerun=False "
        "filtering_rerun=False persistence_rerun=False production_binding_rerun=False",
        flush=True,
    )
    print(
        "production_target_changed=False orbit_thresholds_changed=False "
        "dense_consistency_threshold_relaxation_authorized=False",
        flush=True,
    )
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
