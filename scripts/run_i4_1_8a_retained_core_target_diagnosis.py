#!/usr/bin/env python3
"""I4.1.8A retained-Core object-target diagnosis using saved production evidence.

No PyCOLMAP, DA3, dense-pair, fusion, filtering, persistence, or production
binding stage is rerun.  The script reads a completed DEGRADED I4.1.8 scene,
its retained Core NPZ, and qualified I4.1.7B evidence, then writes diagnosis-
only JSON/NPZ artifacts.
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
from core.reconstruction.offline_dense_i4_retained_core_target_diagnosis import (  # noqa: E402
    I4_1_8A_SCHEMA_VERSION,
    build_i4_1_8a_retained_core_target_diagnosis,
)

_DEFAULT_I4_1_7B = Path("/tmp/depth3d_i4_1_7b_box_bounded_target_01")
_DEFAULT_I4_1_8 = Path("/tmp/depth3d_i4_1_8_box_production_binding_01")
_DEFAULT_OUTPUT = Path("/tmp/depth3d_i4_1_8a_retained_core_target_diagnosis")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-1-7b-dir", type=Path, default=_DEFAULT_I4_1_7B)
    p.add_argument("--i4-1-8-dir", type=Path, default=_DEFAULT_I4_1_8)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.8A cannot read JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.1.8A JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_i418_diagnostics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            target = np.asarray(payload["retained_core_target_reconstruction"], dtype=np.float64)
            poses = np.asarray(payload["production_selected_camera_matrices"], dtype=np.float64)
            sources = np.asarray(payload["production_selected_source_sequence_indices"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.8A cannot read I4.1.8 orbit diagnostics: {path}") from exc
    return target, poses, sources


def _load_core(path: Path) -> tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            voxel_size = float(np.asarray(payload["voxel_size_reconstruction_units"], dtype=np.float64).reshape(()))
            support = (
                np.asarray(payload["view_observation_counts"], dtype=np.float64)
                if "view_observation_counts" in payload.files
                else None
            )
            confidence = (
                np.asarray(payload["filter_confidence"], dtype=np.float64)
                if "filter_confidence" in payload.files
                else None
            )
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.8A cannot read retained Core NPZ: {path}") from exc
    return points, voxel_size, support, confidence


def main() -> int:
    args = _args()
    b_dir = args.i4_1_7b_dir.expanduser().resolve()
    i418_dir = args.i4_1_8_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.8A output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i418_summary_path = i418_dir / "i4_1_8_summary.json"
    b_summary_path = b_dir / "i4_1_7b_summary.json"
    i418 = _read_json(i418_summary_path)
    b_summary = _read_json(b_summary_path)
    if not isinstance(i418.get("artifacts"), Mapping):
        raise ReconstructionValidationError("I4.1.8A I4.1.8 artifact map is missing.")
    artifacts = i418["artifacts"]
    diag_path = Path(str(artifacts.get("final_orbit_diagnostics_npz", ""))).expanduser().resolve()
    f_summary_path = Path(str(artifacts.get("f_summary", ""))).expanduser().resolve()
    f_summary = _read_json(f_summary_path)
    f_artifacts = f_summary.get("artifacts")
    if not isinstance(f_artifacts, Mapping):
        raise ReconstructionValidationError("I4.1.8A F artifact map is missing.")
    core_path = Path(str(f_artifacts.get("core_npz", ""))).expanduser().resolve()

    current_target, poses, sources = _load_i418_diagnostics(diag_path)
    core_points, voxel_size, support, confidence = _load_core(core_path)
    result = build_i4_1_8a_retained_core_target_diagnosis(
        source_i4_1_8_summary=i418,
        source_i4_1_7b_summary=b_summary,
        production_selected_camera_matrices=poses,
        retained_core_points=core_points,
        retained_core_voxel_size=voxel_size,
        current_retained_core_target=current_target,
        core_support_counts=support,
        core_filter_confidence=confidence,
    )
    result["source_artifacts"] = {
        "i4_1_7b_dir": str(b_dir),
        "i4_1_7b_summary": str(b_summary_path),
        "i4_1_8_dir": str(i418_dir),
        "i4_1_8_summary": str(i418_summary_path),
        "i4_1_8_final_orbit_diagnostics_npz": str(diag_path),
        "f_summary": str(f_summary_path),
        "retained_core_npz": str(core_path),
    }

    targets = result["targets"]
    plane = result["retained_core"]["dominant_plane"]
    diag_npz = output / "i4_1_8a_target_diagnostics.npz"
    npz_payload: dict[str, np.ndarray] = {
        "production_selected_camera_matrices": np.asarray(poses, dtype=np.float64),
        "production_selected_source_sequence_indices": np.asarray(sources, dtype=np.int64),
        "dominant_plane_normal_reconstruction": np.asarray(plane["normal_reconstruction"], dtype=np.float64),
        "dominant_plane_offset_reconstruction": np.asarray(plane["offset_reconstruction"], dtype=np.float64),
        "schema_version": np.asarray(I4_1_8A_SCHEMA_VERSION, dtype=np.int32),
    }
    for name, record in targets.items():
        npz_payload[f"target__{name}"] = np.asarray(record["target_reconstruction"], dtype=np.float64)
    np.savez_compressed(diag_npz, **npz_payload)
    result["artifacts"] = {"diagnostic_npz": str(diag_npz.resolve())}
    summary_path = _write_json(output / "i4_1_8a_summary.json", result)

    diagnosis = result["diagnosis"]
    current = targets["current_whole_core_coordinate_median"]
    circle = targets["production_bound_trajectory_circle"]
    attention = targets.get("retained_core_camera_attention_candidate")
    print("I4.1.8A Retained-Core Object-Target Diagnosis: DIAGNOSIS_READY", flush=True)
    print(
        f"classification={diagnosis['classification']} "
        f"recommends_i4_1_8b={diagnosis['recommends_i4_1_8b_bounded_target_correction_design']}",
        flush=True,
    )
    print(
        f"dominant_plane fraction={plane['core_inlier_fraction']:.3f} strong={plane['strong_dominant_plane']} "
        f"threshold={plane['inlier_threshold_voxels']:.2f}vox",
        flush=True,
    )
    current_metric = current["orbit_evaluation"]["current_metric"]
    circle_metric = circle["orbit_evaluation"]["current_metric"]
    print(
        f"current_core_median azimuth={current_metric['azimuth_coverage_degrees']:.1f}deg "
        f"bins={current_metric['occupied_azimuth_bins']}/12 opposed={current_metric['maximum_opposed_view_angle_degrees']:.1f}deg "
        f"plane_distance={current['structure']['dominant_plane_distance_voxels']:.2f}vox",
        flush=True,
    )
    print(
        f"production_trajectory_circle azimuth={circle_metric['azimuth_coverage_degrees']:.1f}deg "
        f"bins={circle_metric['occupied_azimuth_bins']}/12 opposed={circle_metric['maximum_opposed_view_angle_degrees']:.1f}deg "
        f"plane_distance={circle['structure']['dominant_plane_distance_voxels']:.2f}vox "
        f"robust={circle['orbit_evaluation']['robust_locked_orbit_pass']}",
        flush=True,
    )
    if attention is not None:
        am = attention["orbit_evaluation"]["current_metric"]
        print(
            f"retained_core_attention_candidate azimuth={am['azimuth_coverage_degrees']:.1f}deg "
            f"bins={am['occupied_azimuth_bins']}/12 opposed={am['maximum_opposed_view_angle_degrees']:.1f}deg "
            f"plane_distance={attention['structure']['dominant_plane_distance_voxels']:.2f}vox "
            f"robust={attention['orbit_evaluation']['robust_locked_orbit_pass']}",
            flush=True,
        )
    probe = result["retained_core"]["coordinate_median_invariance_probe"]
    print(
        f"coordinate_median_rigid_probe max_drift={probe['maximum_drift_voxels']:.2f}vox "
        f"axis_sensitive={diagnosis['evidence_flags']['coordinatewise_core_median_is_axis_sensitive']}",
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
