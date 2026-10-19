#!/usr/bin/env python3
"""I4.1.2 SfM-invariant orbit geometry + temporal sampling diagnosis.

Reuses a completed DEGRADED I4.1 campaign.  It is intentionally able to inspect
both the ordinary-baseline-quality stop and the earlier I1-baseline stop.  No
SfM/DA3/fusion/filter/persistence stage is rerun.
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
from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import (  # noqa: E402
    I4_1_2_SCHEMA_VERSION,
    build_i4_1_2_diagnosis,
)
from core.reconstruction.offline_dense_i4_orbit_qualification import camera_poses_from_sfm_report  # noqa: E402

_DEFAULT_I4 = Path("/tmp/depth3d_i4_object_orbit_campaign")
_DEFAULT_OUTPUT = Path("/tmp/depth3d_i4_1_2_orbit_geometry_sampling_diagnosis")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-dir", type=Path, default=_DEFAULT_I4)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--expected-max-sampled-frames", type=int, default=40)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.2 cannot read JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.1.2 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _workspace_order_to_keyframe(sfm_report: Mapping[str, Any]) -> dict[int, str]:
    sfm = sfm_report.get("sfm_result")
    workspace = sfm.get("workspace") if isinstance(sfm, Mapping) else None
    images = workspace.get("images") if isinstance(workspace, Mapping) else None
    if not isinstance(images, list):
        raise ReconstructionValidationError("I4.1.2 SfM workspace images are missing.")
    result: dict[int, str] = {}
    for item in images:
        if not isinstance(item, Mapping):
            raise ReconstructionValidationError("I4.1.2 workspace image record must be an object.")
        order = int(item.get("order", -1))
        keyframe = str(item.get("keyframe_id", "")).strip()
        if order < 0 or not keyframe or order in result:
            raise ReconstructionValidationError("I4.1.2 workspace order/keyframe mapping is invalid.")
        result[order] = keyframe
    return result


def _sparse_landmarks(sfm_report: Mapping[str, Any]) -> np.ndarray:
    sfm = sfm_report.get("sfm_result")
    landmarks = sfm.get("landmarks") if isinstance(sfm, Mapping) else None
    if not isinstance(landmarks, list) or len(landmarks) < 64:
        raise ReconstructionValidationError("I4.1.2 SfM report requires at least 64 sparse landmarks.")
    points = np.asarray([item.get("position_reconstruction") for item in landmarks], dtype=np.float64)
    if points.shape != (len(landmarks), 3) or not np.isfinite(points).all():
        raise ReconstructionValidationError("I4.1.2 sparse landmark positions are invalid.")
    return points


def _aligned_ids(i1_dir: Path, mapping: Mapping[int, str]) -> tuple[str, ...]:
    dense_dir = i1_dir / "02_dense_views"
    if not dense_dir.is_dir():
        return ()
    result: list[str] = []
    for order, keyframe in sorted(mapping.items()):
        cloud = dense_dir / f"candidate_order_{order:02d}" / "canonical_cloud.npz"
        if cloud.is_file():
            result.append(keyframe)
    return tuple(result)


def _selected_ids(i1_dir: Path, i1_summary: Mapping[str, Any], mapping: Mapping[int, str]) -> tuple[str, ...]:
    e3_path = i1_dir / "04_qualified_fusion" / "e3_summary.json"
    if e3_path.is_file():
        e3 = _read_json(e3_path)
        values = tuple(str(v) for v in e3.get("selected_keyframe_ids", []) if str(v))
        if values:
            return values
    selected = i1_summary.get("selected_path")
    orders = selected.get("orders") if isinstance(selected, Mapping) else None
    if not isinstance(orders, list):
        return ()
    missing = [int(order) for order in orders if int(order) not in mapping]
    if missing:
        raise ReconstructionValidationError(f"I4.1.2 selected path contains unmapped workspace orders: {missing!r}")
    return tuple(mapping[int(order)] for order in orders)


def _core_points_if_available(i1_dir: Path) -> np.ndarray | None:
    path = i1_dir / "05_filtering" / "f_filtered_extended.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        if "points_reconstruction" not in payload or "class_code" not in payload:
            return None
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        classes = np.asarray(payload["class_code"], dtype=np.uint8).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or classes.shape != (len(points),):
        raise ReconstructionValidationError("I4.1.2 persisted Extended/Core geometry arrays are misaligned.")
    core = points[classes == 3]
    return core if len(core) >= 64 else None


def main() -> int:
    args = _args()
    i4_dir = args.i4_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.2 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i4 = _read_json(i4_dir / "i4_1_summary.json")
    i1_dir = i4_dir / "01_i1_baseline"
    i1 = _read_json(i1_dir / "i1_campaign_summary.json")
    sfm_report = _read_json(i1_dir / "01_sfm" / "real_sfm_report.json")
    camera_ids, camera_matrices = camera_poses_from_sfm_report(sfm_report)
    landmarks = _sparse_landmarks(sfm_report)
    order_to_keyframe = _workspace_order_to_keyframe(sfm_report)
    aligned_ids = _aligned_ids(i1_dir, order_to_keyframe)
    selected_ids = _selected_ids(i1_dir, i1, order_to_keyframe)
    core_points = _core_points_if_available(i1_dir)

    diagnosis = build_i4_1_2_diagnosis(
        source_i4_summary=i4,
        sfm_report=sfm_report,
        registered_camera_ids=camera_ids,
        registered_camera_matrices=camera_matrices,
        sparse_landmarks=landmarks,
        aligned_keyframe_ids=aligned_ids,
        selected_keyframe_ids=selected_ids,
        core_points=core_points,
        expected_max_sampled_frames=int(args.expected_max_sampled_frames),
    )

    targets = diagnosis["targets"]
    artifact_npz = output / "i4_1_2_orbit_geometry_sampling_diagnostics.npz"
    arrays: dict[str, Any] = {
        "registered_camera_ids": np.asarray(camera_ids, dtype="U160"),
        "registered_camera_matrices": np.asarray(camera_matrices, dtype=np.float64),
        "sparse_landmarks_reconstruction": np.asarray(landmarks, dtype=np.float64),
        "aligned_camera_ids": np.asarray(aligned_ids, dtype="U160"),
        "selected_camera_ids": np.asarray(selected_ids, dtype="U160"),
        "schema_version": np.asarray(I4_1_2_SCHEMA_VERSION, dtype=np.int32),
    }
    for name, record in targets.items():
        arrays[f"target_{name}"] = np.asarray(record["target_reconstruction"], dtype=np.float64)
    np.savez_compressed(artifact_npz, **arrays)

    diagnosis["source"] = {
        "i4_dir": str(i4_dir),
        "i1_dir": str(i1_dir),
        "sfm_report": str((i1_dir / "01_sfm" / "real_sfm_report.json").resolve()),
    }
    diagnosis["artifacts"] = {"diagnostics_npz": str(artifact_npz.resolve())}
    summary = _write_json(output / "i4_1_2_summary.json", diagnosis)

    sampling = diagnosis["temporal_sampling"]
    counts = diagnosis["counts"]
    diag = diagnosis["diagnosis"]
    print("I4.1.2 SfM-Invariant Orbit Geometry & Temporal Sampling Diagnosis: DIAGNOSIS_READY")
    print(
        f"views registered={counts['registered_cameras']} aligned={counts['aligned_cameras']} "
        f"selected={counts['selected_cameras']} sparse_landmarks={counts['sparse_landmarks']}"
    )
    duration = sampling["estimated_source_duration_seconds"]
    temporal = sampling["sampled_temporal_span_fraction"]
    last_frac = sampling["sampled_last_source_index_fraction"]
    print(
        f"source_duration={duration if duration is not None else 'unknown'}s "
        f"sampled_span={sampling['sampled_timestamp_span_seconds']:.3f}s "
        f"temporal_fraction={temporal if temporal is not None else 'unknown'} "
        f"last_source_fraction={last_frac if last_frac is not None else 'unknown'}"
    )
    print(
        f"sample_rate={sampling['inferred_sample_rate_hz']:.3f}Hz cap_reached={sampling['cap_reached']} "
        f"prefix_max_frame_truncation_likely={sampling['prefix_max_frame_truncation_likely']}"
    )
    print(
        f"sampling_classification={diag['sampling_classification']} "
        f"sampling_policy_correction_candidate={diag['sampling_policy_correction_candidate']} authorized=False"
    )
    for set_name, evaluation in diagnosis["camera_set_evaluations"].items():
        values = []
        for target_name, target_eval in evaluation["targets"].items():
            values.append(f"{target_name}:{target_eval['coverage_min_degrees']:.1f}-{target_eval['coverage_max_degrees']:.1f}")
        print(
            f"{set_name}_orbit invariant_coverage_deg=" + ",".join(values) +
            f" trajectory_centered={evaluation['trajectory_centered_pca']['azimuth_coverage_degrees']:.1f}"
        )
    print(
        f"orbit_geometry_classification={diag['orbit_geometry_classification']} "
        f"best_registered_target={diag['best_registered_target']} "
        f"best_registered_coverage={diag['best_registered_invariant_coverage_degrees']:.1f}deg"
    )
    print(
        f"current_i4_metric_reproduction_exact={diagnosis['current_i4_metric_reproduction_exact']} "
        f"orbit_metric_correction_candidate={diag['orbit_metric_correction_candidate']} authorized=False"
    )
    print("sfm_rerun=False da3_rerun=False fusion_rerun=False sampling_policy_changed=False orbit_metric_policy_changed=False")
    print("threshold_relaxation=False metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print("i4_2_detail_preservation_authorized=False mesh_authorization=False")
    print(f"next={diagnosis['decision']['next_action']}")
    print(f"summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
