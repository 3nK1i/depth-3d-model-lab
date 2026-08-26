#!/usr/bin/env python3
"""I4.1.4 decoupled global-coverage / local-density sampling diagnosis.

Reuses existing I4 Horse evidence only.  It parses the exact I1 pair-status log
and SfM camera/workspace metadata to measure dense-consistency pass rate versus
temporal spacing and camera-viewpoint separation.  Optional historical runs
allow apples-to-apples comparison with the same source before I4.1.3 and with
the earlier 30-second control segment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import optical_axis_consensus_target  # noqa: E402
from core.reconstruction.offline_dense_i4_sampling_density_diagnosis import (  # noqa: E402
    I4_1_4_SCHEMA_VERSION,
    build_i4_1_4_diagnosis,
    build_pair_observations,
    summarize_pair_observations,
)

_DEFAULT_CURRENT = Path("/tmp/eia_i4_object_orbit_campaign")
_DEFAULT_PREFIX = Path("/tmp/eia_i4_horse_pre_i4_1_3_degraded")
_DEFAULT_CONTROL = Path("/tmp/eia_i4_horse_30s_degraded")
_DEFAULT_OUTPUT = Path("/tmp/eia_i4_1_4_decoupled_sampling_density_diagnosis")
_EDGE_RE = re.compile(r"^\s*edge\s+(\d+)<->(\d+)\s+status=(pass|degraded|inconclusive)\s*$", re.IGNORECASE)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current-i4-dir", type=Path, default=_DEFAULT_CURRENT)
    p.add_argument("--same-source-prefix-dir", type=Path, default=_DEFAULT_PREFIX)
    p.add_argument("--short-control-dir", type=Path, default=_DEFAULT_CONTROL)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--allow-missing-historical", action="store_true")
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.4 cannot read JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I4.1.4 JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _edge_statuses(log_path: Path) -> dict[tuple[int, int], str]:
    if not log_path.is_file():
        raise ReconstructionValidationError(f"I4.1.4 I1 log is missing: {log_path}")
    result: dict[tuple[int, int], str] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EDGE_RE.match(line)
        if not match:
            continue
        a, b = sorted((int(match.group(1)), int(match.group(2))))
        key = (a, b)
        status = match.group(3).lower()
        if key in result and result[key] != status:
            raise ReconstructionValidationError(f"I4.1.4 conflicting pair status for {key}.")
        result[key] = status
    if not result:
        raise ReconstructionValidationError(f"I4.1.4 found no dense-consistency edge lines in {log_path}")
    return result


def _sfm_metadata(report: Mapping[str, Any]) -> tuple[dict[int, int], dict[int, int], dict[int, np.ndarray], np.ndarray]:
    sfm = report.get("sfm_result")
    workspace = sfm.get("workspace") if isinstance(sfm, Mapping) else None
    images = workspace.get("images") if isinstance(workspace, Mapping) else None
    cameras = sfm.get("cameras") if isinstance(sfm, Mapping) else None
    if not isinstance(images, list) or not isinstance(cameras, list):
        raise ReconstructionValidationError("I4.1.4 SfM workspace/cameras are missing.")
    order_to_ts: dict[int, int] = {}
    order_to_source: dict[int, int] = {}
    keyframe_to_order: dict[str, int] = {}
    for item in images:
        if not isinstance(item, Mapping):
            raise ReconstructionValidationError("I4.1.4 workspace image record must be an object.")
        order = int(item.get("order", -1)); kid = str(item.get("keyframe_id", ""))
        if order < 0 or not kid or order in order_to_ts:
            raise ReconstructionValidationError("I4.1.4 workspace image mapping is invalid.")
        order_to_ts[order] = int(item.get("timestamp_ns", 0))
        order_to_source[order] = int(item.get("source_sequence_index", order))
        keyframe_to_order[kid] = order
    order_to_matrix: dict[int, np.ndarray] = {}
    matrices: list[np.ndarray] = []
    for item in cameras:
        if not isinstance(item, Mapping):
            continue
        kid = str(item.get("keyframe_id", "")); order = keyframe_to_order.get(kid)
        if order is None:
            continue
        matrix = np.asarray(item.get("T_reconstruction_from_camera"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ReconstructionValidationError("I4.1.4 SfM camera matrix is invalid.")
        order_to_matrix[order] = matrix
        matrices.append(matrix)
    if len(order_to_matrix) < 3:
        raise ReconstructionValidationError("I4.1.4 requires at least three registered cameras.")
    return order_to_ts, order_to_source, order_to_matrix, np.stack(matrices, axis=0)


def _run_summary(i4_dir: Path, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    i4_dir = i4_dir.expanduser().resolve()
    i1_dir = i4_dir / "01_i1_baseline"
    report = _read_json(i1_dir / "01_sfm" / "real_sfm_report.json")
    statuses = _edge_statuses(i4_dir / "logs" / "01_i1_baseline.log")
    order_to_ts, order_to_source, order_to_matrix, registered_matrices = _sfm_metadata(report)
    target, target_diag = optical_axis_consensus_target(registered_matrices)
    observations = build_pair_observations(
        edge_statuses=statuses,
        order_to_timestamp_ns=order_to_ts,
        order_to_source_index=order_to_source,
        order_to_camera_matrix=order_to_matrix,
        target_reconstruction=target,
    )
    summary = summarize_pair_observations(observations)
    summary.update({
        "label": label,
        "i4_dir": str(i4_dir),
        "target_mode": "optical_axis_consensus",
        "target_diagnostics": target_diag,
        "workspace_sample_count": len(order_to_ts),
        "registered_camera_count": len(order_to_matrix),
        "source_index_first": min(order_to_source.values()),
        "source_index_last": max(order_to_source.values()),
        "sampled_timestamp_span_seconds": float((max(order_to_ts.values()) - min(order_to_ts.values())) / 1e9),
    })
    return summary, [item.as_dict() for item in observations]


def _optional_summary(path: Path, label: str, allow_missing: bool) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not path.expanduser().exists():
        if allow_missing:
            return None, []
        raise ReconstructionValidationError(
            f"I4.1.4 historical evidence is missing: {path}. Use --allow-missing-historical to run current-only diagnosis."
        )
    return _run_summary(path, label)


def main() -> int:
    args = _args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.4 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    current, current_rows = _run_summary(args.current_i4_dir, "current_full_span")
    prefix, prefix_rows = _optional_summary(args.same_source_prefix_dir, "same_source_dense_prefix", args.allow_missing_historical)
    control, control_rows = _optional_summary(args.short_control_dir, "short_control", args.allow_missing_historical)

    diagnosis = build_i4_1_4_diagnosis(
        current_summary=current,
        same_source_prefix_summary=prefix,
        short_control_summary=control,
    )
    diagnosis["sources"] = {
        "current_i4_dir": str(args.current_i4_dir.expanduser().resolve()),
        "same_source_prefix_dir": None if prefix is None else str(args.same_source_prefix_dir.expanduser().resolve()),
        "short_control_dir": None if control is None else str(args.short_control_dir.expanduser().resolve()),
    }
    artifact = output / "i4_1_4_pair_observations.json"
    _write_json(artifact, {
        "schema_version": I4_1_4_SCHEMA_VERSION,
        "current_full_span": current_rows,
        "same_source_dense_prefix": prefix_rows,
        "short_control": control_rows,
    })
    diagnosis["artifacts"] = {"pair_observations_json": str(artifact.resolve())}
    summary_path = _write_json(output / "i4_1_4_summary.json", diagnosis)

    d = diagnosis["diagnosis"]
    print("I4.1.4 Decoupled Global-Coverage / Local-Density Sampling Diagnosis: DIAGNOSIS_READY")
    for name, record in (("current", current), ("same_source_prefix", prefix), ("short_control", control)):
        if record is None:
            print(f"{name}: unavailable")
            continue
        print(
            f"{name}: edges={record['decisive_edge_count']} pass={record['pass_count']} "
            f"degraded={record['degraded_count']} pass_fraction={record['pass_fraction']:.3f} "
            f"sampled_span={record['sampled_timestamp_span_seconds']:.3f}s"
        )
        buckets = record["time_buckets"]
        print(
            f"  time_pass le0.75={buckets['le_0_75s']['pass_fraction']} "
            f"0.75-1.5={buckets['gt_0_75_le_1_5s']['pass_fraction']} "
            f"1.5-3={buckets['gt_1_5_le_3_0s']['pass_fraction']} gt3={buckets['gt_3_0s']['pass_fraction']}"
        )
        print(
            f"  pass_dt_p50={record['delta_t_seconds']['pass']['p50']} "
            f"degraded_dt_p50={record['delta_t_seconds']['degraded']['p50']} "
            f"pass_view_angle_p50={record['target_view_angle_degrees']['pass']['p50']} "
            f"degraded_view_angle_p50={record['target_view_angle_degrees']['degraded']['p50']}"
        )
    print(
        f"classification={d['classification']} local_density_signal={d['local_density_signal']} "
        f"capture_regime_signal={d['capture_regime_signal']}"
    )
    print(
        f"same_source_prefix_gain={d['same_source_prefix_gain_over_current']} "
        f"short_control_gain={d['short_control_gain_over_same_source_prefix']}"
    )
    print(
        f"decoupled_sampling_architecture_candidate={d['decoupled_sampling_architecture_candidate']} "
        f"cleaner_static_capture_recommended={d['cleaner_static_capture_recommended']}"
    )
    print("sampling_policy_change_authorized=False dense_consistency_threshold_relaxation_authorized=False")
    print("sfm_rerun=False da3_rerun=False dense_pair_rerun=False fusion_rerun=False filter_policy_changed=False persistence_changed=False")
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print("i4_2_detail_preservation_authorized=False mesh_authorization=False")
    print(f"next={diagnosis['decision']['next_action']}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
