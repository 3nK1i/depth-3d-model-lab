#!/usr/bin/env python3
"""I4.1.6B real DA3 + dense-consistency A/B for decoupled candidate plans.

This runner is intentionally opt-in and non-production-authoritative.  It reuses
the already-authorized global PyCOLMAP reconstruction and I4.1.6A candidate
localizations, runs real DA3 only for extra observations, aligns every accepted
extra depth map into the same arbitrary SfM gauge, then evaluates the unchanged
I1 dense-consistency gates for both bounded local candidate planners.

It does not change the production default, fusion/filter/persistence policy,
metric authority, measurement authority, or mesh authority.
"""
from __future__ import annotations

import argparse
import gc
import json
from math import isfinite
from pathlib import Path
import re
import resource
import selectors
import subprocess
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import project_sfm_keyframe_dense_rgb, reconstruction_camera_from_sfm_model  # noqa: E402
from core.reconstruction.da3 import DA3_EFFICIENT_MODEL_ID, probe_da3_environment  # noqa: E402
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_campaign import pair_key, select_longest_monotonic_pass_path  # noqa: E402
from core.reconstruction.offline_dense_i4_decoupled_dense_ab import (  # noqa: E402
    DenseABPlanSummary,
    align_localized_candidate_depth,
    camera_estimate_from_localization,
    choose_real_dense_plan,
)
from core.reconstruction.offline_dense_i4_decoupled_sampling_integration import (  # noqa: E402
    DenseCandidatePlannerMode,
    build_dense_candidate_plan,
)
from scripts.run_b2_sfm_dense_real_alignment import _adapt_da3_depth_to_sfm_pixel_space  # noqa: E402
from scripts.run_d1_three_view_canonical_accumulation import _all_sparse_landmarks, _sfm_result_from_payload  # noqa: E402
from scripts.run_e3_qualified_scene_fusion_expansion import _diagnose_pair, _robust_diagonal  # noqa: E402
from scripts.run_i1_offline_dense_campaign import _DENSE_THRESHOLDS, _load_reused_dense_candidate  # noqa: E402

_DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/depth3d-da3/bin/python3.11"
_DEFAULT_DA3_REPO = Path.home() / "Desktop/depth3d-da3-backends/depth-anything-3"
_DA3_WORKER = ROOT / "scripts/da3_workers/i4_1_6b_da3_sequential_worker.py"
_EDGE_RE = re.compile(r"^\s*edge\s+(\d+)<->(\d+)\s+status=(pass|degraded|inconclusive)\s*$", re.IGNORECASE)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-dir", type=Path, default=Path("/tmp/depth3d_i4_object_orbit_campaign"))
    p.add_argument("--i4-1-6a-dir", type=Path, default=Path("/tmp/depth3d_i4_1_6_decoupled_sampling_integration"))
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/depth3d_i4_1_6b_decoupled_dense_ab"))
    p.add_argument("--da3-python", type=Path, default=_DEFAULT_DA3_PYTHON)
    p.add_argument("--da3-repo", type=Path, default=_DEFAULT_DA3_REPO)
    p.add_argument("--da3-model", default=DA3_EFFICIENT_MODEL_ID)
    p.add_argument("--da3-model-revision", default="")
    p.add_argument("--process-res", type=int, default=392)
    p.add_argument("--da3-total-timeout-seconds", type=float, default=7200.0)
    p.add_argument("--dense-candidate-cap", type=int, default=120)
    p.add_argument("--local-target-fps", type=float, default=2.0)
    p.add_argument("--max-bridge-gap", type=int, default=3)
    p.add_argument("--minimum-correspondences", type=int, default=8)
    p.add_argument("--maximum-relative-residual", type=float, default=0.35)
    p.add_argument("--max-points-per-direction", type=int, default=120_000)
    p.add_argument("--nn-max-points-per-view", type=int, default=60_000)
    p.add_argument("--final-fusion-cap", type=int, default=40)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.6B cannot read JSON evidence: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.1.6B JSON root must be object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _resolve_sfm_report(i4_dir: Path) -> Path:
    candidates = [
        i4_dir / "01_i1_baseline/01_sfm/real_sfm_report.json",
        i4_dir / "01_sfm/real_sfm_report.json",
        i4_dir / "real_sfm_report.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise ReconstructionValidationError("I4.1.6B cannot find the authoritative real_sfm_report.json.")


def _stream_command(command: list[str], *, log_path: Path, timeout_s: float) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                if time.monotonic() - started > float(timeout_s):
                    process.kill(); process.wait()
                    raise ReconstructionValidationError("I4.1.6B DA3 worker exceeded total timeout.")
                events = selector.select(timeout=0.5)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line:
                        print(line, end="", flush=True); log.write(line); log.flush()
                if process.poll() is not None:
                    for line in process.stdout:
                        print(line, end="", flush=True); log.write(line)
                    log.flush()
                    break
        finally:
            selector.close()
            if process.poll() is None:
                process.kill(); process.wait()
        return int(process.returncode)


def _baseline_selected_path(i1_summary: Mapping[str, Any]) -> Mapping[str, Any]:
    dense_qualification = i1_summary.get("dense_qualification")
    if isinstance(dense_qualification, Mapping):
        selected = dense_qualification.get("selected_pass_path")
        if isinstance(selected, Mapping):
            return selected

    selected = i1_summary.get("selected_path")
    if isinstance(selected, Mapping):
        return selected

    raise ReconstructionValidationError(
        "I4.1.6B baseline I1 summary lacks "
        "dense_qualification.selected_pass_path and legacy selected_path."
    )


def _baseline_dense_metrics(i4_dir: Path) -> dict[str, Any]:
    i1_summary = _read_json(i4_dir / "01_i1_baseline/i1_campaign_summary.json")
    selected = _baseline_selected_path(i1_summary)
    selected_views = int(selected.get("view_count", 0))
    log_path = i4_dir / "logs/01_i1_baseline.log"
    if not log_path.is_file():
        raise ReconstructionValidationError(f"I4.1.6B baseline I1 log missing: {log_path}")
    statuses: list[str] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EDGE_RE.match(line)
        if match:
            statuses.append(match.group(3).lower())
    if not statuses:
        raise ReconstructionValidationError("I4.1.6B found no baseline dense pair statuses.")
    passed = sum(v == "pass" for v in statuses)
    degraded = sum(v == "degraded" for v in statuses)
    inconclusive = sum(v == "inconclusive" for v in statuses)
    decisive = passed + degraded
    return {
        "candidate_mode": "global_anchors_only",
        "pair_edges": len(statuses),
        "pair_pass": passed,
        "pair_degraded": degraded,
        "pair_inconclusive": inconclusive,
        "pair_pass_fraction": float(passed / max(1, decisive)),
        "selected_qualified_views": selected_views,
        "selected_path": dict(selected),
        "source": str(i4_dir),
        "rerun": False,
    }


def _localization_payload(
    i4_1_6a_dir: Path,
    source_index: int,
    *,
    require_accepted: bool = True,
) -> dict[str, Any]:
    path = i4_1_6a_dir / "localization_evidence" / f"candidate_{source_index:08d}.json"
    payload = _read_json(path)
    if require_accepted and not bool(payload.get("accepted")):
        raise ReconstructionValidationError(f"I4.1.6B candidate {source_index} was not accepted by I4.1.6A.")
    return payload


def _partition_extra_localization_candidates(
    i4_1_6a_dir: Path,
    extra_indices: list[int],
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    accepted: list[int] = []
    rejected: dict[int, dict[str, Any]] = {}
    for source_index in extra_indices:
        payload = _localization_payload(
            i4_1_6a_dir, source_index, require_accepted=False
        )
        if bool(payload.get("accepted")):
            accepted.append(int(source_index))
        else:
            rejected[int(source_index)] = payload
    return accepted, rejected


def _localization_rejection_summary(
    *,
    source_index: int,
    localization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_sequence_index": int(source_index),
        "keyframe_id": str(localization.get("keyframe_id", "")),
        "alignment_accepted": False,
        "rejection_code": "localization_not_accepted",
        "localization_status": str(localization.get("status", "unknown")),
        "candidate_correspondences": 0,
        "valid_correspondences": 0,
        "inlier_correspondences": int(localization.get("inlier_count", 0) or 0),
        "median_relative_residual": None,
        "p90_relative_residual": None,
        "pixel_adapter": None,
        "alignment_evidence_json": None,
        "da3_npz": None,
        "localized_observation": False,
        "da3_skipped_due_to_localization_rejection": True,
    }


def _prepare_da3(
    *,
    extra_indices: list[int],
    i4_1_6a_dir: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    da3_python = args.da3_python.expanduser().resolve(); da3_repo = args.da3_repo.expanduser().resolve()
    probe = probe_da3_environment(da3_python, da3_repo)
    if not probe.available:
        raise ReconstructionValidationError("I4.1.6B DA3 environment unavailable: " + json.dumps(probe.as_dict(), sort_keys=True))
    records = []
    for index in extra_indices:
        frame = i4_1_6a_dir / "candidate_frames" / f"candidate_{index:08d}.jpg"
        if not frame.is_file():
            raise ReconstructionValidationError(f"I4.1.6B missing I4.1.6A candidate frame: {frame}")
        records.append({"source_sequence_index": int(index), "path": str(frame.resolve())})
    da3_dir = output / "01_da3_extra_candidates"
    request = {
        "schema_version": 1,
        "output_dir": str(da3_dir.resolve()),
        "images": records,
        "profile": {
            "model_id": str(args.da3_model),
            "model_revision": str(args.da3_model_revision),
            "process_res": int(args.process_res),
            "process_res_method": "upper_bound_resize",
            "ref_view_strategy": "saddle_balanced",
        },
    }
    request_path = _write_json(output / "i4_1_6b_da3_request.json", request)
    command = [str(da3_python), str(_DA3_WORKER.resolve()), "--request", str(request_path.resolve())]
    code = _stream_command(command, log_path=output / "logs/01_da3_extra_candidates.log", timeout_s=float(args.da3_total_timeout_seconds))
    if code != 0:
        raise ReconstructionValidationError(f"I4.1.6B DA3 worker failed with exit code {code}.")
    manifest = _read_json(da3_dir / "i4_1_6b_da3_manifest.json")
    if int(manifest.get("candidate_count", -1)) != len(extra_indices):
        raise ReconstructionValidationError("I4.1.6B DA3 candidate accounting mismatch.")
    return manifest


def _extra_record(
    *,
    source_index: int,
    sfm_result: Any,
    camera_model: Any,
    i4_1_6a_dir: Path,
    da3_dir: Path,
    output: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    localization = _localization_payload(i4_1_6a_dir, source_index)
    frame_path = i4_1_6a_dir / "candidate_frames" / f"candidate_{source_index:08d}.jpg"
    bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ReconstructionValidationError(f"I4.1.6B cannot decode candidate frame {frame_path}")
    source_h, source_w = bgr.shape[:2]
    da3_path = da3_dir / f"candidate_{source_index:08d}.npz"
    try:
        with np.load(da3_path, allow_pickle=False) as payload:
            raw_depth = np.asarray(payload["depth"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:
        raise ReconstructionValidationError(f"I4.1.6B cannot read DA3 depth {da3_path}: {exc}") from exc
    depth_sfm, pixel_adapter = _adapt_da3_depth_to_sfm_pixel_space(
        raw_depth,
        sfm_width=camera_model.image_size.width,
        sfm_height=camera_model.image_size.height,
    )
    evidence = align_localized_candidate_depth(
        sfm_result,
        localization=localization,
        camera_model=camera_model,
        dense_depth_sfm_grid=depth_sfm,
        source_image_width=source_w,
        source_image_height=source_h,
        depth_provenance=(
            f"Depth Anything 3 I4.1.6B localized candidate source_index={source_index};"
            "same authoritative offline SfM frame; arbitrary gauge"
        ),
        minimum_correspondences=int(args.minimum_correspondences),
        maximum_relative_residual=float(args.maximum_relative_residual),
    )
    root = output / "02_extra_candidate_clouds" / f"candidate_{source_index:08d}"
    root.mkdir(parents=True, exist_ok=True)
    evidence_path = _write_json(root / "alignment.json", evidence.as_dict())
    summary = {
        "source_sequence_index": int(source_index),
        "keyframe_id": str(localization.get("keyframe_id", "")),
        "alignment_accepted": bool(evidence.is_accepted and evidence.alignment is not None),
        "rejection_code": None if evidence.rejection_code is None else evidence.rejection_code.value,
        "candidate_correspondences": int(evidence.candidate_observations),
        "valid_correspondences": int(evidence.valid_correspondences),
        "inlier_correspondences": int(evidence.inlier_correspondences),
        "median_relative_residual": evidence.median_relative_residual,
        "p90_relative_residual": evidence.p90_relative_residual,
        "pixel_adapter": pixel_adapter,
        "alignment_evidence_json": str(evidence_path.resolve()),
        "da3_npz": str(da3_path.resolve()),
        "localized_observation": True,
    }
    if not evidence.is_accepted or evidence.alignment is None:
        return None, summary
    estimate = camera_estimate_from_localization(localization, camera_model=camera_model)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (camera_model.image_size.width, camera_model.image_size.height), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    projection = project_sfm_keyframe_dense_rgb(
        estimate,
        camera_model,
        timestamp_ns=int(localization.get("timestamp_ns", 0)),
        rgb=rgb,
        dense_depth_source_units=depth_sfm,
        alignment_evidence=evidence,
        pixel_stride=1,
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    cloud_path = root / "canonical_cloud.npz"
    np.savez_compressed(
        cloud_path,
        points_reconstruction=projection.points_reconstruction,
        colors_rgb=projection.colors_rgb,
        aligned_depth_reconstruction_units=np.asarray(depth_sfm * evidence.alignment.scale_to_reconstruction_units, dtype=np.float64),
        T_reconstruction_from_camera=estimate.T_reconstruction_from_camera,
        camera_center_reconstruction=estimate.camera_center_reconstruction,
        keyframe_id=np.asarray(estimate.keyframe_id),
        source_sequence_index=np.asarray(source_index, dtype=np.int64),
    )
    camera = reconstruction_camera_from_sfm_model(
        camera_model,
        keyframe_id=estimate.keyframe_id,
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    record = {
        "workspace_order": -1,
        "source_sequence_index": int(source_index),
        "keyframe_id": str(estimate.keyframe_id),
        "frame_id": str(estimate.frame_id),
        "camera_id": str(estimate.camera_id),
        "camera_model_name": str(camera_model.backend_model_name),
        "source": "i4_1_6b_localized_extra_candidate",
        "cloud_npz": str(cloud_path.resolve()),
        "points": np.asarray(projection.points_reconstruction, dtype=np.float64),
        "colors": np.asarray(projection.colors_rgb, dtype=np.uint8),
        "aligned_depth": np.asarray(depth_sfm * evidence.alignment.scale_to_reconstruction_units, dtype=np.float64),
        "T_reconstruction_from_camera": estimate.T_reconstruction_from_camera,
        "camera_center_reconstruction": estimate.camera_center_reconstruction,
        "camera_estimate": estimate,
        "camera": camera,
        "scale_to_reconstruction_units": float(evidence.alignment.scale_to_reconstruction_units),
    }
    summary.update({
        "scale_to_reconstruction_units": float(evidence.alignment.scale_to_reconstruction_units),
        "relative_scale_uncertainty": float(evidence.alignment.relative_scale_uncertainty),
        "point_count": int(projection.points_reconstruction.shape[0]),
        "cloud_npz": str(cloud_path.resolve()),
    })
    return record, summary



def _compact_dense_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop bulk arrays after the canonical cloud has been persisted.

    I4.1.6B may admit well over one hundred full-resolution observations. Keeping
    every points/depth/color array resident makes host RAM grow linearly with the
    candidate count. The canonical NPZ is already the authoritative persisted
    payload for pair diagnostics, so retain only lightweight metadata in memory.
    """
    if record is None:
        return None
    compact = dict(record)
    for key in ("points", "colors", "aligned_depth"):
        compact.pop(key, None)
    return compact


def _hydrate_dense_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Load bulk dense arrays for one record from its persisted canonical NPZ."""
    cloud_path = Path(str(record.get("cloud_npz", ""))).expanduser().resolve()
    if not cloud_path.is_file():
        raise ReconstructionValidationError(
            f"I4.1.6B dense record canonical cloud missing: {cloud_path}"
        )
    try:
        with np.load(cloud_path, allow_pickle=False) as payload:
            points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
            aligned_depth = np.asarray(
                payload["aligned_depth_reconstruction_units"], dtype=np.float64
            )
    except (OSError, ValueError, KeyError) as exc:
        raise ReconstructionValidationError(
            f"I4.1.6B unable to hydrate canonical cloud {cloud_path}: {exc}"
        ) from exc
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ReconstructionValidationError(
            f"I4.1.6B hydrated canonical cloud arrays are invalid: {cloud_path}"
        )
    hydrated = dict(record)
    hydrated["points"] = points
    hydrated["colors"] = colors
    hydrated["aligned_depth"] = aligned_depth
    return hydrated


def _plan_dense_ab(
    *,
    plan: Any,
    sfm_result: Any,
    global_records_by_source: Mapping[int, tuple[dict[str, Any] | None, dict[str, Any]]],
    extra_records_by_source: Mapping[int, tuple[dict[str, Any] | None, dict[str, Any]]],
    sparse_points: np.ndarray,
    scene_scale: float,
    output: Path,
    args: argparse.Namespace,
) -> tuple[DenseABPlanSummary, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []
    for plan_order, spec in enumerate(plan.candidates):
        source_index = int(spec.source_sequence_index)
        source = global_records_by_source if spec.is_global_anchor else extra_records_by_source
        record, base = source[source_index]
        row = dict(base)
        row.update({
            "plan_order": int(plan_order),
            "source_sequence_index": source_index,
            "is_global_anchor": bool(spec.is_global_anchor),
            "global_workspace_order": spec.global_workspace_order,
        })
        candidate_rows.append(row)
        if record is not None:
            clone = dict(record)
            clone["workspace_order"] = int(plan_order)
            clone["source_sequence_index"] = source_index
            clone["global_workspace_order"] = spec.global_workspace_order
            records[plan_order] = clone

    aligned_orders = tuple(sorted(records))
    pair_summaries: dict[tuple[int, int], dict[str, Any]] = {}
    pair_dir = output / "03_dense_pair_ab" / plan.mode.value
    pair_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nI4.1.6B real dense graph plan={plan.mode.value}", flush=True)
    started = time.perf_counter()
    for pos, order_a in enumerate(aligned_orders):
        for order_b in aligned_orders[pos + 1:]:
            if order_b - order_a > int(args.max_bridge_gap):
                break
            record_a = _hydrate_dense_record(records[order_a])
            record_b = _hydrate_dense_record(records[order_b])
            try:
                pair = _diagnose_pair(
                    a=record_a, b=record_b, thresholds=dict(_DENSE_THRESHOLDS),
                    scene_scale=float(scene_scale), sparse_landmarks=sparse_points, output_dir=pair_dir,
                    max_points_per_direction=int(args.max_points_per_direction),
                    nn_max_points_per_view=int(args.nn_max_points_per_view),
                )
            finally:
                del record_a, record_b
                gc.collect()
            pair["campaign_stage"] = "I4.1.6B"
            pair["plan_name"] = plan.mode.value
            pair["source_sequence_indices"] = [
                int(records[order_a]["source_sequence_index"]),
                int(records[order_b]["source_sequence_index"]),
            ]
            pair_summaries[pair_key(order_a, order_b)] = pair
            print(
                f"  edge {order_a}<->{order_b} source={pair['source_sequence_indices'][0]}<->{pair['source_sequence_indices'][1]} status={pair['status']}",
                flush=True,
            )
    pair_duration = time.perf_counter() - started
    selected = select_longest_monotonic_pass_path(aligned_orders, pair_summaries, max_bridge_gap=int(args.max_bridge_gap))
    selected_sources = [int(records[o]["source_sequence_index"]) for o in selected.orders if o in records]
    statuses = [str(v.get("status", "")) for v in pair_summaries.values()]
    passed = sum(v == "pass" for v in statuses); degraded = sum(v == "degraded" for v in statuses); inconclusive = sum(v == "inconclusive" for v in statuses)
    plan_summary = DenseABPlanSummary(
        plan_name=plan.mode.value,
        planned_candidates=len(plan.candidates),
        alignment_accepted=len(records),
        pair_edges=len(statuses),
        pair_pass=passed,
        pair_degraded=degraded,
        pair_inconclusive=inconclusive,
        selected_qualified_views=int(selected.view_count),
        full_span=bool(plan.source_span_fraction >= 0.98),
        final_fusion_cap=int(args.final_fusion_cap),
    )
    payload = plan_summary.as_dict()
    payload.update({
        "candidate_rows": candidate_rows,
        "selected_path": selected.as_dict(),
        "selected_source_sequence_indices": selected_sources,
        "pair_duration_s": float(pair_duration),
        "thresholds": dict(_DENSE_THRESHOLDS),
        "threshold_relaxation": False,
        "max_bridge_gap": int(args.max_bridge_gap),
    })
    _write_json(output / "plan_summaries" / f"{plan.mode.value}.json", payload)
    print(
        f"plan={plan.mode.value} aligned={len(records)}/{len(plan.candidates)} "
        f"pairs pass={passed} degraded={degraded} inconclusive={inconclusive} "
        f"pass_fraction={plan_summary.pair_pass_fraction} selected={selected.view_count} dense_gate={plan_summary.dense_gate_pass}",
        flush=True,
    )
    return plan_summary, payload


def main() -> int:
    args = _args()
    i4_dir = args.i4_dir.expanduser().resolve(); a_dir = args.i4_1_6a_dir.expanduser().resolve()
    video = args.video.expanduser().resolve(); output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.6B output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not video.is_file():
        raise ReconstructionValidationError(f"I4.1.6B video missing: {video}")

    a_summary = _read_json(a_dir / "i4_1_6a_summary.json")
    decision = a_summary.get("decision")
    if not isinstance(decision, Mapping) or not bool(decision.get("dense_ab_execution_authorized")):
        raise ReconstructionValidationError("I4.1.6B requires I4.1.6A dense_ab_execution_authorized=True.")
    sfm_report_path = _resolve_sfm_report(i4_dir)
    sfm_report = _read_json(sfm_report_path)
    sfm_payload = sfm_report.get("sfm_result")
    if not isinstance(sfm_payload, dict):
        raise ReconstructionValidationError("I4.1.6B authoritative SfM report lacks sfm_result.")
    sfm_result = _sfm_result_from_payload(sfm_payload)
    if sfm_result.gauge.unit_to_metre is not None:
        raise ReconstructionValidationError("I4.1.6B refuses metric-scale promotion.")
    if len(sfm_result.camera_models) != 1:
        raise ReconstructionValidationError("I4.1.6B currently requires one shared SfM camera model.")
    camera_model = sfm_result.camera_models[0]

    source = a_summary.get("source") if isinstance(a_summary.get("source"), Mapping) else {}
    total_frames = int(source.get("source_total_frames", 0)); source_fps = float(source.get("source_fps", 0.0))
    if total_frames < 2 or not isfinite(source_fps) or source_fps <= 0.0:
        raise ReconstructionValidationError("I4.1.6B I4.1.6A source metadata invalid.")
    modes = (
        DenseCandidatePlannerMode.UNIFORM_LOCALIZED_DENSE_POOL,
        DenseCandidatePlannerMode.GLOBAL_ANCHORS_PLUS_LOCAL_NEIGHBORS,
    )
    plans = tuple(
        build_dense_candidate_plan(
            sfm_result,
            source_total_frames=total_frames,
            source_fps=source_fps,
            mode=mode,
            dense_candidate_cap=int(args.dense_candidate_cap),
            local_target_fps=float(args.local_target_fps),
        )
        for mode in modes
    )
    # Protect the A/B run from silently using a different plan than 1.6A.
    a_plan_counts = {
        str(item.get("plan", {}).get("mode")): int(item.get("planned_candidates", -1))
        for item in a_summary.get("plans", ()) if isinstance(item, Mapping) and isinstance(item.get("plan"), Mapping)
    }
    for plan in plans:
        if a_plan_counts.get(plan.mode.value) != len(plan.candidates):
            raise ReconstructionValidationError(f"I4.1.6B plan {plan.mode.value} differs from I4.1.6A evidence.")

    registered_keyframes = {item.keyframe_id for item in sfm_result.cameras}
    workspace_by_source = {
        int(item.source_sequence_index): item for item in sfm_result.workspace.images
        if item.keyframe_id in registered_keyframes
    }
    extra_indices = sorted({
        int(spec.source_sequence_index) for plan in plans for spec in plan.candidates if not spec.is_global_anchor
    })
    accepted_extra_indices, rejected_localizations = _partition_extra_localization_candidates(
        a_dir, extra_indices
    )
    print(
        f"I4.1.6B real A/B: planned_extra_candidates={len(extra_indices)} "
        f"accepted_localized_extras={len(accepted_extra_indices)} "
        f"localization_rejected_extras={len(rejected_localizations)} "
        f"plans={','.join(p.mode.value for p in plans)}",
        flush=True,
    )
    da3_manifest = _prepare_da3(
        extra_indices=accepted_extra_indices, i4_1_6a_dir=a_dir, output=output, args=args
    )

    dense_dir = i4_dir / "01_i1_baseline/02_dense_views"
    global_records_by_source: dict[int, tuple[dict[str, Any] | None, dict[str, Any]]] = {}
    for source_index, workspace in workspace_by_source.items():
        record, summary = _load_reused_dense_candidate(order=int(workspace.order), dense_dir=dense_dir, sfm_result=sfm_result)
        base = dict(summary); base["source_sequence_index"] = int(source_index); base["global_anchor_reused"] = True
        if record is not None:
            clone = dict(record); clone["source_sequence_index"] = int(source_index); record = _compact_dense_record(clone)
        global_records_by_source[int(source_index)] = (record, base)

    extra_records_by_source: dict[int, tuple[dict[str, Any] | None, dict[str, Any]]] = {}
    started_alignment = time.perf_counter()
    for ordinal, source_index in enumerate(extra_indices, start=1):
        print(f"I4.1.6B aligning/projecting extra {ordinal}/{len(extra_indices)} source_index={source_index}", flush=True)
        rejected = rejected_localizations.get(source_index)
        if rejected is not None:
            summary = _localization_rejection_summary(
                source_index=source_index, localization=rejected
            )
            extra_records_by_source[source_index] = (None, summary)
            print(
                f"  localization=REJECTED status={summary['localization_status']} "
                "da3=SKIPPED alignment=NOT_RUN",
                flush=True,
            )
            continue
        record, summary = _extra_record(
            source_index=source_index, sfm_result=sfm_result, camera_model=camera_model,
            i4_1_6a_dir=a_dir, da3_dir=output / "01_da3_extra_candidates",
            output=output, args=args,
        )
        alignment_accepted = record is not None
        extra_records_by_source[source_index] = (_compact_dense_record(record), summary)
        del record
        gc.collect()
        print(
            f"  alignment={'PASS' if alignment_accepted else 'REJECTED'} "
            f"scale={summary.get('scale_to_reconstruction_units')} reason={summary.get('rejection_code')}",
            flush=True,
        )
    alignment_duration = time.perf_counter() - started_alignment

    sparse_points, _ = _all_sparse_landmarks(sfm_result)
    scene_scale = _robust_diagonal(sparse_points)
    if not isfinite(scene_scale) or scene_scale <= 0.0:
        raise ReconstructionValidationError("I4.1.6B robust sparse scene scale invalid.")

    plan_objects: list[DenseABPlanSummary] = []; plan_payloads: list[dict[str, Any]] = []
    for plan in plans:
        summary_obj, payload = _plan_dense_ab(
            plan=plan, sfm_result=sfm_result, global_records_by_source=global_records_by_source,
            extra_records_by_source=extra_records_by_source, sparse_points=sparse_points,
            scene_scale=scene_scale, output=output, args=args,
        )
        plan_objects.append(summary_obj); plan_payloads.append(payload)

    baseline = _baseline_dense_metrics(i4_dir)
    decision = choose_real_dense_plan(
        plan_objects,
        baseline_selected_views=int(baseline["selected_qualified_views"]),
        baseline_pair_pass_fraction=float(baseline["pair_pass_fraction"]),
    )
    ru = resource.getrusage(resource.RUSAGE_SELF)
    summary = {
        "phase": "I4.1.6B",
        "schema_version": 1,
        "status": "real_dense_ab_ready",
        "source": {
            "video": str(video),
            "i4_dir": str(i4_dir),
            "i4_1_6a_dir": str(a_dir),
            "sfm_report": str(sfm_report_path),
            "sfm_session_id": str(sfm_result.session_id),
        },
        "baseline_global_only": baseline,
        "plans": plan_payloads,
        "da3": da3_manifest,
        "performance": {
            "extra_alignment_projection_duration_s": float(alignment_duration),
            "host_peak_rss_mib": float(ru.ru_maxrss / 1024.0),
            "da3_peak_vram_mib": da3_manifest.get("peak_vram_mib"),
            "da3_inference_duration_s_total": da3_manifest.get("inference_duration_s_total"),
        },
        "decision": {
            **decision,
            "single_authoritative_sfm_frame": True,
            "global_full_span_sampling_preserved": True,
            "extra_dense_candidates_are_observations_not_second_map": True,
            "extra_dense_candidates_pose_localized": True,
            "final_fusion_cap_preserved": int(args.final_fusion_cap),
            "production_default_changed": False,
            "filter_policy_changed": False,
            "persistence_changed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "execution": {
            "pycolmap_rerun": False,
            "da3_extra_candidates_executed": True,
            "global_anchor_da3_reused": True,
            "dense_pair_rerun": True,
            "fusion_rerun": False,
            "filtering_rerun": False,
            "persistence_rerun": False,
        },
        "thresholds": {
            "dense_consistency": dict(_DENSE_THRESHOLDS),
            "threshold_relaxation": False,
            "max_bridge_gap": int(args.max_bridge_gap),
        },
        "next": (
            "If real_dense_ab_validation_pass=True, I4.1.6C may bind the winning planner additively into the production offline path while preserving the global SfM authority and final fusion cap. "
            "If False, preserve this evidence, keep I4.1.3 global sampling, and move final I4 qualification to a cleaner static orbit capture rather than relaxing dense thresholds."
        ),
    }
    summary_path = _write_json(output / "i4_1_6b_summary.json", summary)
    print("\nI4.1.6B Real Decoupled Dense A/B: REAL_DENSE_AB_READY", flush=True)
    print(
        f"baseline selected={baseline['selected_qualified_views']} pair_pass_fraction={baseline['pair_pass_fraction']:.3f}",
        flush=True,
    )
    for item in plan_objects:
        print(
            f"plan={item.plan_name} aligned={item.alignment_accepted}/{item.planned_candidates} "
            f"pair_pass_fraction={item.pair_pass_fraction} selected={item.selected_qualified_views} dense_gate={item.dense_gate_pass}",
            flush=True,
        )
    print(
        f"recommended_plan={decision['recommended_plan']} real_dense_ab_validation_pass={decision['real_dense_ab_validation_pass']} "
        f"i4_1_6c_production_binding_authorized={decision['i4_1_6c_production_binding_authorized']}",
        flush=True,
    )
    print(
        f"baseline_comparison_rule={decision['comparison_rule']} "
        f"baseline_pair_saturated={decision['baseline_pair_pass_fraction_saturated']} "
        f"pair_quality_regressed={decision['pair_quality_regressed']} "
        f"selected_gain={decision['selected_view_gain_over_global_only']}",
        flush=True,
    )
    print("single_authoritative_sfm_frame=True production_default_changed=False", flush=True)
    print("production_sampling_integration_authorized=False dense_consistency_threshold_relaxation_authorized=False", flush=True)
    print("fusion_rerun=False filtering_rerun=False persistence_rerun=False", flush=True)
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False", flush=True)
    print("i4_2_detail_preservation_authorized=False mesh_authorization=False", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
