#!/usr/bin/env python3
"""I4.1.6A real decoupled dense-candidate localization preflight.

Uses an existing I4 global SfM result, keeps that reconstruction as the single
geometry authority, and localizes bounded extra dense candidates into the same
frame.  This preflight intentionally stops before DA3/dense-consistency/fusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_i4_decoupled_sampling_integration import (  # noqa: E402
    DenseCandidateLocalizationPolicy,
    DenseCandidatePlannerMode,
    SfmAnchorDescriptorIndex,
    build_dense_candidate_plan,
    global_anchor_localization_result,
    probe_video_metadata,
    summarize_localization_results,
)
from scripts.run_d1_three_view_canonical_accumulation import _sfm_result_from_payload  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plans",
        nargs="+",
        default=["uniform_localized_dense_pool", "global_anchors_plus_local_neighbors"],
        choices=[
            "uniform_localized_dense_pool",
            "global_anchors_plus_local_neighbors",
            "global_anchors_only",
        ],
    )
    parser.add_argument("--dense-candidate-cap", type=int, default=120)
    parser.add_argument("--local-target-fps", type=float, default=2.0)
    parser.add_argument("--nearest-anchor-count", type=int, default=4)
    parser.add_argument("--sift-features", type=int, default=6000)
    parser.add_argument("--maximum-landmarks-per-anchor", type=int, default=5000)
    parser.add_argument("--descriptor-ratio-test", type=float, default=0.78)
    parser.add_argument("--minimum-correspondences", type=int, default=40)
    parser.add_argument("--minimum-inliers", type=int, default=30)
    parser.add_argument("--minimum-inlier-fraction", type=float, default=0.35)
    parser.add_argument("--minimum-extra-localization-fraction", type=float, default=0.80)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{path} must contain a JSON object.")
    return payload


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_sfm_report(i4_dir: Path) -> Path:
    candidates = [
        i4_dir / "01_i1_baseline/01_sfm/real_sfm_report.json",
        i4_dir / "01_sfm/real_sfm_report.json",
        i4_dir / "real_sfm_report.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise ReconstructionValidationError(
        "I4.1.6 could not find real_sfm_report.json under the supplied I4 directory."
    )


def _materialize_union(video_path: Path, indices: set[int], output_dir: Path) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not indices:
        return {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ReconstructionValidationError(f"Could not open video: {video_path}")
    wanted = set(int(v) for v in indices)
    max_index = max(wanted)
    paths: dict[int, Path] = {}
    index = 0
    try:
        while index <= max_index and wanted:
            ok, image = capture.read()
            if not ok:
                break
            if index in wanted:
                path = output_dir / f"candidate_{index:08d}.jpg"
                if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 97]):
                    raise ReconstructionValidationError(f"Could not write dense candidate frame {index}.")
                paths[index] = path
                wanted.remove(index)
            index += 1
    finally:
        capture.release()
    if wanted:
        raise ReconstructionValidationError(
            f"I4.1.6 could not decode all dense candidates; missing source indices: {sorted(wanted)[:12]}"
        )
    return paths


def main() -> int:
    args = _parse_args()
    i4_dir = args.i4_dir.expanduser().resolve()
    video = args.video.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.6 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    sfm_report_path = _resolve_sfm_report(i4_dir)
    sfm_report = _read_json(sfm_report_path)
    sfm_payload = sfm_report.get("sfm_result")
    if not isinstance(sfm_payload, dict):
        raise ReconstructionValidationError("I4.1.6 real SfM report is missing sfm_result.")
    sfm_result = _sfm_result_from_payload(sfm_payload)
    if sfm_result.gauge.unit_to_metre is not None:
        raise ReconstructionValidationError("I4.1.6 does not promote or consume metric scale.")
    if len(sfm_result.camera_models) != 1:
        raise ReconstructionValidationError("I4.1.6 currently requires one shared SfM camera model.")
    camera_model = sfm_result.camera_models[0]

    total_frames, source_fps, duration = probe_video_metadata(video)
    modes = tuple(DenseCandidatePlannerMode(value) for value in args.plans)
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
    registered_keyframes = {item.keyframe_id for item in sfm_result.cameras}
    anchor_indices = {
        int(item.source_sequence_index) for item in sfm_result.workspace.images
        if item.keyframe_id in registered_keyframes
    }
    extra_union = {
        item.source_sequence_index
        for plan in plans for item in plan.candidates
        if item.source_sequence_index not in anchor_indices
    }
    frame_paths = _materialize_union(video, extra_union, output / "candidate_frames")

    policy = DenseCandidateLocalizationPolicy(
        nearest_anchor_count=int(args.nearest_anchor_count),
        sift_features=int(args.sift_features),
        maximum_landmarks_per_anchor=int(args.maximum_landmarks_per_anchor),
        descriptor_ratio_test=float(args.descriptor_ratio_test),
        minimum_correspondences=int(args.minimum_correspondences),
        minimum_inliers=int(args.minimum_inliers),
        minimum_inlier_fraction=float(args.minimum_inlier_fraction),
    )
    print("I4.1.6A: building authoritative SfM anchor descriptor index...", flush=True)
    anchor_index = SfmAnchorDescriptorIndex(sfm_result, policy)
    print(f"I4.1.6A: indexed_global_anchors={len(anchor_index.records)}", flush=True)

    global_specs = {
        item.source_sequence_index: item
        for plan in plans for item in plan.candidates if item.is_global_anchor
    }
    localized_by_index = {
        index: global_anchor_localization_result(spec, sfm_result=sfm_result)
        for index, spec in global_specs.items()
    }
    extra_specs = {
        item.source_sequence_index: item
        for plan in plans for item in plan.candidates if not item.is_global_anchor
    }
    for ordinal, index in enumerate(sorted(extra_specs), start=1):
        spec = extra_specs[index]
        print(
            f"I4.1.6A localizing extra candidate {ordinal}/{len(extra_specs)} "
            f"source_index={index}...",
            flush=True,
        )
        result = anchor_index.localize(
            source_sequence_index=index,
            timestamp_ns=spec.timestamp_ns,
            image_path=frame_paths[index],
            camera_model=camera_model,
        )
        localized_by_index[index] = result
        print(
            f"  status={result.status.value} matches={result.correspondence_count} "
            f"inliers={result.inlier_count} ratio={result.inlier_fraction:.3f} "
            f"median_px={result.median_reprojection_error_px}",
            flush=True,
        )
        evidence_dir = output / "localization_evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            evidence_dir / f"candidate_{index:08d}.json",
            result.as_dict(include_matches=True),
        )

    plan_summaries = []
    for plan in plans:
        results = [localized_by_index[item.source_sequence_index] for item in plan.candidates]
        summary = summarize_localization_results(
            plan,
            results,
            minimum_extra_localization_fraction=float(args.minimum_extra_localization_fraction),
        )
        plan_summaries.append(summary)
        print(
            f"plan={plan.mode.value} candidates={summary['planned_candidates']} "
            f"extra={summary['extra_candidates']} accepted_extra={summary['accepted_extra_candidates']} "
            f"localization_fraction={summary['extra_localization_fraction']:.3f} "
            f"full_span={summary['full_span']} gate={summary['localization_gate_pass']}",
            flush=True,
        )

    dense_ab_authorized = len(plan_summaries) >= 1 and any(
        bool(item["localization_gate_pass"]) for item in plan_summaries
    )
    summary = {
        "phase": "I4.1.6A",
        "schema_version": 1,
        "status": "localization_ready" if dense_ab_authorized else "localization_degraded",
        "source": {
            "video": str(video),
            "source_total_frames": total_frames,
            "source_fps": source_fps,
            "source_duration_seconds": duration,
            "source_sha256": _sha256_file(video),
            "i4_dir": str(i4_dir),
            "sfm_report": str(sfm_report_path),
            "sfm_session_id": str(sfm_result.session_id),
            "registered_global_anchors": len(sfm_result.cameras),
        },
        "localization_policy": policy.as_dict(),
        "plans": plan_summaries,
        "decision": {
            "single_authoritative_sfm_frame": True,
            "global_full_span_sampling_preserved": True,
            "extra_dense_candidates_are_observations_not_second_map": True,
            "extra_dense_candidates_require_pose_localization": True,
            "dense_ab_execution_authorized": bool(dense_ab_authorized),
            "production_sampling_integration_authorized": False,
            "production_default_changed": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "i4_2_detail_preservation_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
            "mesh_authorization": False,
        },
        "execution": {
            "pycolmap_rerun": False,
            "da3_rerun": False,
            "dense_pair_rerun": False,
            "fusion_rerun": False,
            "filter_policy_changed": False,
            "persistence_changed": False,
        },
        "next": (
            "If at least one bounded local plan passes real localization, continue I4.1.6B with real DA3/dense-consistency A/B on localized candidates while keeping the same SfM authority and unchanged dense thresholds. "
            "If localization is weak, diagnose candidate registration rather than relaxing dense-consistency gates."
        ),
    }
    summary_path = _write_json(output / "i4_1_6a_summary.json", summary)
    print(
        "I4.1.6A Decoupled Dense-Candidate Localization: "
        + ("LOCALIZATION_READY" if dense_ab_authorized else "LOCALIZATION_DEGRADED"),
        flush=True,
    )
    print("single_authoritative_sfm_frame=True production_default_changed=False", flush=True)
    print("pycolmap_rerun=False da3_rerun=False dense_pair_rerun=False fusion_rerun=False", flush=True)
    print("production_sampling_integration_authorized=False dense_consistency_threshold_relaxation_authorized=False", flush=True)
    print(f"dense_ab_execution_authorized={dense_ab_authorized} i4_2_detail_preservation_authorized=False mesh_authorization=False", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0 if dense_ab_authorized else 3


if __name__ == "__main__":
    raise SystemExit(main())
