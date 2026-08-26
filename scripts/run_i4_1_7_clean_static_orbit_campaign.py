#!/usr/bin/env python3
"""I4.1.7 clean static object/orbit qualification campaign.

Runs a fresh global I1 baseline on the clean capture, then reuses the already
validated I4.1.6A localization and I4.1.6B real dense A/B machinery without
binding either local planner into production.  The winning/selected plan is
finally checked against the locked object-orbit gates in the same authoritative
SfM frame.

The I4.1.7 orbit target is provisional optical-axis consensus because the
production decoupled path has not yet been fused/filtered.  I4.1.8 must
recompute the final target from retained Core geometry before closing I4.1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import (  # noqa: E402
    I4_1_7_SCHEMA_VERSION,
    assemble_selected_pose_sequence,
    build_clean_static_orbit_assessment,
    selected_plan_payload,
    validate_prior_decoupled_evidence,
)
from core.reconstruction.offline_dense_i4_decoupled_sampling_integration import probe_video_metadata  # noqa: E402
from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import (  # noqa: E402
    optical_axis_consensus_target,
)
from scripts.run_d1_three_view_canonical_accumulation import _sfm_result_from_payload  # noqa: E402

_DEFAULT_SFM_PYTHON = Path.home() / "anaconda3/envs/eia-pycolmap-probe/bin/python3.11"
_DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/eia-da3/bin/python3.11"
_DEFAULT_DA3_REPO = Path.home() / "Desktop/eia-da3-backends/depth-anything-3"
_DEFAULT_PRIOR_6B = Path("/tmp/eia_i4_1_6b_decoupled_dense_ab/i4_1_6b_summary.json")
_DEFAULT_OUTPUT = Path("/tmp/eia_i4_1_7_clean_static_orbit")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--prior-i4-1-6b-summary", type=Path, default=_DEFAULT_PRIOR_6B)
    p.add_argument("--confirm-static-capture", action="store_true")
    p.add_argument(
        "--qualification-plan",
        choices=("auto", "uniform_localized_dense_pool", "global_anchors_plus_local_neighbors"),
        default="auto",
    )
    p.add_argument("--sfm-python", type=Path, default=_DEFAULT_SFM_PYTHON)
    p.add_argument("--da3-python", type=Path, default=_DEFAULT_DA3_PYTHON)
    p.add_argument("--da3-repo", type=Path, default=_DEFAULT_DA3_REPO)
    p.add_argument("--profile", choices=("efficient", "balanced", "high_quality"), default="efficient")
    p.add_argument("--matching", choices=("sequential", "exhaustive"), default="exhaustive")
    p.add_argument("--max-keyframes", type=int, default=40)
    p.add_argument("--max-dense-views", type=int, default=40)
    p.add_argument("--video-target-fps", type=float, default=2.0)
    p.add_argument("--local-target-fps", type=float, default=2.0)
    p.add_argument("--dense-candidate-cap", type=int, default=120)
    p.add_argument("--process-res", type=int, default=392)
    p.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    p.add_argument("--da3-timeout-seconds", type=float, default=600.0)
    p.add_argument("--da3-total-timeout-seconds", type=float, default=7200.0)
    p.add_argument("--contact-sheet-frames", type=int, default=12)
    args = p.parse_args()
    if args.max_keyframes < 10 or args.max_dense_views < 10:
        p.error("I4.1.7 requires at least 10 global keyframes/dense views")
    if args.dense_candidate_cap < args.max_keyframes:
        p.error("--dense-candidate-cap must be >= --max-keyframes")
    if args.contact_sheet_frames < 6 or args.contact_sheet_frames > 24:
        p.error("--contact-sheet-frames must be in [6,24]")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.7 cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.1.7 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _run(command: list[str], *, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("\nI4.1.7 command: " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return int(process.wait())


def _make_contact_sheet(video: Path, output: Path, *, count: int) -> dict[str, Any]:
    total_frames, fps, duration = probe_video_metadata(video)
    indices = np.linspace(0, max(0, total_frames - 1), num=int(count), dtype=np.int64)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ReconstructionValidationError(f"I4.1.7 cannot open video: {video}")
    frames: list[np.ndarray] = []
    width = height = 0
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ReconstructionValidationError(f"I4.1.7 cannot decode contact-sheet frame {int(index)}.")
            height, width = frame.shape[:2]
            target_w = 320
            target_h = max(1, int(round(height * target_w / max(1, width))))
            small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cv2.putText(
                small,
                f"{int(index)}  {int(index) / fps:.1f}s",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frames.append(small)
    finally:
        capture.release()
    columns = 4
    rows = int(np.ceil(len(frames) / columns))
    tile_h, tile_w = frames[0].shape[:2]
    sheet = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
    for ordinal, frame in enumerate(frames):
        row, col = divmod(ordinal, columns)
        sheet[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = frame
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise ReconstructionValidationError(f"I4.1.7 cannot write contact sheet: {output}")
    return {
        "source_total_frames": int(total_frames),
        "source_fps": float(fps),
        "source_duration_seconds": float(duration),
        "source_width": int(width),
        "source_height": int(height),
        "contact_sheet_source_indices": [int(v) for v in indices],
        "contact_sheet": str(output.resolve()),
        "capture_advisory": {
            "duration_20_to_90_seconds": bool(20.0 <= duration <= 90.0),
            "resolution_at_least_720p": bool(width >= 1280 and height >= 720),
            "fps_at_least_20": bool(fps >= 20.0),
            "manual_static_object_review_required": True,
            "manual_no_cuts_review_required": True,
            "manual_motion_blur_review_required": True,
        },
    }


def _write_degraded(output: Path, *, stage: str, reason: str, details: Mapping[str, Any] | None = None) -> int:
    summary = {
        "phase": "I4.1.7",
        "schema_version": I4_1_7_SCHEMA_VERSION,
        "status": "degraded",
        "failed_stage": str(stage),
        "reason": str(reason),
        "details": dict(details or {}),
        "decision": {
            "i4_1_8_production_binding_authorized": False,
            "production_sampling_integration_authorized": False,
            "i4_2_detail_preservation_authorized": False,
            "mesh_authorization": False,
        },
        "authority": {
            "single_authoritative_sfm_frame": True,
            "production_default_changed": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
    }
    path = _write_json(output / "i4_1_7_summary.json", summary)
    print("\nI4.1.7 Clean Static Orbit Qualification: DEGRADED", flush=True)
    print(f"failed_stage={stage} reason={reason}", flush=True)
    print("i4_1_8_production_binding_authorized=False production_sampling_integration_authorized=False", flush=True)
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False", flush=True)
    print(f"summary={path}", flush=True)
    return 3


def _pose_maps(sfm_result: Any, *, a_dir: Path, selected_sources: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    camera_by_keyframe = {item.keyframe_id: item for item in sfm_result.cameras}
    global_by_source: dict[int, np.ndarray] = {}
    for workspace in sfm_result.workspace.images:
        camera = camera_by_keyframe.get(workspace.keyframe_id)
        if camera is not None:
            global_by_source[int(workspace.source_sequence_index)] = np.asarray(
                camera.T_reconstruction_from_camera, dtype=np.float64
            )
    localized: dict[int, np.ndarray] = {}
    for source_index in selected_sources:
        if source_index in global_by_source:
            continue
        evidence_path = a_dir / "localization_evidence" / f"candidate_{source_index:08d}.json"
        evidence = _read_json(evidence_path)
        if not bool(evidence.get("accepted")):
            raise ReconstructionValidationError(
                f"I4.1.7 selected extra source {source_index} lacks accepted localization evidence."
            )
        localized[source_index] = np.asarray(evidence.get("T_reconstruction_from_camera"), dtype=np.float64)
    return global_by_source, localized


def main() -> int:
    args = _args()
    video = args.video.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    prior_path = args.prior_i4_1_6b_summary.expanduser().resolve()
    if not video.is_file():
        raise ReconstructionValidationError(f"I4.1.7 video missing: {video}")
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.7 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    prior = _read_json(prior_path)
    prior_guardrails = validate_prior_decoupled_evidence(prior)
    capture = _make_contact_sheet(video, output / "00_capture_preflight/contact_sheet.jpg", count=int(args.contact_sheet_frames))
    _write_json(output / "00_capture_preflight/capture_preflight.json", capture)
    print("I4.1.7 prior I4.1.6B guardrails: PASS", flush=True)
    print(
        f"capture duration={capture['source_duration_seconds']:.3f}s "
        f"resolution={capture['source_width']}x{capture['source_height']} fps={capture['source_fps']:.3f}",
        flush=True,
    )
    print(f"contact_sheet={capture['contact_sheet']}", flush=True)
    if not args.confirm_static_capture:
        print("I4.1.7 note: --confirm-static-capture not supplied; automated work may run but final authorization will remain false.", flush=True)

    i1_dir = output / "01_i1_baseline"
    i1_command = [
        sys.executable, str(ROOT / "scripts/run_i1_offline_dense_campaign.py"),
        "--video", str(video),
        "--video-target-fps", str(args.video_target_fps),
        "--output-dir", str(i1_dir),
        "--sfm-python", str(args.sfm_python.expanduser().resolve()),
        "--da3-python", str(args.da3_python.expanduser().resolve()),
        "--da3-repo", str(args.da3_repo.expanduser().resolve()),
        "--profile", str(args.profile),
        "--matching", str(args.matching),
        "--max-keyframes", str(args.max_keyframes),
        "--max-dense-views", str(args.max_dense_views),
        "--process-res", str(args.process_res),
        "--sfm-timeout-minutes", str(args.sfm_timeout_minutes),
        "--da3-timeout-seconds", str(args.da3_timeout_seconds),
    ]
    i1_code = _run(i1_command, log=output / "logs/01_i1_baseline.log")
    i1_summary_path = i1_dir / "i1_campaign_summary.json"
    if not i1_summary_path.is_file():
        return _write_degraded(output, stage="global_i1_baseline", reason="I1 did not emit i1_campaign_summary.json.", details={"returncode": i1_code})
    i1_summary = _read_json(i1_summary_path)
    i1_status = str(i1_summary.get("status"))
    i1_failed_stage = str(i1_summary.get("failed_stage", ""))
    if i1_code != 0 and not (i1_status == "degraded" and i1_failed_stage == "dense_pair_qualification"):
        return _write_degraded(
            output,
            stage="global_i1_baseline",
            reason="Clean capture did not reach the dense-pair stage with at least ten aligned global observations.",
            details={"returncode": i1_code, "i1_status": i1_status, "i1_failed_stage": i1_failed_stage, "i1_reason": i1_summary.get("reason")},
        )
    print(
        "I4.1.7 global baseline usable: "
        + ("I1 PASS" if i1_status == "pass" else "global-only dense path DEGRADED; continuing decoupled qualification"),
        flush=True,
    )

    a_dir = output / "02_decoupled_localization"
    a_command = [
        sys.executable, str(ROOT / "scripts/run_i4_1_6_decoupled_sampling_integration.py"),
        "--i4-dir", str(output),
        "--video", str(video),
        "--output-dir", str(a_dir),
        "--plans", "uniform_localized_dense_pool", "global_anchors_plus_local_neighbors",
        "--dense-candidate-cap", str(args.dense_candidate_cap),
        "--local-target-fps", str(args.local_target_fps),
    ]
    a_code = _run(a_command, log=output / "logs/02_decoupled_localization.log")
    if a_code != 0:
        return _write_degraded(output, stage="decoupled_localization", reason="I4.1.6A localization did not authorize real dense A/B.", details={"returncode": a_code})

    b_dir = output / "03_real_dense_ab"
    b_command = [
        sys.executable, str(ROOT / "scripts/run_i4_1_6b_decoupled_dense_ab.py"),
        "--i4-dir", str(output),
        "--i4-1-6a-dir", str(a_dir),
        "--video", str(video),
        "--output-dir", str(b_dir),
        "--da3-python", str(args.da3_python.expanduser().resolve()),
        "--da3-repo", str(args.da3_repo.expanduser().resolve()),
        "--process-res", str(args.process_res),
        "--da3-total-timeout-seconds", str(args.da3_total_timeout_seconds),
        "--dense-candidate-cap", str(args.dense_candidate_cap),
        "--local-target-fps", str(args.local_target_fps),
    ]
    b_code = _run(b_command, log=output / "logs/03_real_dense_ab.log")
    if b_code != 0:
        return _write_degraded(output, stage="real_dense_ab", reason="I4.1.6B real dense A/B execution failed.", details={"returncode": b_code})
    b_summary = _read_json(b_dir / "i4_1_6b_summary.json")
    b_decision = b_summary.get("decision")
    if not isinstance(b_decision, Mapping):
        raise ReconstructionValidationError("I4.1.7 clean I4.1.6B decision is missing.")
    plan_name = str(b_decision.get("recommended_plan", "")) if args.qualification_plan == "auto" else str(args.qualification_plan)
    plan = selected_plan_payload(b_summary, plan_name)
    selected_sources = [int(v) for v in plan.get("selected_source_sequence_indices", [])]

    sfm_report_path = i1_dir / "01_sfm/real_sfm_report.json"
    sfm_report = _read_json(sfm_report_path)
    sfm_payload = sfm_report.get("sfm_result")
    if not isinstance(sfm_payload, dict):
        raise ReconstructionValidationError("I4.1.7 authoritative SfM report lacks sfm_result.")
    sfm_result = _sfm_result_from_payload(sfm_payload)
    if sfm_result.gauge.unit_to_metre is not None:
        raise ReconstructionValidationError("I4.1.7 refuses metric-scale promotion.")
    registered_matrices = np.asarray([item.T_reconstruction_from_camera for item in sfm_result.cameras], dtype=np.float64)
    target, target_diag = optical_axis_consensus_target(registered_matrices)
    global_pose_by_source, localized_pose_by_source = _pose_maps(
        sfm_result,
        a_dir=a_dir,
        selected_sources=selected_sources,
    )
    selected_matrices, selected_provenance = assemble_selected_pose_sequence(
        selected_sources,
        global_pose_by_source=global_pose_by_source,
        localized_pose_by_source=localized_pose_by_source,
    )
    assessment = build_clean_static_orbit_assessment(
        input_global_views=len(sfm_result.workspace.images),
        registered_global_views=len(sfm_result.cameras),
        registered_camera_matrices=registered_matrices,
        selected_camera_matrices=selected_matrices,
        selected_source_indices=selected_sources,
        selected_pose_provenance=selected_provenance,
        provisional_target_reconstruction=target,
        target_diagnostics=target_diag,
        plan_payload=plan,
        baseline_payload=b_summary.get("baseline_global_only", {}),
        prior_guardrails=prior_guardrails,
        capture_confirmed_static=bool(args.confirm_static_capture),
    )

    diagnostics = output / "i4_1_7_orbit_diagnostics.npz"
    np.savez_compressed(
        diagnostics,
        target_reconstruction=np.asarray(target, dtype=np.float64),
        registered_camera_matrices=registered_matrices,
        selected_camera_matrices=selected_matrices,
        selected_source_sequence_indices=np.asarray(selected_sources, dtype=np.int64),
        selected_pose_provenance=np.asarray(selected_provenance, dtype="U48"),
        schema_version=np.asarray(I4_1_7_SCHEMA_VERSION, dtype=np.int32),
    )
    passed = bool(assessment.get("pass"))
    summary = {
        "phase": "I4.1.7",
        "schema_version": I4_1_7_SCHEMA_VERSION,
        "status": "qualified_for_i4_1_8" if passed else "degraded",
        "source": {
            "video": str(video),
            **capture,
        },
        "prior_i4_1_6b": {
            "summary": str(prior_path),
            **prior_guardrails,
        },
        "global_i1": {
            "directory": str(i1_dir),
            "status": i1_status,
            "failed_stage": i1_failed_stage or None,
            "returncode": int(i1_code),
        },
        "decoupled_localization_dir": str(a_dir),
        "real_dense_ab_dir": str(b_dir),
        "qualification_plan": plan_name,
        "assessment": assessment,
        "artifacts": {
            "contact_sheet": capture["contact_sheet"],
            "orbit_diagnostics_npz": str(diagnostics.resolve()),
            "i4_1_6a_summary": str((a_dir / "i4_1_6a_summary.json").resolve()),
            "i4_1_6b_summary": str((b_dir / "i4_1_6b_summary.json").resolve()),
        },
        "decision": {
            "clean_static_orbit_qualified": passed,
            "i4_1_8_production_binding_authorized": passed,
            "production_sampling_integration_authorized": False,
            "production_default_changed": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "i4_2_detail_preservation_authorized": False,
            "mesh_authorization": False,
        },
        "authority": {
            "single_authoritative_sfm_frame": True,
            "provisional_orbit_target_only": True,
            "i4_1_8_core_target_requalification_required": True,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "execution": {
            "fresh_global_i1_executed": True,
            "fresh_i4_1_6a_localization_executed": True,
            "fresh_i4_1_6b_real_dense_ab_executed": True,
            "production_binding_executed": False,
            "fusion_of_decoupled_plan_executed": False,
            "filter_policy_changed": False,
            "persistence_changed": False,
        },
        "next": (
            "If i4_1_8_production_binding_authorized=True, bind the validated planner additively in I4.1.8, run bounded fusion/F/I2, recompute the orbit target from retained Core geometry, and close I4.1. "
            "If False, diagnose the clean capture/gate that failed; do not relax dense-consistency or orbit thresholds."
        ),
    }
    summary_path = _write_json(output / "i4_1_7_summary.json", summary)
    counts = assessment["counts"]
    selected_orbit = assessment.get("selected_orbit") or {}
    print("\nI4.1.7 Clean Static Orbit Qualification: " + ("QUALIFIED" if passed else "DEGRADED"), flush=True)
    print(
        f"plan={plan_name} registered={counts['registered_global_views']}/{counts['input_global_views']} "
        f"aligned={counts['dense_alignment_accepted_views']}/{counts['planned_dense_candidates']} "
        f"selected={counts['selected_qualified_views']} pair_pass_fraction={counts['pair_pass_fraction']}",
        flush=True,
    )
    if selected_orbit:
        print(
            f"selected_azimuth={selected_orbit['azimuth_coverage_degrees']:.1f}deg "
            f"bins={selected_orbit['occupied_azimuth_bins']}/12 "
            f"opposed={selected_orbit['maximum_opposed_view_angle_degrees']:.1f}deg "
            f"facing={selected_orbit['target_facing_fraction']:.3f} "
            f"standoff_p90_p10={selected_orbit['standoff_p90_to_p10_ratio']:.3f}",
            flush=True,
        )
    print("target_mode=provisional_optical_axis_consensus i4_1_8_core_target_requalification_required=True", flush=True)
    print(f"i4_1_8_production_binding_authorized={passed} production_sampling_integration_authorized=False", flush=True)
    print("dense_consistency_threshold_relaxation_authorized=False i4_2_detail_preservation_authorized=False mesh_authorization=False", flush=True)
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
