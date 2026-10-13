#!/usr/bin/env python3
"""Run a real non-UI Phase 8.3D PyCOLMAP sparse reconstruction smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    BackendOperationContext,
    ExternalPycolmapSfmBackend,
    FrameQualityPolicy,
    FrameSamplingMode,
    FrameSamplingPolicy,
    ImageSequenceReader,
    IngestionCachePolicy,
    KeyframeSelectionPolicy,
    OfflineSfmConfiguration,
    OfflineSfmRequest,
    RecordedVideoReader,
    ReconstructionCancelledError,
    ReconstructionIngestionPipeline,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    SfmMatchingStrategy,
    SfmResumePolicy,
    SfmScaleStatus,
    materialize_sfm_workspace,
    new_reconstruction_session_id,
)

_MIB = 1024**2


def _write_sparse_ply(result, path: Path) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        "comment Phase 8.3D real PyCOLMAP sparse reconstruction",
        "comment arbitrary reconstruction gauge; coordinates are NOT metres",
        f"element vertex {len(result.landmarks)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for landmark in result.landmarks:
        x, y, z = landmark.position_reconstruction
        r, g, b = landmark.color_rgb or (180, 180, 180)
        lines.append(f"{x:.9g} {y:.9g} {z:.9g} {r} {g} {b}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _build_source(args: argparse.Namespace):
    if args.images is not None:
        return ImageSequenceReader.from_directory(
            args.images.expanduser().resolve(),
            recursive=args.recursive,
            frame_rate_hz=args.image_sequence_fps,
        ), FrameSamplingPolicy(
            mode=FrameSamplingMode.EVERY_FRAME,
            maximum_sampled_frames=args.max_keyframes,
        )

    source = RecordedVideoReader(args.video.expanduser().resolve())
    return source, FrameSamplingPolicy(
        mode=FrameSamplingMode.TARGET_FPS,
        target_fps_hz=args.video_target_fps,
        maximum_sampled_frames=args.max_keyframes,
    )


def _print_event(event) -> None:
    if event.progress_fraction is not None:
        print(
            f"[{event.progress_fraction * 100:5.1f}%] {event.message}",
            flush=True,
        )
    elif event.message:
        print(event.message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--images", type=Path, help="Directory of overlapping images.")
    source_group.add_argument("--video", type=Path, help="Recorded video to reconstruct.")
    parser.add_argument(
        "--sfm-python",
        required=True,
        help="Python executable from the isolated PyCOLMAP 4.1.1 environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_3d_real_sfm"),
    )
    parser.add_argument(
        "--profile",
        choices=[item.value for item in ReconstructionQualityProfile],
        default=ReconstructionQualityProfile.EFFICIENT.value,
    )
    parser.add_argument(
        "--matching",
        choices=[item.value for item in SfmMatchingStrategy],
        default=SfmMatchingStrategy.SEQUENTIAL.value,
        help="Use sequential for ordered sequences/video; exhaustive for small unordered sets.",
    )
    parser.add_argument("--max-keyframes", type=int, default=80)
    parser.add_argument("--maximum-image-size", type=int, default=None)
    parser.add_argument("--sequential-overlap", type=int, default=None)
    parser.add_argument("--video-target-fps", type=float, default=2.0)
    parser.add_argument("--image-sequence-fps", type=float, default=10.0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA SIFT/matching.")
    parser.add_argument("--timeout-minutes", type=float, default=60.0)
    parser.add_argument(
        "--resume-policy",
        choices=[item.value for item in SfmResumePolicy],
        default=SfmResumePolicy.REUSE_COMPLETED_STAGES.value,
    )
    parser.add_argument(
        "--exercise-resume",
        action="store_true",
        help=(
            "After matching completes, cancel the first worker attempt and rerun the same "
            "request to validate feature/match checkpoint reuse."
        ),
    )
    args = parser.parse_args()

    if args.max_keyframes < 3:
        parser.error("--max-keyframes must be at least 3 for real SfM.")
    if args.maximum_image_size is not None and args.maximum_image_size <= 0:
        parser.error("--maximum-image-size must be positive.")
    if args.sequential_overlap is not None and args.sequential_overlap <= 0:
        parser.error("--sequential-overlap must be positive.")
    if args.timeout_minutes <= 0.0:
        parser.error("--timeout-minutes must be positive.")
    if args.exercise_resume and args.resume_policy == SfmResumePolicy.DISABLED.value:
        parser.error("--exercise-resume requires checkpoint reuse to be enabled.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source, sampling_policy = _build_source(args)
    session_id = new_reconstruction_session_id()

    ingestion = ReconstructionIngestionPipeline(
        session_id=session_id,
        source=source,
        sampling_policy=sampling_policy,
        keyframe_policy=KeyframeSelectionPolicy(
            minimum_spacing_ns=1,
            forced_spacing_ns=1,
            minimum_visual_change=0.0,
            minimum_quality_score=0.0,
            maximum_keyframes=args.max_keyframes,
        ),
        quality_policy=FrameQualityPolicy(),
        cache_policy=IngestionCachePolicy(
            maximum_cpu_bytes=512 * _MIB,
            maximum_frames=min(8, args.max_keyframes),
            gpu_staging_bytes=0,
        ),
    ).run()
    if len(ingestion.keyframes) < 3:
        raise RuntimeError(
            "Fewer than three valid keyframes survived ingestion. "
            "Use a sharper, better exposed, overlapping sequence."
        )

    sampled_records = [record for record in ingestion.records if record.sampled]
    sampled_indices = [record.frame.sequence_index for record in sampled_records]
    sampled_timestamps = [record.frame.timestamp_ns for record in sampled_records]
    estimated_total = ingestion.sampling_plan.estimated_total_frames
    source_index_denominator = max(1, int(estimated_total or 1) - 1)
    actual_sampling = {
        "sampled_frame_count": len(sampled_records),
        "first_source_sequence_index": sampled_indices[0] if sampled_indices else None,
        "last_source_sequence_index": sampled_indices[-1] if sampled_indices else None,
        "source_sequence_span_fraction": (
            (sampled_indices[-1] - sampled_indices[0]) / source_index_denominator
            if len(sampled_indices) >= 2 and estimated_total is not None and estimated_total > 1
            else None
        ),
        "sampled_timestamp_span_seconds": (
            (sampled_timestamps[-1] - sampled_timestamps[0]) / 1_000_000_000.0
            if len(sampled_timestamps) >= 2
            else 0.0
        ),
    }

    workspace = materialize_sfm_workspace(
        ingestion,
        output_dir / "workspace",
        overwrite=True,
    )
    profile = ReconstructionQualityProfile(args.profile)
    request = OfflineSfmRequest(
        session_id=session_id,
        profile=profile,
        workspace=workspace,
        configuration=OfflineSfmConfiguration(
            matching_strategy=SfmMatchingStrategy(args.matching),
            use_gpu=not args.cpu,
            maximum_image_size=args.maximum_image_size,
            sequential_overlap=args.sequential_overlap,
            resume_policy=SfmResumePolicy(args.resume_policy),
        ),
    )

    backend = ExternalPycolmapSfmBackend(
        args.sfm_python,
        timeout_s=args.timeout_minutes * 60.0,
    )
    events = []
    resume_exercised = False
    first_attempt_cancelled = False

    try:
        if args.exercise_resume:
            first_control = ReconstructionOperationControl()
            first_events = []

            def on_first_event(event) -> None:
                first_events.append(event)
                _print_event(event)
                if (
                    event.progress_fraction is not None
                    and event.progress_fraction >= 0.60
                    and not first_control.cancellation_requested
                ):
                    first_control.request_cancel()

            try:
                backend.run_sparse_reconstruction(
                    request,
                    BackendOperationContext(
                        session_id=session_id,
                        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
                        profile=profile,
                        control=first_control,
                        emit_event=on_first_event,
                    ),
                )
            except ReconstructionCancelledError:
                first_attempt_cancelled = True
                print(
                    "First SfM attempt cancelled after matching; starting resume validation.",
                    flush=True,
                )
            if not first_attempt_cancelled:
                raise RuntimeError(
                    "Resume exercise did not cancel the first worker attempt as intended."
                )
            resume_exercised = True

        final_control = ReconstructionOperationControl()

        def on_final_event(event) -> None:
            events.append(event)
            _print_event(event)

        result = backend.run_sparse_reconstruction(
            request,
            BackendOperationContext(
                session_id=session_id,
                stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
                profile=profile,
                control=final_control,
                emit_event=on_final_event,
            ),
        )
        probe = backend.environment_probe
    finally:
        backend.shutdown()

    result_payload = result.as_dict()
    metadata = result_payload["metadata"]
    quality = metadata.get("quality_assessment", {})
    registration_ratio = result.statistics.registered_keyframes / len(workspace.images)
    report = {
        "phase": "8.3D",
        "real_sfm_executed": True,
        "resume_exercised": resume_exercised,
        "first_attempt_cancelled": first_attempt_cancelled,
        "source": ingestion.source.as_dict(),
        "ingestion_summary": ingestion.as_dict()["summary"],
        "sampling_plan": ingestion.sampling_plan.as_dict(),
        "sampling_execution": actual_sampling,
        "external_pycolmap_probe": None if probe is None else probe.as_dict(),
        "registration_ratio": registration_ratio,
        "quality_assessment": quality,
        "event_count": len(events),
        "events": [event.as_dict() for event in events],
        "sfm_result": result_payload,
    }
    report_path = output_dir / "real_sfm_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_path = output_dir / "real_sfm_summary.json"
    summary = {
        "phase": "8.3D",
        "real_sfm_executed": True,
        "source_kind": ingestion.source.kind.value,
        "input_keyframes": len(workspace.images),
        "registered_keyframes": result.statistics.registered_keyframes,
        "registration_ratio": registration_ratio,
        "camera_models": len(result.camera_models),
        "landmarks": result.statistics.landmarks,
        "observations": result.statistics.observations,
        "mean_reprojection_error_px": result.statistics.mean_reprojection_error_px,
        "quality_assessment": quality,
        "gauge": result.gauge.as_dict(),
        "warnings": list(result.warnings),
        "matching_strategy": metadata.get("matching_strategy"),
        "resume_policy": metadata.get("resume_policy"),
        "resume_from_stage": metadata.get("resume_from_stage"),
        "resume_exercised": resume_exercised,
        "worker_request_fingerprint": metadata.get("worker_request_fingerprint"),
        "selected_model_id": metadata.get("selected_model_id"),
        "candidate_models": metadata.get("candidate_models", []),
        "resolved_configuration": metadata.get("resolved_configuration", {}),
        "pycolmap_version": metadata.get("pycolmap_version"),
        "external_cuda_devices": metadata.get("external_cuda_devices"),
        "sampling_plan": ingestion.sampling_plan.as_dict(),
        "sampling_execution": actual_sampling,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ply_path = output_dir / "real_sparse_arbitrary_scale.ply"
    _write_sparse_ply(result, ply_path)

    resume_ok = True
    if args.exercise_resume:
        resume_ok = metadata.get("resume_from_stage") == "matching"
    passed = (
        metadata.get("real_sfm") is True
        and result.gauge.scale_status is SfmScaleStatus.ARBITRARY
        and result.gauge.unit_to_metre is None
        and result.statistics.registered_keyframes >= 3
        and result.statistics.landmarks > 0
        and len(result.camera_models) > 0
        and quality.get("status") != "rejected"
        and resume_ok
    )

    print()
    print(f"Phase 8.3D real PyCOLMAP smoke: {'PASS' if passed else 'FAIL'}")
    print(
        f"input_keyframes={len(workspace.images)} "
        f"registered={result.statistics.registered_keyframes} "
        f"registration_ratio={registration_ratio:.1%}"
    )
    print(
        f"camera_models={len(result.camera_models)} "
        f"landmarks={result.statistics.landmarks} "
        f"observations={result.statistics.observations} "
        f"mean_reprojection_error_px={result.statistics.mean_reprojection_error_px}"
    )
    print(
        f"quality_status={quality.get('status')} "
        f"mean_track_length={quality.get('mean_track_length')}"
    )
    print(
        f"gauge={result.gauge.scale_status.value} "
        f"unit_to_metre={result.gauge.unit_to_metre}"
    )
    print(
        f"matching={metadata.get('matching_strategy')} "
        f"resume_from_stage={metadata.get('resume_from_stage')} "
        f"resume_exercised={resume_exercised}"
    )
    print(
        f"sampling={ingestion.sampling_plan.mode.value} "
        f"requested_fps={ingestion.sampling_plan.requested_target_fps_hz} "
        f"effective_fps={ingestion.sampling_plan.effective_target_fps_hz} "
        f"sampled={actual_sampling['sampled_frame_count']} "
        f"source_span_fraction={actual_sampling['source_sequence_span_fraction']}"
    )
    print("real_sfm_executed=True")
    print(f"workspace={workspace.workspace_root}")
    print(f"report={report_path}")
    print(f"summary={summary_path}")
    print(f"sparse_ply={ply_path}")
    print(
        "NOTE: sparse coordinates are arbitrary-scale monocular SfM coordinates, not metres."
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
