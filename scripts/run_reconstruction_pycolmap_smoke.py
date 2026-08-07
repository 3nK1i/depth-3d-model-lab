#!/usr/bin/env python3
"""Run a real non-UI Phase 8.3C PyCOLMAP sparse reconstruction smoke."""

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
    ReconstructionIngestionPipeline,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    SfmMatchingStrategy,
    SfmScaleStatus,
    materialize_sfm_workspace,
    new_reconstruction_session_id,
)

_MIB = 1024**2


def _write_sparse_ply(result, path: Path) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        "comment EIA TECH Phase 8.3C real PyCOLMAP sparse reconstruction",
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
        default=Path("/tmp/phase8_3c_real_sfm"),
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
    parser.add_argument("--video-target-fps", type=float, default=2.0)
    parser.add_argument("--image-sequence-fps", type=float, default=10.0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA SIFT/matching.")
    parser.add_argument("--timeout-minutes", type=float, default=60.0)
    args = parser.parse_args()

    if args.max_keyframes < 3:
        parser.error("--max-keyframes must be at least 3 for real SfM.")
    if args.maximum_image_size is not None and args.maximum_image_size <= 0:
        parser.error("--maximum-image-size must be positive.")
    if args.timeout_minutes <= 0.0:
        parser.error("--timeout-minutes must be positive.")

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
        ),
    )

    events = []

    def on_event(event) -> None:
        events.append(event)
        if event.progress_fraction is not None:
            print(
                f"[{event.progress_fraction * 100:5.1f}%] "
                f"{event.message}",
                flush=True,
            )
        elif event.message:
            print(event.message, flush=True)

    context = BackendOperationContext(
        session_id=session_id,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=profile,
        control=ReconstructionOperationControl(),
        emit_event=on_event,
    )
    backend = ExternalPycolmapSfmBackend(
        args.sfm_python,
        timeout_s=args.timeout_minutes * 60.0,
    )
    try:
        result = backend.run_sparse_reconstruction(request, context)
        probe = backend.environment_probe
    finally:
        backend.shutdown()

    result_payload = result.as_dict()
    registration_ratio = result.statistics.registered_keyframes / len(workspace.images)
    report = {
        "phase": "8.3C",
        "real_sfm_executed": True,
        "source": ingestion.source.as_dict(),
        "ingestion_summary": ingestion.as_dict()["summary"],
        "external_pycolmap_probe": None if probe is None else probe.as_dict(),
        "registration_ratio": registration_ratio,
        "event_count": len(events),
        "events": [event.as_dict() for event in events],
        "sfm_result": result_payload,
    }
    report_path = output_dir / "real_sfm_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ply_path = output_dir / "real_sparse_arbitrary_scale.ply"
    _write_sparse_ply(result, ply_path)

    passed = (
        result_payload["metadata"].get("real_sfm") is True
        and result.gauge.scale_status is SfmScaleStatus.ARBITRARY
        and result.gauge.unit_to_metre is None
        and result.statistics.registered_keyframes >= 3
        and result.statistics.landmarks > 0
        and len(result.camera_models) > 0
    )

    print()
    print(f"Phase 8.3C real PyCOLMAP smoke: {'PASS' if passed else 'FAIL'}")
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
        f"gauge={result.gauge.scale_status.value} "
        f"unit_to_metre={result.gauge.unit_to_metre}"
    )
    print("real_sfm_executed=True")
    print(f"workspace={workspace.workspace_root}")
    print(f"report={report_path}")
    print(f"sparse_ply={ply_path}")
    print(
        "NOTE: sparse coordinates are arbitrary-scale monocular SfM coordinates, not metres."
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
