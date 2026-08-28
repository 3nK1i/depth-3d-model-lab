#!/usr/bin/env python3
"""Production Phase 8.8A3 offline reconstruction worker.

This worker accepts real image-sequence or recorded-video input from the
application controller. It runs production ingestion in ``depth`` and launches
PyCOLMAP only through the qualified isolated interpreter. It intentionally stops
at sparse camera reconstruction. D1 additionally emits a bounded visualization
companion artifact; sparse SfM still remains partial/non-canonical geometry.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any

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
    ReconstructionEventRecorder,
    ReconstructionIngestionPipeline,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    SfmMatchingStrategy,
    SfmResumePolicy,
    SfmScaleStatus,
    materialize_sfm_workspace,
)
from core.hardware.gpu_info import detect_hardware
from core.reconstruction.adaptive_production_planning import build_adaptive_pre_sfm_frame_plan
from core.reconstruction.adaptive_view_selection import (
    AdaptiveSelectionResourceSnapshot,
    provider_view_selection_requirements,
)
from core.reconstruction.dense_provider_architecture import dense_provider_implementation_contracts
from core.reconstruction.errors import ReconstructionValidationError

PROTOCOL = "eia.reconstruction.offline.v1"
OFFLINE_VISUALIZATION_SCHEMA = "eia.reconstruction.offline.visualization.v1"
OFFLINE_VISUALIZATION_LANDMARK_CAP = 8_192
_MIB = 1024**2


def _emit(event: str, *, stage: str, message: str, progress_fraction: float | None = None, **extra: Any) -> None:
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "event": event,
        "stage": stage,
        "message": message,
    }
    if progress_fraction is not None:
        payload["progress_fraction"] = float(progress_fraction)
    payload.update(extra)
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def _write_sparse_ply(result, path: Path) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        "comment EIA TECH Phase 8.8A3 real PyCOLMAP sparse reconstruction",
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




def _sample_indices(count: int, cap: int) -> tuple[int, ...]:
    """Return deterministic first/last-preserving indices for a bounded artifact."""

    count = int(count)
    cap = int(cap)
    if count <= 0:
        return ()
    if cap < 2:
        raise ValueError("Visualization sample cap must be at least 2.")
    if count <= cap:
        return tuple(range(count))
    scale = float(count - 1) / float(cap - 1)
    indices = tuple(int(round(index * scale)) for index in range(cap))
    # scale > 1 when decimating, so rounding remains strictly increasing.
    if len(indices) != len(set(indices)) or indices[0] != 0 or indices[-1] != count - 1:
        raise RuntimeError("Offline visualization sampling failed to preserve bounded endpoints.")
    return indices


def _write_offline_visualization(result, path: Path) -> None:
    """Write a bounded, dependency-neutral companion artifact for the D1 UI."""

    indices = _sample_indices(len(result.landmarks), OFFLINE_VISUALIZATION_LANDMARK_CAP)
    points: list[list[float]] = []
    colors: list[list[int]] = []
    for index in indices:
        landmark = result.landmarks[index]
        points.append([float(value) for value in landmark.position_reconstruction])
        color = landmark.color_rgb or (184, 199, 214)
        colors.append([int(value) for value in color])

    cameras = [
        {
            "estimate_id": camera.estimate_id,
            "keyframe_id": camera.keyframe_id,
            "frame_id": camera.frame_id,
            "T_reconstruction_from_camera": camera.T_reconstruction_from_camera.tolist(),
        }
        for camera in result.cameras
    ]
    payload = {
        "schema": OFFLINE_VISUALIZATION_SCHEMA,
        "session_id": result.session_id,
        "backend_id": result.backend_id,
        "backend_version": result.backend_version,
        "pose_convention": "T_reconstruction_from_camera",
        "scale_status": result.gauge.scale_status.value,
        "alignment_status": result.gauge.alignment_status.value,
        "registered_camera_total": len(result.cameras),
        "registered_cameras": cameras,
        "landmark_total": len(result.landmarks),
        "landmarks_reconstruction": points,
        "landmark_colors_rgb": colors,
        "mean_reprojection_error_px": result.statistics.mean_reprojection_error_px,
        "certified_metrology": False,
        "canonical_geometry_authoritative": False,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images-dir", type=Path)
    source.add_argument("--video", type=Path)
    source.add_argument("--image", action="append", type=Path, dest="images")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sfm-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=[item.value for item in ReconstructionQualityProfile],
        default=ReconstructionQualityProfile.EFFICIENT.value,
    )
    parser.add_argument(
        "--matching",
        choices=[item.value for item in SfmMatchingStrategy],
        default=SfmMatchingStrategy.SEQUENTIAL.value,
    )
    parser.add_argument("--max-keyframes", type=int, default=None)
    parser.add_argument("--maximum-image-size", type=int, default=None)
    parser.add_argument("--sequential-overlap", type=int, default=None)
    parser.add_argument("--video-target-fps", type=float, default=2.0)
    parser.add_argument("--image-sequence-fps", type=float, default=10.0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--timeout-minutes", type=float, default=60.0)
    args = parser.parse_args()
    if args.max_keyframes is not None and args.max_keyframes < 3:
        parser.error("--max-keyframes must be at least 3 when provided")
    if args.maximum_image_size is not None and args.maximum_image_size <= 0:
        parser.error("--maximum-image-size must be positive")
    if args.sequential_overlap is not None and args.sequential_overlap <= 0:
        parser.error("--sequential-overlap must be positive")
    if args.video_target_fps <= 0.0 or args.image_sequence_fps <= 0.0:
        parser.error("input frame rates must be positive")
    if args.timeout_minutes <= 0.0:
        parser.error("--timeout-minutes must be positive")
    return args


def _available_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _adaptive_resources(*, cpu_only: bool) -> AdaptiveSelectionResourceSnapshot:
    hardware = detect_hardware(0)
    return AdaptiveSelectionResourceSnapshot(
        gpu_name=hardware.name,
        total_vram_bytes=hardware.total_memory_bytes,
        free_vram_bytes=hardware.free_memory_bytes,
        available_ram_bytes=_available_ram_bytes(),
        cuda_ready=bool(hardware.cuda_ready and not cpu_only),
    )


def _build_source(args: argparse.Namespace, resources: AdaptiveSelectionResourceSnapshot):
    if args.images_dir is not None:
        source = ImageSequenceReader.from_directory(
            args.images_dir,
            recursive=bool(args.recursive),
            frame_rate_hz=float(args.image_sequence_fps),
        )
        source_kind = "image_directory"
    elif args.images:
        source = ImageSequenceReader.from_paths(
            args.images,
            frame_rate_hz=float(args.image_sequence_fps),
            preserve_order=True,
        )
        source_kind = "image_paths"
    else:
        source = RecordedVideoReader(args.video)
        source_kind = "recorded_video"

    pre_plan = build_adaptive_pre_sfm_frame_plan(
        source_kind=source_kind,
        estimated_total_frames=source.estimated_total_frames,
        expected_frame_rate_hz=source.metadata.expected_frame_rate_hz,
        resources=resources,
        engineer_max_keyframes_override=args.max_keyframes,
    )
    target = pre_plan.target_sampled_frames
    total = source.estimated_total_frames
    reduced_known_source = target is not None and total is not None and target < total
    if source_kind == "recorded_video":
        sampling = FrameSamplingPolicy(
            mode=FrameSamplingMode.TARGET_FPS,
            target_fps_hz=float(args.video_target_fps),
            maximum_sampled_frames=target,
        )
    elif reduced_known_source:
        # TARGET_FPS + a finite cap activates the existing deterministic
        # first/last-preserving full-span sampler rather than a prefix cut.
        sampling = FrameSamplingPolicy(
            mode=FrameSamplingMode.TARGET_FPS,
            target_fps_hz=float(args.image_sequence_fps),
            maximum_sampled_frames=target,
        )
    else:
        sampling = FrameSamplingPolicy(
            mode=FrameSamplingMode.EVERY_FRAME,
            maximum_sampled_frames=None,
        )
    return source, sampling, source_kind, pre_plan


def _write_result(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "offline_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Isolate this application worker and every backend child in one process
    # group. ReconstructionController can then hard-kill the complete group if
    # cooperative SIGTERM cancellation ever exceeds its bounded grace window.
    if hasattr(os, "setpgrp"):
        try:
            os.setpgrp()
        except OSError:
            pass

    control = ReconstructionOperationControl()
    cancellation_signal = {"value": ""}

    def request_cancel(signum, _frame) -> None:
        cancellation_signal["value"] = signal.Signals(signum).name
        control.request_cancel()
        _emit(
            "status",
            stage="cancellation",
            message="Offline reconstruction cancellation requested.",
            cancellation_signal=cancellation_signal["value"],
        )

    signal.signal(signal.SIGTERM, request_cancel)
    signal.signal(signal.SIGINT, request_cancel)

    backend = None
    source_kind = ""
    try:
        _emit("status", stage="source", message="Validating offline reconstruction source.")
        resources = _adaptive_resources(cpu_only=bool(args.cpu))
        source, sampling_policy, source_kind, pre_sfm_plan = _build_source(args, resources)
        profile = ReconstructionQualityProfile(args.profile)
        pre_sfm_path = output_dir / "adaptive_pre_sfm_plan.json"
        pre_sfm_path.write_text(
            json.dumps(pre_sfm_plan.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _emit(
            "status",
            stage="source_planning",
            message="Adaptive pre-SfM frame plan resolved.",
            adaptive_pre_sfm_plan=pre_sfm_plan.as_dict(),
        )
        events = ReconstructionEventRecorder(args.session_id)
        _emit("progress", stage="source_ingestion", message="Offline ingestion started.", progress_fraction=0.0)
        ingestion = ReconstructionIngestionPipeline(
            session_id=args.session_id,
            source=source,
            sampling_policy=sampling_policy,
            keyframe_policy=KeyframeSelectionPolicy(
                minimum_spacing_ns=1,
                forced_spacing_ns=1,
                minimum_visual_change=0.0,
                minimum_quality_score=0.0,
                maximum_keyframes=pre_sfm_plan.target_sampled_frames,
            ),
            quality_policy=FrameQualityPolicy(),
            cache_policy=IngestionCachePolicy(
                maximum_cpu_bytes=512 * _MIB,
                maximum_frames=8,
                gpu_staging_bytes=0,
            ),
        ).run(control=control, events=events)
        if len(ingestion.keyframes) < 3:
            raise ReconstructionValidationError(
                "Fewer than three valid keyframes survived ingestion. Use a sharper, overlapping sequence."
            )
        ingestion_path = output_dir / "ingestion_result.json"
        ingestion_path.write_text(
            json.dumps(ingestion.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _emit(
            "progress",
            stage="source_ingestion",
            message=f"Selected {len(ingestion.keyframes)} reconstruction keyframes.",
            progress_fraction=1.0,
            keyframes=len(ingestion.keyframes),
        )

        control.checkpoint()
        _emit("progress", stage="camera_pose", message="Preparing sparse reconstruction workspace.", progress_fraction=0.0)
        workspace = materialize_sfm_workspace(
            ingestion,
            output_dir / "sfm_workspace",
            overwrite=True,
        )
        request = OfflineSfmRequest(
            session_id=args.session_id,
            profile=profile,
            workspace=workspace,
            configuration=OfflineSfmConfiguration(
                matching_strategy=SfmMatchingStrategy(args.matching),
                use_gpu=not bool(args.cpu),
                maximum_image_size=args.maximum_image_size,
                sequential_overlap=args.sequential_overlap,
                resume_policy=SfmResumePolicy.REUSE_COMPLETED_STAGES,
            ),
        )
        backend = ExternalPycolmapSfmBackend(
            str(args.sfm_python.expanduser().resolve()),
            timeout_s=float(args.timeout_minutes) * 60.0,
        )

        def on_sfm_event(event) -> None:
            payload = event.as_dict()
            _emit(
                "progress" if event.progress_fraction is not None else "status",
                stage="camera_pose",
                message=event.message,
                progress_fraction=event.progress_fraction,
                reconstruction_event=payload,
            )

        sfm_result = backend.run_sparse_reconstruction(
            request,
            BackendOperationContext(
                session_id=args.session_id,
                stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
                profile=profile,
                control=control,
                emit_event=on_sfm_event,
            ),
        )
        sfm_payload = sfm_result.as_dict()
        sfm_result_path = output_dir / "sfm_result.json"
        sfm_result_path.write_text(
            json.dumps(sfm_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        adaptive_dense_readiness = {
            "schema_version": 1,
            "phase": "I4.3F2",
            "status": "ready",
            "registered_view_count": sfm_result.statistics.registered_keyframes,
            "provider_selected": False,
            "auto_policy_implemented": False,
            "provider_execution_started": False,
            "permanent_global_view_cap": None,
            "resources": resources.as_dict(),
            "provider_requirements": [
                provider_view_selection_requirements(item.implementation).as_dict()
                for item in dense_provider_implementation_contracts()
            ],
            "authority": {
                "camera_pose_authority_owner": "pycolmap_sfm",
                "authoritative_sfm_modified": False,
                "metric_scale_claimed": False,
                "measurement_authoritative": False,
                "certified_metrology": False,
                "surface_mesh_generated": False,
            },
        }
        adaptive_dense_path = output_dir / "adaptive_dense_selection_ready.json"
        adaptive_dense_path.write_text(
            json.dumps(adaptive_dense_readiness, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        sparse_ply = output_dir / "sparse_reconstruction.ply"
        _write_sparse_ply(sfm_result, sparse_ply)
        offline_visualization = output_dir / "offline_visualization.json"
        _write_offline_visualization(sfm_result, offline_visualization)
        metadata = sfm_payload.get("metadata", {})
        quality = metadata.get("quality_assessment", {}) if isinstance(metadata, dict) else {}
        registration_ratio = sfm_result.statistics.registered_keyframes / len(workspace.images)
        passed = bool(
            sfm_result.statistics.registered_keyframes >= 3
            and sfm_result.statistics.landmarks > 0
            and len(sfm_result.camera_models) > 0
            and sfm_result.gauge.scale_status is SfmScaleStatus.ARBITRARY
            and sfm_result.gauge.unit_to_metre is None
            and (not isinstance(quality, dict) or quality.get("status") != "rejected")
        )
        result = {
            "phase": "8.8A3",
            "pass": passed,
            "status": "completed" if passed else "failed_quality_gate",
            "session_id": args.session_id,
            "production_input_assembly": True,
            "tests_fixture_used": False,
            "source_kind": source_kind,
            "source": ingestion.source.as_dict(),
            "ingestion_summary": ingestion.as_dict()["summary"],
            "adaptive_pre_sfm_plan": pre_sfm_plan.as_dict(),
            "sfm_summary": {
                "backend_id": sfm_result.backend_id,
                "backend_version": sfm_result.backend_version,
                "input_keyframes": len(workspace.images),
                "registered_keyframes": sfm_result.statistics.registered_keyframes,
                "registration_ratio": registration_ratio,
                "camera_models": len(sfm_result.camera_models),
                "landmarks": sfm_result.statistics.landmarks,
                "observations": sfm_result.statistics.observations,
                "mean_reprojection_error_px": sfm_result.statistics.mean_reprojection_error_px,
                "quality_assessment": quality,
                "gauge": sfm_result.gauge.as_dict(),
            },
            "artifacts": {
                "ingestion_result_json": str(ingestion_path.resolve()),
                "adaptive_pre_sfm_plan_json": str(pre_sfm_path.resolve()),
                "adaptive_dense_selection_ready_json": str(adaptive_dense_path.resolve()),
                "sfm_result_json": str(sfm_result_path.resolve()),
                "sparse_reconstruction_ply": str(sparse_ply.resolve()),
                "offline_visualization_json": str(offline_visualization.resolve()),
                "sfm_workspace": str(Path(workspace.workspace_root).resolve()),
            },
            "execution_boundary": {
                "application_worker_python": str(Path(sys.executable).resolve()),
                "pycolmap_python": str(args.sfm_python.expanduser().resolve()),
                "pycolmap_external_process": True,
                "production_depth_imports_pycolmap": False,
            },
            "camera_pose_available": True,
            "canonical_geometry_authoritative": False,
            "canonical_geometry_deferred_to": "subsequent integrated offline stages",
            "adaptive_dense_provider_wiring_ready": True,
            "permanent_global_view_cap": None,
            "persistent_scene_ready": False,
            "persistent_scene_checkpoint_bridge_deferred_to": "8.8A4",
            "certified_metrology": False,
            "reconstruction_ui_enabled": False,
        }
        _write_result(output_dir, result)
        _emit(
            "completed" if passed else "failed",
            stage="camera_pose",
            message="Offline reconstruction execution bridge completed." if passed else "Offline reconstruction quality gate failed.",
            progress_fraction=1.0,
            session_id=args.session_id,
        )
        return 0 if passed else 2
    except ReconstructionCancelledError:
        result = {
            "phase": "8.8A3",
            "pass": False,
            "status": "cancelled",
            "session_id": args.session_id,
            "source_kind": source_kind,
            "cancellation_signal": cancellation_signal["value"],
            "tests_fixture_used": False,
            "certified_metrology": False,
            "reconstruction_ui_enabled": False,
        }
        _write_result(output_dir, result)
        _emit("cancelled", stage="cancellation", message="Offline reconstruction cancelled.")
        return 130
    except Exception as exc:
        result = {
            "phase": "8.8A3",
            "pass": False,
            "status": "failed",
            "session_id": args.session_id,
            "source_kind": source_kind,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "tests_fixture_used": False,
            "certified_metrology": False,
            "reconstruction_ui_enabled": False,
        }
        _write_result(output_dir, result)
        _emit("failed", stage="offline", message=str(exc), error_type=type(exc).__name__)
        return 1
    finally:
        if backend is not None:
            backend.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
