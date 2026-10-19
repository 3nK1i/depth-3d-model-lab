#!/usr/bin/env python3
"""B2 real single-keyframe PyCOLMAP ↔ dense-depth alignment validation.

This validation intentionally stops before RGB-D projection and multi-view
fusion. It runs a real offline PyCOLMAP reconstruction, selects one registered
keyframe with strong sparse support, obtains one real DA3 dense-depth prediction
through the isolated ``depth3d-da3`` runtime, adapts that depth map back to the SfM
keyframe pixel grid when DA3 inference resolution differs, and executes the B1
scale-only alignment bridge.

The result remains in the arbitrary offline-SfM reconstruction gauge. This
script does not establish physical scale, canonical geometry authority,
measurement authority, or certified metrology.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
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
    SfmResumePolicy,
    SfmScaleStatus,
    align_sfm_keyframe_dense_depth,
    materialize_sfm_workspace,
    new_reconstruction_session_id,
)
from core.reconstruction.da3 import (  # noqa: E402
    DA3_EFFICIENT_MODEL_ID,
    probe_da3_environment,
    validate_da3_npz_result,
)
from core.reconstruction.da3_provider import (  # noqa: E402
    DA3ExternalWorkerRunner,
    DA3InputManifest,
    DA3ProviderProfile,
    write_da3_input_manifest,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402

_MIB = 1024**2
_DEFAULT_SFM_PYTHON = Path.home() / "anaconda3/envs/depth3d-pycolmap-probe/bin/python"
_DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/depth3d-da3/bin/python"
_DEFAULT_DA3_REPO = Path.home() / "Desktop/depth3d-da3-backends/depth-anything-3"
_DA3_WORKER = ROOT / "scripts/da3_workers/da3_worker.py"
_MAX_PIXEL_SCALE_ANISOTROPY = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="Directory of overlapping images.")
    source.add_argument("--video", type=Path, help="Recorded video with camera overlap/motion.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/b2_sfm_dense_real_alignment"))
    parser.add_argument("--sfm-python", type=Path, default=_DEFAULT_SFM_PYTHON)
    parser.add_argument("--da3-python", type=Path, default=_DEFAULT_DA3_PYTHON)
    parser.add_argument("--da3-repo", type=Path, default=_DEFAULT_DA3_REPO)
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
    parser.add_argument("--max-keyframes", type=int, default=40)
    parser.add_argument("--maximum-image-size", type=int, default=None)
    parser.add_argument("--sequential-overlap", type=int, default=None)
    parser.add_argument("--video-target-fps", type=float, default=2.0)
    parser.add_argument("--image-sequence-fps", type=float, default=10.0)
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA for PyCOLMAP SIFT/matching.")
    parser.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    parser.add_argument("--keyframe-id", default="", help="Optional registered SfM keyframe to validate.")
    parser.add_argument("--da3-model", default=DA3_EFFICIENT_MODEL_ID)
    parser.add_argument("--da3-model-revision", default="")
    parser.add_argument("--process-res", type=int, default=392)
    parser.add_argument("--da3-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--minimum-correspondences", type=int, default=8)
    parser.add_argument("--maximum-relative-residual", type=float, default=0.35)
    args = parser.parse_args()

    if args.max_keyframes < 3:
        parser.error("--max-keyframes must be at least 3.")
    if args.maximum_image_size is not None and args.maximum_image_size <= 0:
        parser.error("--maximum-image-size must be positive.")
    if args.sequential_overlap is not None and args.sequential_overlap <= 0:
        parser.error("--sequential-overlap must be positive.")
    if args.video_target_fps <= 0.0 or args.image_sequence_fps <= 0.0:
        parser.error("input frame rates must be positive.")
    if args.sfm_timeout_minutes <= 0.0 or args.da3_timeout_seconds <= 0.0:
        parser.error("backend timeouts must be positive.")
    if args.minimum_correspondences < 3:
        parser.error("--minimum-correspondences must be at least 3.")
    if not 0.0 < args.maximum_relative_residual < 1.0:
        parser.error("--maximum-relative-residual must be in (0, 1).")
    return args


def _build_source(args: argparse.Namespace):
    if args.images is not None:
        return ImageSequenceReader.from_directory(
            args.images.expanduser().resolve(),
            recursive=bool(args.recursive),
            frame_rate_hz=float(args.image_sequence_fps),
        ), FrameSamplingPolicy(
            mode=FrameSamplingMode.EVERY_FRAME,
            maximum_sampled_frames=int(args.max_keyframes),
        )
    return RecordedVideoReader(args.video.expanduser().resolve()), FrameSamplingPolicy(
        mode=FrameSamplingMode.TARGET_FPS,
        target_fps_hz=float(args.video_target_fps),
        maximum_sampled_frames=int(args.max_keyframes),
    )


def _observation_counts(sfm_result) -> Counter[str]:
    registered = {camera.keyframe_id for camera in sfm_result.cameras}
    counts: Counter[str] = Counter()
    for landmark in sfm_result.landmarks:
        for observation in landmark.observations:
            if observation.keyframe_id in registered:
                counts[observation.keyframe_id] += 1
    return counts


def _select_registered_keyframe(sfm_result, *, requested_keyframe_id: str = ""):
    """Return (camera, workspace_image, observation_count) for B2 validation."""

    requested = str(requested_keyframe_id).strip()
    cameras = {camera.keyframe_id: camera for camera in sfm_result.cameras}
    workspace = {image.keyframe_id: image for image in sfm_result.workspace.images}
    counts = _observation_counts(sfm_result)

    if requested:
        if requested not in cameras:
            raise ReconstructionValidationError(
                f"Requested B2 keyframe {requested!r} is not registered by offline SfM."
            )
        selected_id = requested
    else:
        candidates = [keyframe_id for keyframe_id in cameras if keyframe_id in workspace]
        if not candidates:
            raise ReconstructionValidationError("Offline SfM returned no materialized registered keyframe.")
        selected_id = max(candidates, key=lambda value: (counts.get(value, 0), value))

    if selected_id not in workspace:
        raise ReconstructionValidationError(
            f"Registered B2 keyframe {selected_id!r} is missing from the SfM workspace."
        )
    return cameras[selected_id], workspace[selected_id], int(counts.get(selected_id, 0))


def _adapt_da3_depth_to_sfm_pixel_space(
    depth: object,
    *,
    sfm_width: int,
    sfm_height: int,
    maximum_scale_anisotropy: float = _MAX_PIXEL_SCALE_ANISOTROPY,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return DA3 depth sampled on the SfM image grid for B2 validation.

    DA3's qualified worker may emit depth at its bounded inference resolution.
    B1 intentionally requires an exact shared pixel grid. For B2 only, when the
    DA3 output differs in size, this helper accepts a near-uniform resize and
    resamples depth back to the known SfM image dimensions. A large width/height
    scale disagreement is rejected instead of silently assuming crop/padding.
    """

    array = np.asarray(depth, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ReconstructionValidationError("B2 DA3 depth must be a non-empty HxW array.")
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ReconstructionValidationError("B2 DA3 depth must be finite and strictly positive.")
    width = int(sfm_width)
    height = int(sfm_height)
    if width <= 0 or height <= 0:
        raise ReconstructionValidationError("B2 SfM image dimensions must be positive.")
    maximum = float(maximum_scale_anisotropy)
    if not isfinite(maximum) or not 0.0 <= maximum < 1.0:
        raise ReconstructionValidationError("maximum_scale_anisotropy must be finite in [0,1).")

    source_height, source_width = array.shape
    if source_width == width and source_height == height:
        result = np.ascontiguousarray(array.copy())
        result.setflags(write=False)
        return result, {
            "mode": "identity",
            "source_size": [source_width, source_height],
            "target_sfm_size": [width, height],
            "scale_x": 1.0,
            "scale_y": 1.0,
            "relative_scale_anisotropy": 0.0,
            "interpolation": "none",
            "validation_only": True,
        }

    scale_x = float(source_width) / float(width)
    scale_y = float(source_height) / float(height)
    anisotropy = abs(scale_x - scale_y) / max(scale_x, scale_y, 1e-12)
    if anisotropy > maximum:
        raise ReconstructionValidationError(
            "DA3 output pixel scaling is not sufficiently uniform to prove a safe B2 resize "
            f"back to the SfM grid (anisotropy={anisotropy:.6f}, limit={maximum:.6f})."
        )

    resized = cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = np.asarray(resized, dtype=np.float32)
    if resized.shape != (height, width) or not np.isfinite(resized).all() or np.any(resized <= 0.0):
        raise ReconstructionValidationError("B2 depth resize produced invalid dense-depth evidence.")
    result = np.ascontiguousarray(resized.copy())
    result.setflags(write=False)
    return result, {
        "mode": "aspect_preserving_resize_to_sfm_grid",
        "source_size": [source_width, source_height],
        "target_sfm_size": [width, height],
        "scale_x": scale_x,
        "scale_y": scale_y,
        "relative_scale_anisotropy": anisotropy,
        "maximum_allowed_scale_anisotropy": maximum,
        "interpolation": "opencv_linear",
        "validation_only": True,
        "authority_note": (
            "B2 validation-side pixel-grid adaptation only; this operation does not establish "
            "metric scale or geometry authority."
        ),
    }


def _candidate_observation_xy(sfm_result, keyframe_id: str) -> tuple[tuple[float, float], ...]:
    values: list[tuple[float, float]] = []
    for landmark in sfm_result.landmarks:
        for observation in landmark.observations:
            if observation.keyframe_id == keyframe_id:
                values.append((float(observation.feature_xy[0]), float(observation.feature_xy[1])))
                break
    return tuple(values)


def _depth_visualization(depth: np.ndarray) -> np.ndarray:
    finite = np.asarray(depth, dtype=np.float32)
    lo, hi = np.percentile(finite, [2.0, 98.0])
    denom = max(float(hi - lo), 1e-6)
    normalized = np.clip((finite - float(lo)) / denom, 0.0, 1.0)
    gray = np.asarray(np.round(normalized * 255.0), dtype=np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def _write_diagnostic(
    *,
    rgb_path: Path,
    depth_sfm_grid: np.ndarray,
    observation_xy: tuple[tuple[float, float], ...],
    evidence,
    output_path: Path,
    pixel_adapter: dict[str, Any],
) -> Path:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ReconstructionValidationError(f"Unable to read selected B2 keyframe: {rgb_path}")
    height, width = depth_sfm_grid.shape
    if rgb.shape[:2] != (height, width):
        raise ReconstructionValidationError(
            "Selected workspace RGB and B2 depth must share the same final pixel grid."
        )

    rgb_overlay = rgb.copy()
    depth_overlay = _depth_visualization(depth_sfm_grid)
    for x, y in observation_xy:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            cv2.circle(rgb_overlay, (xi, yi), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(depth_overlay, (xi, yi), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    body = np.hstack([rgb_overlay, depth_overlay])
    header_height = 116
    header = np.zeros((header_height, body.shape[1], 3), dtype=np.uint8)
    status = "ACCEPTED" if evidence.is_accepted else "REJECTED"
    line1 = f"B2 SfM <-> dense depth: {status} | keyframe={evidence.keyframe_id}"
    line2 = (
        f"candidate={evidence.candidate_observations}  valid={evidence.valid_correspondences}  "
        f"inliers={evidence.inlier_correspondences}"
    )
    if evidence.alignment is not None:
        line3 = (
            f"scale={evidence.alignment.scale_to_reconstruction_units:.6g}  "
            f"median_rel_residual={evidence.median_relative_residual:.4f}  "
            f"p90={evidence.p90_relative_residual:.4f}"
        )
    else:
        line3 = f"reason={evidence.rejection_code.value if evidence.rejection_code else 'unknown'}"
    line4 = (
        f"pixel_adapter={pixel_adapter['mode']} | arbitrary SfM gauge | metric scale NOT claimed"
    )
    for index, text in enumerate((line1, line2, line3, line4)):
        cv2.putText(
            header,
            text,
            (16, 25 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    diagnostic = np.vstack([header, body])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), diagnostic):
        raise ReconstructionValidationError(f"Unable to write B2 diagnostic image: {output_path}")
    return output_path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _print_sfm_event(event) -> None:
    if event.progress_fraction is not None:
        print(f"[SfM {event.progress_fraction * 100:5.1f}%] {event.message}", flush=True)
    elif event.message:
        print(f"[SfM] {event.message}", flush=True)


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session_id = new_reconstruction_session_id()
    source, sampling_policy = _build_source(args)
    profile = ReconstructionQualityProfile(args.profile)

    print("B2: running real offline SfM...", flush=True)
    ingestion = ReconstructionIngestionPipeline(
        session_id=session_id,
        source=source,
        sampling_policy=sampling_policy,
        keyframe_policy=KeyframeSelectionPolicy(
            minimum_spacing_ns=1,
            forced_spacing_ns=1,
            minimum_visual_change=0.0,
            minimum_quality_score=0.0,
            maximum_keyframes=int(args.max_keyframes),
        ),
        quality_policy=FrameQualityPolicy(),
        cache_policy=IngestionCachePolicy(
            maximum_cpu_bytes=512 * _MIB,
            maximum_frames=min(8, int(args.max_keyframes)),
            gpu_staging_bytes=0,
        ),
    ).run()
    if len(ingestion.keyframes) < 3:
        raise ReconstructionValidationError(
            "B2 requires at least three valid overlapping keyframes for real offline SfM."
        )

    workspace = materialize_sfm_workspace(
        ingestion,
        output_dir / "sfm_workspace",
        overwrite=True,
    )
    sfm_request = OfflineSfmRequest(
        session_id=session_id,
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
    sfm_backend = ExternalPycolmapSfmBackend(
        args.sfm_python.expanduser().resolve(),
        timeout_s=float(args.sfm_timeout_minutes) * 60.0,
    )
    try:
        sfm_result = sfm_backend.run_sparse_reconstruction(
            sfm_request,
            BackendOperationContext(
                session_id=session_id,
                stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
                profile=profile,
                control=ReconstructionOperationControl(),
                emit_event=_print_sfm_event,
            ),
        )
        sfm_probe = sfm_backend.environment_probe
    finally:
        sfm_backend.shutdown()

    if sfm_result.statistics.registered_keyframes < 3 or sfm_result.statistics.landmarks <= 0:
        raise ReconstructionValidationError("B2 real SfM did not produce sufficient registered sparse geometry.")
    if sfm_result.gauge.scale_status is not SfmScaleStatus.ARBITRARY:
        raise ReconstructionValidationError(
            "B2 expects the raw monocular offline-SfM result to remain arbitrary-scale."
        )

    camera, workspace_image, observation_count = _select_registered_keyframe(
        sfm_result,
        requested_keyframe_id=args.keyframe_id,
    )
    rgb_path = Path(workspace.workspace_root).resolve() / workspace_image.staged_image_path
    if not rgb_path.is_file():
        raise ReconstructionValidationError(f"Selected B2 workspace image is missing: {rgb_path}")

    print(
        f"B2: selected keyframe={camera.keyframe_id} sparse_observations={observation_count}",
        flush=True,
    )
    print("B2: probing isolated DA3 runtime...", flush=True)
    da3_python = args.da3_python.expanduser().resolve()
    da3_repo = args.da3_repo.expanduser().resolve()
    da3_probe = probe_da3_environment(da3_python, da3_repo)
    if not da3_probe.available:
        raise ReconstructionValidationError(
            "B2 DA3 environment is not qualified: " + json.dumps(da3_probe.as_dict(), sort_keys=True)
        )

    da3_root = output_dir / "da3_single_keyframe"
    input_manifest_path = write_da3_input_manifest(
        da3_root / "input_manifest.json",
        DA3InputManifest(
            request_id=f"b2-{session_id}",
            image_paths=(str(rgb_path.resolve()),),
            metadata={
                "phase": "B2",
                "session_id": session_id,
                "sfm_keyframe_id": camera.keyframe_id,
                "purpose": "real_single_keyframe_sfm_dense_alignment",
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
            },
        ),
    )
    da3_runner = DA3ExternalWorkerRunner(
        python_executable=da3_python,
        worker_script=_DA3_WORKER.resolve(),
        profile=DA3ProviderProfile(
            model_id=args.da3_model,
            model_revision=args.da3_model_revision,
            process_res=int(args.process_res),
            process_res_method="upper_bound_resize",
        ),
        timeout_s=float(args.da3_timeout_seconds),
    )
    print("B2: running real DA3 dense depth for the selected keyframe...", flush=True)
    da3_manifest_path, da3_manifest = da3_runner.run(
        input_manifest_path,
        da3_root / "worker_output",
    )
    da3_arrays = validate_da3_npz_result(da3_manifest)
    if da3_manifest.frame_count != 1 or da3_arrays["depth"].shape[0] != 1:
        raise ReconstructionValidationError("B2 single-keyframe DA3 run returned an unexpected frame count.")

    depth_sfm_grid, pixel_adapter = _adapt_da3_depth_to_sfm_pixel_space(
        da3_arrays["depth"][0],
        sfm_width=workspace_image.image_size.width,
        sfm_height=workspace_image.image_size.height,
    )
    depth_provenance = (
        f"Depth Anything 3 model={da3_manifest.model_id} revision={da3_manifest.model_revision};"
        f"source_revision={da3_manifest.source_revision};pixel_adapter={pixel_adapter['mode']};"
        "B2 real validation only; no metric scale claim"
    )
    evidence = align_sfm_keyframe_dense_depth(
        sfm_result,
        keyframe_id=camera.keyframe_id,
        dense_depth_source_units=depth_sfm_grid,
        depth_provenance=depth_provenance,
        minimum_correspondences=int(args.minimum_correspondences),
        maximum_relative_residual=float(args.maximum_relative_residual),
    )

    sfm_result_json = _write_json(output_dir / "sfm_result.json", sfm_result.as_dict())
    evidence_json = _write_json(output_dir / "b2_alignment_evidence.json", evidence.as_dict())
    diagnostic_path = _write_diagnostic(
        rgb_path=rgb_path,
        depth_sfm_grid=depth_sfm_grid,
        observation_xy=_candidate_observation_xy(sfm_result, camera.keyframe_id),
        evidence=evidence,
        output_path=output_dir / "b2_alignment_diagnostic.png",
        pixel_adapter=pixel_adapter,
    )

    observation_counts = _observation_counts(sfm_result)
    top_keyframes = sorted(observation_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    summary = {
        "phase": "B2",
        "pass": bool(evidence.is_accepted),
        "real_sfm_executed": True,
        "real_dense_depth_executed": True,
        "session_id": session_id,
        "source": ingestion.source.as_dict(),
        "ingestion_summary": ingestion.as_dict()["summary"],
        "sfm": {
            "backend_id": sfm_result.backend_id,
            "backend_version": sfm_result.backend_version,
            "environment_probe": None if sfm_probe is None else sfm_probe.as_dict(),
            "registered_keyframes": sfm_result.statistics.registered_keyframes,
            "landmarks": sfm_result.statistics.landmarks,
            "observations": sfm_result.statistics.observations,
            "mean_reprojection_error_px": sfm_result.statistics.mean_reprojection_error_px,
            "gauge": sfm_result.gauge.as_dict(),
            "selected_keyframe_id": camera.keyframe_id,
            "selected_frame_id": camera.frame_id,
            "selected_camera_id": camera.camera_id,
            "selected_workspace_image": str(rgb_path.resolve()),
            "selected_sparse_observations": observation_count,
            "top_registered_keyframe_observation_counts": [
                {"keyframe_id": keyframe_id, "observations": count}
                for keyframe_id, count in top_keyframes
            ],
        },
        "dense_depth": {
            "provider": "depth-anything-3",
            "environment_probe": da3_probe.as_dict(),
            "manifest": da3_manifest.as_dict(),
            "manifest_path": str(da3_manifest_path.resolve()),
            "pixel_space_adapter": pixel_adapter,
        },
        "alignment_evidence": evidence.as_dict(),
        "authority": {
            "sfm_scale_status": sfm_result.gauge.scale_status.value,
            "unit_to_metre": sfm_result.gauge.unit_to_metre,
            "metric_scale_claimed": False,
            "canonical_geometry_authoritative": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
        },
        "artifacts": {
            "sfm_result_json": str(sfm_result_json.resolve()),
            "alignment_evidence_json": str(evidence_json.resolve()),
            "alignment_diagnostic_png": str(diagnostic_path.resolve()),
            "sfm_workspace": str(Path(workspace.workspace_root).resolve()),
            "da3_result_manifest": str(da3_manifest_path.resolve()),
        },
    }
    summary_path = _write_json(output_dir / "b2_summary.json", summary)

    print()
    print(f"B2 real SfM ↔ dense-depth alignment: {'PASS' if evidence.is_accepted else 'REJECTED'}")
    print(
        f"keyframe={camera.keyframe_id} candidate={evidence.candidate_observations} "
        f"valid={evidence.valid_correspondences} inliers={evidence.inlier_correspondences}"
    )
    if evidence.alignment is not None:
        print(
            f"scale_to_reconstruction_units={evidence.alignment.scale_to_reconstruction_units:.9g} "
            f"relative_scale_uncertainty={evidence.alignment.relative_scale_uncertainty:.6f}"
        )
        print(
            f"median_relative_residual={evidence.median_relative_residual:.6f} "
            f"p90_relative_residual={evidence.p90_relative_residual:.6f} "
            f"robust_limit={evidence.robust_relative_residual_limit:.6f}"
        )
    else:
        print(
            f"rejection_code={evidence.rejection_code.value if evidence.rejection_code else 'unknown'} "
            f"message={evidence.message}"
        )
    print(
        f"da3_output={da3_manifest.width}x{da3_manifest.height} "
        f"sfm_grid={workspace_image.image_size.width}x{workspace_image.image_size.height} "
        f"pixel_adapter={pixel_adapter['mode']}"
    )
    print(
        f"sfm_scale_status={sfm_result.gauge.scale_status.value} "
        f"unit_to_metre={sfm_result.gauge.unit_to_metre} metric_scale_claimed=False"
    )
    print("canonical_geometry_authoritative=False certified_metrology=False")
    print(f"diagnostic={diagnostic_path}")
    print(f"summary={summary_path}")
    return 0 if evidence.is_accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
