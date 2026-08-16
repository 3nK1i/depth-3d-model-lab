#!/usr/bin/env python3
"""Depth-side orchestration worker for Phase 9.6E2 full-scene DA3 video builds."""
from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import shutil
import selectors
import subprocess
import time
import sys
import traceback

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.da3_canonical_scene_persistence import materialize_da3_canonical_persistent_scene
from app.reconstruction_da3_full_scene import (
    DA3_FULL_SCENE_PROTOCOL,
    estimate_overlap_similarity,
    plan_overlapping_windows,
    transform_chunk_to_global_gauge,
)
from core.perception.reviewed_analysis import source_image_sha256
from core.reconstruction import read_persistent_reconstruction_scene, verify_persistent_scene_file_references
from core.reconstruction.da3 import (
    DA3_EFFICIENT_MODEL_ID,
    DA3ResultManifest,
    probe_da3_environment,
)
from core.reconstruction.da3_canonical_scene import integrate_da3_result_to_canonical_scene
from core.reconstruction.da3_provider import DA3InputManifest, write_da3_input_manifest
from core.reconstruction.keyframes import visual_change_score
from core.reconstruction.quality import FrameQualityPolicy, analyze_frame_quality

EXTERNAL_WORKER = PROJECT_ROOT / "scripts" / "da3_workers" / "da3_full_scene_worker.py"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--da3-python", required=True)
    p.add_argument("--da3-repo", required=True)
    p.add_argument("--model", default=DA3_EFFICIENT_MODEL_ID)
    p.add_argument("--target-fps", type=float, default=5.0)
    p.add_argument("--max-keyframes", type=int, default=96)
    p.add_argument("--minimum-spacing-seconds", type=float, default=0.5)
    p.add_argument("--forced-spacing-seconds", type=float, default=2.0)
    p.add_argument("--minimum-visual-change", type=float, default=0.055)
    p.add_argument("--minimum-quality-score", type=float, default=0.12)
    p.add_argument("--process-res", type=int, default=392)
    p.add_argument("--pixel-stride", type=int, default=6)
    p.add_argument("--per-window-timeout", type=float, default=180.0)
    p.add_argument("--reuse-external-summary", default="")
    p.add_argument("--reuse-keyframes-dir", default="")
    return p.parse_args()


def _emit(job_id: str, event: str, *, stage: str, message: str, **payload) -> None:
    out = {
        "protocol": DA3_FULL_SCENE_PROTOCOL,
        "job_id": job_id,
        "event": event,
        "stage": stage,
        "message": message,
        "canonical_geometry_authoritative": False,
        "metric_scale_authoritative": False,
        "measurement_authoritative": False,
        "qc_authoritative": False,
        "certified_metrology": False,
        **payload,
    }
    print(json.dumps(out, sort_keys=True, allow_nan=False), flush=True)


def _select_video_keyframes(args: argparse.Namespace, root: Path) -> tuple[list[Path], list[int], list[int], dict]:
    video = Path(args.video).expanduser().resolve()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open recorded video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not isfinite(fps) or fps <= 0.0:
        fps = 30.0
    duration = (float(total) / fps) if total > 0 else 0.0
    target_step = max(1, int(round(fps / float(args.target_fps))))
    max_keyframes = int(args.max_keyframes)
    adaptive_min_spacing = max(
        float(args.minimum_spacing_seconds),
        (duration / max(1, max_keyframes - 1)) if duration > 0 else float(args.minimum_spacing_seconds),
    )
    forced_spacing = max(float(args.forced_spacing_seconds), adaptive_min_spacing * 1.5)
    quality_policy = FrameQualityPolicy(minimum_quality_score=float(args.minimum_quality_score))
    output = root / "selected-keyframes"
    output.mkdir(parents=True, exist_ok=True)
    selected_paths: list[Path] = []
    selected_indices: list[int] = []
    selected_timestamps: list[int] = []
    selected_last_image: np.ndarray | None = None
    selected_last_time = -1e9
    candidate_count = 0
    rejected_quality = 0
    index = 0
    last_good: tuple[int, int, np.ndarray] | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if index % target_step != 0:
                index += 1
                continue
            candidate_count += 1
            time_s = index / fps
            metrics = analyze_frame_quality(frame, quality_policy)
            if not metrics.accepted or metrics.quality_score < float(args.minimum_quality_score):
                rejected_quality += 1
                index += 1
                continue
            last_good = (index, int(round(time_s * 1_000_000_000)), frame.copy())
            elapsed = time_s - selected_last_time
            should_select = not selected_paths
            if selected_paths and elapsed >= adaptive_min_spacing:
                change = visual_change_score(selected_last_image, frame) if selected_last_image is not None else 1.0
                should_select = change >= float(args.minimum_visual_change) or elapsed >= forced_spacing
            if should_select and len(selected_paths) < max_keyframes:
                destination = output / f"kf-{len(selected_paths):04d}-frame-{index:08d}.png"
                if not cv2.imwrite(str(destination), frame):
                    raise RuntimeError(f"Unable to materialize selected keyframe: {destination}")
                selected_paths.append(destination.resolve())
                selected_indices.append(index)
                selected_timestamps.append(int(round(time_s * 1_000_000_000)))
                selected_last_image = frame.copy()
                selected_last_time = time_s
            index += 1
    finally:
        cap.release()
    if last_good is not None and len(selected_paths) < max_keyframes:
        final_index, final_timestamp, final_frame = last_good
        if not selected_indices or final_index != selected_indices[-1]:
            final_time = final_timestamp / 1_000_000_000.0
            if final_time - selected_last_time >= adaptive_min_spacing * 0.75:
                destination = output / f"kf-{len(selected_paths):04d}-frame-{final_index:08d}.png"
                if not cv2.imwrite(str(destination), final_frame):
                    raise RuntimeError(f"Unable to materialize final keyframe: {destination}")
                selected_paths.append(destination.resolve())
                selected_indices.append(final_index)
                selected_timestamps.append(final_timestamp)
    if len(selected_paths) < 3:
        raise RuntimeError("Full-scene DA3 requires at least three sharp, diverse keyframes from the video.")
    return selected_paths, selected_indices, selected_timestamps, {
        "reported_frame_count": total,
        "reported_fps": fps,
        "duration_seconds": duration,
        "target_sampling_fps": float(args.target_fps),
        "candidate_count": candidate_count,
        "quality_rejected_count": rejected_quality,
        "selected_keyframe_count": len(selected_paths),
        "adaptive_minimum_spacing_seconds": adaptive_min_spacing,
        "forced_spacing_seconds": forced_spacing,
    }


def _copy_reuse_keyframes(directory: Path, root: Path) -> tuple[list[Path], list[int], list[int], dict]:
    images = sorted(path.resolve() for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(images) < 3:
        raise RuntimeError("Reuse keyframe directory requires at least three images.")
    target = root / "selected-keyframes"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, source in enumerate(images):
        destination = target / f"kf-{index:04d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        copied.append(destination.resolve())
    return copied, list(range(len(copied))), [index * 1_000_000_000 for index in range(len(copied))], {
        "test_reuse": True,
        "selected_keyframe_count": len(copied),
    }


def _run_external(args: argparse.Namespace, root: Path, paths: list[Path], windows: tuple[tuple[int, ...], ...], job_id: str) -> dict:
    external_dir = root / "external-da3"
    external_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_external_summary:
        source = Path(args.reuse_external_summary).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        copied_chunks = []
        for chunk in payload["chunks"]:
            source_npz = Path(chunk["npz_path"]).resolve()
            destination = external_dir / Path(source_npz).name
            shutil.copy2(source_npz, destination)
            copied = dict(chunk)
            copied["npz_path"] = str(destination.resolve())
            copied_chunks.append(copied)
        payload["chunks"] = copied_chunks
        summary_path = external_dir / "full_scene_external_summary.json"
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    da3_python = Path(args.da3_python).resolve()
    da3_repo = Path(args.da3_repo).resolve()
    probe = probe_da3_environment(da3_python, da3_repo)
    if not probe.available:
        raise RuntimeError(f"Qualified isolated DA3 runtime is unavailable: {probe.error or 'qualification_failed'}")
    request = {
        "image_paths": [str(path) for path in paths],
        "windows": [list(window) for window in windows],
        "output_dir": str(external_dir),
        "model_id": str(args.model),
        "process_res": int(args.process_res),
    }
    request_path = root / "external-da3-request.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    process = subprocess.Popen(
        [str(da3_python), str(EXTERNAL_WORKER), "--request", str(request_path)],
        cwd=str(da3_repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    completed = 0
    timeout_s = float(args.per_window_timeout)
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    # Model import/checkpoint load precedes the first window, so give startup a
    # wider bounded allowance than steady-state per-window inference.
    deadline = time.monotonic() + max(600.0, timeout_s)
    try:
        while True:
            if process.poll() is not None:
                # Drain any final buffered protocol lines before exit handling.
                for raw in process.stdout:
                    text = raw.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("event") == "window_completed":
                        completed += 1
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
                raise RuntimeError(
                    f"Full-scene DA3 window exceeded the {timeout_s:.0f}s bounded timeout."
                )
            ready = selector.select(timeout=min(1.0, remaining))
            if not ready:
                continue
            raw = process.stdout.readline()
            if not raw:
                continue
            text = raw.strip()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if payload.get("event") == "window_completed":
                completed += 1
                deadline = time.monotonic() + timeout_s
                fraction = 0.20 + 0.55 * (completed / max(1, len(windows)))
                _emit(
                    job_id,
                    "progress",
                    stage="da3_windows",
                    message=f"Processed DA3 window {completed} / {len(windows)}.",
                    progress_fraction=fraction,
                    completed_units=completed,
                    total_units=len(windows),
                )
    finally:
        selector.close()
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"External full-scene DA3 worker exited with code {code}.")
    if completed != len(windows):
        raise RuntimeError(
            f"External full-scene DA3 worker completed {completed} / {len(windows)} declared windows."
        )
    summary_path = external_dir / "full_scene_external_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("External full-scene DA3 summary is missing.")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("source_revision") != probe.source_revision:
        raise RuntimeError("Full-scene DA3 source revision does not match the qualified external checkout.")
    if payload.get("gpu_name") not in probe.device_names:
        raise RuntimeError("Full-scene DA3 GPU identity is not present in the qualified runtime probe.")
    return payload


def _homogeneous_w2c_batch(value: object) -> np.ndarray:
    """Normalize DA3/OpenCV W2C extrinsics to one homogeneous Nx4x4 shape."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] not in {(3, 4), (4, 4)}:
        raise RuntimeError("Full-scene DA3 extrinsics must have shape (N,3,4) or (N,4,4).")
    if not np.isfinite(array).all():
        raise RuntimeError("Full-scene DA3 extrinsics contain non-finite values.")
    if array.shape[1:] == (4, 4):
        return np.ascontiguousarray(array, dtype=np.float32)
    output = np.repeat(np.eye(4, dtype=np.float32)[None, ...], array.shape[0], axis=0)
    output[:, :3, :] = array
    return np.ascontiguousarray(output, dtype=np.float32)



def _consolidate_fixed_camera_intrinsics(
    raw_per_frame: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict]:
    """Consolidate DA3-predicted intrinsics for one fixed-lens recorded video.

    DA3 predicts intrinsics independently for each bounded window.  A recorded
    video captured without zoom still represents one physical camera, so E2 may
    robustly collapse prediction jitter to one *estimated, not calibrated*
    camera matrix.  The admission checks below intentionally fail closed when
    the predictions look like a genuinely changing camera model.
    """

    per_frame = np.asarray(raw_per_frame, dtype=np.float64)
    if per_frame.ndim != 3 or per_frame.shape[1:] != (3, 3):
        raise RuntimeError("Full-scene DA3 per-frame intrinsics must have shape (N,3,3).")
    if not np.isfinite(per_frame).all():
        raise RuntimeError("Full-scene DA3 intrinsics contain non-finite values.")

    fx = per_frame[:, 0, 0]
    fy = per_frame[:, 1, 1]
    cx = per_frame[:, 0, 2]
    cy = per_frame[:, 1, 2]
    if np.any(fx <= 0.0) or np.any(fy <= 0.0):
        raise RuntimeError("Full-scene DA3 intrinsics contain non-positive focal lengths.")

    representative = np.median(per_frame, axis=0)
    representative[2, :] = (0.0, 0.0, 1.0)
    if representative[0, 0] <= 0.0 or representative[1, 1] <= 0.0:
        raise RuntimeError("Full-scene DA3 representative intrinsics are invalid.")

    focal_rel = np.maximum(
        np.abs(fx - representative[0, 0]) / max(abs(float(representative[0, 0])), 1.0),
        np.abs(fy - representative[1, 1]) / max(abs(float(representative[1, 1])), 1.0),
    )
    focal_max = float(np.max(focal_rel))
    focal_p95 = float(np.quantile(focal_rel, 0.95))

    aspect = fx / fy
    aspect_median = float(np.median(aspect))
    aspect_rel_max = float(np.max(np.abs(aspect - aspect_median) / max(abs(aspect_median), 1e-12)))

    principal_dx = float(np.max(np.abs(cx - np.median(cx))))
    principal_dy = float(np.max(np.abs(cy - np.median(cy))))
    principal_fraction = max(principal_dx / max(float(width), 1.0), principal_dy / max(float(height), 1.0))

    # E2 is explicitly a fixed-camera recorded-video workflow.  These are
    # admission bounds for DA3 prediction jitter, not calibration tolerances.
    # The downstream F3 10% single-camera guard remains unchanged and sees the
    # consolidated representative only after these checks pass.
    if focal_max > 0.30 or focal_p95 > 0.25:
        raise RuntimeError(
            "Full-scene DA3 focal predictions are too inconsistent for fixed-camera consolidation "
            f"(max={focal_max:.6f}, p95={focal_p95:.6f}). Capture without zoom or calibrate the camera."
        )
    if aspect_rel_max > 0.05:
        raise RuntimeError(
            "Full-scene DA3 fx/fy ratio varies too much for one fixed-camera model "
            f"(max_relative_spread={aspect_rel_max:.6f})."
        )
    if principal_fraction > 0.01:
        raise RuntimeError(
            "Full-scene DA3 principal point varies too much for one fixed-camera model "
            f"(max_image_fraction={principal_fraction:.6f})."
        )

    consolidated = np.repeat(
        representative.astype(np.float32)[None, ...],
        per_frame.shape[0],
        axis=0,
    )
    return np.ascontiguousarray(consolidated), {
        "intrinsics_consolidation": "robust_median_fixed_camera",
        "intrinsics_source": "da3_predicted_per_window",
        "fixed_camera_assumption": True,
        "camera_calibrated": False,
        "raw_intrinsics_prediction_count": int(per_frame.shape[0]),
        "raw_focal_max_relative_deviation": focal_max,
        "raw_focal_p95_relative_deviation": focal_p95,
        "raw_focal_aspect_ratio_max_relative_deviation": aspect_rel_max,
        "raw_principal_point_max_image_fraction": float(principal_fraction),
        "representative_fx": float(representative[0, 0]),
        "representative_fy": float(representative[1, 1]),
        "representative_cx": float(representative[0, 2]),
        "representative_cy": float(representative[1, 2]),
    }

def _consolidate_chunks(summary: dict, frame_count: int, root: Path) -> tuple[Path, dict]:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    chunks = list(summary.get("chunks", []))
    if not chunks:
        raise RuntimeError("Full-scene DA3 produced no chunk evidence.")
    global_depth: dict[int, np.ndarray] = {}
    global_conf: dict[int, np.ndarray] = {}
    global_ext: dict[int, np.ndarray] = {}
    global_intr: dict[int, np.ndarray] = {}
    alignment_scales = []
    for ordinal, chunk in enumerate(chunks):
        indices = [int(value) for value in chunk["frame_indices"]]
        data = np.load(Path(chunk["npz_path"]).resolve(), allow_pickle=False)
        depth = np.asarray(data["depth"], dtype=np.float32)
        conf = np.asarray(data["confidence"], dtype=np.float32)
        ext = _homogeneous_w2c_batch(data["extrinsics_w2c"])
        intr = np.asarray(data["intrinsics"], dtype=np.float32)
        if intr.shape != (3, 3, 3) or not np.isfinite(intr).all():
            raise RuntimeError("Full-scene DA3 chunk intrinsics contract drift detected.")
        if len(indices) != 3 or depth.shape[0] != 3 or conf.shape != depth.shape:
            raise RuntimeError("Full-scene DA3 chunk contract drift detected.")
        if ordinal == 0:
            transformed_ext, transformed_depth = ext, depth
        else:
            shared = indices[:2]
            if any(index not in global_depth for index in shared):
                raise RuntimeError("DA3 sequential window overlap is missing from the accumulated global gauge.")
            scale, rotation, translation = estimate_overlap_similarity(
                [global_ext[index] for index in shared],
                [ext[0], ext[1]],
                [global_depth[index] for index in shared],
                [depth[0], depth[1]],
            )
            transformed_ext, transformed_depth = transform_chunk_to_global_gauge(
                ext, depth, scale=scale, rotation=rotation, translation=translation
            )
            alignment_scales.append(scale)
        for local_pos, frame_index in enumerate(indices):
            if frame_index in global_depth:
                continue
            global_depth[frame_index] = transformed_depth[local_pos]
            global_conf[frame_index] = conf[local_pos]
            global_ext[frame_index] = transformed_ext[local_pos]
            global_intr[frame_index] = intr[local_pos]
    if sorted(global_depth) != list(range(frame_count)):
        raise RuntimeError("Full-scene DA3 consolidated evidence does not cover every selected keyframe.")
    depth = np.stack([global_depth[index] for index in range(frame_count)], axis=0)
    conf = np.stack([global_conf[index] for index in range(frame_count)], axis=0)
    ext = np.stack([global_ext[index] for index in range(frame_count)], axis=0)
    raw_intr = np.stack([global_intr[index] for index in range(frame_count)], axis=0)
    intr, intrinsics_meta = _consolidate_fixed_camera_intrinsics(
        raw_intr,
        width=int(depth.shape[2]),
        height=int(depth.shape[1]),
    )
    output = root / "consolidated-da3.npz"
    np.savez_compressed(
        output,
        depth=depth,
        confidence=conf,
        extrinsics_w2c=ext,
        intrinsics=intr,
        raw_intrinsics_per_frame=raw_intr,
    )
    return output, {
        "alignment_method": "sequential_overlap_pose_similarity_with_depth_scale",
        "overlap_views": 2,
        "window_size": 3,
        "window_count": len(chunks),
        "alignment_scale_min": min(alignment_scales) if alignment_scales else 1.0,
        "alignment_scale_max": max(alignment_scales) if alignment_scales else 1.0,
        "loop_closure": False,
        "quality_certified": False,
        "metric_scale_qualified": False,
        **intrinsics_meta,
    }


def _load_rgb(paths: list[Path], width: int, height: int) -> tuple[np.ndarray, ...]:
    frames = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unable to reload selected keyframe: {path}")
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8))
    return tuple(frames)


def main() -> int:
    args = _args()
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    job_id = str(args.job_id).strip()
    try:
        _emit(job_id, "started", stage="video_analysis", message="Analyzing recorded video for full-scene keyframes.", progress_fraction=0.01)
        if args.reuse_keyframes_dir:
            paths, frame_indices, timestamps, selection_meta = _copy_reuse_keyframes(Path(args.reuse_keyframes_dir).resolve(), root)
        else:
            paths, frame_indices, timestamps, selection_meta = _select_video_keyframes(args, root)
        windows = plan_overlapping_windows(len(paths))
        _emit(
            job_id,
            "progress",
            stage="keyframe_selection",
            message=f"Selected {len(paths)} quality-filtered keyframes across the recorded scene.",
            progress_fraction=0.15,
            completed_units=len(paths),
            total_units=len(paths),
        )
        input_manifest = DA3InputManifest(
            request_id=f"phase9-6e2-{job_id[:8]}",
            image_paths=tuple(str(path) for path in paths),
            metadata={
                "phase": "9.6E2",
                "source": "recorded_video_full_scene",
                "selected_source_frame_indices": frame_indices,
                "alignment_method": "sequential_overlap_pose_similarity_with_depth_scale",
                "loop_closure": False,
                "metric_scale_qualified": False,
                "certified_metrology": False,
            },
        )
        input_manifest_path = write_da3_input_manifest(root / "da3_full_scene_input_manifest.json", input_manifest)
        _emit(job_id, "progress", stage="da3_runtime", message="Starting bounded external DA3 window processing.", progress_fraction=0.20, completed_units=0, total_units=len(windows))
        external = _run_external(args, root, paths, windows, job_id)
        _emit(job_id, "progress", stage="gauge_alignment", message="Aligning overlapping DA3 windows into one arbitrary reconstruction gauge.", progress_fraction=0.78)
        consolidated_npz, alignment_meta = _consolidate_chunks(external, len(paths), root)
        first_chunk = np.load(Path(external["chunks"][0]["npz_path"]), allow_pickle=False)
        height, width = map(int, np.asarray(first_chunk["depth"]).shape[1:])
        result_manifest = DA3ResultManifest(
            request_id=input_manifest.request_id,
            model_id=str(external["model_id"]),
            model_revision=str(external.get("model_revision", "")),
            source_revision=str(external["source_revision"]),
            output_npz_path=str(consolidated_npz.resolve()),
            frame_count=len(paths),
            width=width,
            height=height,
            process_res=int(external["process_res"]),
            inference_duration_s=max(1e-6, float(external["inference_duration_s"])),
            peak_vram_mib=float(external["peak_vram_mib"]),
            gpu_name=str(external["gpu_name"]),
            metadata={
                "phase": "9.6E2",
                "full_scene_temporal_coverage": True,
                "selected_keyframes": len(paths),
                **selection_meta,
                **alignment_meta,
            },
        )
        result_manifest_path = root / "da3_full_scene_result_manifest.json"
        result_manifest_path.write_text(json.dumps(result_manifest.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _emit(job_id, "progress", stage="canonicalizing", message="Canonicalizing full-scene DA3 supporting evidence.", progress_fraction=0.88)
        fingerprints_list = []
        for path in paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Unable to read selected keyframe for fingerprinting: {path}")
            fingerprints_list.append(source_image_sha256(bgr))
        fingerprints = tuple(fingerprints_list)
        integration = integrate_da3_result_to_canonical_scene(
            result_manifest,
            session_id=job_id,
            request_id=f"phase9-6e2-{job_id[:8]}",
            rgb_frames=_load_rgb(paths, width, height),
            frame_ids=tuple(f"da3-full-scene-{index:04d}" for index in range(len(paths))),
            source_frame_sha256=fingerprints,
            timestamps_ns=tuple(timestamps),
            source_label="phase9-6e2-recorded-video-full-scene",
            pixel_stride=int(args.pixel_stride),
            relative_scale_uncertainty=0.35,
        )
        scene, scene_manifest, surface_path = materialize_da3_canonical_persistent_scene(
            integration,
            root / "persistent-scene",
            input_manifest_path=input_manifest_path,
            result_manifest_path=result_manifest_path,
        )
        reloaded = read_persistent_reconstruction_scene(scene_manifest)
        verify_persistent_scene_file_references(reloaded)
        summary = {
            "schema_version": 1,
            "phase": "9.6E2",
            "job_id": job_id,
            "pass": True,
            "source_video": str(Path(args.video).expanduser().resolve()),
            "selected_keyframes": len(paths),
            "source_frame_indices": frame_indices,
            "window_count": len(windows),
            "persistent_scene_manifest": str(Path(scene_manifest).resolve()),
            "canonical_surface_path": str(Path(surface_path).resolve()),
            "da3_input_manifest": str(Path(input_manifest_path).resolve()),
            "da3_result_manifest": str(result_manifest_path.resolve()),
            "geometry_scale": "relative_arbitrary_scale",
            "unit_to_metre": None,
            "alignment_method": alignment_meta["alignment_method"],
            "intrinsics_consolidation": alignment_meta["intrinsics_consolidation"],
            "raw_focal_max_relative_deviation": alignment_meta["raw_focal_max_relative_deviation"],
            "raw_focal_p95_relative_deviation": alignment_meta["raw_focal_p95_relative_deviation"],
            "camera_calibrated": False,
            "loop_closure": False,
            "quality_certified": False,
            "canonical_geometry_authoritative": False,
            "metric_scale_authoritative": False,
            "measurement_authoritative": False,
            "qc_authoritative": False,
            "certified_metrology": False,
            "scene_id": scene.scene_id,
        }
        summary_path = root / "phase9_6e2_full_scene_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _emit(
            job_id,
            "ready",
            stage="completed",
            message=f"Full-scene DA3 geometry ready from {len(paths)} selected video keyframes.",
            progress_fraction=1.0,
            completed_units=len(windows),
            total_units=len(windows),
            result_path=str(summary_path.resolve()),
        )
        return 0
    except Exception as exc:
        _emit(job_id, "status", stage="failed", message=str(exc), error=str(exc))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
