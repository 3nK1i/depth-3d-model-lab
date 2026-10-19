#!/usr/bin/env python3
"""Depth-environment worker for Phase 9.3H DA3 -> canonical integration.

The heavy DA3 model remains in the qualified ``depth3d-da3`` interpreter.  This
worker only orchestrates source materialization, external DA3 execution, and
in-process canonical dense/surface integration in the production ``depth``
environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.da3_canonical_scene_persistence import materialize_da3_canonical_persistent_scene
from core.reconstruction.da3 import (
    DA3_EFFICIENT_MODEL_ID,
    probe_da3_environment,
    read_da3_result_manifest,
)
from core.reconstruction.da3_canonical_scene import integrate_da3_result_to_canonical_scene
from core.reconstruction import read_persistent_reconstruction_scene, verify_persistent_scene_file_references
from core.reconstruction.da3_provider import (
    DA3ExternalWorkerRunner,
    DA3InputManifest,
    DA3ProviderProfile,
    read_da3_input_manifest,
    write_da3_input_manifest,
)
from core.reconstruction.sources import SUPPORTED_IMAGE_EXTENSIONS
from core.perception.reviewed_analysis import source_image_sha256

PROTOCOL = "depth3d.reconstruction.da3-canonical.v1"
DA3_WORKER = PROJECT_ROOT / "scripts" / "da3_workers" / "da3_worker.py"


def _event(kind: str, **payload) -> None:
    print(json.dumps({"protocol": PROTOCOL, "event": kind, **payload}, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--session-id", required=True)
    p.add_argument("--output-dir", required=True)
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--images-dir")
    source.add_argument("--video")
    p.add_argument("--da3-python", required=False, default="")
    p.add_argument("--da3-repo", required=False, default="")
    p.add_argument("--model", default=DA3_EFFICIENT_MODEL_ID)
    p.add_argument("--process-res", type=int, default=392)
    p.add_argument("--pixel-stride", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=3)
    p.add_argument("--timeout-seconds", type=float, default=900.0)
    # Dependency-light validation hook: reuse already-qualified DA3 artifacts.
    p.add_argument("--reuse-da3-input-manifest", default="")
    p.add_argument("--reuse-da3-result-manifest", default="")
    return p.parse_args()


def _select_image_paths(directory: Path, count: int) -> tuple[Path, ...]:
    images = tuple(
        item.resolve()
        for item in sorted(directory.iterdir())
        if item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if len(images) < 2:
        raise RuntimeError("DA3 canonical integration requires at least two supported source images.")
    take = min(max(2, count), len(images))
    start = max(0, (len(images) - take) // 2)
    return images[start : start + take]


def _materialize_video_frames(video: Path, output: Path, count: int) -> tuple[Path, ...]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open recorded video: {video}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        take = max(2, int(count))
        if total < take:
            raise RuntimeError("Recorded video does not contain enough frames for DA3 integration.")
        # Keep a compact overlapping burst around the temporal midpoint.  A small
        # interval is more useful to joint depth/pose than first/middle/last frames.
        stride = max(1, int(round(fps * 0.20))) if fps > 0 else 1
        # Bound the temporal stride so even a very short valid clip cannot
        # produce a frame index beyond the decoded range.
        max_stride = max(1, (total - 1) // max(1, take - 1))
        stride = min(stride, max_stride)
        span = stride * (take - 1)
        start = max(0, min(total - 1 - span, total // 2 - span // 2))
        output.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index in range(take):
            frame_index = start + index * stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Unable to decode video frame {frame_index}.")
            path = output / f"frame-{index:04d}.png"
            if not cv2.imwrite(str(path), bgr):
                raise RuntimeError(f"Unable to materialize video frame: {path}")
            paths.append(path.resolve())
        return tuple(paths)
    finally:
        cap.release()


def _qualified_da3_import_source(da3_python: Path, da3_repo: Path) -> Path:
    """Verify the external interpreter imports DA3 from the qualified checkout.

    The RTX-2060 compatibility patch is intentionally an external-checkout
    modification, so matching only the Git revision is insufficient: another
    checkout at the same revision could lack that patch.
    """

    code = (
        "import inspect; "
        "from depth_anything_3.api import DepthAnything3; "
        "print(inspect.getfile(DepthAnything3))"
    )
    completed = subprocess.run(
        [str(da3_python), "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30.0,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = completed.stderr.strip() or f"exit_{completed.returncode}"
        raise RuntimeError(f"Unable to resolve DA3 import source: {detail}")
    imported = Path(completed.stdout.strip().splitlines()[-1]).expanduser().resolve()
    repository = da3_repo.expanduser().resolve()
    try:
        imported.relative_to(repository)
    except ValueError as exc:
        raise RuntimeError(
            f"Qualified DA3 interpreter imports {imported}, outside expected checkout {repository}."
        ) from exc
    return imported




def _source_frame_fingerprints(paths: tuple[Path, ...]) -> tuple[str, ...]:
    fingerprints: list[str] = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unable to read DA3 source image for fingerprinting: {path}")
        fingerprints.append(source_image_sha256(bgr))
    return tuple(fingerprints)

def _load_rgb_frames(paths: tuple[Path, ...], *, width: int, height: int) -> tuple[np.ndarray, ...]:
    frames = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unable to read DA3 source image: {path}")
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return tuple(frames)


def _copy_reuse_artifacts(input_path: Path, result_path: Path, root: Path):
    source_input = read_da3_input_manifest(input_path)
    source_result = read_da3_result_manifest(result_path)
    if source_input.request_id != source_result.request_id:
        raise RuntimeError("Reused DA3 input/result request IDs differ.")
    reuse_root = root / "da3-reuse"
    reuse_root.mkdir(parents=True, exist_ok=True)
    copied_images = []
    for index, path_text in enumerate(source_input.image_paths):
        source = Path(path_text)
        destination = reuse_root / f"source-{index:04d}{source.suffix or '.png'}"
        shutil.copy2(source, destination)
        copied_images.append(str(destination.resolve()))
    copied_npz = reuse_root / "da3_output.npz"
    shutil.copy2(source_result.output_npz_path, copied_npz)
    request_id = f"phase9-3h-{root.name}"
    input_manifest = DA3InputManifest(
        request_id=request_id,
        image_paths=tuple(copied_images),
        metadata={"phase9_3h_reused_qualified_evidence": True},
    )
    input_manifest_path = write_da3_input_manifest(reuse_root / "input_manifest.json", input_manifest)
    payload = source_result.as_dict()
    payload["request_id"] = request_id
    payload["output_npz_path"] = str(copied_npz.resolve())
    result_manifest_path = reuse_root / "result_manifest.json"
    result_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return input_manifest_path, read_da3_result_manifest(result_manifest_path), result_manifest_path


def main() -> int:
    args = _parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    result_file = root / "offline_result.json"
    session_id = str(args.session_id).strip()
    try:
        _event("started", session_id=session_id)
        reuse = bool(args.reuse_da3_input_manifest or args.reuse_da3_result_manifest)
        if reuse:
            if not args.reuse_da3_input_manifest or not args.reuse_da3_result_manifest:
                raise RuntimeError("Both reuse DA3 manifests are required together.")
            _event("da3_reuse", runtime_probed=False, backend_started=False)
            input_manifest_path, manifest, result_manifest_path = _copy_reuse_artifacts(
                Path(args.reuse_da3_input_manifest).resolve(),
                Path(args.reuse_da3_result_manifest).resolve(),
                root,
            )
            input_manifest = read_da3_input_manifest(input_manifest_path)
            source_paths = tuple(Path(value).resolve() for value in input_manifest.image_paths)
            da3_runtime_probed = False
            da3_backend_started = False
            da3_runtime_lock_valid = False
            da3_turing_compatibility_patch_applied = False
            imported_da3_source = None
        else:
            if not args.images_dir and not args.video:
                raise RuntimeError("Image-sequence or recorded-video source is required.")
            da3_python = Path(args.da3_python).expanduser().resolve()
            da3_repo = Path(args.da3_repo).expanduser().resolve()
            _event("probing_da3", python=str(da3_python), repository=str(da3_repo))
            probe = probe_da3_environment(da3_python, da3_repo)
            if not probe.available:
                raise RuntimeError(f"Qualified isolated DA3 runtime is unavailable: {probe.error or 'qualification_failed'}")
            imported_da3_source = _qualified_da3_import_source(da3_python, da3_repo)
            _event(
                "da3_qualified",
                source_revision=probe.source_revision,
                devices=list(probe.device_names),
                runtime_lock_valid=probe.runtime_lock_valid,
                import_source=str(imported_da3_source),
            )
            if args.images_dir:
                source_paths = _select_image_paths(Path(args.images_dir).expanduser().resolve(), args.max_frames)
            else:
                source_paths = _materialize_video_frames(
                    Path(args.video).expanduser().resolve(), root / "materialized-video-frames", args.max_frames
                )
            request_id = f"phase9-3h-{session_id[:8]}"
            input_manifest = DA3InputManifest(
                request_id=request_id,
                image_paths=tuple(str(path) for path in source_paths),
                metadata={
                    "phase": "9.3H",
                    "calibration_required": False,
                    "metric_scale_qualified": False,
                },
            )
            input_manifest_path = write_da3_input_manifest(root / "da3_input_manifest.json", input_manifest)
            profile = DA3ProviderProfile(
                model_id=str(args.model),
                process_res=int(args.process_res),
            )
            _event("da3_inference_started", frame_count=len(source_paths), model_id=profile.model_id)
            runner = DA3ExternalWorkerRunner(
                python_executable=da3_python,
                worker_script=DA3_WORKER,
                profile=profile,
                timeout_s=float(args.timeout_seconds),
            )
            result_manifest_path, manifest = runner.run(input_manifest_path, root / "da3-worker-output")
            if manifest.source_revision != probe.source_revision:
                raise RuntimeError(
                    "DA3 worker source revision does not match the qualified external checkout."
                )
            if manifest.gpu_name not in probe.device_names:
                raise RuntimeError(
                    "DA3 worker GPU identity is not present in the qualified runtime probe."
                )
            da3_runtime_lock_valid = bool(probe.runtime_lock_valid)
            da3_turing_compatibility_patch_applied = bool(
                probe.turing_compatibility_patch_applied
            )
            _event(
                "da3_inference_completed",
                frame_count=manifest.frame_count,
                peak_vram_mib=manifest.peak_vram_mib,
                gpu_name=manifest.gpu_name,
            )
            da3_runtime_probed = True
            da3_backend_started = True

        if input_manifest.request_id != manifest.request_id:
            raise RuntimeError("DA3 input/result request identity mismatch.")
        source_fingerprints = _source_frame_fingerprints(source_paths)
        rgbs = _load_rgb_frames(source_paths, width=manifest.width, height=manifest.height)
        _event("canonicalizing", frame_count=manifest.frame_count, scale="relative_arbitrary_scale")
        integration = integrate_da3_result_to_canonical_scene(
            manifest,
            session_id=session_id,
            request_id=f"phase9-3h-{session_id[:8]}",
            rgb_frames=rgbs,
            frame_ids=tuple(f"da3-{index:04d}" for index in range(manifest.frame_count)),
            source_frame_sha256=source_fingerprints,
            source_label=("phase9-3h-recorded-video" if args.video else "phase9-3h-image-sequence"),
            pixel_stride=int(args.pixel_stride),
        )
        scene, scene_manifest, surface_path = materialize_da3_canonical_persistent_scene(
            integration,
            root / "persistent-scene",
            input_manifest_path=input_manifest_path,
            result_manifest_path=result_manifest_path,
        )
        # Fail before publishing a successful worker result unless the serialized
        # scene can be read back and its materialized canonical resources pass
        # integrity verification. The application repeats this check on admission.
        reloaded_scene = read_persistent_reconstruction_scene(scene_manifest)
        verify_persistent_scene_file_references(reloaded_scene)
        if reloaded_scene.scene_id != scene.scene_id or reloaded_scene.session_id != scene.session_id:
            raise RuntimeError("Serialized DA3 canonical scene identity changed during read-back verification.")
        payload = {
            "pass": True,
            "status": "completed",
            "phase": "9.3H",
            "session_id": session_id,
            "persistent_scene_ready": True,
            "persistent_scene_manifest": str(scene_manifest),
            "canonical_surface_obj": str(surface_path),
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
            "metric_geometry_status": "arbitrary",
            "unit_to_metre": None,
            "calibration_required": False,
            "calibration_available": False,
            "da3_supporting_authority_only": True,
            "da3_runtime_probed": da3_runtime_probed,
            "da3_backend_started": da3_backend_started,
            "da3_runtime_lock_valid": da3_runtime_lock_valid,
            "da3_turing_compatibility_patch_applied": da3_turing_compatibility_patch_applied,
            "da3_import_source": str(imported_da3_source) if imported_da3_source is not None else "",
            "da3_source_revision": manifest.source_revision,
            "da3_model_id": manifest.model_id,
            "da3_model_revision": manifest.model_revision,
            "da3_peak_vram_mib": manifest.peak_vram_mib,
            "da3_gpu_name": manifest.gpu_name,
            "camera_pose_available": True,
            "canonical_vertex_count": integration.surface.mesh.vertex_count,
            "canonical_face_count": integration.surface.mesh.face_count,
            "semantic_context_available": False,
            "inspection_qc_available": False,
            "semantic_geometry_fabricated": False,
            "inspection_qc_fabricated": False,
            "scene_state": scene.state.value,
        }
        result_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _event("completed", **payload)
        return 0
    except Exception as exc:  # worker boundary: serialize a bounded failure
        payload = {
            "pass": False,
            "status": "failed",
            "phase": "9.3H",
            "session_id": session_id,
            "persistent_scene_ready": False,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        try:
            result_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError:
            pass
        _event("failed", **payload)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
