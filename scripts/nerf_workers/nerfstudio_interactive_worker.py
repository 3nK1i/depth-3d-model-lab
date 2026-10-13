#!/usr/bin/env python3
"""Phase 9.5G-hardened isolated Nerfstudio/Nerfacto interactive renderer.

The worker is intentionally line-protocol driven and owns every Nerfstudio,
tiny-cuda-nn and torch import.  The production ``depth`` interpreter sends only
camera/render requests and receives PNG paths plus timing evidence.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROTOCOL_PREFIX = "DEPTH3D_NERF_JSON "
EXPECTED_NERFSTUDIO_REVISION = "6b60855003011b2ca23c2fe3f8e2ca6314c69924"
EXPECTED_TCNN_REVISION = "b3473c81396fe927293bdfd5a6be32df8769927c"
MAX_RENDER_DIMENSION = 1280
MIN_RENDER_DIMENSION = 160
MAX_RETAINED_FRAMES = 2




def _cuda_telemetry(torch) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "total_vram_mib": 0.0,
            "cuda_allocated_mib": 0.0,
            "cuda_reserved_mib": 0.0,
            "cuda_peak_allocated_mib": 0.0,
            "cuda_peak_reserved_mib": 0.0,
        }
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    mib = 1024.0 * 1024.0
    return {
        "total_vram_mib": float(props.total_memory) / mib,
        "cuda_allocated_mib": float(torch.cuda.memory_allocated(device)) / mib,
        "cuda_reserved_mib": float(torch.cuda.memory_reserved(device)) / mib,
        "cuda_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device)) / mib,
        "cuda_peak_reserved_mib": float(torch.cuda.max_memory_reserved(device)) / mib,
    }


def _prune_frames(frames_dir: Path, *, keep: int = MAX_RETAINED_FRAMES) -> int:
    frames = sorted(frames_dir.glob("frame-*.png"), key=lambda item: item.stat().st_mtime_ns)
    while len(frames) > max(1, int(keep)):
        victim = frames.pop(0)
        try:
            victim.unlink()
        except FileNotFoundError:
            pass
    return sum(1 for _ in frames_dir.glob("frame-*.png"))

def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(PROTOCOL_PREFIX + json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def _git_state(root: Path, expected_revision: str) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=10,
    )
    if revision.returncode != 0:
        raise RuntimeError(f"cannot read Git revision for {root}: {revision.stderr.strip()}")
    value = revision.stdout.strip().lower()
    if value != expected_revision:
        raise RuntimeError(f"checkout revision mismatch for {root}: {value}")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=10,
    )
    if status.returncode != 0:
        raise RuntimeError(f"cannot read Git status for {root}: {status.stderr.strip()}")
    return value, not bool(status.stdout.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_load_spec(path: Path) -> dict[str, Any]:
    spec = _load_json(path, label="Phase 9.5D runtime load spec")
    if spec.get("phase") != "9.5D":
        raise RuntimeError("interactive renderer requires a Phase 9.5D load spec")
    if spec.get("backend_kind") != "nerfstudio_nerfacto_external":
        raise RuntimeError("interactive renderer requires isolated Nerfstudio/Nerfacto")
    if spec.get("method_name") != "nerfacto":
        raise RuntimeError("interactive renderer supports only the qualified Nerfacto method")
    if spec.get("load_strategy") != "isolated_staged_rebind_from_immutable_training_config":
        raise RuntimeError("unsupported persistent NeRF load strategy")
    if spec.get("scene_role") != "appearance_only":
        raise RuntimeError("persistent NeRF changed appearance-only authority")
    for key in (
        "canonical_geometry_authoritative",
        "metric_scale_authoritative",
        "measurement_authoritative",
        "qc_authoritative",
        "certified_metrology",
        "backend_start_performed",
        "rendering_performed",
    ):
        if bool(spec.get(key, False)):
            raise RuntimeError(f"persistent NeRF load spec has forbidden field {key}=true")
    return spec


def _existing_file(value: Any, *, label: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"{label} missing: {path}")
    return path


def _environment_value(
    name: str,
    *,
    legacy_name: str | None = None,
) -> str:
    raw = os.environ.get(name, "").strip()
    if not raw and legacy_name:
        raw = os.environ.get(legacy_name, "").strip()
    return raw


def _existing_dir(value: Any, *, label: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{label} missing: {path}")
    return path


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("step-"):
        raise RuntimeError(f"unsupported Nerfstudio checkpoint name: {path.name}")
    return int(stem.split("-", 1)[1])


def _load_pipeline(
    *,
    config_path: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    staging_root: Path,
):
    # Keep backend chatter out of the line protocol.  Only lines prefixed with
    # DEPTH3D_NERF_JSON are consumed by the application.
    with redirect_stdout(sys.stderr):
        import torch
        import tinycudann  # noqa: F401
        import nerfstudio  # noqa: F401
        from nerfstudio.utils.eval_utils import eval_setup

        step = _checkpoint_step(checkpoint_path)

        def update_config(config):
            # Nerfstudio 1.1.5 eval_setup intentionally overwrites ``load_dir``
            # with ``config.get_checkpoint_dir()`` after this callback.  9.5D
            # therefore specified an explicit staged-rebind strategy: point the
            # config output root at a temporary session directory, materialize the
            # immutable persisted checkpoint at the exact derived checkpoint
            # location, and leave the persisted training config untouched.
            config.data = dataset_root
            config.output_dir = staging_root / "nerfstudio-output"
            if hasattr(config, "load_step"):
                config.load_step = step
            expected_dir = Path(config.get_checkpoint_dir()).resolve()
            expected_dir.mkdir(parents=True, exist_ok=True)
            staged_checkpoint = expected_dir / f"step-{step:09d}.ckpt"
            if staged_checkpoint.exists():
                staged_checkpoint.unlink()
            try:
                os.link(checkpoint_path, staged_checkpoint)
            except OSError:
                shutil.copy2(checkpoint_path, staged_checkpoint)

            datamanager = getattr(getattr(config, "pipeline", None), "datamanager", None)
            if datamanager is not None and hasattr(datamanager, "data"):
                datamanager.data = dataset_root
            dataparser = getattr(datamanager, "dataparser", None)
            if dataparser is not None and hasattr(dataparser, "data"):
                dataparser.data = dataset_root
            if dataparser is not None:
                if hasattr(dataparser, "orientation_method"):
                    dataparser.orientation_method = "none"
                if hasattr(dataparser, "center_method"):
                    dataparser.center_method = "none"
                if hasattr(dataparser, "auto_scale_poses"):
                    dataparser.auto_scale_poses = False
                if hasattr(dataparser, "scale_factor"):
                    dataparser.scale_factor = 1.0
                if hasattr(dataparser, "eval_mode"):
                    dataparser.eval_mode = "filename"
            model_config = getattr(getattr(config, "pipeline", None), "model", None)
            camera_optimizer = getattr(model_config, "camera_optimizer", None)
            if camera_optimizer is not None and hasattr(camera_optimizer, "mode"):
                camera_optimizer.mode = "off"
            return config

        loaded_config, pipeline, loaded_checkpoint, loaded_step = eval_setup(
            config_path,
            test_mode="test",
            update_config_callback=update_config,
        )
        pipeline.eval()
        torch.cuda.synchronize()
    return torch, pipeline, Path(loaded_checkpoint).resolve(), int(loaded_step), loaded_config


def _render_request(torch, pipeline, request: dict[str, Any], frames_dir: Path) -> dict[str, Any]:
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from PIL import Image

    request_id = int(request.get("request_id", -1))
    width = int(request.get("width", 0))
    height = int(request.get("height", 0))
    if min(width, height) < MIN_RENDER_DIMENSION or max(width, height) > MAX_RENDER_DIMENSION:
        raise RuntimeError("interactive render resolution outside Phase 9.5E bounds")
    c2w = torch.tensor(request.get("camera_to_world_gl"), dtype=torch.float32)
    if tuple(c2w.shape) != (4, 4) or not bool(torch.isfinite(c2w).all().item()):
        raise RuntimeError("interactive render camera_to_world_gl must be finite 4x4")
    if request.get("scene_role") != "appearance_only":
        raise RuntimeError("interactive render request changed NeRF appearance authority")
    for key in (
        "canonical_geometry_authoritative",
        "metric_scale_authoritative",
        "measurement_authoritative",
        "qc_authoritative",
        "certified_metrology",
    ):
        if bool(request.get(key, False)):
            raise RuntimeError(f"interactive render request has forbidden authority {key}")

    camera = Cameras(
        camera_to_worlds=c2w[:3, :4].unsqueeze(0),
        fx=float(request["fx"]),
        fy=float(request["fy"]),
        cx=float(request["cx"]),
        cy=float(request["cy"]),
        width=width,
        height=height,
        camera_type=CameraType.PERSPECTIVE,
    ).to(pipeline.device)

    with redirect_stdout(sys.stderr):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with redirect_stdout(sys.stderr):
        outputs = pipeline.model.get_outputs_for_camera(camera)
        torch.cuda.synchronize()
    elapsed_ms = (time.monotonic() - started) * 1000.0
    rgb = outputs.get("rgb")
    if rgb is None or not bool(torch.isfinite(rgb).all().item()):
        raise RuntimeError("interactive Nerfacto render did not produce finite RGB")
    image = (
        rgb.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    )
    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"unexpected interactive RGB shape: {image.shape}")
    image = image[..., :3]
    frame_path = (frames_dir / f"frame-{request_id:06d}.png").resolve()
    Image.fromarray(image, mode="RGB").save(frame_path)
    telemetry = _cuda_telemetry(torch)
    retained = _prune_frames(frames_dir)
    return {
        "type": "frame",
        "status": "PASS",
        "request_id": request_id,
        "frame_path": str(frame_path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "render_ms": float(elapsed_ms),
        "scene_id": str(request.get("scene_id", "")),
        "artifact_id": str(request.get("artifact_id", "")),
        "retained_frame_count": int(retained),
        "max_retained_frames": MAX_RETAINED_FRAMES,
        "resource_hardening_phase": "9.5G",
        **telemetry,
        "scene_role": "appearance_only",
        "canonical_geometry_authoritative": False,
        "metric_scale_authoritative": False,
        "measurement_authoritative": False,
        "qc_authoritative": False,
        "certified_metrology": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-spec", required=True)
    parser.add_argument("--session-dir", required=True)
    args = parser.parse_args()

    session_dir = Path(args.session_dir).expanduser().resolve()
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    spec_path = _existing_file(args.load_spec, label="runtime load spec")
    try:
        spec = _validate_load_spec(spec_path)
        config = _existing_file(spec.get("immutable_training_config_path"), label="immutable training config")
        checkpoint = _existing_file(spec.get("checkpoint_path"), label="persisted checkpoint")
        dataset_root = _existing_dir(spec.get("dataset_root"), label="persisted dataset")
        transforms = _existing_file(spec.get("transforms_path"), label="persisted transforms")
        try:
            transforms.relative_to(dataset_root)
        except ValueError as exc:
            raise RuntimeError("persisted transforms escaped dataset root") from exc

        nerfstudio_root = _existing_dir(
            _environment_value(
                "DEPTH3D_NERFSTUDIO_ROOT",
                legacy_name="EIA_NERFSTUDIO_ROOT",
            ),
            label="Nerfstudio checkout",
        )
        tcnn_root = _existing_dir(
            _environment_value(
                "DEPTH3D_TCNN_ROOT",
                legacy_name="EIA_TCNN_ROOT",
            ),
            label="tiny-cuda-nn checkout",
        )
        ns_revision, ns_clean = _git_state(nerfstudio_root, EXPECTED_NERFSTUDIO_REVISION)
        tcnn_revision, tcnn_clean = _git_state(tcnn_root, EXPECTED_TCNN_REVISION)
        if not ns_clean or not tcnn_clean:
            raise RuntimeError("interactive NeRF refuses dirty external backend checkouts")

        staging_root = (session_dir / "runtime-staging").resolve()
        staging_root.mkdir(parents=True, exist_ok=True)
        torch, pipeline, loaded_checkpoint, step, _config = _load_pipeline(
            config_path=config,
            checkpoint_path=checkpoint,
            dataset_root=dataset_root,
            staging_root=staging_root,
        )
        if _sha256(loaded_checkpoint) != _sha256(checkpoint):
            raise RuntimeError(
                "interactive Nerfacto staged checkpoint does not match the persisted checkpoint"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("interactive Nerfacto requires CUDA on the qualified target GPU")
        gpu_name = torch.cuda.get_device_name(0)
        with redirect_stdout(sys.stderr):
            torch.cuda.reset_peak_memory_stats()
        ready_telemetry = _cuda_telemetry(torch)
        _emit(
            {
                "type": "ready",
                "status": "PASS",
                "phase": "9.5E",
                "resource_hardening_phase": "9.5G",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "staged_checkpoint_path": str(loaded_checkpoint),
                "staged_checkpoint_sha256": _sha256(loaded_checkpoint),
                "checkpoint_step": int(step),
                "load_strategy": "isolated_staged_rebind_from_immutable_training_config",
                "dataset_root": str(dataset_root),
                "dataset_fingerprint_sha256": str(spec.get("dataset_fingerprint_sha256", "")),
                "gpu_name": gpu_name,
                "worker_pid": int(os.getpid()),
                "retained_frame_count": int(_prune_frames(frames_dir)),
                "max_retained_frames": MAX_RETAINED_FRAMES,
                **ready_telemetry,
                "nerfstudio_revision": ns_revision,
                "tiny_cuda_nn_revision": tcnn_revision,
                "scene_role": "appearance_only",
                "canonical_geometry_authoritative": False,
                "metric_scale_authoritative": False,
                "measurement_authoritative": False,
                "qc_authoritative": False,
                "certified_metrology": False,
            }
        )

        for line in sys.stdin:
            raw = line.strip()
            if not raw:
                continue
            try:
                request = json.loads(raw)
                if not isinstance(request, dict):
                    raise RuntimeError("protocol request must be a JSON object")
                request_type = str(request.get("type", ""))
                if request_type == "shutdown":
                    _emit({
                        "type": "stopped",
                        "status": "PASS",
                        "phase": "9.5E",
                        "resource_hardening_phase": "9.5G",
                        "worker_pid": int(os.getpid()),
                        "retained_frame_count": int(_prune_frames(frames_dir)),
                        **_cuda_telemetry(torch),
                    })
                    return 0
                if request_type != "render":
                    raise RuntimeError(f"unsupported interactive NeRF request type: {request_type}")
                _emit(_render_request(torch, pipeline, request, frames_dir))
            except Exception as exc:  # keep the already-loaded renderer alive after a bad request
                _emit(
                    {
                        "type": "error",
                        "status": "FAIL",
                        "request_id": int(request.get("request_id", -1)) if isinstance(request, dict) else -1,
                        "error": str(exc),
                    }
                )
        return 0
    except Exception as exc:
        _emit({"type": "fatal", "status": "FAIL", "phase": "9.5E", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
