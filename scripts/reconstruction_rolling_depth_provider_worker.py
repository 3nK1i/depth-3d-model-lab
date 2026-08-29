#!/usr/bin/env python3
"""Isolated official-RollingDepth worker for I4.3D2.

This module intentionally imports no project ``core`` package at module load.
It runs inside the dedicated RollingDepth environment/source checkout and only
returns native temporally co-aligned relative-depth evidence.  Canonical SfM
alignment/projection/fusion remains in the main application environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
PHASE = "I4.3D2"
DEPTH_SEMANTICS = "temporally_coaligned_relative_depth"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("phase") != PHASE:
        raise RuntimeError("invalid I4.3D2 input manifest")
    return payload


def _git_revision(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not resolve RollingDepth revision: {completed.stderr.strip()}")
    return completed.stdout.strip().lower()


def _as_positive_int(value: Any, *, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise RuntimeError(f"{name} must be positive")
    return result


def _try_enable_xformers(pipe: Any) -> bool:
    """Enable optional xFormers attention when a usable operator is available.

    RollingDepth can execute with the standard attention path.  An installed
    xFormers package may still be unusable for the active Torch/CUDA/GPU build;
    its operator probe raises ``NotImplementedError`` in that case.  Treat that
    exactly like an unavailable optional acceleration path rather than failing
    the isolated provider worker.
    """
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except (ImportError, ModuleNotFoundError, NotImplementedError):
        return False
    return True


def run(input_manifest: Path, output_dir: Path) -> Path:
    payload = _read_manifest(input_manifest)
    repo = Path(str(payload["rolling_depth_repository"])).expanduser().resolve()
    checkpoint = Path(str(payload["checkpoint_path"])).expanduser().resolve()
    video_info = payload.get("prepared_video")
    config = payload.get("config")
    if not isinstance(video_info, dict) or not isinstance(config, dict):
        raise RuntimeError("prepared_video/config missing from I4.3D2 manifest")
    video_path = Path(str(video_info["video_path"])).expanduser().resolve()
    expected_frames = _as_positive_int(payload["expected_frame_count"], name="expected_frame_count")
    expected_height = _as_positive_int(payload["expected_height"], name="expected_height")
    expected_width = _as_positive_int(payload["expected_width"], name="expected_width")
    source_revision = str(payload["source_revision"]).strip().lower()
    precision = str(payload.get("precision", "fp16")).strip().lower()
    device_text = str(payload.get("device", "cuda:0")).strip()

    if not (repo / "rollingdepth" / "__init__.py").is_file():
        raise RuntimeError("official RollingDepth source checkout is incomplete")
    if _git_revision(repo) != source_revision:
        raise RuntimeError("RollingDepth source revision drifted after environment probe")
    if not checkpoint.is_dir() or not (checkpoint / "model_index.json").is_file():
        raise RuntimeError("local RollingDepth checkpoint is missing/incomplete")
    if not video_path.is_file() or _sha256_file(video_path) != str(video_info.get("sha256", "")):
        raise RuntimeError("prepared lossless input video is missing or changed")
    if bool(payload.get("authority", {}).get("per_frame_alignment", True)):
        raise RuntimeError("I4.3D2 worker refuses per-frame reconstruction alignment")

    sys.path.insert(0, str(repo))
    import torch
    from rollingdepth import RollingDepthPipeline

    if precision == "fp16":
        dtype = torch.float16
    elif precision == "fp32":
        dtype = torch.float32
    else:
        raise RuntimeError(f"unsupported precision: {precision}")
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device = torch.device(device_text)

    pipe = RollingDepthPipeline.from_pretrained(str(checkpoint), torch_dtype=dtype)
    xformers_enabled = _try_enable_xformers(pipe)
    pipe = pipe.to(device)

    seed = int(config.get("seed", 42))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    dilations = [int(v) for v in config.get("dilations", [1, 25])]
    snippet_lengths = [int(v) for v in config.get("snippet_lengths", [3])]

    with torch.no_grad():
        pipe_out = pipe(
            input_video_path=video_path,
            start_frame=0,
            frame_count=expected_frames,
            processing_res=int(config.get("processing_resolution", 512)),
            resample_method="BILINEAR",
            dilations=dilations,
            cap_dilation=True,
            snippet_lengths=snippet_lengths,
            init_infer_steps=[1],
            strides=[1],
            coalign_kwargs=None,
            refine_step=int(config.get("refine_step", 0)),
            refine_snippet_len=3,
            refine_start_dilation=6,
            generator=generator,
            verbose=False,
            max_vae_bs=int(config.get("max_vae_batch_size", 1)),
            restore_res=bool(config.get("restore_resolution", True)),
            unload_snippet=bool(config.get("unload_snippet", True)),
        )
    depth_tensor = pipe_out.depth_pred
    depth = depth_tensor.detach().float().cpu().numpy()
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (expected_frames, expected_height, expected_width):
        raise RuntimeError(
            "RollingDepth output must be restored to the authoritative pixel grid: "
            f"got={depth.shape} expected={(expected_frames, expected_height, expected_width)}"
        )
    if not np.isfinite(depth).all() or float(np.ptp(depth)) <= 1e-9:
        raise RuntimeError("RollingDepth output is non-finite or has no relative-depth dynamic range")

    output_dir.mkdir(parents=True, exist_ok=False)
    npz_path = output_dir / "rolling_depth_evidence.npz"
    np.savez_compressed(npz_path, depth=depth)
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "pass",
        "source_revision": source_revision,
        "checkpoint_path": str(checkpoint),
        "frame_count": expected_frames,
        "height": expected_height,
        "width": expected_width,
        "depth_npz_path": str(npz_path.resolve()),
        "depth_semantics": DEPTH_SEMANTICS,
        "native_temporal_coalignment": True,
        "per_frame_alignment": False,
        "native_metric": False,
        "native_camera_pose_authority": False,
        "processing_resolution": int(config.get("processing_resolution", 512)),
        "dilations": dilations,
        "snippet_lengths": snippet_lengths,
        "refine_step": int(config.get("refine_step", 0)),
        "max_vae_batch_size": int(config.get("max_vae_batch_size", 1)),
        "unload_snippet": bool(config.get("unload_snippet", True)),
        "seed": seed,
        "precision": precision,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": str(torch.cuda.get_device_name(device)) if device.type == "cuda" else "",
        "xformers_enabled": xformers_enabled,
        "prepared_video_sha256": _sha256_file(video_path),
        "depth_npz_sha256": _sha256_file(npz_path),
    }
    result_path = output_dir / "rolling_depth_result_manifest.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(Path(args.input_manifest).resolve(), Path(args.output_dir).resolve())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
