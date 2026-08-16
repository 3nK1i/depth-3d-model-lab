#!/usr/bin/env python3
"""Resident external DA3 worker for Phase 9.6E2 bounded video windows."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from huggingface_hub import HfApi
from depth_anything_3.api import DepthAnything3


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    return parser.parse_args()


def _source_revision() -> str:
    api_file = Path(inspect.getfile(DepthAnything3)).resolve()
    repo = api_file.parents[2]
    if not (repo / ".git").exists():
        raise RuntimeError(f"Unable to resolve DA3 Git checkout from API module: {api_file}")
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    request = json.loads(Path(_args().request).read_text(encoding="utf-8"))
    images = [Path(item).resolve() for item in request["image_paths"]]
    windows = [tuple(int(index) for index in item) for item in request["windows"]]
    output_dir = Path(request["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(not image.is_file() for image in images):
        raise RuntimeError("Full-scene DA3 input keyframes are missing.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-scene DA3 execution.")
    model_id = str(request["model_id"])
    revision = str(request.get("model_revision", "")).strip().lower()
    if not revision:
        revision = str(HfApi().model_info(model_id).sha or "").strip().lower()
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise RuntimeError("Unable to resolve immutable DA3 model revision.")
    model = DepthAnything3.from_pretrained(model_id, revision=revision).to(torch.device("cuda"))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    source_revision = _source_revision()
    chunks = []
    total_duration = 0.0
    for ordinal, indices in enumerate(windows):
        paths = [str(images[index]) for index in indices]
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = model.inference(
            paths,
            process_res=int(request["process_res"]),
            process_res_method="upper_bound_resize",
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
        )
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        total_duration += duration
        depth = np.asarray(prediction.depth, dtype=np.float32)
        confidence = np.asarray(prediction.conf, dtype=np.float32)
        extrinsics = np.asarray(prediction.extrinsics, dtype=np.float32)
        intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)
        if depth.shape[0] != len(indices) or confidence.shape != depth.shape:
            raise RuntimeError("Unexpected DA3 full-scene depth/confidence shape.")
        chunk_path = output_dir / f"chunk-{ordinal:04d}.npz"
        np.savez_compressed(
            chunk_path,
            depth=depth,
            confidence=confidence,
            extrinsics_w2c=extrinsics,
            intrinsics=intrinsics,
        )
        chunk = {
            "ordinal": ordinal,
            "frame_indices": list(indices),
            "npz_path": str(chunk_path),
            "frame_count": int(depth.shape[0]),
            "height": int(depth.shape[1]),
            "width": int(depth.shape[2]),
            "inference_duration_s": float(duration),
        }
        chunks.append(chunk)
        print(json.dumps({"event": "window_completed", **chunk}, sort_keys=True), flush=True)
    summary = {
        "schema_version": 1,
        "model_id": model_id,
        "model_revision": revision,
        "source_revision": source_revision,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "peak_vram_mib": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
        "inference_duration_s": float(total_duration),
        "process_res": int(request["process_res"]),
        "chunks": chunks,
        "geometry_scale": "relative_arbitrary_scale",
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }
    summary_path = output_dir / "full_scene_external_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", "summary_path": str(summary_path)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
