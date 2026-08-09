#!/usr/bin/env python3
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = json.loads(Path(args.request).read_text())
    input_manifest = json.loads(Path(request["input_manifest_path"]).read_text())
    profile = request["profile"]
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [str(Path(p)) for p in input_manifest["image_paths"]]
    if not images or any(not Path(p).is_file() for p in images):
        raise RuntimeError("DA3 worker input images are missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for F2 real qualification")
    model_id = str(profile["model_id"])
    requested_model_revision = str(profile.get("model_revision", "")).strip().lower()
    if requested_model_revision:
        if len(requested_model_revision) != 40 or any(ch not in "0123456789abcdef" for ch in requested_model_revision):
            raise RuntimeError("DA3 model_revision must be a full 40-character commit when supplied")
        model_revision = requested_model_revision
    else:
        model_revision = str(HfApi().model_info(model_id).sha or "").strip().lower()
        if len(model_revision) != 40 or any(ch not in "0123456789abcdef" for ch in model_revision):
            raise RuntimeError(f"Unable to resolve immutable Hugging Face revision for {model_id}")
    model = DepthAnything3.from_pretrained(model_id, revision=model_revision)
    loaded_revision = str(getattr(model, "_commit_hash", "") or "").strip().lower()
    if loaded_revision and loaded_revision != model_revision:
        raise RuntimeError(f"DA3 model revision mismatch: requested {model_revision}, loaded {loaded_revision}")
    model = model.to(device=torch.device("cuda"))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    prediction = model.inference(
        images,
        process_res=int(profile["process_res"]),
        process_res_method=str(profile["process_res_method"]),
        use_ray_pose=bool(profile["use_ray_pose"]),
        ref_view_strategy=str(profile["ref_view_strategy"]),
    )
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    depth = np.asarray(prediction.depth, dtype=np.float32)
    confidence = np.asarray(prediction.conf, dtype=np.float32)
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float32)
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)
    if depth.ndim != 3 or confidence.shape != depth.shape:
        raise RuntimeError(f"unexpected depth/conf shapes: {depth.shape} {confidence.shape}")
    if extrinsics.shape not in {(depth.shape[0], 3, 4), (depth.shape[0], 4, 4)}:
        raise RuntimeError(f"unexpected extrinsics shape: {extrinsics.shape}")
    if intrinsics.shape != (depth.shape[0], 3, 3):
        raise RuntimeError(f"unexpected intrinsics shape: {intrinsics.shape}")
    npz_path = output_dir / "da3_output.npz"
    np.savez_compressed(
        npz_path,
        depth=depth,
        confidence=confidence,
        extrinsics_w2c=extrinsics,
        intrinsics=intrinsics,
    )
    repo_root = Path(__file__).resolve()
    source_revision = ""
    for parent in repo_root.parents:
        if (parent / ".git").exists() and (parent / "src/depth_anything_3").exists():
            source_revision = subprocess.check_output(["git", "-C", str(parent), "rev-parse", "HEAD"], text=True).strip()
            break
    if not source_revision:
        # The top-level depth_anything_3 package may be a namespace package and
        # therefore have __file__ == None. Resolve the concrete API module that
        # defines DepthAnything3 instead; editable installs point this file back
        # into the pinned external checkout under <repo>/src/depth_anything_3.
        api_file = Path(inspect.getfile(DepthAnything3)).resolve()
        da3_repo = api_file.parents[2]
        if not (da3_repo / ".git").exists():
            raise RuntimeError(f"Unable to resolve DA3 Git checkout from API module: {api_file}")
        source_revision = subprocess.check_output(
            ["git", "-C", str(da3_repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    payload = {
        "schema_version": 1,
        "request_id": str(input_manifest["request_id"]),
        "model_id": model_id,
        "model_revision": model_revision,
        "source_revision": source_revision,
        "output_npz_path": str(npz_path.resolve()),
        "frame_count": int(depth.shape[0]),
        "height": int(depth.shape[1]),
        "width": int(depth.shape[2]),
        "process_res": int(profile["process_res"]),
        "inference_duration_s": float(duration),
        "peak_vram_mib": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "camera_axes": "opencv_x_right_y_down_z_forward",
        "extrinsics_convention": "camera_from_reconstruction_w2c",
        "geometry_scale": "relative_arbitrary_scale",
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "metadata": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda or ""),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "use_ray_pose": bool(profile["use_ray_pose"]),
            "ref_view_strategy": str(profile["ref_view_strategy"]),
        },
    }
    (output_dir / "result_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
