#!/usr/bin/env python3
"""I4.1.6B DA3 worker: load the model once, infer candidate frames one-by-one.

The one-frame inference shape deliberately matches the historical I1/E3 DA3
usage while avoiding one Python/model reload per candidate.  Each candidate is
saved independently so an interrupted run can safely resume without changing
reconstruction policy.
"""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", required=True)
    return p.parse_args()


def _resolve_source_revision() -> str:
    api_file = Path(inspect.getfile(DepthAnything3)).resolve()
    for parent in api_file.parents:
        if (parent / ".git").exists() and (parent / "src/depth_anything_3").exists():
            return subprocess.check_output(["git", "-C", str(parent), "rev-parse", "HEAD"], text=True).strip()
    raise RuntimeError(f"Unable to resolve DA3 Git checkout from API module: {api_file}")


def _valid_cached(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            depth = np.asarray(payload["depth"], dtype=np.float32)
            conf = np.asarray(payload["confidence"], dtype=np.float32)
        return depth.ndim == 2 and conf.shape == depth.shape and depth.size > 0 and np.isfinite(depth).all() and np.all(depth > 0.0)
    except Exception:
        return False


def main() -> int:
    args = _args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output = Path(request["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = request.get("images")
    profile = request.get("profile") or {}
    if not isinstance(records, list) or not records:
        raise RuntimeError("I4.1.6B worker requires non-empty images list")
    parsed = []
    for item in records:
        source_index = int(item["source_sequence_index"])
        path = Path(item["path"]).resolve()
        if source_index < 0 or not path.is_file():
            raise RuntimeError(f"Invalid I4.1.6B candidate record: {item}")
        parsed.append((source_index, path))
    if len({idx for idx, _ in parsed}) != len(parsed):
        raise RuntimeError("Duplicate source sequence indices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for I4.1.6B real DA3 execution")

    model_id = str(profile["model_id"])
    requested_revision = str(profile.get("model_revision", "")).strip().lower()
    if requested_revision:
        model_revision = requested_revision
    else:
        model_revision = str(HfApi().model_info(model_id).sha or "").strip().lower()
    if len(model_revision) != 40 or any(ch not in "0123456789abcdef" for ch in model_revision):
        raise RuntimeError(f"Unable to resolve immutable DA3 model revision for {model_id}")

    pending = [(idx, path) for idx, path in parsed if not _valid_cached(output / f"candidate_{idx:08d}.npz")]
    source_revision = _resolve_source_revision()
    start_all = time.perf_counter()
    model = None
    processed = 0
    reused = len(parsed) - len(pending)
    durations: list[float] = []
    peak_mib = 0.0
    gpu_name = str(torch.cuda.get_device_name(0))
    try:
        if pending:
            model = DepthAnything3.from_pretrained(model_id, revision=model_revision)
            loaded_revision = str(getattr(model, "_commit_hash", "") or "").strip().lower()
            if loaded_revision and loaded_revision != model_revision:
                raise RuntimeError(f"DA3 model revision mismatch: requested {model_revision}, loaded {loaded_revision}")
            model = model.to(device=torch.device("cuda"))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for ordinal, (source_index, path) in enumerate(pending, start=1):
                print(f"I4.1.6B DA3 {ordinal}/{len(pending)} source_index={source_index}", flush=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                prediction = model.inference(
                    [str(path)],
                    process_res=int(profile.get("process_res", 392)),
                    process_res_method=str(profile.get("process_res_method", "upper_bound_resize")),
                    use_ray_pose=False,
                    ref_view_strategy=str(profile.get("ref_view_strategy", "saddle_balanced")),
                )
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
                depth = np.asarray(prediction.depth, dtype=np.float32)
                confidence = np.asarray(prediction.conf, dtype=np.float32)
                if depth.ndim != 3 or depth.shape[0] != 1 or confidence.shape != depth.shape:
                    raise RuntimeError(f"Unexpected DA3 candidate shapes: {depth.shape} {confidence.shape}")
                if not np.isfinite(depth).all() or np.any(depth <= 0.0):
                    raise RuntimeError(f"DA3 candidate {source_index} produced invalid depth")
                np.savez_compressed(
                    output / f"candidate_{source_index:08d}.npz",
                    depth=depth[0],
                    confidence=confidence[0],
                    source_sequence_index=np.asarray(source_index, dtype=np.int64),
                )
                processed += 1
                durations.append(float(duration))
                peak_mib = max(peak_mib, float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)))
                if ordinal % 12 == 0:
                    torch.cuda.empty_cache()
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "phase": "I4.1.6B",
        "model_id": model_id,
        "model_revision": model_revision,
        "source_revision": source_revision,
        "candidate_count": len(parsed),
        "processed_candidate_count": processed,
        "reused_candidate_count": reused,
        "process_res": int(profile.get("process_res", 392)),
        "inference_mode": "single_frame_sequential_model_loaded_once",
        "inference_duration_s_total": float(sum(durations)),
        "wall_duration_s": float(time.perf_counter() - start_all),
        "inference_duration_s_p50": None if not durations else float(np.median(np.asarray(durations))),
        "peak_vram_mib": peak_mib,
        "gpu_name": gpu_name,
        "canonical_geometry_authoritative": False,
        "metric_scale_claimed": False,
        "certified_metrology": False,
    }
    (output / "i4_1_6b_da3_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
