#!/usr/bin/env python3
"""Isolated I4.3D1 Video Depth Anything temporal-depth worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models.adapters.video_depth_anything_adapter import VideoDepthAnythingGeometryModel
from core.models.catalog import VIDEO_DEPTH_ANYTHING_MODEL_ID
from core.runtime.messages import FrameRequest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("I4.3D1 worker input manifest must be a JSON object.")
    return payload


def main() -> int:
    args = _args()
    manifest_path = args.input_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"I4.3D1 worker output directory must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(manifest_path)
    sequence = manifest.get("sequence")
    if not isinstance(sequence, list) or len(sequence) < 2:
        raise RuntimeError("I4.3D1 worker requires at least two ordered temporal frames.")
    model_id = str(manifest.get("model_id", ""))
    if model_id != VIDEO_DEPTH_ANYTHING_MODEL_ID:
        raise RuntimeError("I4.3D1 worker model_id mismatch.")
    runtime_config = manifest.get("runtime_config")
    if not isinstance(runtime_config, dict):
        raise RuntimeError("I4.3D1 worker runtime_config must be an object.")
    device = str(manifest.get("device", "")).strip()
    precision = str(manifest.get("precision", "")).strip().lower()
    source_revision = str(manifest.get("source_revision", "")).strip().lower()

    model = VideoDepthAnythingGeometryModel()
    depths: list[np.ndarray] = []
    frame_records: list[dict] = []
    started = time.perf_counter()
    try:
        model.load(device, precision, dict(runtime_config))
        model.warmup()
        for index, item in enumerate(sequence):
            if not isinstance(item, dict):
                raise RuntimeError("I4.3D1 worker sequence item must be an object.")
            path = Path(str(item.get("image_path", ""))).expanduser().resolve()
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"I4.3D1 worker could not decode frame: {path}")
            result = model.infer(
                FrameRequest(
                    session_id=1,
                    frame_id=index + 1,
                    image=np.ascontiguousarray(bgr, dtype=np.uint8),
                    point_budget=1,
                )
            )
            if result.depth is None:
                raise RuntimeError("I4.3D1 worker VDA inference returned no depth.")
            depth = np.asarray(result.depth, dtype=np.float32)
            if depth.ndim != 2 or depth.shape != bgr.shape[:2]:
                raise RuntimeError("I4.3D1 worker VDA depth did not restore to input frame resolution.")
            if not np.isfinite(depth).all() or np.any(depth < 0.0):
                raise RuntimeError("I4.3D1 worker VDA depth is invalid.")
            depths.append(np.ascontiguousarray(depth))
            frame_records.append(
                {
                    "order": index,
                    "keyframe_id": str(item.get("keyframe_id", "")),
                    "source_sequence_index": int(item.get("source_sequence_index", index)),
                    "timestamp_ns": int(item.get("timestamp_ns", 0)),
                    "latency_ms": float(result.latency_ms),
                    "depth_scale_native": result.depth_scale,
                    "intrinsics_source_native": result.intrinsics_source,
                    "native_is_metric": bool(result.is_metric),
                }
            )
    finally:
        try:
            model.unload()
        except Exception:
            pass

    shapes = {tuple(value.shape) for value in depths}
    if len(shapes) != 1:
        raise RuntimeError("I4.3D1 worker requires one stable frame size for temporal depth evidence.")
    stack = np.stack(depths, axis=0).astype(np.float32, copy=False)
    npz_path = output_dir / "vda_depth_evidence.npz"
    np.savez_compressed(npz_path, depth=stack)
    result_manifest = {
        "schema_version": 1,
        "phase": "I4.3D1",
        "status": "pass",
        "model_id": VIDEO_DEPTH_ANYTHING_MODEL_ID,
        "source_revision": source_revision,
        "frame_count": len(depths),
        "depth_npz_path": str(npz_path),
        "frame_records": frame_records,
        "elapsed_s": time.perf_counter() - started,
        "authority": {
            "native_metric_depth_preserved_as_evidence": True,
            "native_estimated_intrinsics_preserved_as_evidence": True,
            "native_metric_depth_promoted_to_canonical_metric_scale": False,
            "native_estimated_intrinsics_promoted_to_canonical_projection": False,
            "camera_pose_authority": "pycolmap_sfm",
        },
    }
    (output_dir / "vda_result_manifest.json").write_text(
        json.dumps(result_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
