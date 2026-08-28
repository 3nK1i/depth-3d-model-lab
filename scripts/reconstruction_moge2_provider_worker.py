#!/usr/bin/env python3
"""I4.3E3 isolated MoGe-2 production-provider worker.

Heavy model/CUDA imports remain inside run(). MoGe native points, normals and
predicted intrinsics are retained only as evidence; canonical geometry is built
from depth later using authoritative PyCOLMAP cameras.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

I43E3_PHASE = "I4.3E3"
I43E3_SCHEMA_VERSION = 1
_ALLOWED_MODEL_IDS = {"moge-2"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("worker manifest root must be an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(manifest_path: Path, output_dir: Path) -> Path:
    # Heavy/project imports remain inside run() so importing this worker does not
    # allocate CUDA state or load optional model runtimes.
    import cv2
    import numpy as np

    from core.models.registry import build_model, get_model_descriptor
    from core.models.validation import normalize_geometry_result
    from core.runtime.messages import FrameRequest

    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    payload = _read_json(manifest_path)
    if payload.get("phase") != I43E3_PHASE or int(payload.get("schema_version", -1)) != I43E3_SCHEMA_VERSION:
        raise ValueError("worker received an unsupported phase/schema manifest")
    model_id = str(payload.get("model_id", "")).strip()
    if model_id not in _ALLOWED_MODEL_IDS:
        raise ValueError(f"model_id {model_id!r} is not authorized by I4.3E3")
    sequence = payload.get("sequence")
    if not isinstance(sequence, list) or len(sequence) < 2:
        raise ValueError("worker requires at least two registered image records")
    runtime_config = payload.get("runtime_config")
    if not isinstance(runtime_config, dict):
        raise ValueError("runtime_config must be an object")
    device = str(payload.get("device", "cpu"))
    precision = str(payload.get("precision", "fp32"))

    descriptor = get_model_descriptor(model_id)
    model = build_model(model_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    unload_metadata: dict[str, Any] = {}
    model_metadata: dict[str, Any] = {}
    try:
        model.load(device=device, precision=precision, config=dict(runtime_config))
        model.warmup()
        model_metadata = dict(model.get_metadata() or {})
        for index, item in enumerate(sequence):
            if not isinstance(item, dict):
                raise ValueError("sequence entries must be objects")
            if int(item.get("order", -1)) != index:
                raise ValueError("sequence order must remain contiguous")
            image_path = Path(str(item.get("image_path", ""))).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"staged image missing: {image_path}")
            expected_sha = str(item.get("image_sha256", ""))
            if _sha256_file(image_path) != expected_sha:
                raise ValueError(f"staged image checksum mismatch: {item.get('keyframe_id')}")
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"could not decode staged image: {image_path}")
            expected_hw = (int(item.get("height", 0)), int(item.get("width", 0)))
            if bgr.shape[:2] != expected_hw:
                raise ValueError("decoded image size drifted from authoritative SfM manifest")

            request = FrameRequest(
                session_id=1,
                frame_id=index + 1,
                image=bgr,
                submitted_at=time.monotonic(),
                source_timestamp=float(item.get("timestamp_ns", 0)) / 1_000_000_000.0,
            )
            result = normalize_geometry_result(model.infer(request), descriptor, request)
            if result.depth is None:
                raise ValueError("MoGe-2 returned no depth map")
            depth = np.asarray(result.depth, dtype=np.float32)
            if depth.shape != expected_hw:
                raise ValueError("MoGe-2 depth was not restored to the input/SfM pixel grid")
            valid_mask = (
                np.asarray(result.valid_mask, dtype=bool)
                if result.valid_mask is not None
                else (np.isfinite(depth) & (depth > 0.0))
            )
            if valid_mask.shape != expected_hw:
                raise ValueError("MoGe-2 valid mask pixel grid drifted")
            if result.is_metric is not True or result.depth_scale != "metric":
                raise ValueError("MoGe-2 native metric-depth semantics drifted")
            if result.intrinsics_source != "model_predicted" or result.intrinsics is None:
                raise ValueError("MoGe-2 native model-predicted intrinsics semantics drifted")
            if result.confidence_kind not in {"", "none"} or result.confidence is not None:
                raise ValueError("MoGe-2 must remain a no-confidence provider")
            if result.points is None:
                raise ValueError("MoGe-2 direct 3D-points evidence is missing")
            if result.normals is None:
                raise ValueError("MoGe-2 normal evidence is missing")

            model_intrinsics = np.asarray(result.intrinsics, dtype=np.float32)
            if model_intrinsics.shape != (3, 3) or not np.isfinite(model_intrinsics).all():
                raise ValueError("MoGe-2 intrinsics evidence must be finite 3x3")
            direct_points = np.asarray(result.points, dtype=np.float32)
            normals = np.asarray(result.normals, dtype=np.float32)
            if direct_points.ndim != 3 or direct_points.shape[-1] != 3 or not np.isfinite(direct_points).all():
                raise ValueError("MoGe-2 direct points must be finite HxWx3 model evidence")
            if normals.ndim != 3 or normals.shape[-1] != 3 or not np.isfinite(normals).all():
                raise ValueError("MoGe-2 normals must be finite HxWx3 model evidence")

            evidence_path = output_dir / f"frame_{index:05d}_{item['keyframe_id']}.npz"
            np.savez_compressed(
                evidence_path,
                depth=np.ascontiguousarray(depth, dtype=np.float32),
                valid_mask=np.ascontiguousarray(valid_mask, dtype=bool),
                direct_points_model=np.ascontiguousarray(direct_points, dtype=np.float32),
                normals_model=np.ascontiguousarray(normals, dtype=np.float32),
                model_intrinsics=np.ascontiguousarray(model_intrinsics, dtype=np.float32),
            )
            records.append(
                {
                    "order": index,
                    "keyframe_id": str(item["keyframe_id"]),
                    "frame_id": str(item["frame_id"]),
                    "source_sequence_index": int(item.get("source_sequence_index", index)),
                    "evidence_npz_path": str(evidence_path.resolve()),
                    "width": int(expected_hw[1]),
                    "height": int(expected_hw[0]),
                    "native_is_metric": True,
                    "native_depth_scale": "metric",
                    "intrinsics_source_native": "model_predicted",
                    "confidence_kind_native": "none",
                    "direct_points_native": True,
                    "normals_native": True,
                    "model_intrinsics_are_canonical_authority": False,
                    "model_direct_points_are_canonical_geometry": False,
                    "model_normals_are_canonical_geometry": False,
                    "model_metric_depth_establishes_metric_scale": False,
                    "latency_ms": float(result.latency_ms),
                    "vram_peak_reserved_gib": result.vram_peak_reserved_gib,
                }
            )
    finally:
        try:
            unload_metadata = dict(model.unload() or {})
        except Exception:  # noqa: BLE001 - do not mask the primary worker failure
            unload_metadata = {"cleanup_failed": True}

    result_manifest = output_dir / "moge2_result_manifest.json"
    _write_json(
        result_manifest,
        {
            "schema_version": I43E3_SCHEMA_VERSION,
            "phase": I43E3_PHASE,
            "status": "pass",
            "model_id": model_id,
            "source_revision": str(payload.get("source_revision", "")),
            "checkpoint_repository": str(payload.get("checkpoint_repository", "")),
            "checkpoint_revision": str(payload.get("checkpoint_revision", "")),
            "frame_count": len(records),
            "frame_records": records,
            "runtime": {
                "device": device,
                "precision": precision,
                "elapsed_s": time.perf_counter() - started,
                "model_metadata": model_metadata,
                "unload_metadata": unload_metadata,
            },
            "authority": {
                "camera_pose_authority_owner": "pycolmap_sfm",
                "model_intrinsics_are_canonical_authority": False,
                "model_direct_points_are_canonical_geometry": False,
                "model_normals_are_canonical_geometry": False,
                "model_metric_depth_establishes_metric_scale": False,
            },
        },
    )
    return result_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.manifest, args.output_dir)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
