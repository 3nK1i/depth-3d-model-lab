#!/usr/bin/env python3
"""I4.3E6 isolated Marigold production-provider worker.

Heavy model/CUDA imports remain inside run(). Relative depth, validity,
estimated intrinsics and optional uncertainty are persisted as native evidence.
Adapter-projected points are discarded; canonical projection occurs later with
PyCOLMAP cameras.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

I43E6_PHASE = "I4.3E6"
I43E6_SCHEMA_VERSION = 1
_ALLOWED_MODEL_IDS = {"marigold"}


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
    import cv2
    import numpy as np

    from core.models.registry import build_model, get_model_descriptor
    from core.models.validation import normalize_geometry_result
    from core.runtime.messages import FrameRequest

    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    payload = _read_json(manifest_path)
    if payload.get("phase") != I43E6_PHASE or int(payload.get("schema_version", -1)) != I43E6_SCHEMA_VERSION:
        raise ValueError("worker received an unsupported phase/schema manifest")
    model_id = str(payload.get("model_id", "")).strip()
    if model_id not in _ALLOWED_MODEL_IDS:
        raise ValueError(f"model_id {model_id!r} is not authorized by I4.3E6")
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
        prediction_type = str(model_metadata.get("prediction_type", runtime_config.get("prediction_type", "depth"))).lower()
        if prediction_type not in {"depth", "disparity"}:
            raise ValueError("Marigold runtime reported unsupported prediction_type")
        for index, item in enumerate(sequence):
            if not isinstance(item, dict):
                raise ValueError("sequence entries must be objects")
            if int(item.get("order", -1)) != index:
                raise ValueError("sequence order must remain contiguous")
            image_path = Path(str(item.get("image_path", ""))).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"staged image missing: {image_path}")
            if _sha256_file(image_path) != str(item.get("image_sha256", "")):
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
                raise ValueError("Marigold returned no relative-depth map")
            relative_depth = np.asarray(result.depth, dtype=np.float32)
            if relative_depth.shape != expected_hw:
                raise ValueError("Marigold depth was not restored to the input/SfM pixel grid")
            valid_mask = np.asarray(result.valid_mask, dtype=bool) if result.valid_mask is not None else (np.isfinite(relative_depth) & (relative_depth > 0.0))
            if valid_mask.shape != expected_hw:
                raise ValueError("Marigold valid mask pixel grid drifted")
            if result.is_metric is not False or result.depth_scale != "relative":
                raise ValueError("Marigold affine-invariant relative-depth semantics drifted")
            if result.intrinsics_source != "estimated" or result.intrinsics is None:
                raise ValueError("Marigold estimated-intrinsics semantics drifted")
            if result.normals is not None:
                raise ValueError("Marigold must not fabricate native normals")
            confidence_kind = str(result.confidence_kind or "none")
            if confidence_kind not in {"none", "uncertainty"}:
                raise ValueError("Marigold confidence semantics must be none or uncertainty")
            uncertainty = None
            if result.confidence is not None:
                if confidence_kind != "uncertainty":
                    raise ValueError("Marigold confidence map must be uncertainty")
                uncertainty = np.asarray(result.confidence, dtype=np.float32)
                if uncertainty.shape != expected_hw or not np.isfinite(uncertainty).all():
                    raise ValueError("Marigold uncertainty evidence is invalid")
                if np.any(uncertainty < 0.0):
                    raise ValueError("Marigold uncertainty must be non-negative")
            elif confidence_kind == "uncertainty":
                raise ValueError("Marigold uncertainty kind cannot be declared without an uncertainty map")

            estimated_intrinsics = np.asarray(result.intrinsics, dtype=np.float32)
            if estimated_intrinsics.shape != (3, 3) or not np.isfinite(estimated_intrinsics).all():
                raise ValueError("Marigold estimated intrinsics evidence must be finite 3x3")

            evidence_path = output_dir / f"frame_{index:05d}_{item['keyframe_id']}.npz"
            arrays = {
                "relative_depth": np.ascontiguousarray(relative_depth, dtype=np.float32),
                "valid_mask": np.ascontiguousarray(valid_mask, dtype=bool),
                "estimated_intrinsics": np.ascontiguousarray(estimated_intrinsics, dtype=np.float32),
            }
            if uncertainty is not None:
                arrays["uncertainty"] = np.ascontiguousarray(uncertainty, dtype=np.float32)
            np.savez_compressed(evidence_path, **arrays)
            records.append({
                "order": index,
                "keyframe_id": str(item["keyframe_id"]),
                "frame_id": str(item["frame_id"]),
                "source_sequence_index": int(item.get("source_sequence_index", index)),
                "evidence_npz_path": str(evidence_path.resolve()),
                "width": int(expected_hw[1]),
                "height": int(expected_hw[0]),
                "native_is_metric": False,
                "native_depth_scale": "relative",
                "depth_representation_native": "affine_invariant_relative_depth_or_disparity",
                "prediction_type": prediction_type,
                "intrinsics_source_native": "estimated",
                "confidence_kind_native": confidence_kind,
                "uncertainty_available": uncertainty is not None,
                "direct_points_native": False,
                "normals_native": False,
                "adapter_projected_points_discarded": True,
                "estimated_model_intrinsics_are_canonical_authority": False,
                "adapter_projected_points_are_canonical_geometry": False,
                "relative_depth_establishes_metric_scale": False,
                "latency_ms": float(result.latency_ms),
                "vram_peak_reserved_gib": result.vram_peak_reserved_gib,
            })
    finally:
        try:
            unload_metadata = dict(model.unload() or {})
        except Exception:  # noqa: BLE001
            unload_metadata = {"cleanup_failed": True}

    result_manifest = output_dir / "marigold_result_manifest.json"
    _write_json(result_manifest, {
        "schema_version": I43E6_SCHEMA_VERSION,
        "phase": I43E6_PHASE,
        "status": "pass",
        "model_id": model_id,
        "runtime_distribution": str(payload.get("runtime_distribution", "")),
        "runtime_distribution_version": str(payload.get("runtime_distribution_version", "")),
        "profile": str(payload.get("profile", "")),
        "checkpoint_repository": str(payload.get("checkpoint_repository", "")),
        "checkpoint_resolved_revision": str(payload.get("checkpoint_resolved_revision", "")),
        "checkpoint_path": str(payload.get("checkpoint_path", "")),
        "checkpoint_marker_sha256": str(payload.get("checkpoint_marker_sha256", "")),
        "model_index_sha256": str(payload.get("model_index_sha256", "")),
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
            "estimated_model_intrinsics_are_canonical_authority": False,
            "adapter_projected_points_are_canonical_geometry": False,
            "relative_depth_establishes_metric_scale": False,
        },
    })
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
