#!/usr/bin/env python3
"""Dependency-light helpers for isolated Phase 8.7D3 benchmark workers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_WORKER_SCHEMA_VERSION = 1
SUPPORTED_BACKENDS = {"gsplat_external", "graphdeco_reference_external"}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_benchmark_request(
    request_path: Path,
    *,
    expected_backend_kind: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    request = load_json(request_path)
    if int(request.get("schema_version", 0)) != BENCHMARK_WORKER_SCHEMA_VERSION:
        raise ValueError("unsupported Gaussian benchmark worker schema")
    if expected_backend_kind not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported expected backend kind: {expected_backend_kind}")
    if request.get("backend_kind") != expected_backend_kind:
        raise ValueError("benchmark worker received a request for another backend")

    dataset_path = Path(str(request["dataset_manifest_path"])).resolve()
    seed_path = Path(str(request["seed_path"])).resolve()
    if not dataset_path.is_file() or not seed_path.is_file():
        raise ValueError("dataset manifest or Gaussian seed is missing")

    dataset = load_json(dataset_path)
    dataset_fingerprint = str(request.get("dataset_fingerprint_sha256", "")).lower()
    if dataset.get("dataset_fingerprint_sha256") != dataset_fingerprint:
        raise ValueError("dataset fingerprint mismatch")
    frames = dataset.get("frames")
    if not isinstance(frames, list) or len(frames) < 3:
        raise ValueError("real-scene benchmark requires at least three dataset frames")

    frame_ids = [str(frame.get("frame_id", "")) for frame in frames]
    if any(not value for value in frame_ids) or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("dataset frame IDs must be non-empty and unique")
    train = tuple(str(value) for value in request.get("training_frame_ids", ()))
    validation = tuple(str(value) for value in request.get("validation_frame_ids", ()))
    if len(train) < 2 or not validation:
        raise ValueError("benchmark request requires training and held-out validation views")
    if set(train).intersection(validation):
        raise ValueError("training and validation views must be disjoint")
    if set(train).union(validation) != set(frame_ids):
        raise ValueError("benchmark split must partition every dataset frame exactly once")

    benchmark_fingerprint = str(request.get("benchmark_fingerprint_sha256", "")).lower()
    if len(benchmark_fingerprint) != 64:
        raise ValueError("benchmark fingerprint must be SHA-256")

    with np.load(seed_path, allow_pickle=False) as loaded:
        required = {
            "positions_reconstruction",
            "colors_rgb",
            "log_scales_reconstruction",
            "opacities",
        }
        if not required.issubset(loaded.files):
            raise ValueError("Gaussian seed is missing required arrays")
        seed = {name: np.asarray(loaded[name]) for name in required}
    count = int(len(seed["positions_reconstruction"]))
    if count < 1 or any(len(seed[name]) != count for name in required):
        raise ValueError("Gaussian seed arrays must be non-empty and aligned")

    return request, dataset, seed


def frame_map(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(frame["frame_id"]): frame for frame in dataset["frames"]}


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_render_manifest(
    *,
    path: Path,
    request: dict[str, Any],
    renders: list[dict[str, Any]],
) -> Path:
    validation_ids = [str(value) for value in request["validation_frame_ids"]]
    render_ids = [str(value["frame_id"]) for value in renders]
    if len(render_ids) != len(validation_ids) or set(render_ids) != set(validation_ids):
        raise ValueError("held-out render set does not exactly match validation split")
    return write_json(
        path,
        {
            "schema_version": 1,
            "backend_kind": request["backend_kind"],
            "benchmark_fingerprint_sha256": request["benchmark_fingerprint_sha256"],
            "dataset_fingerprint_sha256": request["dataset_fingerprint_sha256"],
            "validation_frame_ids": validation_ids,
            "renders": renders,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    )


def write_backend_result(
    *,
    path: Path,
    request: dict[str, Any],
    backend_version: str,
    scene_path: Path,
    render_manifest_path: Path,
    gaussian_count: int,
    training_steps: int,
    training_duration_s: float,
    device: str,
) -> Path:
    return write_json(
        path,
        {
            "schema_version": 1,
            "backend_kind": request["backend_kind"],
            "backend_version": str(backend_version),
            "benchmark_fingerprint_sha256": request["benchmark_fingerprint_sha256"],
            "dataset_fingerprint_sha256": request["dataset_fingerprint_sha256"],
            "profile": request["profile"],
            "training_frame_ids": list(request["training_frame_ids"]),
            "validation_frame_ids": list(request["validation_frame_ids"]),
            "scene_path": str(scene_path.resolve()),
            "render_manifest_path": str(render_manifest_path.resolve()),
            "gaussian_count": int(gaussian_count),
            "training_steps": int(training_steps),
            "training_duration_s": float(training_duration_s),
            "device": str(device),
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    )
