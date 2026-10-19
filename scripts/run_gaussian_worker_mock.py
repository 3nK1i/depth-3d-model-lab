#!/usr/bin/env python3
"""External development worker for Phase 8.7D1 protocol validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

SCHEMA_VERSION = 1


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    request_path = Path(args.request).resolve()
    result_path = Path(args.result).resolve()
    request = _load_json(request_path)
    if int(request.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("unsupported request schema")
    if request.get("backend_kind") != "development_mock":
        raise ValueError("development mock only accepts development_mock backend requests")
    backend_provenance = request.get("backend_provenance")
    if not isinstance(backend_provenance, dict):
        raise ValueError("backend provenance missing from request")
    if backend_provenance.get("backend_kind") != "development_mock":
        raise ValueError("backend provenance does not match development mock request")
    dataset_path = Path(str(request["dataset_manifest_path"]))
    seed_path = Path(str(request["seed_path"]))
    output_dir = Path(str(request["output_dir"]))
    dataset = _load_json(dataset_path)
    if dataset.get("dataset_fingerprint_sha256") != request.get("dataset_fingerprint_sha256"):
        raise ValueError("dataset fingerprint mismatch")
    with np.load(seed_path, allow_pickle=False) as seed:
        positions = np.asarray(seed["positions_reconstruction"], dtype=np.float32)
        colors = np.asarray(seed["colors_rgb"], dtype=np.uint8)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 1:
            raise ValueError("invalid Gaussian seed")
        if colors.shape != (len(positions), 3):
            raise ValueError("invalid Gaussian seed colors")

    started = time.monotonic()
    artifact_dir = output_dir / "development-mock"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scene_path = artifact_dir / "seed_scene.npz"
    shutil.copy2(seed_path, scene_path)
    manifest_path = artifact_dir / "artifact_manifest.json"
    artifact_payload = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "backend_id": "development-gaussian-mock",
        "backend_version": "1",
        "backend_provenance": backend_provenance,
        "optimization_state": "seed_only",
        "scene_role": "appearance_only",
        "artifact_dir": str(artifact_dir.resolve()),
        "scene_path": str(scene_path.resolve()),
        "dataset_fingerprint_sha256": request["dataset_fingerprint_sha256"],
        "input_frame_count": int(dataset["frame_count"]),
        "gaussian_count": int(len(positions)),
        "training_steps": 0,
        "training_duration_s": float(time.monotonic() - started),
        "render_capable": False,
        "device": "development-mock-cpu",
        "peak_vram_mib": None,
        "pose_convention": dataset["pose_convention"],
        "reconstruction_frame_id": dataset["reconstruction_frame_id"],
        "metric_geometry_status": dataset["metric_geometry_status"],
        "unit_to_metre": dataset["unit_to_metre"],
        "relative_scale_uncertainty": dataset["relative_scale_uncertainty"],
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }
    manifest_path.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Gaussian development worker: seed-only artifact written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
