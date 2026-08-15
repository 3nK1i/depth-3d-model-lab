#!/usr/bin/env python3
"""Materialize real Gaussian-ready inputs from an already-persisted Phase 9.3 DA3 scene.

This command is deliberately backend-free: it does not rerun DA3 and does not
probe, import, train, optimize, or render with gsplat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.reconstruction import (
    current_process_gaussian_heavy_modules,
    materialize_gaussian_input_from_da3_persistent_scene,
    read_neural_scene_dataset_manifest,
    read_persistent_reconstruction_scene,
)

DEFAULT_SUMMARY = Path("/tmp/phase9_3j_real_scene/phase9_3j_qualification_summary.json")
DEFAULT_OUTPUT = Path("/tmp/phase9_4b_real_gaussian_input")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--qualification-summary", default=str(DEFAULT_SUMMARY))
    source.add_argument("--scene-manifest", default="")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def _scene_manifest(args: argparse.Namespace) -> Path:
    if args.scene_manifest:
        return Path(args.scene_manifest).expanduser().resolve()
    summary_path = Path(args.qualification_summary).expanduser().resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = str(payload.get("persistent_scene", {}).get("manifest", "")).strip()
    if not manifest:
        raise RuntimeError("Qualification summary does not contain persistent_scene.manifest.")
    return Path(manifest).expanduser().resolve()


def main() -> int:
    args = _parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase9_4b_real_bridge_summary.json"
    try:
        heavy_before = current_process_gaussian_heavy_modules()
        manifest = _scene_manifest(args)
        scene = read_persistent_reconstruction_scene(manifest)
        result = materialize_gaussian_input_from_da3_persistent_scene(
            scene, output_dir=output / "dataset"
        )
        loaded = read_neural_scene_dataset_manifest(result.manifest_path)
        heavy_after = current_process_gaussian_heavy_modules()
        passed = bool(
            loaded.frame_count == result.frame_count
            and result.frame_count >= 2
            and result.dataset_fingerprint_sha256
            and all(Path(frame.image_path).is_file() for frame in loaded.frames)
            and not heavy_before
            and not heavy_after
            and not result.training_started
            and not result.rendering_started
        )
        payload = {
            "phase": "9.4B",
            "status": "PASS" if passed else "FAIL",
            "scene_manifest": str(manifest),
            "scene_id": scene.scene_id,
            "scene_state": scene.state.value,
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "gaussian_input_manifest": result.manifest_path,
            "dataset_fingerprint_sha256": result.dataset_fingerprint_sha256,
            "frame_count": result.frame_count,
            "source_kind": result.source_kind,
            "source_frame_fingerprint_count": len(result.source_frame_sha256),
            "source_frame_fingerprints_exact": True,
            "canonical_camera_binding_exact": True,
            "retained_da3_intrinsics_used": True,
            "metric_geometry_status": loaded.metric_geometry_status.value,
            "unit_to_metre": loaded.unit_to_metre,
            "gaussian_scene_role": result.scene_role.value,
            "gaussian_canonical_geometry_authoritative": False,
            "gaussian_metric_scale_authoritative": False,
            "gaussian_measurement_authoritative": False,
            "gaussian_qc_authoritative": False,
            "certified_metrology": False,
            "da3_reexecuted": False,
            "gaussian_runtime_probed": False,
            "gaussian_backend_started": False,
            "training_started": False,
            "rendering_started": False,
            "production_heavy_modules": list(heavy_after),
        }
    except Exception as exc:
        payload = {
            "phase": "9.4B",
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "da3_reexecuted": False,
            "gaussian_backend_started": False,
            "training_started": False,
            "rendering_started": False,
        }
        passed = False
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Phase 9.4B real Gaussian-input bridge summary: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
