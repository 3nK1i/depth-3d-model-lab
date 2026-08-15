#!/usr/bin/env python3
"""Persist the real Phase 9.4C gsplat artifact into the existing scene contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    GaussianSceneArtifact,
    current_process_gaussian_heavy_modules,
    persist_gaussian_representation,
    read_persistent_reconstruction_scene,
)

DEFAULT_OPTIMIZATION_SUMMARY = Path("/tmp/phase9_4c_real_gaussian_optimization/phase9_4c_real_optimization_summary.json")
DEFAULT_BRIDGE_SUMMARY = Path("/tmp/phase9_4b_real_gaussian_input/phase9_4b_real_bridge_summary.json")
DEFAULT_OUTPUT = Path("/tmp/phase9_4d_persistent_gaussian_scene")


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-summary", type=Path, default=DEFAULT_OPTIMIZATION_SUMMARY)
    parser.add_argument("--bridge-summary", type=Path, default=DEFAULT_BRIDGE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args()

    optimization_path = args.optimization_summary.expanduser().resolve()
    bridge_path = args.bridge_summary.expanduser().resolve()
    if not optimization_path.is_file():
        raise SystemExit(f"Phase 9.4C optimization summary is missing: {optimization_path}")
    if not bridge_path.is_file():
        raise SystemExit(f"Phase 9.4B bridge summary is missing: {bridge_path}")
    optimization = _json(optimization_path)
    bridge = _json(bridge_path)
    if optimization.get("status") != "PASS" or optimization.get("optimization_completed") is not True:
        raise SystemExit("Phase 9.4D refuses a non-PASS/non-completed 9.4C optimization summary.")
    if bridge.get("status") != "PASS":
        raise SystemExit("Phase 9.4D refuses a non-PASS 9.4B bridge summary.")
    if optimization.get("scene_id") != bridge.get("scene_id"):
        raise SystemExit("9.4B/9.4C scene identities differ.")
    if optimization.get("dataset_fingerprint_sha256") != bridge.get("dataset_fingerprint_sha256"):
        raise SystemExit("9.4B/9.4C dataset fingerprints differ.")
    if optimization.get("canonical_geometry_authoritative") is not True:
        raise SystemExit("9.4C source summary lost canonical geometry authority.")
    if optimization.get("gaussian_canonical_geometry_authoritative") is not False:
        raise SystemExit("9.4C Gaussian source summary claims forbidden canonical authority.")

    scene_manifest = Path(str(bridge["scene_manifest"])).expanduser().resolve()
    base_scene = read_persistent_reconstruction_scene(scene_manifest)
    artifact_manifest = Path(str(optimization["artifact_manifest"])).expanduser().resolve()
    artifact = GaussianSceneArtifact.from_payload(_json(artifact_manifest))
    if artifact.dataset_fingerprint_sha256 != str(bridge["dataset_fingerprint_sha256"]):
        raise SystemExit("Gaussian artifact fingerprint differs from 9.4B bridge evidence.")

    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.keep_output:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    result = persist_gaussian_representation(
        base_scene=base_scene,
        gaussian_artifact=artifact,
        preview_path=Path(str(optimization["preview_path"])),
        qualification_metrics_path=Path(str(optimization["qualification_metrics"])),
        source_artifact_manifest_path=artifact_manifest,
        source_optimization_summary_path=optimization_path,
        expected_dataset_fingerprint_sha256=str(bridge["dataset_fingerprint_sha256"]),
        output_dir=output,
    )
    summary = result.as_dict()
    summary.update(
        {
            "status": "PASS",
            "source_scene_manifest": str(scene_manifest),
            "source_optimization_summary": str(optimization_path),
            "source_bridge_summary": str(bridge_path),
            "source_scene_state": base_scene.state.value,
            "persistent_scene_state": result.scene.state.value,
            "persistent_scene_schema_version": result.scene.schema_version,
            "canonical_geometry_artifact_id": result.scene.canonical_geometry_artifact_id,
            "source_canonical_geometry_artifact_id": base_scene.canonical_geometry_artifact_id,
            "production_heavy_modules": list(current_process_gaussian_heavy_modules()),
            "gaussian_backend_reexecuted": False,
            "da3_reexecuted": False,
            "nerf_started": False,
        }
    )
    required = (
        summary["source_scene_unchanged"] is True
        and summary["restart_reload_verified"] is True
        and summary["canonical_geometry_artifact_id"] == summary["source_canonical_geometry_artifact_id"]
        and summary["canonical_geometry_authoritative"] is True
        and summary["gaussian_canonical_geometry_authoritative"] is False
        and summary["metric_scale_authoritative"] is False
        and summary["measurement_authoritative"] is False
        and summary["qc_authoritative"] is False
        and summary["certified_metrology"] is False
        and summary["backend_started"] is False
        and summary["production_heavy_modules"] == []
        and Path(str(summary["scene_manifest_path"])).is_file()
        and Path(str(summary["gaussian_scene_path"])).is_file()
    )
    if not required:
        summary["status"] = "FAIL"
    summary_path = output / "phase9_4d_persistence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Phase 9.4D persistent Gaussian summary: {summary_path}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
