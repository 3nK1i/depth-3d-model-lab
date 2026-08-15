#!/usr/bin/env python3
"""Persist the real Phase 9.5C Nerfacto artifact into the existing scene schema."""
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
    NerfBackendKind,
    NerfSceneArtifact,
    NerfSceneRole,
    NerfTrainingState,
    current_process_nerf_heavy_modules,
    nerf_backend_provenance,
    persist_nerf_representation,
    read_neural_scene_dataset_manifest,
    read_persistent_reconstruction_scene,
)

DEFAULT_OPTIMIZATION_SUMMARY = Path("/tmp/phase9_5c_real_nerf_optimization/phase9_5c_real_optimization_summary.json")
DEFAULT_BRIDGE_SUMMARY = Path("/tmp/phase9_5b_real_nerf_input/phase9_5b_real_bridge_summary.json")
DEFAULT_OUTPUT = Path("/tmp/phase9_5d_persistent_nerf_scene")


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
        raise SystemExit(f"Phase 9.5C optimization summary is missing: {optimization_path}")
    if not bridge_path.is_file():
        raise SystemExit(f"Phase 9.5B bridge summary is missing: {bridge_path}")
    optimization = _json(optimization_path)
    bridge = _json(bridge_path)
    required_optimization = (
        optimization.get("status") == "PASS"
        and optimization.get("real_cuda_execution") is True
        and optimization.get("optimization_completed") is True
        and optimization.get("training_completed_without_oom") is True
        and optimization.get("preview_render_written") is True
        and optimization.get("render_capable") is True
        and optimization.get("nerf_scene_role") == "appearance_only"
        and optimization.get("nerf_canonical_geometry_authoritative") is False
        and optimization.get("certified_metrology") is False
    )
    if not required_optimization:
        raise SystemExit("Phase 9.5D refuses a non-qualified 9.5C optimization summary.")
    if bridge.get("status") != "PASS" or bridge.get("source_canonical_geometry_authoritative") is not True:
        raise SystemExit("Phase 9.5D refuses a non-qualified 9.5B bridge summary.")
    if optimization.get("scene_id") != bridge.get("scene_id"):
        raise SystemExit("9.5B/9.5C scene identities differ.")
    if optimization.get("dataset_fingerprint_sha256") != bridge.get("dataset_fingerprint_sha256"):
        raise SystemExit("9.5B/9.5C dataset fingerprints differ.")

    dataset_manifest = Path(str(optimization["dataset_manifest"])).expanduser().resolve()
    dataset = read_neural_scene_dataset_manifest(dataset_manifest)
    optimization_manifest_path = Path(str(optimization["optimization_manifest"])).expanduser().resolve()
    optimization_manifest = _json(optimization_manifest_path)
    runtime_evidence_path = Path(str(optimization["runtime_evidence_path"])).expanduser().resolve()
    runtime_evidence = _json(runtime_evidence_path)
    preview_path = Path(str(optimization["preview_path"])).expanduser().resolve()
    artifact_root = preview_path.parent.resolve()
    artifact = NerfSceneArtifact(
        request_id=str(optimization_manifest["request_id"]),
        backend_id="nerfstudio-nerfacto",
        backend_version=str(optimization["backend_version"]),
        backend_revision=str(optimization["backend_revision"]),
        backend_provenance=nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL),
        method_name=str(optimization["method_name"]),
        training_state=NerfTrainingState.TRAINED,
        scene_role=NerfSceneRole.APPEARANCE_ONLY,
        artifact_dir=str(artifact_root),
        config_path=str(Path(str(optimization["config_path"])).expanduser().resolve()),
        checkpoint_path=str(Path(str(optimization["checkpoint_path"])).expanduser().resolve()),
        dataset_fingerprint_sha256=str(optimization["dataset_fingerprint_sha256"]),
        input_frame_count=int(optimization["frame_count"]),
        training_steps=int(optimization["training_steps"]),
        training_duration_s=float(optimization["training_duration_s"]),
        render_capable=True,
        device=str(optimization_manifest["device"]),
        pose_convention=dataset.pose_convention,
        reconstruction_frame_id=dataset.reconstruction_frame_id,
        metric_geometry_status=dataset.metric_geometry_status,
        unit_to_metre=dataset.unit_to_metre,
        relative_scale_uncertainty=dataset.relative_scale_uncertainty,
        canonical_geometry_authoritative=False,
        certified_metrology=False,
    )
    if runtime_evidence.get("training_completed_without_oom") is not True:
        raise SystemExit("9.5C runtime evidence lost OOM-free training qualification.")

    scene_manifest = Path(str(bridge["scene_manifest"])).expanduser().resolve()
    base_scene = read_persistent_reconstruction_scene(scene_manifest)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.keep_output:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    result = persist_nerf_representation(
        base_scene=base_scene,
        nerf_artifact=artifact,
        preview_path=preview_path,
        runtime_evidence_path=runtime_evidence_path,
        source_optimization_manifest_path=optimization_manifest_path,
        source_optimization_summary_path=optimization_path,
        source_worker_console_path=Path(str(optimization["worker_console_log"])).expanduser().resolve(),
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
            "preview_path": str(Path(result.artifact.artifact_dir) / "preview.png"),
            "runtime_evidence_path": str(Path(result.artifact.artifact_dir) / "e2_runtime_evidence.json"),
            "production_heavy_modules": list(current_process_nerf_heavy_modules()),
            "nerf_backend_reexecuted": False,
            "nerf_training_reexecuted": False,
            "nerf_rendering_reexecuted": False,
            "da3_reexecuted": False,
            "gaussian_backend_started": False,
            "interactive_viewer_started": False,
            "quality_evaluated": False,
            "performance_qualified": False,
        }
    )
    required = (
        summary["source_scene_unchanged"] is True
        and summary["restart_reload_verified"] is True
        and summary["canonical_geometry_artifact_id"] == summary["source_canonical_geometry_artifact_id"]
        and summary["canonical_geometry_authoritative"] is True
        and summary["nerf_canonical_geometry_authoritative"] is False
        and summary["metric_scale_authoritative"] is False
        and summary["measurement_authoritative"] is False
        and summary["qc_authoritative"] is False
        and summary["certified_metrology"] is False
        and summary["backend_started"] is False
        and summary["rendering_started"] is False
        and summary["production_heavy_modules"] == []
        and Path(str(summary["scene_manifest_path"])).is_file()
        and Path(str(summary["nerf_checkpoint_path"])).is_file()
        and Path(str(summary["nerf_config_path"])).is_file()
        and Path(str(summary["preview_path"])).is_file()
        and Path(str(summary["dataset_root"]), "transforms.json").is_file()
    )
    if not required:
        summary["status"] = "FAIL"
    summary_path = output / "phase9_5d_persistence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Phase 9.5D persistent NeRF summary: {summary_path}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
