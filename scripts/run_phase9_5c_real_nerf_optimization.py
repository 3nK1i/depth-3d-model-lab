#!/usr/bin/env python3
"""Run real Phase 9.5C Nerfstudio/Nerfacto optimization from a 9.5B bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    NerfOptimizationProfile,
    current_process_nerf_heavy_modules,
    probe_phase9_nerf_interactive_runtime,
    read_neural_scene_dataset_manifest,
    run_supervised_phase9_nerf_optimization,
)

DEFAULT_BRIDGE_SUMMARY = Path("/tmp/phase9_5b_real_nerf_input/phase9_5b_real_bridge_summary.json")
DEFAULT_OUTPUT = Path("/tmp/phase9_5c_real_nerf_optimization")


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-summary", type=Path, default=DEFAULT_BRIDGE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("efficient", "balanced", "high_quality"), default="efficient")
    parser.add_argument("--training-steps", type=int, default=64)
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args()

    bridge_path = args.bridge_summary.expanduser().resolve()
    if not bridge_path.is_file():
        raise SystemExit(f"Phase 9.5B real bridge summary is missing: {bridge_path}")
    bridge = _json(bridge_path)
    required_bridge = (
        bridge.get("status") == "PASS"
        and bridge.get("nerf_scene_role") == "appearance_only"
        and bridge.get("source_frame_fingerprints_exact") is True
        and bridge.get("canonical_camera_binding_exact") is True
        and bridge.get("source_canonical_geometry_authoritative") is True
        and bridge.get("nerf_backend_started") is False
        and bridge.get("training_started") is False
        and bridge.get("rendering_started") is False
    )
    if not required_bridge:
        raise SystemExit("Phase 9.5C refuses a non-qualified 9.5B bridge summary.")

    dataset_manifest = Path(str(bridge["nerf_input_manifest"])).expanduser().resolve()
    dataset = read_neural_scene_dataset_manifest(dataset_manifest)
    runtime = probe_phase9_nerf_interactive_runtime()
    if not runtime.ready:
        print(json.dumps(runtime.as_dict(), indent=2, sort_keys=True))
        raise SystemExit("Phase 9.5C isolated Nerfstudio runtime is not READY.")

    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.keep_output:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    worker = (ROOT / "scripts" / "nerf_workers" / "nerfstudio_worker.py").resolve()
    report = run_supervised_phase9_nerf_optimization(
        runtime_report=runtime,
        dataset=dataset,
        dataset_manifest_path=dataset_manifest,
        output_dir=output,
        worker_script=worker,
        request_id=f"phase9-5c-{uuid.uuid4()}",
        profile=NerfOptimizationProfile(args.profile),
        training_steps=int(args.training_steps),
        maximum_input_frames=max(3, dataset.frame_count),
    )

    evidence = _json(Path(report.runtime_evidence_path))
    artifact = report.artifact
    summary = {
        "phase": "9.5C",
        "status": "PASS",
        "source_bridge_summary": str(bridge_path),
        "scene_id": bridge["scene_id"],
        "dataset_manifest": str(dataset_manifest),
        "dataset_fingerprint_sha256": artifact.dataset_fingerprint_sha256,
        "frame_count": artifact.input_frame_count,
        "backend_kind": artifact.backend_provenance.backend_kind.value,
        "backend_version": artifact.backend_version,
        "backend_revision": artifact.backend_revision,
        "method_name": artifact.method_name,
        "runtime_python": runtime.spec.python_executable,
        "runtime_checkout_revision": runtime.nerfstudio_checkout.revision,
        "runtime_checkout_clean": runtime.nerfstudio_checkout.clean,
        "tiny_cuda_nn_revision": runtime.tiny_cuda_nn_checkout.revision,
        "tiny_cuda_nn_checkout_clean": runtime.tiny_cuda_nn_checkout.clean,
        "gpu_name": evidence.get("gpu_name", ""),
        "compute_capability": evidence.get("compute_capability", []),
        "cuda_visible_vram_mib": evidence.get("total_vram_mib"),
        "training_steps": artifact.training_steps,
        "training_duration_s": artifact.training_duration_s,
        "training_completed_without_oom": evidence.get("training_completed_without_oom") is True,
        "real_cuda_execution": str(artifact.device).lower().startswith("cuda:"),
        "optimization_completed": artifact.training_state.value == "trained",
        "preview_render_written": Path(report.preview_path).is_file(),
        "render_capable": artifact.render_capable,
        "config_path": artifact.config_path,
        "checkpoint_path": artifact.checkpoint_path,
        "preview_path": report.preview_path,
        "runtime_evidence_path": report.runtime_evidence_path,
        "optimization_manifest": report.optimization_manifest_path,
        "worker_console_log": report.worker_console_log_path,
        "nerf_scene_role": artifact.scene_role.value,
        "source_canonical_geometry_authoritative": bridge["source_canonical_geometry_authoritative"],
        "nerf_canonical_geometry_authoritative": artifact.canonical_geometry_authoritative,
        "nerf_metric_scale_authoritative": False,
        "nerf_measurement_authoritative": False,
        "nerf_qc_authoritative": False,
        "certified_metrology": artifact.certified_metrology,
        "metric_geometry_status": artifact.metric_geometry_status.value,
        "unit_to_metre": artifact.unit_to_metre,
        "quality_evaluated": False,
        "performance_qualified": False,
        "production_heavy_modules": list(current_process_nerf_heavy_modules()),
        "da3_reexecuted": False,
        "gaussian_backend_started": False,
        "persistence_started": False,
        "interactive_viewer_started": False,
    }
    required = (
        summary["real_cuda_execution"] is True
        and summary["optimization_completed"] is True
        and summary["preview_render_written"] is True
        and summary["training_completed_without_oom"] is True
        and summary["render_capable"] is True
        and summary["runtime_checkout_clean"] is True
        and summary["tiny_cuda_nn_checkout_clean"] is True
        and summary["nerf_canonical_geometry_authoritative"] is False
        and summary["certified_metrology"] is False
        and summary["production_heavy_modules"] == []
        and Path(str(summary["checkpoint_path"])).is_file()
        and Path(str(summary["config_path"])).is_file()
    )
    if not required:
        summary["status"] = "FAIL"
    summary_path = output / "phase9_5c_real_optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Phase 9.5C real NeRF optimization summary: {summary_path}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
