#!/usr/bin/env python3
"""Run the real Phase 9.4C gsplat optimization on a 9.4B canonical input bundle."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from core.reconstruction import (
    GaussianCanonicalSeedEvidence,
    GaussianOptimizationProfile,
    build_gaussian_seed_from_canonical_arrays,
    current_process_gaussian_heavy_modules,
    probe_phase9_gaussian_interactive_runtime,
    read_neural_scene_dataset_manifest,
    read_persistent_reconstruction_scene,
    run_supervised_phase9_gaussian_optimization,
)

DEFAULT_BRIDGE_SUMMARY = Path("/tmp/phase9_4b_real_gaussian_input/phase9_4b_real_bridge_summary.json")
DEFAULT_OUTPUT = Path("/tmp/phase9_4c_real_gaussian_optimization")


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-summary", type=Path, default=DEFAULT_BRIDGE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("efficient", "balanced", "high_quality"), default="efficient")
    parser.add_argument("--max-gaussians", type=int, default=2048)
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args()

    summary_path = args.bridge_summary.expanduser().resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Phase 9.4B real bridge summary is missing: {summary_path}")
    bridge = _json(summary_path)
    if bridge.get("status") != "PASS":
        raise SystemExit("Phase 9.4C refuses a non-PASS 9.4B bridge summary.")
    if bridge.get("gaussian_scene_role") != "appearance_only":
        raise SystemExit("Phase 9.4B Gaussian role is not appearance_only.")
    if bridge.get("gaussian_backend_started") is not False or bridge.get("training_started") is not False:
        raise SystemExit("Phase 9.4B bridge unexpectedly reports backend/training execution.")

    scene_manifest = Path(str(bridge["scene_manifest"])).expanduser().resolve()
    dataset_manifest = Path(str(bridge["gaussian_input_manifest"])).expanduser().resolve()
    scene = read_persistent_reconstruction_scene(scene_manifest)
    dataset = read_neural_scene_dataset_manifest(dataset_manifest)
    if scene.scene_id != str(bridge["scene_id"]):
        raise SystemExit("Persistent scene ID differs from the 9.4B bridge summary.")

    mesh = load_canonical_viewer_mesh(scene)
    if mesh is None:
        raise SystemExit("Persistent canonical surface is unavailable for Gaussian seeding.")
    seed = build_gaussian_seed_from_canonical_arrays(
        mesh.vertices_reconstruction,
        mesh.faces,
        mesh.vertex_colors_rgba,
        max_gaussians=int(args.max_gaussians),
    )
    seed_evidence = GaussianCanonicalSeedEvidence(
        source_surface_path=mesh.source_path,
        source_surface_sha256=mesh.source_sha256,
        gaussian_count=seed.gaussian_count,
        source_vertex_count=mesh.vertex_count,
        source_face_count=mesh.face_count,
    )
    if _sha256_file(Path(mesh.source_path)) != mesh.source_sha256:
        raise SystemExit("Canonical surface hash changed after integrity-bound loading.")

    runtime = probe_phase9_gaussian_interactive_runtime()
    if not runtime.ready:
        print(json.dumps(runtime.as_dict(), indent=2, sort_keys=True))
        raise SystemExit("Phase 9.4C isolated gsplat runtime is not READY.")

    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.keep_output:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    worker_script = (ROOT / "scripts" / "gaussian_workers" / "gsplat_worker.py").resolve()
    request_id = f"phase9-4c-{uuid.uuid4()}"
    report = run_supervised_phase9_gaussian_optimization(
        runtime_report=runtime,
        dataset=dataset,
        dataset_manifest_path=dataset_manifest,
        seed=seed,
        seed_evidence=seed_evidence,
        output_dir=output,
        worker_script=worker_script,
        request_id=request_id,
        profile=GaussianOptimizationProfile(args.profile),
        maximum_input_frames=max(3, dataset.frame_count),
    )

    metrics = _json(Path(report.qualification_metrics_path))
    artifact = report.artifact
    summary = {
        "phase": "9.4C",
        "status": "PASS",
        "scene_id": scene.scene_id,
        "dataset_manifest": str(dataset_manifest),
        "dataset_fingerprint_sha256": artifact.dataset_fingerprint_sha256,
        "canonical_surface_path": mesh.source_path,
        "canonical_surface_sha256": mesh.source_sha256,
        "canonical_surface_vertex_count": mesh.vertex_count,
        "canonical_surface_face_count": mesh.face_count,
        "gaussian_seed_count": seed.gaussian_count,
        "gaussian_scene_role": artifact.scene_role.value,
        "backend_kind": artifact.backend_provenance.backend_kind.value,
        "backend_version": artifact.backend_version,
        "runtime_checkout_revision": runtime.checkout_evidence.revision,
        "runtime_checkout_clean": runtime.checkout_evidence.clean,
        "runtime_python": runtime.spec.python_executable,
        "gpu_name": runtime.runtime_evidence.device_names[0] if runtime.runtime_evidence.device_names else "",
        "compute_capability": list(runtime.runtime_evidence.device_capabilities[0]) if runtime.runtime_evidence.device_capabilities else [],
        "training_steps": artifact.training_steps,
        "training_duration_s": artifact.training_duration_s,
        "peak_vram_mib": artifact.peak_vram_mib,
        "gaussian_count": artifact.gaussian_count,
        "scene_path": artifact.scene_path,
        "preview_path": report.preview_path,
        "artifact_manifest": str((Path(artifact.artifact_dir) / "artifact_manifest.json").resolve()),
        "qualification_metrics": report.qualification_metrics_path,
        "worker_console_log": report.worker_console_log_path,
        "initial_l1": metrics.get("initial_l1"),
        "final_l1": metrics.get("final_l1"),
        "finite_loss": metrics.get("finite_loss"),
        "real_cuda_execution": metrics.get("real_cuda_execution") is True,
        "optimization_completed": metrics.get("optimization_completed") is True,
        "preview_render_written": metrics.get("render_written") is True,
        "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
        "gaussian_canonical_geometry_authoritative": artifact.canonical_geometry_authoritative,
        "gaussian_metric_scale_authoritative": False,
        "gaussian_measurement_authoritative": False,
        "gaussian_qc_authoritative": False,
        "certified_metrology": artifact.certified_metrology,
        "metric_geometry_status": artifact.metric_geometry_status.value,
        "unit_to_metre": artifact.unit_to_metre,
        "production_heavy_modules": list(current_process_gaussian_heavy_modules()),
        "da3_reexecuted": False,
        "graphdeco_started": False,
        "nerf_started": False,
    }
    required = (
        summary["real_cuda_execution"] is True
        and summary["optimization_completed"] is True
        and summary["preview_render_written"] is True
        and float(summary["peak_vram_mib"] or 0.0) > 0.0
        and summary["gaussian_canonical_geometry_authoritative"] is False
        and summary["certified_metrology"] is False
        and summary["production_heavy_modules"] == []
        and Path(str(summary["scene_path"])).is_file()
        and Path(str(summary["preview_path"])).is_file()
    )
    if not required:
        summary["status"] = "FAIL"
    summary_path_out = output / "phase9_4c_real_optimization_summary.json"
    summary_path_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Phase 9.4C real Gaussian optimization summary: {summary_path_out}")
    if summary["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
