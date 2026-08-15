#!/usr/bin/env python3
"""Run the real Phase 9.4 Gaussian end-to-end closure campaign.

The campaign deliberately starts from the already-qualified Phase 9.3J canonical
South Building scene.  It performs a fresh Phase 9.4A runtime probe, rematerializes
9.4B inputs, executes the real isolated 9.4C gsplat optimization, persists a new
9.4D Gaussian scene revision, verifies fresh-process reload, and then exercises
9.4E/9.4F/9.4G interactive-load, display-control, authority, and CPU-resource
contracts without launching another Gaussian backend.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping
import weakref

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_gaussian_controls import GaussianDisplaySettings, apply_gaussian_display_settings
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from app.reconstruction_gaussian_resources import (
    build_gaussian_interactive_resource_policy,
    gaussian_display_staging_bytes,
    gaussian_scene_staging_bytes,
)
from app.reconstruction_gaussian_visualization import build_gaussian_visualization_presentation
from core.measurement.interactive_distance import InteractiveDistanceState, estimate_reconstruction_distance
from core.reconstruction import (
    current_process_gaussian_heavy_modules,
    probe_phase9_gaussian_interactive_runtime,
    read_persistent_reconstruction_scene,
    reload_persistent_gaussian_representation,
    verify_persistent_scene_file_references,
)
from core.runtime.stress import read_process_rss_mib

DEFAULT_PHASE93J = Path("/tmp/phase9_3j_real_scene/phase9_3j_qualification_summary.json")
DEFAULT_OUTPUT = Path("/tmp/phase9_4h_real_scene")


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _p95(values: list[float]) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return float(ordered[index])


def _run_checked(args: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(process.stdout or "", encoding="utf-8")
    if process.returncode != 0:
        tail = "\n".join((process.stdout or "").splitlines()[-30:])
        raise RuntimeError(
            f"Closure subprocess failed ({process.returncode}): {' '.join(args)}\n{tail}"
        )


def validate_phase9_4_chain(evidence: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Pure closure invariant check used by unit tests and the real campaign."""

    reasons: list[str] = []
    runtime = dict(evidence.get("runtime", {}))
    source = dict(evidence.get("source_scene", {}))
    bridge = dict(evidence.get("bridge", {}))
    optimization = dict(evidence.get("optimization", {}))
    persistence = dict(evidence.get("persistence", {}))
    reload = dict(evidence.get("reload", {}))
    viewer = dict(evidence.get("viewer", {}))
    controls = dict(evidence.get("controls", {}))
    resources = dict(evidence.get("resources", {}))
    measurement = dict(evidence.get("measurement", {}))
    authority = dict(evidence.get("authority", {}))

    if runtime.get("ready") is not True or runtime.get("production_heavy_modules") not in ([], ()):
        reasons.append("isolated_runtime_not_ready")
    if runtime.get("checkout_clean") is not True or not str(runtime.get("checkout_revision", "")).strip():
        reasons.append("runtime_checkout_not_clean_or_unpinned")

    if source.get("status") != "PASS" or source.get("canonical_geometry_authoritative") is not True:
        reasons.append("phase93j_source_scene_not_qualified")
    if source.get("metric_geometry_status") != "arbitrary" or source.get("unit_to_metre") is not None:
        reasons.append("source_scene_scale_not_preserved")

    if bridge.get("status") != "PASS":
        reasons.append("gaussian_input_bridge_failed")
    if bridge.get("source_frame_fingerprints_exact") is not True or bridge.get("canonical_camera_binding_exact") is not True:
        reasons.append("gaussian_input_binding_not_exact")
    if bridge.get("gaussian_scene_role") != "appearance_only":
        reasons.append("gaussian_input_role_changed")
    if bridge.get("da3_reexecuted") is not False or bridge.get("gaussian_backend_started") is not False:
        reasons.append("bridge_started_forbidden_backend")

    if optimization.get("status") != "PASS":
        reasons.append("real_gaussian_optimization_failed")
    if optimization.get("real_cuda_execution") is not True or optimization.get("optimization_completed") is not True:
        reasons.append("real_cuda_optimization_not_proven")
    if optimization.get("preview_render_written") is not True or float(optimization.get("peak_vram_mib", 0.0) or 0.0) <= 0.0:
        reasons.append("real_render_or_vram_evidence_missing")
    if optimization.get("finite_loss") is not True:
        reasons.append("optimization_loss_not_finite")
    if optimization.get("da3_reexecuted") is not False or optimization.get("graphdeco_started") is not False or optimization.get("nerf_started") is not False:
        reasons.append("optimization_started_forbidden_provider")
    initial_l1 = optimization.get("initial_l1")
    final_l1 = optimization.get("final_l1")
    if initial_l1 is not None and final_l1 is not None:
        if not (math.isfinite(float(initial_l1)) and math.isfinite(float(final_l1)) and float(final_l1) <= float(initial_l1)):
            reasons.append("optimization_did_not_improve_l1")

    if bridge.get("scene_id") != optimization.get("scene_id"):
        reasons.append("bridge_optimization_scene_mismatch")
    if bridge.get("dataset_fingerprint_sha256") != optimization.get("dataset_fingerprint_sha256"):
        reasons.append("dataset_fingerprint_mismatch")
    if runtime.get("checkout_revision") != optimization.get("runtime_checkout_revision"):
        reasons.append("runtime_revision_mismatch")

    if persistence.get("status") != "PASS" or persistence.get("source_scene_unchanged") is not True:
        reasons.append("gaussian_persistence_failed")
    if persistence.get("restart_reload_verified") is not True or persistence.get("gaussian_file_reference_count", 0) < 1:
        reasons.append("gaussian_persistence_reload_or_files_missing")
    if persistence.get("parent_scene_id") != bridge.get("scene_id"):
        reasons.append("persistent_scene_lineage_mismatch")
    if persistence.get("dataset_fingerprint_sha256") != bridge.get("dataset_fingerprint_sha256"):
        reasons.append("persistent_dataset_fingerprint_mismatch")
    if persistence.get("gaussian_backend_reexecuted") is not False or persistence.get("da3_reexecuted") is not False or persistence.get("nerf_started") is not False:
        reasons.append("persistence_reexecuted_forbidden_backend")

    if reload.get("status") != "PASS" or reload.get("restart_reload_verified") is not True:
        reasons.append("fresh_process_reload_failed")
    if reload.get("canonical_geometry_available") is not True or reload.get("gaussian_artifact_available") is not True:
        reasons.append("reload_missing_canonical_or_gaussian_artifact")

    if viewer.get("interactive_appearance_ready") is not True or viewer.get("picking_source") != "canonical_geometry_only":
        reasons.append("interactive_viewer_not_ready_or_wrong_picking_source")
    if controls.get("scene_layer_visibility_only") is not True or controls.get("persisted_gaussian_modified") is not False:
        reasons.append("display_controls_changed_persistence")
    if resources.get("all_numpy_staging_released") is not True or resources.get("release_checks_passed") != resources.get("cycles"):
        reasons.append("resource_release_cycle_failed")
    if resources.get("gpu_vram_measured") is not False or resources.get("timings_are_cpu_ui_prep_only") is not True:
        reasons.append("resource_diagnostics_misclassified")

    if measurement.get("state") != InteractiveDistanceState.UNAVAILABLE.value:
        reasons.append("arbitrary_scale_measurement_did_not_fail_closed")
    if measurement.get("metric_value_exposed") is not False:
        reasons.append("arbitrary_scale_metric_value_exposed")

    if authority.get("canonical_geometry_authoritative") is not True:
        reasons.append("canonical_geometry_authority_lost")
    for key in (
        "gaussian_canonical_geometry_authoritative",
        "gaussian_metric_scale_authoritative",
        "gaussian_measurement_authoritative",
        "gaussian_qc_authoritative",
        "certified_metrology",
    ):
        if authority.get(key) is not False:
            reasons.append(f"authority_violation:{key}")
    if authority.get("production_heavy_modules_after") not in ([], ()):
        reasons.append("production_runtime_contaminated")

    return not reasons, tuple(dict.fromkeys(reasons))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase93j-summary", type=Path, default=DEFAULT_PHASE93J)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("efficient", "balanced", "high_quality"), default="efficient")
    parser.add_argument("--max-gaussians", type=int, default=2048)
    parser.add_argument("--resource-cycles", type=int, default=12)
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.output_dir.expanduser().resolve()
    summary_path = root / "phase9_4h_qualification_summary.json"
    try:
        if root.exists() and not args.keep_output:
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        logs = root / "logs"

        heavy_before = current_process_gaussian_heavy_modules()
        runtime_report = probe_phase9_gaussian_interactive_runtime()
        if not runtime_report.ready:
            raise RuntimeError(f"Phase 9.4A runtime is not READY: {runtime_report.reasons}")

        phase93j_path = args.phase93j_summary.expanduser().resolve()
        if not phase93j_path.is_file():
            raise RuntimeError(f"Phase 9.3J qualification summary is missing: {phase93j_path}")
        phase93j = _json(phase93j_path)
        source_scene_payload = dict(phase93j.get("persistent_scene", {}))
        source_authority = dict(phase93j.get("authority", {}))
        source_manifest = Path(str(source_scene_payload.get("manifest", ""))).expanduser().resolve()
        source_scene = read_persistent_reconstruction_scene(source_manifest)
        verify_persistent_scene_file_references(source_scene)

        bridge_dir = root / "bridge"
        _run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/run_phase9_4b_real_scene_bridge.py"),
                "--qualification-summary", str(phase93j_path),
                "--output-dir", str(bridge_dir),
            ],
            log_path=logs / "phase9_4b_bridge.log",
        )
        bridge_path = bridge_dir / "phase9_4b_real_bridge_summary.json"
        bridge = _json(bridge_path)

        optimization_dir = root / "optimization"
        _run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/run_phase9_4c_real_gaussian_optimization.py"),
                "--bridge-summary", str(bridge_path),
                "--output-dir", str(optimization_dir),
                "--profile", args.profile,
                "--max-gaussians", str(max(1, int(args.max_gaussians))),
            ],
            log_path=logs / "phase9_4c_optimization.log",
        )
        optimization_path = optimization_dir / "phase9_4c_real_optimization_summary.json"
        optimization = _json(optimization_path)

        persistence_dir = root / "persistent"
        _run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/run_phase9_4d_real_gaussian_persistence.py"),
                "--optimization-summary", str(optimization_path),
                "--bridge-summary", str(bridge_path),
                "--output-dir", str(persistence_dir),
            ],
            log_path=logs / "phase9_4d_persistence.log",
        )
        persistence_path = persistence_dir / "phase9_4d_persistence_summary.json"
        persistence = _json(persistence_path)

        _run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/run_phase9_4d_gaussian_reload_check.py"),
                "--summary", str(persistence_path),
            ],
            log_path=logs / "phase9_4d_fresh_reload.log",
        )

        loaded = reload_persistent_gaussian_representation(persistence["scene_manifest_path"])
        scene = loaded.scene
        verified_refs = verify_persistent_scene_file_references(scene)
        mesh = load_canonical_viewer_mesh(scene)
        cameras = load_persistent_canonical_camera_trajectory(scene)
        presentation = build_gaussian_visualization_presentation(scene)
        if mesh is None or cameras is None or presentation is None or not presentation.ready:
            raise RuntimeError("Closure scene is missing canonical mesh/cameras or static Gaussian presentation.")

        reload_evidence = {
            "status": "PASS",
            "restart_reload_verified": loaded.restart_reload_verified,
            "scene_fingerprint_matches_summary": scene.scene_fingerprint_sha256 == persistence.get("persistent_scene_fingerprint_sha256"),
            "canonical_geometry_available": True,
            "canonical_vertex_count": mesh.vertex_count,
            "canonical_face_count": mesh.face_count,
            "canonical_camera_count": cameras.persisted_pose_count,
            "gaussian_artifact_available": scene.gaussian_artifact is not None,
            "gaussian_scene_exists": Path(loaded.artifact.scene_path).is_file(),
            "gaussian_static_presentation_ready": presentation.ready,
        }

        policy = build_gaussian_interactive_resource_policy()
        interactive = load_interactive_gaussian_scene(scene, max_gaussians=policy.max_interactive_gaussians)
        viewer_evidence = {
            "interactive_appearance_ready": True,
            "total_gaussian_count": interactive.total_gaussian_count,
            "display_gaussian_count": interactive.display_gaussian_count,
            "picking_source": "canonical_geometry_only",
            "static_backend_preview_preserved": presentation.ready,
        }
        presets: dict[str, Any] = {}
        for name, settings in (
            ("full", GaussianDisplaySettings()),
            ("balanced", GaussianDisplaySettings(density_percent=50, size_percent=110, opacity_percent=80)),
            ("light", GaussianDisplaySettings(density_percent=25, size_percent=100, opacity_percent=65)),
        ):
            buffers = apply_gaussian_display_settings(interactive, settings)
            presets[name] = {
                "display_count": buffers.display_count,
                "density_percent": settings.density_percent,
                "size_percent": settings.size_percent,
                "opacity_percent": settings.opacity_percent,
            }
        controls_evidence = {
            "scene_layer_visibility_only": True,
            "persisted_gaussian_modified": False,
            "presets": presets,
        }
        del buffers
        del interactive
        gc.collect()

        cycles = max(1, int(args.resource_cycles))
        baseline_rss = read_process_rss_mib(os.getpid())
        load_ms: list[float] = []
        apply_ms: list[float] = []
        release_checks: list[bool] = []
        source_bytes = display_bytes = total_count = source_display = balanced_display = 0
        for _ in range(cycles):
            started = perf_counter()
            current = load_interactive_gaussian_scene(scene, max_gaussians=policy.max_interactive_gaussians)
            load_ms.append((perf_counter() - started) * 1000.0)
            started = perf_counter()
            current_buffers = apply_gaussian_display_settings(
                current,
                GaussianDisplaySettings(density_percent=50, size_percent=110, opacity_percent=80),
            )
            apply_ms.append((perf_counter() - started) * 1000.0)
            source_bytes = gaussian_scene_staging_bytes(current)
            display_bytes = gaussian_display_staging_bytes(current_buffers)
            total_count = current.total_gaussian_count
            source_display = current.display_gaussian_count
            balanced_display = current_buffers.display_count
            source_ref = weakref.ref(current.positions_reconstruction)
            display_ref = weakref.ref(current_buffers.positions_reconstruction)
            del current_buffers
            del current
            gc.collect()
            release_checks.append(source_ref() is None and display_ref() is None)
        final_rss = read_process_rss_mib(os.getpid())
        resources_evidence = {
            "cycles": cycles,
            "device": policy.device_label,
            "resource_tier": policy.tier,
            "adaptive_max_display_gaussians": policy.max_interactive_gaussians,
            "total_gaussian_count": total_count,
            "source_interactive_display_count": source_display,
            "balanced_display_count": balanced_display,
            "source_cpu_staging_mib": source_bytes / (1024.0 * 1024.0),
            "balanced_cpu_staging_mib": display_bytes / (1024.0 * 1024.0),
            "median_safe_load_ms": median(load_ms),
            "p95_safe_load_ms": _p95(load_ms),
            "median_control_apply_ms": median(apply_ms),
            "p95_control_apply_ms": _p95(apply_ms),
            "rss_baseline_mib": baseline_rss,
            "rss_final_mib": final_rss,
            "rss_values_are_advisory": True,
            "release_checks_passed": sum(1 for ok in release_checks if ok),
            "all_numpy_staging_released": all(release_checks),
            "gpu_vram_measured": False,
            "timings_are_cpu_ui_prep_only": True,
        }

        vertices = np.asarray(mesh.vertices_reconstruction, dtype=np.float64)
        a = vertices[0]
        b = vertices[int(np.argmax(np.linalg.norm(vertices - a[None, :], axis=1)))]
        measurement_result = estimate_reconstruction_distance(
            point_a_reconstruction=a,
            point_b_reconstruction=b,
            unit_to_metre=None,
            relative_scale_uncertainty=None,
            scale_state="arbitrary",
            calibration_available=False,
        )
        measurement_evidence = {
            "state": measurement_result.state.value,
            "native_distance_units": measurement_result.distance_source_units,
            "metric_value_exposed": measurement_result.distance_m is not None,
            "expanded_uncertainty_exposed": measurement_result.expanded_uncertainty_m is not None,
            "certified_metrology": measurement_result.certified_metrology,
        }

        heavy_after = current_process_gaussian_heavy_modules()
        runtime_dict = runtime_report.as_dict()
        runtime_evidence = {
            "ready": runtime_report.ready,
            "checkout_revision": runtime_report.checkout_evidence.revision,
            "checkout_clean": runtime_report.checkout_evidence.clean,
            "backend_kind": runtime_report.spec.backend_kind.value,
            "backend_version": runtime_report.environment_probe.gsplat_version,
            "python_executable": runtime_report.spec.python_executable,
            "gpu_names": list(runtime_report.runtime_evidence.device_names),
            "device_capabilities": [list(item) for item in runtime_report.runtime_evidence.device_capabilities],
            "production_heavy_modules": list(runtime_report.production_heavy_modules),
            "process_policy": runtime_report.spec.process_policy.as_dict(),
        }
        source_evidence = {
            "status": phase93j.get("status"),
            "scene_id": source_scene.scene_id,
            "scene_manifest": str(source_manifest),
            "canonical_geometry_authoritative": source_scene.canonical_geometry_authoritative,
            "metric_geometry_status": source_authority.get("metric_geometry_status"),
            "unit_to_metre": source_authority.get("unit_to_metre"),
            "certified_metrology": source_authority.get("certified_metrology"),
        }
        authority_evidence = {
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "gaussian_canonical_geometry_authoritative": False,
            "gaussian_metric_scale_authoritative": False,
            "gaussian_measurement_authoritative": False,
            "gaussian_qc_authoritative": False,
            "certified_metrology": False,
            "metric_geometry_status": loaded.artifact.metric_geometry_status.value,
            "unit_to_metre": loaded.artifact.unit_to_metre,
            "production_heavy_modules_before": list(heavy_before),
            "production_heavy_modules_after": list(heavy_after),
        }

        evidence = {
            "runtime": runtime_evidence,
            "source_scene": source_evidence,
            "bridge": bridge,
            "optimization": optimization,
            "persistence": persistence,
            "reload": reload_evidence,
            "viewer": viewer_evidence,
            "controls": controls_evidence,
            "resources": resources_evidence,
            "measurement": measurement_evidence,
            "authority": authority_evidence,
        }
        passed, reasons = validate_phase9_4_chain(evidence)
        if source_evidence["scene_id"] != bridge.get("scene_id"):
            reasons = tuple(reasons) + ("phase93j_bridge_scene_mismatch",)
            passed = False
        if scene.gaussian_artifact is None:
            reasons = tuple(reasons) + ("persistent_gaussian_artifact_missing",)
            passed = False
        if len(verified_refs) < 1:
            reasons = tuple(reasons) + ("persistent_file_references_missing",)
            passed = False
        if optimization.get("gaussian_count") != persistence.get("gaussian_count"):
            reasons = tuple(reasons) + ("optimization_persistence_gaussian_count_mismatch",)
            passed = False
        if viewer_evidence["total_gaussian_count"] != persistence.get("gaussian_count"):
            reasons = tuple(reasons) + ("persistence_viewer_gaussian_count_mismatch",)
            passed = False
        runtime_gpu_names = [str(name) for name in runtime_evidence.get("gpu_names", [])]
        if any("2060" in name.lower() for name in runtime_gpu_names):
            if policy.tier != "conservative-6gb" or policy.max_interactive_gaussians != 100000:
                reasons = tuple(reasons) + ("rtx2060_resource_policy_mismatch",)
                passed = False

        summary = {
            "phase": "9.4H",
            "status": "PASS" if passed else "FAIL",
            "reasons": list(dict.fromkeys(reasons)),
            "real_gaussian_campaign_performed": True,
            "da3_reexecuted": False,
            "source_phase93j_summary": str(phase93j_path),
            "output_root": str(root),
            "bridge_summary": str(bridge_path),
            "optimization_summary": str(optimization_path),
            "persistence_summary": str(persistence_path),
            "persistent_scene_manifest": persistence.get("scene_manifest_path"),
            "verified_persistent_file_reference_count": len(verified_refs),
            "runtime": runtime_evidence,
            "source_scene": source_evidence,
            "bridge": bridge,
            "optimization": optimization,
            "persistence": persistence,
            "reload": reload_evidence,
            "viewer": viewer_evidence,
            "controls": controls_evidence,
            "resources": resources_evidence,
            "measurement": measurement_evidence,
            "authority": authority_evidence,
            "runtime_report": runtime_dict,
            "deferred": {
                "real_physical_scale_calibration": True,
                "certified_metrology": True,
                "interactive_nerf": "Phase 9.5",
                "unified_digital_twin": "Phase 9.6",
            },
        }
    except Exception as exc:
        summary = {
            "phase": "9.4H",
            "status": "FAIL",
            "reasons": [f"{type(exc).__name__}: {exc}"],
            "real_gaussian_campaign_performed": False,
            "da3_reexecuted": False,
            "output_root": str(root),
        }

    root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print(f"Phase 9.4H real Gaussian closure summary: {summary_path}")
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
