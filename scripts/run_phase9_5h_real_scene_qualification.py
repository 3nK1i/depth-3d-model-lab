#!/usr/bin/env python3
"""Run the real Phase 9.5 NeRF end-to-end closure campaign.

The campaign starts from the already-qualified Phase 9.3J South Building
canonical scene, performs a fresh Phase 9.5A runtime probe, fresh 9.5B bridge,
real 9.5C Nerfacto optimization, 9.5D persistence/reload, 9.5E interactive
render, 9.5F controls audit, 9.5G repeated resource lifecycle qualification,
and 9.5H single-held-out-view quality characterization.  NeRF remains
appearance-only throughout; canonical geometry remains the picking/measurement
authority and arbitrary scale is preserved truthfully.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from core.measurement.interactive_distance import InteractiveDistanceState, estimate_reconstruction_distance
from core.reconstruction import (
    current_process_nerf_heavy_modules,
    probe_phase9_nerf_interactive_runtime,
    read_persistent_reconstruction_scene,
    reload_persistent_nerf_representation,
    verify_persistent_scene_file_references,
)

DEFAULT_PHASE93J = Path("/tmp/phase9_3j_real_scene/phase9_3j_qualification_summary.json")
DEFAULT_OUTPUT = Path("/tmp/phase9_5h_real_scene")
EXPECTED_NS_REVISION = "6b60855003011b2ca23c2fe3f8e2ca6314c69924"
EXPECTED_TCNN_REVISION = "b3473c81396fe927293bdfd5a6be32df8769927c"


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


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
        tail = "\n".join((process.stdout or "").splitlines()[-40:])
        raise RuntimeError(f"Closure subprocess failed ({process.returncode}): {' '.join(args)}\n{tail}")


def _finite_quality_metrics(quality: Mapping[str, Any]) -> bool:
    metrics = dict(quality.get("metrics", {}))
    try:
        psnr = float(metrics["psnr_db"])
        ssim = float(metrics["ssim"])
        lpips = float(metrics["lpips"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(psnr)
        and math.isfinite(ssim)
        and math.isfinite(lpips)
        and psnr >= 0.0
        and 0.0 <= ssim <= 1.0
        and lpips >= 0.0
    )


def validate_phase9_5_chain(evidence: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Pure Phase 9.5A→H closure invariant check."""

    reasons: list[str] = []
    runtime = dict(evidence.get("runtime", {}))
    source = dict(evidence.get("source_scene", {}))
    bridge = dict(evidence.get("bridge", {}))
    optimization = dict(evidence.get("optimization", {}))
    persistence = dict(evidence.get("persistence", {}))
    reload = dict(evidence.get("reload", {}))
    interactive = dict(evidence.get("interactive", {}))
    controls = dict(evidence.get("controls", {}))
    resources = dict(evidence.get("resources", {}))
    quality = dict(evidence.get("quality", {}))
    measurement = dict(evidence.get("measurement", {}))
    authority = dict(evidence.get("authority", {}))

    if runtime.get("ready") is not True or runtime.get("production_heavy_modules") not in ([], ()):
        reasons.append("isolated_runtime_not_ready")
    if runtime.get("nerfstudio_clean") is not True or runtime.get("nerfstudio_revision") != EXPECTED_NS_REVISION:
        reasons.append("nerfstudio_checkout_not_clean_or_pinned")
    if runtime.get("tiny_cuda_nn_clean") is not True or runtime.get("tiny_cuda_nn_revision") != EXPECTED_TCNN_REVISION:
        reasons.append("tiny_cuda_nn_checkout_not_clean_or_pinned")

    if source.get("status") != "PASS" or source.get("canonical_geometry_authoritative") is not True:
        reasons.append("phase93j_source_scene_not_qualified")
    if source.get("metric_geometry_status") != "arbitrary" or source.get("unit_to_metre") is not None:
        reasons.append("source_scene_scale_not_preserved")

    if bridge.get("status") != "PASS":
        reasons.append("nerf_input_bridge_failed")
    if bridge.get("source_frame_fingerprints_exact") is not True or bridge.get("canonical_camera_binding_exact") is not True:
        reasons.append("nerf_input_binding_not_exact")
    if bridge.get("retained_da3_intrinsics_used") is not True:
        reasons.append("retained_intrinsics_not_used")
    if bridge.get("frame_count") != 3 or bridge.get("training_frame_count") != 2 or bridge.get("validation_frame_count") != 1:
        reasons.append("closure_dataset_split_changed")
    if bridge.get("nerf_scene_role") != "appearance_only":
        reasons.append("nerf_input_role_changed")
    if bridge.get("da3_reexecuted") is not False or bridge.get("nerf_backend_started") is not False:
        reasons.append("bridge_started_forbidden_backend")

    if optimization.get("status") != "PASS":
        reasons.append("real_nerf_optimization_failed")
    for key in ("real_cuda_execution", "optimization_completed", "training_completed_without_oom", "preview_render_written", "render_capable"):
        if optimization.get(key) is not True:
            reasons.append(f"optimization_missing:{key}")
    if optimization.get("runtime_checkout_clean") is not True or optimization.get("runtime_checkout_revision") != EXPECTED_NS_REVISION:
        reasons.append("optimization_runtime_revision_mismatch")
    if optimization.get("tiny_cuda_nn_checkout_clean") is not True or optimization.get("tiny_cuda_nn_revision") != EXPECTED_TCNN_REVISION:
        reasons.append("optimization_tcnn_revision_mismatch")
    if optimization.get("da3_reexecuted") is not False or optimization.get("gaussian_backend_started") is not False:
        reasons.append("optimization_started_forbidden_provider")
    if optimization.get("persistence_started") is not False or optimization.get("interactive_viewer_started") is not False:
        reasons.append("optimization_crossed_phase_boundary")

    if bridge.get("scene_id") != optimization.get("scene_id"):
        reasons.append("bridge_optimization_scene_mismatch")
    if bridge.get("dataset_fingerprint_sha256") != optimization.get("dataset_fingerprint_sha256"):
        reasons.append("bridge_optimization_dataset_fingerprint_mismatch")

    if persistence.get("status") != "PASS" or persistence.get("source_scene_unchanged") is not True:
        reasons.append("nerf_persistence_failed")
    if persistence.get("restart_reload_verified") is not True or int(persistence.get("nerf_file_reference_count", 0) or 0) < 10:
        reasons.append("nerf_persistence_reload_or_files_missing")
    if persistence.get("parent_scene_id") != bridge.get("scene_id"):
        reasons.append("persistent_scene_lineage_mismatch")
    if persistence.get("dataset_fingerprint_sha256") != bridge.get("dataset_fingerprint_sha256"):
        reasons.append("persistent_dataset_fingerprint_mismatch")
    for key in ("nerf_backend_reexecuted", "nerf_training_reexecuted", "nerf_rendering_reexecuted", "da3_reexecuted", "gaussian_backend_started"):
        if persistence.get(key) is not False:
            reasons.append(f"persistence_forbidden_execution:{key}")

    if reload.get("status") != "PASS" or reload.get("restart_reload_verified") is not True:
        reasons.append("fresh_process_reload_failed")
    for key in ("scene_fingerprint_matches_summary", "canonical_geometry_available", "nerf_artifact_available", "nerf_checkpoint_exists", "nerf_config_exists", "nerf_dataset_exists"):
        if reload.get(key) is not True:
            reasons.append(f"reload_missing:{key}")

    if interactive.get("status") != "PASS" or interactive.get("interactive_renderer_started") is not True:
        reasons.append("interactive_renderer_not_qualified")
    if interactive.get("real_cuda_execution") is not True or interactive.get("staged_checkpoint_matches_persisted") is not True:
        reasons.append("interactive_cuda_or_checkpoint_binding_failed")
    if interactive.get("runtime_checkout_clean") is not True or interactive.get("runtime_checkout_revision") != EXPECTED_NS_REVISION:
        reasons.append("interactive_runtime_revision_mismatch")
    if interactive.get("tiny_cuda_nn_checkout_clean") is not True or interactive.get("tiny_cuda_nn_revision") != EXPECTED_TCNN_REVISION:
        reasons.append("interactive_tcnn_revision_mismatch")
    if interactive.get("picking_source") != "canonical_geometry_only" or int(interactive.get("worker_exit_code", -1)) != 0:
        reasons.append("interactive_picking_or_worker_exit_failed")

    if controls.get("status") != "PASS":
        reasons.append("nerf_controls_check_failed")
    if controls.get("default_render_profile") != "detail" or controls.get("default_render_resolution") != [640, 360]:
        reasons.append("nerf_default_display_contract_changed")
    if controls.get("auto_refresh_default") is not True or controls.get("nerf_scene_layer_available") is not True:
        reasons.append("nerf_controls_or_scene_layer_missing")
    if controls.get("persistence_modified") is not False or controls.get("backend_started") is not False or controls.get("training_started") is not False:
        reasons.append("nerf_controls_changed_persistence_or_backend")
    if controls.get("picking_source") != "canonical_geometry_only":
        reasons.append("nerf_controls_changed_picking_source")

    if resources.get("status") != "PASS":
        reasons.append("nerf_resource_qualification_failed")
    cycles = int(resources.get("cycles", 0) or 0)
    if cycles < 1 or resources.get("worker_start_count") != cycles or resources.get("worker_release_count") != cycles:
        reasons.append("worker_lifecycle_counts_mismatch")
    if resources.get("all_worker_pids_released_from_os_and_nvidia_smi") is not True or resources.get("all_shutdowns_graceful") is not True:
        reasons.append("worker_release_failed")
    if resources.get("total_render_count") != cycles * 3 or resources.get("performance_characterized") is not True:
        reasons.append("render_performance_characterization_incomplete")
    if resources.get("max_inflight_renders") != 1 or resources.get("max_pending_renders") != 1 or resources.get("max_retained_frames") != 2:
        reasons.append("resource_bounds_changed")
    for key in ("all_frame_retention_checks_passed", "worker_torch_cuda_telemetry_measured", "nvidia_smi_worker_memory_measured"):
        if resources.get(key) is not True:
            reasons.append(f"resource_evidence_missing:{key}")
    if resources.get("runtime_checkout_clean") is not True or resources.get("runtime_checkout_revision") != EXPECTED_NS_REVISION:
        reasons.append("resource_runtime_revision_mismatch")
    if resources.get("tiny_cuda_nn_checkout_clean") is not True or resources.get("tiny_cuda_nn_revision") != EXPECTED_TCNN_REVISION:
        reasons.append("resource_tcnn_revision_mismatch")
    for key in ("training_reexecuted", "persistence_modified", "gaussian_backend_started", "da3_reexecuted"):
        if resources.get(key) is not False:
            reasons.append(f"resource_scope_violation:{key}")

    if quality.get("status") != "PASS" or quality.get("visual_quality_evaluated") is not True or quality.get("quality_characterized") is not True:
        reasons.append("heldout_quality_not_characterized")
    if quality.get("training_frame_count") != 2 or quality.get("validation_frame_count") != 1:
        reasons.append("quality_dataset_split_changed")
    if quality.get("phase8_e3_sanity_policy_applicable") is not False:
        reasons.append("undersized_dataset_misclassified_as_e3_policy_applicable")
    if quality.get("quality_certified") is not False or quality.get("production_quality_certified") is not False:
        reasons.append("undersized_dataset_quality_overclaimed")
    if quality.get("limitation_reason") != "insufficient_views_for_phase8_e3_sanity_policy":
        reasons.append("quality_limitation_not_explicit")
    if not _finite_quality_metrics(quality):
        reasons.append("heldout_quality_metrics_invalid")
    if quality.get("worker_released") is not True:
        reasons.append("quality_worker_not_released")
    if quality.get("production_heavy_modules_before") not in ([], ()) or quality.get("production_heavy_modules_after") not in ([], ()):
        reasons.append("quality_check_contaminated_production_runtime")

    if measurement.get("state") != InteractiveDistanceState.UNAVAILABLE.value:
        reasons.append("arbitrary_scale_measurement_did_not_fail_closed")
    if measurement.get("metric_value_exposed") is not False:
        reasons.append("arbitrary_scale_metric_value_exposed")

    if authority.get("canonical_geometry_authoritative") is not True or authority.get("nerf_scene_role") != "appearance_only":
        reasons.append("canonical_or_nerf_role_authority_changed")
    for key in (
        "nerf_canonical_geometry_authoritative",
        "nerf_metric_scale_authoritative",
        "nerf_measurement_authoritative",
        "nerf_qc_authoritative",
        "certified_metrology",
    ):
        if authority.get(key) is not False:
            reasons.append(f"authority_violation:{key}")
    if authority.get("metric_geometry_status") != "arbitrary" or authority.get("unit_to_metre") is not None:
        reasons.append("closure_scale_authority_changed")
    if authority.get("production_heavy_modules_after") not in ([], ()):
        reasons.append("production_runtime_contaminated")

    return not reasons, tuple(dict.fromkeys(reasons))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase93j-summary", type=Path, default=DEFAULT_PHASE93J)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("efficient", "balanced", "high_quality"), default="efficient")
    parser.add_argument("--training-steps", type=int, default=64)
    parser.add_argument("--resource-cycles", type=int, default=4)
    parser.add_argument(
        "--metrics-python",
        type=Path,
        default=Path.home() / "anaconda3/envs/depth3d-gaussian-metrics/bin/python",
        help="Existing isolated backend-neutral PSNR/SSIM/LPIPS metrics interpreter from Phase 8.7D3.",
    )
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.output_dir.expanduser().resolve()
    summary_path = root / "phase9_5h_qualification_summary.json"
    try:
        if root.exists() and not args.keep_output:
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        logs = root / "logs"

        heavy_before = list(current_process_nerf_heavy_modules())
        runtime_report = probe_phase9_nerf_interactive_runtime()
        if not runtime_report.ready:
            raise RuntimeError(f"Phase 9.5A runtime is not READY: {runtime_report.reasons}")

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
            [sys.executable, str(ROOT / "scripts/run_phase9_5b_real_scene_bridge.py"), "--qualification-summary", str(phase93j_path), "--output-dir", str(bridge_dir)],
            log_path=logs / "phase9_5b_bridge.log",
        )
        bridge_path = bridge_dir / "phase9_5b_real_bridge_summary.json"
        bridge = _json(bridge_path)

        optimization_dir = root / "optimization"
        _run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/run_phase9_5c_real_nerf_optimization.py"),
                "--bridge-summary", str(bridge_path),
                "--output-dir", str(optimization_dir),
                "--profile", args.profile,
                "--training-steps", str(max(1, int(args.training_steps))),
            ],
            log_path=logs / "phase9_5c_optimization.log",
        )
        optimization_path = optimization_dir / "phase9_5c_real_optimization_summary.json"
        optimization = _json(optimization_path)

        persistence_dir = root / "persistent"
        _run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/run_phase9_5d_real_nerf_persistence.py"),
                "--optimization-summary", str(optimization_path),
                "--bridge-summary", str(bridge_path),
                "--output-dir", str(persistence_dir),
            ],
            log_path=logs / "phase9_5d_persistence.log",
        )
        persistence_path = persistence_dir / "phase9_5d_persistence_summary.json"
        persistence = _json(persistence_path)

        _run_checked(
            [sys.executable, str(ROOT / "scripts/run_phase9_5d_nerf_reload_check.py"), "--summary", str(persistence_path)],
            log_path=logs / "phase9_5d_fresh_reload.log",
        )
        loaded = reload_persistent_nerf_representation(persistence["scene_manifest_path"])
        scene = loaded.scene
        verified_refs = verify_persistent_scene_file_references(scene)
        mesh = load_canonical_viewer_mesh(scene)
        cameras = load_persistent_canonical_camera_trajectory(scene)
        if mesh is None or cameras is None or scene.nerf_artifact is None:
            raise RuntimeError("Closure scene is missing canonical mesh/cameras or persisted NeRF appearance")
        reload_evidence = {
            "status": "PASS",
            "restart_reload_verified": loaded.restart_reload_verified,
            "scene_fingerprint_matches_summary": scene.scene_fingerprint_sha256 == persistence.get("persistent_scene_fingerprint_sha256"),
            "canonical_geometry_available": True,
            "canonical_vertex_count": mesh.vertex_count,
            "canonical_face_count": mesh.face_count,
            "canonical_camera_count": cameras.persisted_pose_count,
            "nerf_artifact_available": True,
            "nerf_checkpoint_exists": Path(loaded.artifact.checkpoint_path).is_file(),
            "nerf_config_exists": Path(loaded.artifact.config_path).is_file(),
            "nerf_dataset_exists": Path(loaded.dataset_root).is_dir(),
        }

        interactive_dir = root / "interactive"
        _run_checked(
            [sys.executable, str(ROOT / "scripts/run_phase9_5e_real_interactive_render.py"), "--scene-manifest", str(persistence["scene_manifest_path"]), "--output-dir", str(interactive_dir)],
            log_path=logs / "phase9_5e_interactive.log",
        )
        interactive_path = interactive_dir / "phase9_5e_real_interactive_summary.json"
        interactive = _json(interactive_path)

        controls_path = root / "phase9_5f_controls_summary.json"
        _run_checked(
            [sys.executable, str(ROOT / "scripts/run_phase9_5f_real_nerf_controls_check.py"), "--scene-manifest", str(persistence["scene_manifest_path"]), "--output", str(controls_path)],
            log_path=logs / "phase9_5f_controls.log",
        )
        controls = _json(controls_path)

        resources_dir = root / "resources"
        _run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/run_phase9_5g_real_nerf_resource_check.py"),
                "--scene-manifest", str(persistence["scene_manifest_path"]),
                "--output-dir", str(resources_dir),
                "--cycles", str(max(1, int(args.resource_cycles))),
            ],
            log_path=logs / "phase9_5g_resources.log",
        )
        resources_path = resources_dir / "phase9_5g_real_nerf_resource_summary.json"
        resources = _json(resources_path)

        quality_dir = root / "quality"
        _run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/run_phase9_5h_heldout_quality.py"),
                "--scene-manifest", str(persistence["scene_manifest_path"]),
                "--output-dir", str(quality_dir),
                "--metrics-python", str(args.metrics_python.expanduser().resolve()),
            ],
            log_path=logs / "phase9_5h_quality.log",
        )
        quality_path = quality_dir / "phase9_5h_heldout_quality_summary.json"
        quality = _json(quality_path)

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
        measurement = {
            "state": measurement_result.state.value,
            "native_distance_units": measurement_result.distance_source_units,
            "metric_value_exposed": measurement_result.distance_m is not None,
            "expanded_uncertainty_exposed": measurement_result.expanded_uncertainty_m is not None,
            "certified_metrology": measurement_result.certified_metrology,
        }

        runtime_payload = runtime_report.as_dict()
        runtime = {
            "ready": runtime_report.ready,
            "nerfstudio_revision": runtime_report.nerfstudio_checkout.revision,
            "nerfstudio_clean": runtime_report.nerfstudio_checkout.clean,
            "tiny_cuda_nn_revision": runtime_report.tiny_cuda_nn_checkout.revision,
            "tiny_cuda_nn_clean": runtime_report.tiny_cuda_nn_checkout.clean,
            "python_executable": runtime_report.spec.python_executable,
            "environment_probe": runtime_report.environment_probe.as_dict(),
            "process_policy": runtime_report.spec.process_policy.as_dict(),
            "production_heavy_modules": list(runtime_report.production_heavy_modules),
        }
        source = {
            "status": phase93j.get("status"),
            "scene_id": source_scene.scene_id,
            "scene_manifest": str(source_manifest),
            "canonical_geometry_authoritative": source_scene.canonical_geometry_authoritative,
            "metric_geometry_status": source_authority.get("metric_geometry_status"),
            "unit_to_metre": source_authority.get("unit_to_metre"),
            "certified_metrology": source_authority.get("certified_metrology"),
        }
        heavy_after = list(current_process_nerf_heavy_modules())
        authority = {
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "nerf_scene_role": loaded.artifact.scene_role.value,
            "nerf_canonical_geometry_authoritative": False,
            "nerf_metric_scale_authoritative": False,
            "nerf_measurement_authoritative": False,
            "nerf_qc_authoritative": False,
            "certified_metrology": False,
            "metric_geometry_status": loaded.artifact.metric_geometry_status.value,
            "unit_to_metre": loaded.artifact.unit_to_metre,
            "production_heavy_modules_before": heavy_before,
            "production_heavy_modules_after": heavy_after,
        }
        evidence = {
            "runtime": runtime,
            "source_scene": source,
            "bridge": bridge,
            "optimization": optimization,
            "persistence": persistence,
            "reload": reload_evidence,
            "interactive": interactive,
            "controls": controls,
            "resources": resources,
            "quality": quality,
            "measurement": measurement,
            "authority": authority,
        }
        passed, reasons = validate_phase9_5_chain(evidence)
        extra_reasons: list[str] = list(reasons)
        if source["scene_id"] != bridge.get("scene_id"):
            extra_reasons.append("phase93j_bridge_scene_mismatch")
            passed = False
        if persistence.get("persistent_scene_id") != interactive.get("scene_id"):
            extra_reasons.append("persistent_interactive_scene_mismatch")
            passed = False
        if persistence.get("persistent_scene_id") != resources.get("scene_id"):
            extra_reasons.append("persistent_resource_scene_mismatch")
            passed = False
        if persistence.get("persistent_scene_id") != quality.get("scene_id"):
            extra_reasons.append("persistent_quality_scene_mismatch")
            passed = False
        if len(verified_refs) < int(persistence.get("nerf_file_reference_count", 0) or 0):
            extra_reasons.append("verified_persistent_file_references_incomplete")
            passed = False

        summary = {
            "phase": "9.5H",
            "status": "PASS" if passed else "FAIL",
            "reasons": list(dict.fromkeys(extra_reasons)),
            "real_nerf_campaign_performed": True,
            "source_phase93j_summary": str(phase93j_path),
            "output_root": str(root),
            "bridge_summary": str(bridge_path),
            "optimization_summary": str(optimization_path),
            "persistence_summary": str(persistence_path),
            "persistent_scene_manifest": persistence.get("scene_manifest_path"),
            "interactive_summary": str(interactive_path),
            "controls_summary": str(controls_path),
            "resources_summary": str(resources_path),
            "quality_summary": str(quality_path),
            "verified_persistent_file_reference_count": len(verified_refs),
            "runtime": runtime,
            "source_scene": source,
            "bridge": bridge,
            "optimization": optimization,
            "persistence": persistence,
            "reload": reload_evidence,
            "interactive": interactive,
            "controls": controls,
            "resources": resources,
            "quality": quality,
            "measurement": measurement,
            "authority": authority,
            "runtime_report": runtime_payload,
            "closure": {
                "phase9_5_a_to_g_chain_qualified": passed,
                "quality_characterized_but_not_production_certified": quality.get("quality_characterized") is True and quality.get("production_quality_certified") is False,
                "full_current_tree_regression_required_separately": True,
                "manual_ui_closure_required_separately": True,
                "v0_9_0_tag_created": False,
            },
            "deferred": {
                "real_physical_scale_calibration": True,
                "certified_metrology": True,
                "higher_density_nerf_quality_dataset": True,
                "unified_digital_twin": "Phase 9.6",
            },
        }
    except Exception as exc:
        summary = {
            "phase": "9.5H",
            "status": "FAIL",
            "reasons": [f"{type(exc).__name__}: {exc}"],
            "real_nerf_campaign_performed": False,
            "output_root": str(root),
            "v0_9_0_tag_created": False,
        }

    root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print(f"Phase 9.5H real NeRF closure summary: {summary_path}")
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
