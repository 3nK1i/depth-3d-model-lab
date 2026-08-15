#!/usr/bin/env python3
"""Phase 9.3J real-scene DA3 -> canonical qualification harness.

By default this script performs a *real* execution of the production Phase 9.3H
worker against the existing South Building image sequence.  DA3 still runs only
inside the qualified external ``eia-da3`` interpreter.  The harness then reads
back the persistent canonical scene, verifies file integrity/provenance and
records measured elapsed time / DA3 peak VRAM exactly as reported by the real
worker.

``--existing-worker-output`` is a validation-only mode for automated tests or
for re-checking an already produced worker result. It never counts as a new
real execution in the qualification summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from core.reconstruction.persistent_authority import read_persistent_scene_authority_snapshot
from core.reconstruction.persistent_scene import (
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
)

DEFAULT_IMAGES = Path("/home/eiatech/Desktop/phase8_3c_south_building/south-building/images")
DEFAULT_DA3_PYTHON = Path("/home/eiatech/anaconda3/envs/eia-da3/bin/python")
DEFAULT_DA3_REPO = Path("/home/eiatech/Desktop/eia-da3-backends/depth-anything-3")
WORKER = ROOT / "scripts" / "run_phase9_3h_da3_canonical_job.py"
SUMMARY_NAME = "phase9_3j_qualification_summary.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/phase9_3j_real_scene"))
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    p.add_argument("--da3-python", type=Path, default=DEFAULT_DA3_PYTHON)
    p.add_argument("--da3-repo", type=Path, default=DEFAULT_DA3_REPO)
    p.add_argument("--process-res", type=int, default=392)
    p.add_argument("--pixel-stride", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=3)
    p.add_argument("--timeout-seconds", type=float, default=900.0)
    p.add_argument(
        "--existing-worker-output",
        type=Path,
        default=None,
        help="Validate an existing 9.3H worker output directory/offline_result.json without running DA3.",
    )
    p.add_argument(
        "--authority-manifest",
        type=Path,
        default=None,
        help="Optional Phase 9.1H authority manifest to validate against the generated scene.",
    )
    return p


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Unable to read JSON evidence {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"JSON evidence root must be an object: {path}")
    return raw


def _resolve_worker_result(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.is_dir():
        target = target / "offline_result.json"
    if not target.is_file():
        raise RuntimeError(f"Phase 9.3H worker result does not exist: {target}")
    return target


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_real_worker(args: argparse.Namespace, root: Path) -> tuple[Path, float, Path]:
    job_root = (root / "da3-job").resolve()
    job_root.mkdir(parents=True, exist_ok=True)
    log_path = root / "da3_worker_console.log"
    command = [
        sys.executable,
        str(WORKER),
        "--session-id",
        str(uuid.uuid4()),
        "--output-dir",
        str(job_root),
        "--images-dir",
        str(args.images_dir.expanduser().resolve()),
        "--da3-python",
        str(args.da3_python.expanduser().resolve()),
        "--da3-repo",
        str(args.da3_repo.expanduser().resolve()),
        "--process-res",
        str(int(args.process_res)),
        "--pixel-stride",
        str(int(args.pixel_stride)),
        "--max-frames",
        str(int(args.max_frames)),
        "--timeout-seconds",
        str(float(args.timeout_seconds)),
    ]
    print("Phase 9.3J real DA3 worker command:")
    print(" ".join(command))
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    try:
        with log_path.open("w", encoding="utf-8") as log:
            assert process.stdout is not None
            deadline = started + float(args.timeout_seconds) + 60.0
            while True:
                if time.monotonic() > deadline:
                    _terminate_process_group(process)
                    raise RuntimeError("Phase 9.3J real DA3 worker exceeded the qualification timeout.")
                line = process.stdout.readline()
                if line:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                elif process.poll() is not None:
                    break
                else:
                    time.sleep(0.02)
            return_code = process.wait(timeout=5.0)
    finally:
        if process.poll() is None:
            _terminate_process_group(process)
    elapsed = time.monotonic() - started
    result = job_root / "offline_result.json"
    if return_code != 0:
        detail = _load_json(result).get("error", "worker_failed") if result.is_file() else "worker_failed"
        raise RuntimeError(f"Real Phase 9.3H DA3 worker failed ({return_code}): {detail}")
    if not result.is_file():
        raise RuntimeError("Real DA3 worker completed without offline_result.json.")
    return result, elapsed, log_path


def _is_sha256(value: object) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _module_loaded(prefix: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for name in sys.modules)


def validate_worker_result(
    result_path: Path,
    *,
    real_execution: bool,
    elapsed_seconds: float | None,
    authority_manifest: Path | None = None,
) -> dict[str, Any]:
    result = _load_json(result_path)
    if not bool(result.get("pass")) or str(result.get("status")) != "completed":
        raise RuntimeError(f"Phase 9.3H worker result is not successful: {result.get('error', result.get('status'))}")
    manifest = Path(str(result.get("persistent_scene_manifest", ""))).expanduser().resolve()
    scene = read_persistent_reconstruction_scene(manifest)
    verified_refs = verify_persistent_scene_file_references(scene)
    mesh = load_persistent_canonical_mesh(scene)
    cameras = load_persistent_canonical_camera_trajectory(scene)
    if mesh is None or cameras is None:
        raise RuntimeError("Persistent DA3 scene did not materialize canonical mesh and camera trajectory.")
    fingerprints = tuple(cameras.source_frame_sha256)
    if len(fingerprints) != cameras.persisted_pose_count or not fingerprints:
        raise RuntimeError("Persistent DA3 camera trajectory lacks one exact source fingerprint per pose.")
    if len(set(fingerprints)) != len(fingerprints) or not all(_is_sha256(value) for value in fingerprints):
        raise RuntimeError("Persistent DA3 source-frame fingerprints are malformed or non-unique.")

    if not scene.canonical_geometry_authoritative:
        raise RuntimeError("DA3 qualification scene lost canonical geometry authority after canonicalization.")
    if scene.certified_metrology or bool(result.get("certified_metrology")):
        raise RuntimeError("DA3 qualification scene incorrectly claims certified metrology.")
    if str(result.get("metric_geometry_status")) != "arbitrary" or result.get("unit_to_metre") is not None:
        raise RuntimeError("DA3 qualification invented metric scale.")
    if not bool(result.get("da3_supporting_authority_only")):
        raise RuntimeError("DA3 result did not preserve supporting-only provider authority.")
    if bool(result.get("semantic_context_available")) or bool(result.get("semantic_geometry_fabricated")):
        raise RuntimeError("DA3 qualification fabricated semantic context.")
    if bool(result.get("inspection_qc_available")) or bool(result.get("inspection_qc_fabricated")):
        raise RuntimeError("DA3 qualification fabricated inspection/QC evidence.")

    runtime_checks = {
        "runtime_probed": bool(result.get("da3_runtime_probed")),
        "backend_started": bool(result.get("da3_backend_started")),
        "runtime_lock_valid": bool(result.get("da3_runtime_lock_valid")),
        "turing_compatibility_patch_applied": bool(result.get("da3_turing_compatibility_patch_applied")),
        "import_source": str(result.get("da3_import_source", "")),
        "source_revision": str(result.get("da3_source_revision", "")),
        "model_id": str(result.get("da3_model_id", "")),
        "model_revision": str(result.get("da3_model_revision", "")),
        "gpu_name": str(result.get("da3_gpu_name", "")),
        "peak_vram_mib": float(result.get("da3_peak_vram_mib", 0.0) or 0.0),
    }
    if real_execution:
        if not runtime_checks["runtime_probed"] or not runtime_checks["backend_started"]:
            raise RuntimeError("Real 9.3J qualification did not actually probe/start isolated DA3.")
        if not runtime_checks["runtime_lock_valid"]:
            raise RuntimeError("Real 9.3J qualification did not validate the DA3 runtime lock.")
        if not runtime_checks["import_source"] or not runtime_checks["source_revision"]:
            raise RuntimeError("Real 9.3J qualification lacks DA3 checkout/import provenance.")
        if not runtime_checks["gpu_name"] or not math.isfinite(runtime_checks["peak_vram_mib"]) or runtime_checks["peak_vram_mib"] <= 0.0:
            raise RuntimeError("Real 9.3J qualification lacks measured GPU/VRAM evidence.")
        if "2060" in runtime_checks["gpu_name"].lower() and not runtime_checks["turing_compatibility_patch_applied"]:
            raise RuntimeError("RTX 2060/Turing real qualification did not report the qualified compatibility patch.")

    authority_state: dict[str, Any]
    if authority_manifest is not None:
        authority = read_persistent_scene_authority_snapshot(
            authority_manifest.expanduser().resolve(), expected_scene=scene
        )
        authority_state = {
            "state": "PRESENT",
            "metric_coordinates_allowed": authority.metric_coordinates_allowed,
            "calibrated_metric_scale": authority.calibrated_metric_scale,
            "camera_calibration_present": authority.camera_calibration_profile is not None,
            "certified_metrology": authority.certified_metrology,
        }
    else:
        authority_state = {
            "state": "DEFERRED",
            "metric_coordinates_allowed": False,
            "calibrated_metric_scale": False,
            "camera_calibration_present": False,
            "certified_metrology": False,
            "reason": "No real scene-bound physical calibration/metric-authority sidecar supplied for this campaign.",
        }

    heavy_imports = {
        "depth_anything_3_in_depth_process": _module_loaded("depth_anything_3"),
        "gsplat_in_depth_process": _module_loaded("gsplat"),
        "nerfstudio_in_depth_process": _module_loaded("nerfstudio"),
    }
    if any(heavy_imports.values()):
        raise RuntimeError("9.3J qualification contaminated the production process with an isolated heavy runtime.")

    measured_elapsed = None if elapsed_seconds is None else float(elapsed_seconds)
    if measured_elapsed is not None and (not math.isfinite(measured_elapsed) or measured_elapsed <= 0.0):
        raise RuntimeError("Measured real qualification elapsed time is invalid.")

    return {
        "status": "PASS" if real_execution else "PASS_VALIDATE_ONLY",
        "phase": "9.3J",
        "real_execution_performed": bool(real_execution),
        "real_execution": {
            **runtime_checks,
            "elapsed_seconds": measured_elapsed,
        },
        "persistent_scene": {
            "scene_id": scene.scene_id,
            "scene_state": scene.state.value,
            "manifest": str(manifest),
            "verified_file_reference_count": len(verified_refs),
            "canonical_vertex_count": mesh.vertex_count,
            "canonical_face_count": mesh.face_count,
            "canonical_camera_count": cameras.persisted_pose_count,
            "source_frame_fingerprint_count": len(fingerprints),
            "source_frame_fingerprints_valid": True,
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "certified_metrology": scene.certified_metrology,
        },
        "authority": {
            "da3_supporting_only": True,
            "metric_geometry_status": "arbitrary",
            "unit_to_metre": None,
            "certified_metrology": False,
            "physical_metric_authority": authority_state,
        },
        "semantic_and_qc": {
            "semantic_context_available": False,
            "semantic_geometry_fabricated": False,
            "inspection_qc_available": False,
            "inspection_qc_fabricated": False,
        },
        "measurement_registration": {
            "state": "UNAVAILABLE",
            "canonical_dimension_geometry": False,
            "reason": (
                "Real DA3 scene has arbitrary scale and no qualified real semantic/metric/pose-uncertainty binding; "
                "the existing Measure & Inspect measurement path remains usable without canonical placement."
            ),
            "calibration_required_for_application": False,
        },
        "runtime_isolation": heavy_imports,
        "deferred": {
            "real_physical_checkerboard_charuco_calibration": authority_manifest is None,
            "real_scene_metric_authority_restart_reload": authority_manifest is None,
            "gaussian_interactive_qualification": "Phase 9.4",
            "nerf_interactive_qualification": "Phase 9.5",
            "unified_canonical_gaussian_nerf_qualification": "Phase 9.6",
        },
    }


def load_persistent_canonical_mesh(scene):
    # Kept behind a tiny helper to make the qualification intent explicit.
    return load_canonical_viewer_mesh(scene)


def main() -> int:
    args = _parser().parse_args()
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    real_execution = args.existing_worker_output is None
    elapsed: float | None = None
    log_path: Path | None = None
    try:
        if real_execution:
            images = args.images_dir.expanduser().resolve()
            if not images.is_dir():
                raise RuntimeError(f"Real South Building image directory does not exist: {images}")
            result_path, elapsed, log_path = _run_real_worker(args, root)
        else:
            result_path = _resolve_worker_result(args.existing_worker_output)
        summary = validate_worker_result(
            result_path,
            real_execution=real_execution,
            elapsed_seconds=elapsed,
            authority_manifest=args.authority_manifest,
        )
        summary["worker_result"] = str(result_path)
        summary["worker_console_log"] = "" if log_path is None else str(log_path)
        summary_path = root / SUMMARY_NAME
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Phase 9.3J qualification summary: {summary_path}")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "phase": "9.3J",
            "real_execution_performed": bool(real_execution),
            "error": f"{type(exc).__name__}: {exc}",
        }
        summary_path = root / SUMMARY_NAME
        summary_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
