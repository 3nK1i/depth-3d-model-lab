#!/usr/bin/env python3
"""Real Phase 9.5G RTX-target NeRF performance/resource lifecycle qualification.

This script repeatedly starts the already-qualified isolated interactive Nerfacto
worker, renders all 9.5F display profiles, records worker-reported torch CUDA
telemetry plus nvidia-smi per-process memory, then proves the worker PID leaves
both the OS process table and nvidia-smi after each release. No training occurs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import signal
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_nerf_controls import NERF_RENDER_PROFILES
from app.reconstruction_nerf_interactive import (
    build_interactive_nerf_presentation,
    turntable_state_to_nerf_render_request,
    verify_interactive_nerf_integrity,
)
from app.reconstruction_nerf_resources import build_nerf_interactive_resource_policy
from core.reconstruction.nerf_interactive_runtime import (
    current_process_nerf_heavy_modules,
    default_phase9_nerf_runtime_spec,
)
from core.reconstruction.persistent_scene import read_persistent_reconstruction_scene

PREFIX = "EIA_NERF_JSON "
EXPECTED_NS_REVISION = "6b60855003011b2ca23c2fe3f8e2ca6314c69924"
EXPECTED_TCNN_REVISION = "b3473c81396fe927293bdfd5a6be32df8769927c"


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return float(ordered[index])


def _child_env() -> dict[str, str]:
    spec = default_phase9_nerf_runtime_spec()
    python = Path(spec.python_executable).expanduser().resolve()
    prefix = python.parent.parent
    env = dict(os.environ)
    env["CONDA_PREFIX"] = str(prefix)
    env["CONDA_DEFAULT_ENV"] = prefix.name
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    env["EIA_NERFSTUDIO_ROOT"] = spec.nerfstudio_root
    env["EIA_TCNN_ROOT"] = spec.tiny_cuda_nn_root

    def prepend(key: str, values: list[Path]) -> None:
        parts = [str(value) for value in values if value.is_dir()]
        if env.get(key):
            parts.append(env[key])
        env[key] = os.pathsep.join(parts)

    prepend("PATH", [prefix / "bin"])
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.is_file():
        env["CUDA_HOME"] = str(prefix)
        env["CUDACXX"] = str(nvcc)
    prepend(
        "LD_LIBRARY_PATH",
        [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"],
    )
    prepend(
        "LIBRARY_PATH",
        [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"],
    )
    return env


def _read_protocol(proc: subprocess.Popen[str], *, timeout_s: float, expected: str) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if proc.stdout is None:
            raise RuntimeError("interactive worker stdout is unavailable")
        wait = max(0.0, min(0.5, deadline - time.monotonic()))
        ready, _, _ = select.select([proc.stdout], [], [], wait)
        if not ready:
            if proc.poll() is not None:
                raise RuntimeError(f"interactive worker exited before {expected}: {proc.returncode}")
            continue
        line = proc.stdout.readline()
        if not line:
            continue
        line = line.strip()
        if not line.startswith(PREFIX):
            continue
        payload = json.loads(line[len(PREFIX):])
        if not isinstance(payload, dict):
            continue
        kind = str(payload.get("type", ""))
        if kind in {"fatal", "error"}:
            raise RuntimeError(str(payload.get("error", f"interactive worker {kind}")))
        if kind == expected and payload.get("status") == "PASS":
            return payload
    raise TimeoutError(f"timed out waiting for interactive NeRF {expected}")


def _nvidia_smi_process_memory() -> dict[int, float]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5.0, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Phase 9.5G real qualification requires working nvidia-smi") from exc
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi compute-process query failed: " + completed.stderr.strip())
    values: dict[int, float] = {}
    for line in completed.stdout.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            memory = float(parts[1])
        except ValueError:
            continue
        values[pid] = memory
    return values


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_released(pid: int, *, timeout_s: float = 8.0) -> tuple[bool, float]:
    started = time.monotonic()
    deadline = started + float(timeout_s)
    while time.monotonic() < deadline:
        gpu_processes = _nvidia_smi_process_memory()
        if not _pid_exists(pid) and pid not in gpu_processes:
            return True, (time.monotonic() - started) * 1000.0
        time.sleep(0.1)
    gpu_processes = _nvidia_smi_process_memory()
    return (not _pid_exists(pid) and pid not in gpu_processes), (time.monotonic() - started) * 1000.0


def _graceful_stop(proc: subprocess.Popen[str], *, timeout_s: float) -> tuple[dict[str, Any], bool]:
    stopped: dict[str, Any] = {}
    graceful = False
    if proc.poll() is None:
        try:
            assert proc.stdin is not None
            proc.stdin.write('{"type":"shutdown"}\n')
            proc.stdin.flush()
            stopped = _read_protocol(proc, timeout_s=min(5.0, timeout_s), expected="stopped")
            proc.wait(timeout=min(5.0, timeout_s))
            graceful = proc.returncode == 0
        except (BrokenPipeError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
            graceful = False
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=2.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=2.0)
    return stopped, graceful


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--output-dir", default="/tmp/phase9_5g_real_nerf_resources")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    cycles = max(1, int(args.cycles))
    timeout_s = max(20.0, float(args.timeout_s))

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.scene_manifest).expanduser().resolve()
    scene = read_persistent_reconstruction_scene(manifest)
    presentation = build_interactive_nerf_presentation(scene)
    if presentation is None or not presentation.ready:
        raise RuntimeError("persisted scene has no ready Phase 9.5 NeRF representation")
    verify_interactive_nerf_integrity(scene, presentation)
    mesh = load_canonical_viewer_mesh(scene)
    if mesh is None:
        raise RuntimeError("canonical viewer mesh is unavailable")

    policy = build_nerf_interactive_resource_policy()
    center = (mesh.bounds_min_viewer.astype(np.float64) + mesh.bounds_max_viewer.astype(np.float64)) * 0.5
    span = mesh.bounds_max_viewer.astype(np.float64) - mesh.bounds_min_viewer.astype(np.float64)
    distance = max(float(np.linalg.norm(span)) * 1.44, 1.5)
    runtime = default_phase9_nerf_runtime_spec()
    worker = ROOT / "scripts/nerf_workers/nerfstudio_interactive_worker.py"
    heavy_before = list(current_process_nerf_heavy_modules())

    profile_timings: dict[str, list[float]] = {key: [] for key, _label, _w, _h in NERF_RENDER_PROFILES}
    worker_gpu_memory_samples: list[float] = []
    torch_reserved_samples: list[float] = []
    torch_allocated_samples: list[float] = []
    torch_peak_allocated_samples: list[float] = []
    load_ms_samples: list[float] = []
    release_ms_samples: list[float] = []
    release_checks: list[bool] = []
    graceful_stop_checks: list[bool] = []
    frame_retention_checks: list[bool] = []
    checkout_checks: list[bool] = []
    worker_pids: list[int] = []
    total_render_count = 0
    request_id = 0

    for cycle in range(cycles):
        session = out / f"cycle-{cycle + 1:02d}"
        session.mkdir(parents=True, exist_ok=True)
        console = session / "worker_console.log"
        with console.open("w", encoding="utf-8") as log:
            started = time.monotonic()
            proc = subprocess.Popen(
                [
                    runtime.python_executable,
                    str(worker),
                    "--load-spec",
                    presentation.load_spec_path,
                    "--session-dir",
                    str(session),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
                env=_child_env(),
                cwd=str(ROOT),
                start_new_session=True,
            )
            ready: dict[str, Any] = {}
            stopped: dict[str, Any] = {}
            try:
                ready = _read_protocol(proc, timeout_s=timeout_s, expected="ready")
                load_ms_samples.append((time.monotonic() - started) * 1000.0)
                worker_pid = int(ready.get("worker_pid", proc.pid) or proc.pid)
                worker_pids.append(worker_pid)
                checkout_checks.append(
                    str(ready.get("nerfstudio_revision")) == EXPECTED_NS_REVISION
                    and str(ready.get("tiny_cuda_nn_revision")) == EXPECTED_TCNN_REVISION
                    and str(ready.get("resource_hardening_phase")) == "9.5G"
                )
                gpu_processes = _nvidia_smi_process_memory()
                if worker_pid in gpu_processes:
                    worker_gpu_memory_samples.append(float(gpu_processes[worker_pid]))
                torch_reserved_samples.append(float(ready.get("cuda_reserved_mib", 0.0) or 0.0))
                torch_allocated_samples.append(float(ready.get("cuda_allocated_mib", 0.0) or 0.0))

                for profile_index, (profile, _label, width, height) in enumerate(NERF_RENDER_PROFILES):
                    request_id += 1
                    request = turntable_state_to_nerf_render_request(
                        request_id=request_id,
                        center_viewer=tuple(float(v) for v in center),
                        distance=distance,
                        azimuth_degrees=24.0 + cycle * 7.0 + profile_index * 4.0,
                        elevation_degrees=18.0 + profile_index * 3.0,
                        fov_degrees=45.0,
                        scene_id=presentation.scene_id,
                        artifact_id=presentation.artifact_id,
                        width=width,
                        height=height,
                    )
                    assert proc.stdin is not None
                    proc.stdin.write(json.dumps(request.as_protocol_payload(), separators=(",", ":")) + "\n")
                    proc.stdin.flush()
                    frame = _read_protocol(proc, timeout_s=timeout_s, expected="frame")
                    total_render_count += 1
                    profile_timings[profile].append(float(frame.get("render_ms", 0.0) or 0.0))
                    torch_reserved_samples.append(float(frame.get("cuda_reserved_mib", 0.0) or 0.0))
                    torch_allocated_samples.append(float(frame.get("cuda_allocated_mib", 0.0) or 0.0))
                    torch_peak_allocated_samples.append(float(frame.get("cuda_peak_allocated_mib", 0.0) or 0.0))
                    retained = int(frame.get("retained_frame_count", 0) or 0)
                    frame_retention_checks.append(
                        0 <= retained <= policy.max_retained_frames
                        and int(frame.get("max_retained_frames", 0) or 0) == policy.max_retained_frames
                    )
                    gpu_processes = _nvidia_smi_process_memory()
                    if worker_pid in gpu_processes:
                        worker_gpu_memory_samples.append(float(gpu_processes[worker_pid]))
            finally:
                stopped, graceful = _graceful_stop(proc, timeout_s=timeout_s)
                graceful_stop_checks.append(graceful)

        worker_pid = int(ready.get("worker_pid", proc.pid) or proc.pid)
        released, release_ms = _wait_released(worker_pid)
        release_checks.append(released)
        release_ms_samples.append(release_ms)
        if stopped:
            frame_retention_checks.append(
                int(stopped.get("retained_frame_count", 0) or 0) <= policy.max_retained_frames
            )

    heavy_after = list(current_process_nerf_heavy_modules())
    profile_summary = {
        profile: {
            "samples": len(values),
            "median_render_ms": statistics.median(values) if values else 0.0,
            "p95_render_ms": _p95(values),
            "min_render_ms": min(values) if values else 0.0,
            "max_render_ms": max(values) if values else 0.0,
        }
        for profile, values in profile_timings.items()
    }
    all_profile_samples = all(len(values) == cycles for values in profile_timings.values())
    passed = (
        all_profile_samples
        and total_render_count == cycles * len(NERF_RENDER_PROFILES)
        and all(release_checks)
        and all(graceful_stop_checks)
        and all(frame_retention_checks)
        and all(checkout_checks)
        and bool(worker_gpu_memory_samples)
        and max(torch_reserved_samples or [0.0]) > 0.0
        and max(torch_allocated_samples or [0.0]) > 0.0
        and not heavy_before
        and not heavy_after
        and bool(scene.canonical_geometry_authoritative)
    )
    summary = {
        "phase": "9.5G",
        "status": "PASS" if passed else "FAIL",
        "scene_manifest": str(manifest),
        "scene_id": scene.scene_id,
        "cycles": cycles,
        "worker_start_count": cycles,
        "worker_release_count": sum(1 for value in release_checks if value),
        "all_worker_pids_released_from_os_and_nvidia_smi": all(release_checks),
        "all_shutdowns_graceful": all(graceful_stop_checks),
        "worker_pids": worker_pids,
        "total_render_count": total_render_count,
        "render_profiles": profile_summary,
        "performance_characterized": all_profile_samples,
        "render_timings_are_observation_only": True,
        "visual_quality_evaluated": False,
        "resource_tier": policy.resource_tier,
        "device_label": policy.device_label,
        "max_inflight_renders": policy.max_inflight_renders,
        "max_pending_renders": policy.max_pending_renders,
        "request_queue_policy": policy.queue_text,
        "max_retained_frames": policy.max_retained_frames,
        "all_frame_retention_checks_passed": all(frame_retention_checks),
        "median_worker_load_ms": statistics.median(load_ms_samples) if load_ms_samples else 0.0,
        "p95_worker_load_ms": _p95(load_ms_samples),
        "median_worker_release_ms": statistics.median(release_ms_samples) if release_ms_samples else 0.0,
        "p95_worker_release_ms": _p95(release_ms_samples),
        "worker_torch_cuda_telemetry_measured": bool(torch_reserved_samples),
        "max_worker_torch_cuda_allocated_mib": max(torch_allocated_samples or [0.0]),
        "max_worker_torch_cuda_reserved_mib": max(torch_reserved_samples or [0.0]),
        "max_worker_torch_cuda_peak_allocated_mib": max(torch_peak_allocated_samples or [0.0]),
        "nvidia_smi_worker_memory_measured": bool(worker_gpu_memory_samples),
        "max_nvidia_smi_worker_memory_mib": max(worker_gpu_memory_samples or [0.0]),
        "vram_metrics_scope": "isolated_nerfstudio_worker_process_only",
        "canonical_geometry_authoritative": True,
        "picking_source": "canonical_geometry_only",
        "nerf_scene_role": "appearance_only",
        "nerf_canonical_geometry_authoritative": False,
        "nerf_metric_scale_authoritative": False,
        "nerf_measurement_authoritative": False,
        "nerf_qc_authoritative": False,
        "certified_metrology": False,
        "metric_geometry_status": presentation.metric_geometry_status,
        "unit_to_metre": presentation.unit_to_metre,
        "training_reexecuted": False,
        "persistence_modified": False,
        "gaussian_backend_started": False,
        "da3_reexecuted": False,
        "runtime_checkout_clean": all(checkout_checks),
        "runtime_checkout_revision": EXPECTED_NS_REVISION,
        "tiny_cuda_nn_checkout_clean": all(checkout_checks),
        "tiny_cuda_nn_revision": EXPECTED_TCNN_REVISION,
        "production_heavy_modules_before": heavy_before,
        "production_heavy_modules_after": heavy_after,
    }
    summary_path = out / "phase9_5g_real_nerf_resource_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Phase 9.5G real NeRF resource summary: {summary_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
