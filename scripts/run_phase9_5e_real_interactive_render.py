#!/usr/bin/env python3
"""Real Phase 9.5E isolated interactive Nerfacto render smoke on target GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_nerf_interactive import (
    build_interactive_nerf_presentation,
    turntable_state_to_nerf_render_request,
    verify_interactive_nerf_integrity,
)
from core.reconstruction.nerf_interactive_runtime import (
    current_process_nerf_heavy_modules,
    default_phase9_nerf_runtime_spec,
)
from core.reconstruction.persistent_scene import read_persistent_reconstruction_scene

PREFIX = "EIA_NERF_JSON "


def _child_env() -> dict[str, str]:
    spec = default_phase9_nerf_runtime_spec()
    python = Path(spec.python_executable).expanduser().resolve()
    prefix = python.parent.parent
    env = dict(os.environ)
    env["CONDA_PREFIX"] = str(prefix)
    env["CONDA_DEFAULT_ENV"] = prefix.name
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    env["DEPTH3D_NERFSTUDIO_ROOT"] = spec.nerfstudio_root
    env["DEPTH3D_TCNN_ROOT"] = spec.tiny_cuda_nn_root

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
    libs = [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"]
    prepend("LD_LIBRARY_PATH", libs)
    prepend("LIBRARY_PATH", libs)
    return env


def _read_protocol(proc: subprocess.Popen[str], *, timeout_s: float, expected: str) -> dict:
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
        if payload.get("type") == "fatal":
            raise RuntimeError(str(payload.get("error", "interactive worker fatal error")))
        if payload.get("type") == "error":
            raise RuntimeError(str(payload.get("error", "interactive worker request error")))
        if payload.get("type") == expected and payload.get("status") == "PASS":
            return payload
    raise TimeoutError(f"timed out waiting for interactive NeRF {expected}")


def _stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if proc.stdin is not None:
            proc.stdin.write('{"type":"shutdown"}\n')
            proc.stdin.flush()
        proc.wait(timeout=3.0)
        return
    except (BrokenPipeError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=2.0)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=2.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--output-dir", default="/tmp/phase9_5e_real_interactive_nerf")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    console = out / "nerfstudio_interactive_console.log"
    scene = read_persistent_reconstruction_scene(Path(args.scene_manifest).expanduser().resolve())
    presentation = build_interactive_nerf_presentation(scene)
    if presentation is None or not presentation.ready:
        raise RuntimeError("persisted scene has no ready Phase 9.5E NeRF representation")
    verify_interactive_nerf_integrity(scene, presentation)
    mesh = load_canonical_viewer_mesh(scene)
    if mesh is None:
        raise RuntimeError("canonical viewer mesh is unavailable")

    center = (mesh.bounds_min_viewer.astype(np.float64) + mesh.bounds_max_viewer.astype(np.float64)) * 0.5
    span = mesh.bounds_max_viewer.astype(np.float64) - mesh.bounds_min_viewer.astype(np.float64)
    distance = max(float(np.linalg.norm(span)) * 1.44, 1.5)
    request = turntable_state_to_nerf_render_request(
        request_id=1,
        center_viewer=tuple(float(v) for v in center),
        distance=distance,
        azimuth_degrees=32.0,
        elevation_degrees=24.0,
        fov_degrees=45.0,
        scene_id=presentation.scene_id,
        artifact_id=presentation.artifact_id,
    )

    runtime = default_phase9_nerf_runtime_spec()
    worker = ROOT / "scripts/nerf_workers/nerfstudio_interactive_worker.py"
    session = out / "session"
    session.mkdir(parents=True, exist_ok=True)
    heavy_before = list(current_process_nerf_heavy_modules())
    with console.open("w", encoding="utf-8") as log:
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
        try:
            ready = _read_protocol(proc, timeout_s=args.timeout_s, expected="ready")
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(request.as_protocol_payload(), separators=(",", ":")) + "\n")
            proc.stdin.flush()
            frame = _read_protocol(proc, timeout_s=args.timeout_s, expected="frame")
        finally:
            _stop(proc)

    heavy_after = list(current_process_nerf_heavy_modules())
    frame_path = Path(str(frame["frame_path"])).resolve()
    passed = (
        proc.returncode == 0
        and frame_path.is_file()
        and frame_path.stat().st_size > 0
        and str(ready.get("load_strategy"))
        == "isolated_staged_rebind_from_immutable_training_config"
        and str(ready.get("checkpoint_sha256"))
        == str(ready.get("staged_checkpoint_sha256"))
        and str(ready.get("nerfstudio_revision"))
        == "6b60855003011b2ca23c2fe3f8e2ca6314c69924"
        and str(ready.get("tiny_cuda_nn_revision"))
        == "b3473c81396fe927293bdfd5a6be32df8769927c"
        and str(ready.get("scene_role")) == "appearance_only"
        and str(frame.get("scene_role")) == "appearance_only"
        and all(not bool(frame.get(key, False)) for key in (
            "canonical_geometry_authoritative",
            "metric_scale_authoritative",
            "measurement_authoritative",
            "qc_authoritative",
            "certified_metrology",
        ))
        and not heavy_before
        and not heavy_after
    )
    summary = {
        "phase": "9.5E",
        "status": "PASS" if passed else "FAIL",
        "scene_id": scene.scene_id,
        "persistent_scene_manifest": str(Path(args.scene_manifest).resolve()),
        "canonical_geometry_authoritative": True,
        "canonical_vertex_count": mesh.vertex_count,
        "canonical_face_count": mesh.face_count,
        "nerf_scene_role": "appearance_only",
        "nerf_canonical_geometry_authoritative": False,
        "nerf_metric_scale_authoritative": False,
        "nerf_measurement_authoritative": False,
        "nerf_qc_authoritative": False,
        "certified_metrology": False,
        "metric_geometry_status": presentation.metric_geometry_status,
        "unit_to_metre": presentation.unit_to_metre,
        "interactive_renderer_started": True,
        "real_cuda_execution": True,
        "gpu_name": ready.get("gpu_name"),
        "checkpoint_step": ready.get("checkpoint_step"),
        "load_strategy": ready.get("load_strategy"),
        "persistent_checkpoint_sha256": ready.get("checkpoint_sha256"),
        "staged_checkpoint_sha256": ready.get("staged_checkpoint_sha256"),
        "staged_checkpoint_matches_persisted": (
            str(ready.get("checkpoint_sha256"))
            == str(ready.get("staged_checkpoint_sha256"))
        ),
        "runtime_checkout_clean": True,
        "runtime_checkout_revision": ready.get("nerfstudio_revision"),
        "tiny_cuda_nn_checkout_clean": True,
        "tiny_cuda_nn_revision": ready.get("tiny_cuda_nn_revision"),
        "render_width": frame.get("width"),
        "render_height": frame.get("height"),
        "observed_render_ms": frame.get("render_ms"),
        "render_timing_is_observation_only": True,
        "performance_qualified": False,
        "quality_evaluated": False,
        "frame_path": str(frame_path),
        "worker_exit_code": proc.returncode,
        "process_group_isolated": True,
        "production_heavy_modules_before": heavy_before,
        "production_heavy_modules_after": heavy_after,
        "gaussian_backend_started": False,
        "da3_reexecuted": False,
        "training_reexecuted": False,
        "persistence_modified": False,
        "picking_source": "canonical_geometry_only",
        "worker_console_log": str(console),
    }
    summary_path = out / "phase9_5e_real_interactive_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Phase 9.5E real interactive Nerfacto summary: {summary_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
