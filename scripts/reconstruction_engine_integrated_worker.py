#!/usr/bin/env python3
"""Run the already-qualified F4 pipeline followed by F5 persistence as one engine.

This worker exists so the G1 supervisor can own a single process group containing
the full integrated reconstruction attempt and all external Gaussian/NeRF children.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, stage: str) -> None:
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"engine_child_started stage={stage} pid={child.pid}", flush=True)
    stdout, _ = child.communicate()
    if stdout:
        print(stdout, end="", flush=True)
    if child.returncode != 0:
        raise RuntimeError(f"integrated engine child failed with exit code {child.returncode}")
    print(f"engine_child_completed stage={stage} pid={child.pid}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--nerf-training-steps", type=int, default=16)
    parser.add_argument("--backend-timeout-s", type=float, default=1800.0)
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    expected = os.environ.get("EIA_RECONSTRUCTION_ENGINE_WORKSPACE", "")
    if expected and Path(expected).resolve() != workspace:
        raise RuntimeError("workspace does not match supervisor-owned workspace")
    workspace.mkdir(parents=True, exist_ok=True)

    f4_dir = workspace / "f4"
    f5_dir = workspace / "f5"
    _run(
        [
            sys.executable,
            str((ROOT / "scripts/run_reconstruction_f4_real_backends.py").resolve()),
            "--output-dir",
            str(f4_dir),
            "--nerf-training-steps",
            str(int(args.nerf_training_steps)),
            "--timeout-s",
            str(float(args.backend_timeout_s)),
        ],
        stage="f4",
    )
    _run(
        [
            sys.executable,
            str((ROOT / "scripts/run_reconstruction_f5_real_persistence.py").resolve()),
            "--f4-result",
            str((f4_dir / "pipeline_result.json").resolve()),
            "--f4-summary",
            str((f4_dir / "summary.json").resolve()),
            "--output-dir",
            str(f5_dir),
        ],
        stage="f5",
    )

    f4 = json.loads((f4_dir / "summary.json").read_text(encoding="utf-8"))
    f5 = json.loads((f5_dir / "summary.json").read_text(encoding="utf-8"))
    payload = {
        "phase": "8.7G1",
        "pass": bool(f4.get("pass") and f5.get("pass")),
        "pipeline_status": str(f5.get("pipeline_status", "")),
        "persistent_scene_state": str(f5.get("state", "")),
        "scene_id": str(f5.get("scene_id", "")),
        "scene_fingerprint_sha256": str(f5.get("scene_fingerprint_sha256", "")),
        "canonical_geometry_authoritative": bool(f5.get("canonical_geometry_authoritative", False)),
        "gaussian_authority": str(f5.get("gaussian_authority", "")),
        "nerf_authority": str(f5.get("nerf_authority", "")),
        "certified_metrology": bool(f5.get("certified_metrology", False)),
        "reconstruction_ui_enabled": bool(f5.get("reconstruction_ui_enabled", False)),
    }
    if not payload["pass"]:
        raise RuntimeError("integrated F4/F5 engine result did not pass")
    if not payload["canonical_geometry_authoritative"]:
        raise RuntimeError("integrated engine lost canonical geometry authority")
    if payload["gaussian_authority"] != "appearance_only" or payload["nerf_authority"] != "appearance_only":
        raise RuntimeError("integrated engine appearance authority drifted")
    if payload["certified_metrology"] or payload["reconstruction_ui_enabled"]:
        raise RuntimeError("integrated engine crossed a deferred authority/UI boundary")

    (workspace / "engine_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Phase 8.7G1 integrated engine worker: PASS")
    print(f"scene_id={payload['scene_id']} state={payload['persistent_scene_state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
