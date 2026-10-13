#!/usr/bin/env python3
"""Synthetic whole-engine fault worker for Phase 8.7G1 lifecycle validation.

The worker intentionally creates a descendant process and a partial artifact so
G1 can verify process-group and workspace cleanup without invoking a heavy model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mode", choices=("complete", "hang", "crash"), required=True)
    parser.add_argument("--child-sleep-s", type=float, default=30.0)
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    expected = os.environ.get("DEPTH3D_RECONSTRUCTION_ENGINE_WORKSPACE", "")
    if expected and Path(expected).resolve() != workspace:
        raise RuntimeError("workspace does not match supervisor-owned workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "partial_artifact.bin").write_bytes(b"phase8-7g1-partial")

    child = subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({float(args.child_sleep_s)!r})"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    (workspace / "child_pid.txt").write_text(f"{child.pid}\n", encoding="utf-8")

    if args.mode == "crash":
        # Leave the descendant alive intentionally. The G1 supervisor must reap
        # the entire engine process group after the parent exits abnormally.
        os._exit(17)

    if args.mode == "hang":
        while True:
            time.sleep(0.25)

    child.terminate()
    try:
        child.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=2.0)
    payload = {
        "phase": "8.7G1",
        "mode": args.mode,
        "pass": True,
        "child_reaped_by_worker": True,
    }
    (workspace / "engine_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (workspace / "partial_artifact.bin").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
