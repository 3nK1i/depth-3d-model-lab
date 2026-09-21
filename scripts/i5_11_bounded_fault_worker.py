#!/usr/bin/env python3
"""Bounded synthetic fault worker for I5.11C.

This worker is intentionally tiny and campaign-specific. It never allocates GPU
memory and never fills a filesystem. Fault modes only emit deterministic failure
signatures inside the supervisor-owned workspace/process group.

I5.11B adds and freezes this worker but does not execute any fault mode.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _spawn_descendant(sleep_s: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({float(sleep_s)!r})"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("complete", "cuda_oom", "disk_enospc", "crash_orphan_descendant"),
        required=True,
    )
    parser.add_argument("--child-sleep-s", type=float, default=30.0)
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    expected = os.environ.get("EIA_RECONSTRUCTION_ENGINE_WORKSPACE", "")
    if expected and Path(expected).resolve() != workspace:
        raise RuntimeError("workspace does not match supervisor-owned workspace")

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "i5_11_partial_artifact.bin").write_bytes(b"i5.11-bounded-fault-partial")

    child = _spawn_descendant(args.child_sleep_s)
    (workspace / "child_pid.txt").write_text(f"{child.pid}\n", encoding="utf-8")

    if args.mode == "cuda_oom":
        print("I511_FAULT=cuda_oom simulated cuda out-of-memory", flush=True)
        raise RuntimeError("simulated cuda out-of-memory")

    if args.mode == "disk_enospc":
        print("I511_FAULT=disk_enospc No space left on device", flush=True)
        raise OSError(errno.ENOSPC, "No space left on device")

    if args.mode == "crash_orphan_descendant":
        print("I511_FAULT=crash_orphan_descendant", flush=True)
        os._exit(17)

    child.terminate()
    try:
        child.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=2.0)

    payload = {
        "phase": "I5.11",
        "mode": "complete",
        "pass": True,
        "child_reaped_by_worker": True,
        "real_gpu_exhaustion": False,
        "real_filesystem_fill": False,
    }
    (workspace / "i5_11_recovery_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (workspace / "i5_11_partial_artifact.bin").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
