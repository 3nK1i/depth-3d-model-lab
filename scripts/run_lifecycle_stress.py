#!/usr/bin/env python3
"""Repeated Development Mock load/run/stop/unload lifecycle stress test."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models.catalog import MOCK_MODEL_ID  # noqa: E402
from core.runtime.messages import CommandType  # noqa: E402
from core.runtime.stress import (  # noqa: E402
    make_sample,
    tail_drift,
    wait_for_process_exit,
    write_stress_report,
)
from scripts.stress_support import WorkerHarness  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--frames-per-cycle", type=int, default=2)
    parser.add_argument("--restart-every", type=int, default=0)
    parser.add_argument("--max-parent-rss-drift-mib", type=float, default=64.0)
    parser.add_argument("--max-worker-rss-drift-mib", type=float, default=64.0)
    parser.add_argument("--settle-ms", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stress/lifecycle_stress.json"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    cycles = max(1, args.cycles)
    frames_per_cycle = max(1, args.frames_per_cycle)
    restart_every = max(0, args.restart_every)
    harness = WorkerHarness(command_timeout_ms=5_000)
    samples = []
    observed_pids: list[int] = []
    expected_restarts = 0

    try:
        harness.manager.start_worker()
        harness.wait_until(
            lambda: harness.manager.is_running,
            timeout_s=5.0,
            description="initial worker start",
        )
        if harness.manager.worker_pid is not None:
            observed_pids.append(harness.manager.worker_pid)

        for cycle in range(1, cycles + 1):
            harness.wait_for_ack(
                lambda: harness.manager.load_model(
                    MOCK_MODEL_ID,
                    {"device": "cpu", "precision": "fp32", "latency_ms": 0.0},
                ),
                CommandType.LOAD_MODEL,
            )
            harness.wait_for_ack(harness.manager.resume, CommandType.RESUME)

            image = np.zeros((48, 64, 3), dtype=np.uint8)
            for frame_id in range(frames_per_cycle):
                result = harness.wait_for_result(
                    lambda fid=frame_id: harness.manager.submit_frame(
                        cycle,
                        fid,
                        image,
                        point_budget=1_000,
                        submitted_at=time.monotonic(),
                    ),
                    session_id=cycle,
                    frame_id=frame_id,
                )
                if result.depth is None or result.depth.shape != image.shape[:2]:
                    raise RuntimeError("Development Mock returned an invalid depth map")

            harness.wait_for_ack(
                lambda: harness.manager.stop(cycle),
                CommandType.STOP,
            )
            unload_ack = harness.wait_for_ack(
                harness.manager.unload_model,
                CommandType.UNLOAD_MODEL,
            )
            harness.pump(max(0, args.settle_ms) / 1000.0)
            samples.append(
                make_sample(
                    cycle=cycle,
                    phase="after_unload",
                    worker_pid=harness.manager.worker_pid,
                    worker_generation=harness.manager.generation,
                    metadata=dict(unload_ack.metadata or {}),
                )
            )

            if restart_every and cycle % restart_every == 0 and cycle < cycles:
                old_process = harness.manager._process
                old_pid = harness.manager.worker_pid
                old_generation = harness.manager.generation
                harness.manager.restart_worker()
                expected_restarts += 1
                harness.wait_until(
                    lambda: harness.manager.is_running
                    and harness.manager.generation == old_generation + 1,
                    timeout_s=5.0,
                    description="replacement worker start",
                )
                new_pid = harness.manager.worker_pid
                if new_pid is not None:
                    observed_pids.append(new_pid)
                if old_process is not None and old_process.is_alive():
                    raise RuntimeError(f"Old worker PID {old_pid} survived restart")

        parent_drift = tail_drift(sample.parent_rss_mib for sample in samples)
        worker_drift = tail_drift(sample.worker_rss_mib for sample in samples)
        checks = {
            "no_crash_signal": harness.crashes == 0,
            "no_command_timeout": not harness.timeouts,
            "restart_count_matches": harness.manager.restart_count == expected_restarts,
            "parent_rss_bounded": parent_drift is None
            or parent_drift <= args.max_parent_rss_drift_mib,
            "worker_rss_bounded": worker_drift is None
            or worker_drift <= args.max_worker_rss_drift_mib,
            "parent_tail_drift_mib": parent_drift,
            "worker_tail_drift_mib": worker_drift,
        }
    finally:
        current_pid = harness.manager.worker_pid
        if current_pid is not None:
            observed_pids.append(current_pid)
        harness.close()

    all_workers_exited = all(wait_for_process_exit(pid, 3.0) for pid in set(observed_pids))
    checks["all_workers_exited"] = all_workers_exited
    checks["manager_ipc_cleared"] = (
        not harness.manager.is_running
        and harness.manager._parent_conn is None
        and harness.manager._frame_queue is None
        and harness.manager._result_queue is None
    )

    output = write_stress_report(
        args.output,
        kind="mock_lifecycle",
        settings={
            "cycles": cycles,
            "frames_per_cycle": frames_per_cycle,
            "restart_every": restart_every,
            "pid": os.getpid(),
        },
        samples=samples,
        checks=checks,
    )
    print(f"Stress report: {output}")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    return 0 if all(value is not False for value in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
