#!/usr/bin/env python3
"""Repeatedly crash, detect, replace and reuse the spawned mock worker."""

from __future__ import annotations

import argparse
import os
import signal
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
    wait_for_process_exit,
    write_stress_report,
)
from scripts.stress_support import WorkerHarness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stress/worker_recovery_stress.json"),
    )
    args = parser.parse_args()
    cycles = max(1, args.cycles)
    harness = WorkerHarness(command_timeout_ms=3_000)
    samples = []
    old_pids: list[int] = []

    try:
        harness.manager.start_worker()
        harness.wait_until(
            lambda: harness.manager.is_running,
            timeout_s=5.0,
            description="initial worker",
        )

        for cycle in range(1, cycles + 1):
            harness.wait_for_ack(
                lambda: harness.manager.load_model(
                    MOCK_MODEL_ID,
                    {"device": "cpu", "precision": "fp32", "latency_ms": 0.0},
                ),
                CommandType.LOAD_MODEL,
            )
            harness.wait_for_ack(harness.manager.resume, CommandType.RESUME)
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            harness.wait_for_result(
                lambda: harness.manager.submit_frame(cycle, 0, image),
                session_id=cycle,
                frame_id=0,
            )

            old_process = harness.manager._process
            old_pid = harness.manager.worker_pid
            old_generation = harness.manager.generation
            if old_pid is None or old_process is None:
                raise RuntimeError("Worker has no process identity")
            old_pids.append(old_pid)
            crashes_before = harness.crashes
            os.kill(old_pid, signal.SIGKILL)
            harness.wait_until(
                lambda: harness.crashes == crashes_before + 1,
                timeout_s=5.0,
                description="single crash signal",
            )
            harness.pump(0.05)
            if harness.crashes != crashes_before + 1:
                raise RuntimeError("One worker death emitted duplicate crash signals")

            harness.manager.restart_worker()
            harness.wait_until(
                lambda: harness.manager.is_running
                and harness.manager.generation == old_generation + 1,
                timeout_s=5.0,
                description="fresh worker after crash",
            )
            if old_process.is_alive():
                raise RuntimeError(f"Crashed worker PID {old_pid} was not reaped")
            samples.append(
                make_sample(
                    cycle=cycle,
                    phase="after_recovery",
                    worker_pid=harness.manager.worker_pid,
                    worker_generation=harness.manager.generation,
                )
            )

        checks = {
            "one_crash_signal_per_cycle": harness.crashes == cycles,
            "restart_count_matches": harness.manager.restart_count == cycles,
            "no_command_timeout": not harness.timeouts,
            "all_crashed_workers_reaped": all(
                wait_for_process_exit(pid, 0.1) for pid in old_pids
            ),
        }
    finally:
        final_pid = harness.manager.worker_pid
        harness.close()

    checks["final_worker_exited"] = wait_for_process_exit(final_pid, 3.0)
    output = write_stress_report(
        args.output,
        kind="worker_recovery",
        settings={"cycles": cycles, "pid": os.getpid()},
        samples=samples,
        checks=checks,
    )
    print(f"Stress report: {output}")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    return 0 if all(value is not False for value in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
