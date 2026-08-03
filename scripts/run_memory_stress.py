#!/usr/bin/env python3
"""Measure repeated model load/infer/unload memory stability.

Development Mock is dependency-free.  MoGe-2 mode requires the model package,
a cached/downloadable checkpoint and a CUDA-ready environment.
"""

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

from core.hardware.gpu_info import detect_hardware  # noqa: E402
from core.hardware.profiles import ComputeProfile, resolve_profile  # noqa: E402
from core.models.catalog import MOGE2_MODEL_ID, MOCK_MODEL_ID  # noqa: E402
from core.runtime.messages import CommandType  # noqa: E402
from core.runtime.stress import (  # noqa: E402
    make_sample,
    tail_drift,
    wait_for_process_exit,
    write_stress_report,
)
from scripts.stress_support import WorkerHarness  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mock", "moge2"), default="mock")
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in ComputeProfile),
        default=ComputeProfile.AUTO.value,
    )
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--frames-per-cycle", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--settle-ms", type=int, default=500)
    parser.add_argument("--max-worker-rss-drift-mib", type=float, default=256.0)
    parser.add_argument("--max-unloaded-reserved-gib", type=float, default=0.25)
    parser.add_argument("--max-device-vram-drift-gib", type=float, default=0.5)
    parser.add_argument(
        "--strict-device-vram",
        action="store_true",
        help="Fail when device-wide VRAM drifts; other GPU processes can make this noisy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stress/model_memory_stress.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cycles = max(1, args.cycles)
    frames = max(1, args.frames_per_cycle)
    width = max(32, args.width)
    height = max(32, args.height)

    if args.model == "moge2":
        hardware = detect_hardware()
        if not hardware.cuda_ready:
            raise SystemExit(f"MoGe-2 stress requires CUDA: {hardware.warning or hardware.details}")
        resolved = resolve_profile(ComputeProfile(args.profile), hardware)
        model_id = MOGE2_MODEL_ID
        config = resolved.as_model_config()
        model_description = resolved.spec.model_label
    else:
        hardware = None
        model_id = MOCK_MODEL_ID
        config = {"device": "cpu", "precision": "fp32", "latency_ms": 0.0}
        model_description = "Development Mock"

    harness = WorkerHarness(command_timeout_ms=10_000)
    samples = []
    initial_pid = None
    load_times = []
    unload_times = []
    try:
        harness.manager.start_worker()
        harness.wait_until(
            lambda: harness.manager.is_running,
            timeout_s=5.0,
            description="worker start",
        )
        initial_pid = harness.manager.worker_pid
        image = np.zeros((height, width, 3), dtype=np.uint8)

        for cycle in range(1, cycles + 1):
            load_ack = harness.wait_for_ack(
                lambda: harness.manager.load_model(model_id, config),
                CommandType.LOAD_MODEL,
                timeout_s=240.0,
            )
            if load_ack.load_time_ms is not None:
                load_times.append(float(load_ack.load_time_ms))
            harness.wait_for_ack(harness.manager.resume, CommandType.RESUME)
            for frame_id in range(frames):
                harness.wait_for_result(
                    lambda fid=frame_id: harness.manager.submit_frame(
                        cycle,
                        fid,
                        image,
                        point_budget=int(config.get("point_budget", 250_000)),
                        submitted_at=time.monotonic(),
                    ),
                    session_id=cycle,
                    frame_id=frame_id,
                    timeout_s=120.0,
                )
            harness.wait_for_ack(
                lambda: harness.manager.stop(cycle),
                CommandType.STOP,
            )
            unload_ack = harness.wait_for_ack(
                harness.manager.unload_model,
                CommandType.UNLOAD_MODEL,
                timeout_s=60.0,
            )
            if unload_ack.unload_time_ms is not None:
                unload_times.append(float(unload_ack.unload_time_ms))
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

        worker_drift = tail_drift(sample.worker_rss_mib for sample in samples)
        device_drift = tail_drift(sample.vram_device_used_gib for sample in samples)
        reserved = [
            sample.vram_reserved_gib
            for sample in samples
            if sample.vram_reserved_gib is not None
        ]
        checks = {
            "same_worker_process": harness.manager.worker_pid == initial_pid,
            "no_crash_signal": harness.crashes == 0,
            "no_command_timeout": not harness.timeouts,
            "worker_rss_bounded": worker_drift is None
            or worker_drift <= args.max_worker_rss_drift_mib,
            "worker_tail_drift_mib": worker_drift,
            "unloaded_allocator_reserved_bounded": not reserved
            or max(reserved) <= args.max_unloaded_reserved_gib,
            "max_unloaded_reserved_gib": max(reserved) if reserved else None,
            "device_vram_tail_drift_gib": device_drift,
            "device_vram_bounded": (
                device_drift is None
                or device_drift <= args.max_device_vram_drift_gib
                or not args.strict_device_vram
            ),
            "mean_load_time_ms": sum(load_times) / len(load_times) if load_times else None,
            "mean_unload_time_ms": (
                sum(unload_times) / len(unload_times) if unload_times else None
            ),
        }
    finally:
        final_pid = harness.manager.worker_pid
        harness.close(timeout_s=5.0)

    checks["worker_exited_after_shutdown"] = wait_for_process_exit(final_pid, 5.0)
    output = write_stress_report(
        args.output,
        kind="model_memory",
        settings={
            "model": args.model,
            "model_description": model_description,
            "profile": args.profile,
            "cycles": cycles,
            "frames_per_cycle": frames,
            "image_size": [width, height],
            "strict_device_vram": args.strict_device_vram,
            "pid": os.getpid(),
            "hardware": hardware.short_label if hardware is not None else "CPU",
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
