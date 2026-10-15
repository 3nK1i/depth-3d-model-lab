#!/usr/bin/env python3
"""Dependency-light GPU performance helpers for Phase 8.7D3 Step 3.

These helpers intentionally use ``nvidia-smi`` rather than importing NVML into
production code. They run only inside isolated Gaussian benchmark workers and
can observe the worker process plus child processes spawned by Graphdeco.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable

MIB = 1024.0 * 1024.0


@dataclass(frozen=True, slots=True)
class GpuHardwareProbe:
    gpu_index: int
    name: str
    uuid: str
    total_vram_mib: float
    driver_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "gpu_index": self.gpu_index,
            "gpu_name": self.name,
            "gpu_uuid": self.uuid,
            "total_vram_mib": self.total_vram_mib,
            "driver_version": self.driver_version,
        }


def _run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"command failed: {' '.join(command)}: {detail}")
    return completed.stdout


def probe_gpu_hardware(gpu_index: int = 0) -> GpuHardwareProbe:
    output = _run_text(
        [
            "nvidia-smi",
            f"--id={int(gpu_index)}",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    if not output:
        raise RuntimeError("nvidia-smi returned no GPU hardware row")
    row = output.splitlines()[0]
    parts = [part.strip() for part in row.split(",")]
    if len(parts) != 5:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {row!r}")
    return GpuHardwareProbe(
        gpu_index=int(parts[0]),
        name=parts[1],
        uuid=parts[2],
        total_vram_mib=float(parts[3]),
        driver_version=parts[4],
    )


def _descendant_pids(root_pid: int) -> set[int]:
    """Return root PID plus recursively discovered Linux descendants."""

    root_pid = int(root_pid)
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in descendants:
                continue
            try:
                stat_text = (entry / "stat").read_text(encoding="utf-8")
                # /proc/<pid>/stat wraps the command name in parentheses and the
                # command may itself contain spaces. Fields after the closing
                # parenthesis begin with state then PPID.
                closing = stat_text.rfind(")")
                if closing < 0:
                    continue
                remainder = stat_text[closing + 1:].strip().split()
                ppid = int(remainder[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _query_compute_process_memory_mib(gpu_uuid: str) -> dict[int, float]:
    output = _run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    values: dict[int, float] = {}
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or parts[1] != gpu_uuid:
            continue
        try:
            values[int(parts[0])] = float(parts[2])
        except ValueError:
            continue
    return values


class ProcessTreeGpuSampler:
    """Sample aggregate VRAM for one process tree on one NVIDIA GPU."""

    def __init__(self, *, root_pid: int, gpu_uuid: str, interval_s: float = 0.10) -> None:
        if interval_s <= 0.0:
            raise ValueError("interval_s must be positive")
        self.root_pid = int(root_pid)
        self.gpu_uuid = str(gpu_uuid)
        self.interval_s = float(interval_s)
        self.peak_vram_mib = 0.0
        self.sample_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def _sample_once(self) -> None:
        pids = _descendant_pids(self.root_pid)
        usage = _query_compute_process_memory_mib(self.gpu_uuid)
        total = sum(value for pid, value in usage.items() if pid in pids)
        self.peak_vram_mib = max(self.peak_vram_mib, float(total))
        self.sample_count += 1

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._sample_once()
            except BaseException as exc:  # pragma: no cover - surfaced at stop()
                self._error = exc
                self._stop.set()
                return

    def start(self) -> "ProcessTreeGpuSampler":
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="depth3d-gpu-vram-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4.0))
        self._sample_once()
        if self._error is not None:
            raise RuntimeError(f"GPU sampler failed: {self._error}") from self._error
        if self.sample_count < 1:
            raise RuntimeError("GPU sampler produced no samples")
        return float(self.peak_vram_mib)

    def __enter__(self) -> "ProcessTreeGpuSampler":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def timed_gpu_phase(
    callback: Callable[[], None],
    *,
    gpu_uuid: str,
    root_pid: int | None = None,
    interval_s: float = 0.10,
) -> tuple[float, float, int]:
    """Run a phase and return ``(duration_s, peak_vram_mib, sample_count)``."""

    sampler = ProcessTreeGpuSampler(
        root_pid=os.getpid() if root_pid is None else int(root_pid),
        gpu_uuid=gpu_uuid,
        interval_s=interval_s,
    ).start()
    started = time.monotonic()
    try:
        callback()
    finally:
        duration_s = float(time.monotonic() - started)
        peak_vram_mib = sampler.stop()
    return duration_s, peak_vram_mib, sampler.sample_count
