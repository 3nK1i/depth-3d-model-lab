#!/usr/bin/env python3
"""GPU performance helpers for isolated Phase 8.7E4 NeRF workers.

The sampler observes the worker process tree with ``nvidia-smi``.  It avoids
adding NVML or other profiling packages to either the production environment or
the isolated Nerfstudio environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable


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
    parts = [part.strip() for part in output.splitlines()[0].split(",")]
    if len(parts) != 5:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {output!r}")
    return GpuHardwareProbe(
        gpu_index=int(parts[0]),
        name=parts[1],
        uuid=parts[2],
        total_vram_mib=float(parts[3]),
        driver_version=parts[4],
    )


def _descendant_pids(root_pid: int) -> set[int]:
    descendants = {int(root_pid)}
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
                closing = stat_text.rfind(")")
                if closing < 0:
                    continue
                remainder = stat_text[closing + 1 :].strip().split()
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
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3 or parts[1] != gpu_uuid:
            continue
        try:
            values[int(parts[0])] = float(parts[2])
        except ValueError:
            continue
    return values


class ProcessTreeGpuSampler:
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
            except BaseException as exc:  # pragma: no cover
                self._error = exc
                self._stop.set()
                return

    def start(self) -> "ProcessTreeGpuSampler":
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="eia-nerf-vram-sampler", daemon=True)
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


def timed_gpu_phase(
    callback: Callable[[], None],
    *,
    gpu_uuid: str,
    root_pid: int | None = None,
    interval_s: float = 0.10,
) -> tuple[float, float, int]:
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
