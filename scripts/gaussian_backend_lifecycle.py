#!/usr/bin/env python3
"""Process-group supervision for Phase 8.7D3 Gaussian backend workers.

This module owns only the isolated backend-process boundary. Whole reconstruction
session cancellation/checkpointing remains Phase 8.7G work.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SupervisedProcessOutcome:
    command: tuple[str, ...]
    exit_code: int | None
    stdout: str
    elapsed_s: float
    cancelled: bool = False
    timed_out: bool = False
    process_group_clean: bool = False
    termination_signal: str = ""


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(proc: subprocess.Popen[str], *, grace_s: float = 1.0) -> str:
    if proc.poll() is not None:
        return ""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return ""
    deadline = time.monotonic() + max(0.0, float(grace_s))
    while time.monotonic() < deadline:
        if proc.poll() is not None and not _process_group_exists(proc.pid):
            return "SIGTERM"
        time.sleep(0.02)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return "SIGTERM"
    return "SIGKILL"


def run_supervised_process(
    command: Sequence[str],
    *,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    cancel_after_s: float | None = None,
    terminate_grace_s: float = 1.0,
) -> SupervisedProcessOutcome:
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    if cancel_after_s is not None and cancel_after_s <= 0.0:
        raise ValueError("cancel_after_s must be positive when provided")
    args = tuple(str(value) for value in command)
    started = time.monotonic()
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env) if env is not None else None,
        cwd=str(cwd) if cwd is not None else None,
        start_new_session=True,
    )
    cancelled = False
    timed_out = False
    termination_signal = ""
    while proc.poll() is None:
        elapsed = time.monotonic() - started
        if cancel_after_s is not None and elapsed >= cancel_after_s:
            cancelled = True
            termination_signal = _terminate_process_group(proc, grace_s=terminate_grace_s)
            break
        if elapsed >= timeout_s:
            timed_out = True
            termination_signal = _terminate_process_group(proc, grace_s=terminate_grace_s)
            break
        time.sleep(0.02)
    try:
        stdout, _ = proc.communicate(timeout=max(1.0, terminate_grace_s + 0.5))
    except subprocess.TimeoutExpired:
        termination_signal = _terminate_process_group(proc, grace_s=0.0) or termination_signal
        stdout, _ = proc.communicate(timeout=2.0)
    # Allow the kernel a brief moment to reap descendants after group termination.
    deadline = time.monotonic() + 1.0
    while _process_group_exists(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    return SupervisedProcessOutcome(
        command=args,
        exit_code=proc.returncode,
        stdout=stdout or "",
        elapsed_s=float(time.monotonic() - started),
        cancelled=cancelled,
        timed_out=timed_out,
        process_group_clean=not _process_group_exists(proc.pid),
        termination_signal=termination_signal,
    )
