#!/usr/bin/env python3
"""Process-group supervision for Phase 8.7E4 NeRF backend workers.

This is backend-process hardening only.  Whole reconstruction-session
cancellation/recovery remains Phase 8.7G.
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


def _signal_group(pgid: int, sig: signal.Signals) -> bool:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    return True


def _ensure_process_group_dead(pgid: int, *, grace_s: float) -> str:
    if not _process_group_exists(pgid):
        return ""
    sent = ""
    if _signal_group(pgid, signal.SIGTERM):
        sent = "SIGTERM"
    deadline = time.monotonic() + max(0.0, float(grace_s))
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return sent
        time.sleep(0.02)
    if _signal_group(pgid, signal.SIGKILL):
        sent = "SIGKILL"
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            break
        time.sleep(0.02)
    return sent


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
    pgid = proc.pid
    cancelled = False
    timed_out = False
    termination_signal = ""
    while proc.poll() is None:
        elapsed = time.monotonic() - started
        if cancel_after_s is not None and elapsed >= cancel_after_s:
            cancelled = True
            termination_signal = _ensure_process_group_dead(pgid, grace_s=terminate_grace_s)
            break
        if elapsed >= timeout_s:
            timed_out = True
            termination_signal = _ensure_process_group_dead(pgid, grace_s=terminate_grace_s)
            break
        time.sleep(0.02)
    try:
        stdout, _ = proc.communicate(timeout=max(1.0, terminate_grace_s + 0.5))
    except subprocess.TimeoutExpired:
        termination_signal = _ensure_process_group_dead(pgid, grace_s=0.0) or termination_signal
        stdout, _ = proc.communicate(timeout=2.0)

    # A crashing parent can leave children alive in its session.  E4 explicitly
    # cleans that residual process group as part of the backend boundary.
    if _process_group_exists(pgid):
        cleanup_signal = _ensure_process_group_dead(pgid, grace_s=terminate_grace_s)
        if cleanup_signal:
            termination_signal = cleanup_signal

    return SupervisedProcessOutcome(
        command=args,
        exit_code=proc.returncode,
        stdout=stdout or "",
        elapsed_s=float(time.monotonic() - started),
        cancelled=cancelled,
        timed_out=timed_out,
        process_group_clean=not _process_group_exists(pgid),
        termination_signal=termination_signal,
    )
