"""Phase 8.7G1 whole-engine cancellation, recovery, and resource cleanup.

This module hardens the *complete reconstruction-engine process boundary*.
It is intentionally dependency-light and owns only lifecycle supervision:

- cooperative cancellation propagation into orchestration providers,
- hard process-group cancellation/timeout/crash cleanup,
- owned workspace cleanup after failed/aborted attempts,
- explicit recovery evidence before a later run is considered safe.

Checkpoint/resume remains Phase 8.7G2. Endurance/degradation campaigns remain
Phase 8.7G3. Full regression/UI-readiness remains Phase 8.7G4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError

ENGINE_HARDENING_SCHEMA_VERSION = 1
ENGINE_HARDENING_PHASE = "8.7G1"


class ReconstructionEngineRunState(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    CRASHED = "crashed"


@dataclass(frozen=True, slots=True)
class ReconstructionEngineProcessSpec:
    """One isolated whole-engine process attempt owned by an explicit root."""

    command: tuple[str, ...]
    owned_root: str
    workspace: str
    timeout_s: float
    cancel_after_s: float | None = None
    terminate_grace_s: float = 1.0
    cwd: str = ""
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        command = tuple(str(value).strip() for value in self.command)
        if not command or any(not value for value in command):
            raise ReconstructionValidationError("Whole-engine command must be non-empty.")
        owned_root = Path(self.owned_root).expanduser()
        workspace = Path(self.workspace).expanduser()
        if not owned_root.is_absolute() or not workspace.is_absolute():
            raise ReconstructionValidationError("owned_root and workspace must be absolute paths.")
        owned_root = owned_root.resolve()
        workspace = workspace.resolve()
        if workspace == owned_root or owned_root not in workspace.parents:
            raise ReconstructionValidationError("workspace must be a strict child of owned_root.")
        timeout = float(self.timeout_s)
        cancel_after = None if self.cancel_after_s is None else float(self.cancel_after_s)
        grace = float(self.terminate_grace_s)
        if timeout <= 0.0:
            raise ReconstructionValidationError("timeout_s must be positive.")
        if cancel_after is not None and cancel_after <= 0.0:
            raise ReconstructionValidationError("cancel_after_s must be positive when provided.")
        if cancel_after is not None and cancel_after >= timeout:
            raise ReconstructionValidationError("cancel_after_s must be earlier than timeout_s.")
        if grace < 0.0:
            raise ReconstructionValidationError("terminate_grace_s must be non-negative.")
        cwd = str(self.cwd).strip()
        if cwd and not Path(cwd).expanduser().is_absolute():
            raise ReconstructionValidationError("cwd must be absolute when provided.")
        environment = tuple((str(key), str(value)) for key, value in self.environment)
        if len({key for key, _ in environment}) != len(environment):
            raise ReconstructionValidationError("environment keys must be unique.")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "owned_root", str(owned_root))
        object.__setattr__(self, "workspace", str(workspace))
        object.__setattr__(self, "timeout_s", timeout)
        object.__setattr__(self, "cancel_after_s", cancel_after)
        object.__setattr__(self, "terminate_grace_s", grace)
        object.__setattr__(self, "cwd", str(Path(cwd).expanduser().resolve()) if cwd else "")
        object.__setattr__(self, "environment", environment)

    @property
    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "owned_root": self.owned_root,
            "workspace": self.workspace,
            "timeout_s": self.timeout_s,
            "cancel_after_s": self.cancel_after_s,
            "terminate_grace_s": self.terminate_grace_s,
            "cwd": self.cwd,
            "environment_keys": sorted(key for key, _ in self.environment),
        }


@dataclass(frozen=True, slots=True)
class ReconstructionEngineProcessOutcome:
    state: ReconstructionEngineRunState
    command: tuple[str, ...]
    workspace: str
    exit_code: int | None
    stdout: str
    elapsed_s: float
    process_group_clean: bool
    workspace_exists_after: bool
    failed_workspace_removed: bool
    termination_signal: str = ""
    reason: str = ""
    schema_version: int = ENGINE_HARDENING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.state, ReconstructionEngineRunState):
            raise TypeError("state must be ReconstructionEngineRunState.")
        workspace = Path(self.workspace)
        if not workspace.is_absolute():
            raise ReconstructionValidationError("outcome workspace must be absolute.")
        if self.elapsed_s < 0.0:
            raise ReconstructionValidationError("elapsed_s must be non-negative.")
        if self.state is ReconstructionEngineRunState.COMPLETED:
            if self.exit_code != 0:
                raise ReconstructionValidationError("Completed engine attempt requires exit_code=0.")
            if not self.process_group_clean:
                raise ReconstructionValidationError("Completed engine attempt cannot leave a process group alive.")
            if not self.workspace_exists_after:
                raise ReconstructionValidationError("Completed engine attempt must preserve its workspace.")
            if self.failed_workspace_removed:
                raise ReconstructionValidationError("Successful workspace cannot be marked as failed/removed.")
        else:
            if not self.process_group_clean:
                raise ReconstructionValidationError("Aborted/failed engine attempt must leave no process group alive.")
            if self.workspace_exists_after:
                raise ReconstructionValidationError("Aborted/failed engine workspace must be removed.")
            if not self.failed_workspace_removed:
                raise ReconstructionValidationError("Aborted/failed engine attempt must record workspace cleanup.")
        if int(self.schema_version) != ENGINE_HARDENING_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported engine hardening schema version.")
        object.__setattr__(self, "command", tuple(str(value) for value in self.command))
        object.__setattr__(self, "workspace", str(workspace.resolve()))
        object.__setattr__(self, "stdout", str(self.stdout))
        object.__setattr__(self, "termination_signal", str(self.termination_signal).strip())
        object.__setattr__(self, "reason", str(self.reason).strip())
        object.__setattr__(self, "schema_version", ENGINE_HARDENING_SCHEMA_VERSION)

    @property
    def safe_for_recovery(self) -> bool:
        return self.process_group_clean and (
            self.state is ReconstructionEngineRunState.COMPLETED or self.failed_workspace_removed
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": ENGINE_HARDENING_PHASE,
            "state": self.state.value,
            "command": list(self.command),
            "workspace": self.workspace,
            "exit_code": self.exit_code,
            "elapsed_s": self.elapsed_s,
            "process_group_clean": self.process_group_clean,
            "workspace_exists_after": self.workspace_exists_after,
            "failed_workspace_removed": self.failed_workspace_removed,
            "termination_signal": self.termination_signal,
            "reason": self.reason,
            "safe_for_recovery": self.safe_for_recovery,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionEngineRecoveryReport:
    scenario: str
    fault: ReconstructionEngineProcessOutcome
    recovery: ReconstructionEngineProcessOutcome

    def __post_init__(self) -> None:
        scenario = str(self.scenario).strip()
        if not scenario:
            raise ReconstructionValidationError("Recovery report scenario must be non-empty.")
        if not isinstance(self.fault, ReconstructionEngineProcessOutcome):
            raise TypeError("fault must be ReconstructionEngineProcessOutcome.")
        if not isinstance(self.recovery, ReconstructionEngineProcessOutcome):
            raise TypeError("recovery must be ReconstructionEngineProcessOutcome.")
        if self.fault.state is ReconstructionEngineRunState.COMPLETED:
            raise ReconstructionValidationError("Recovery report fault attempt cannot be completed.")
        object.__setattr__(self, "scenario", scenario)

    @property
    def passed(self) -> bool:
        return (
            self.fault.safe_for_recovery
            and self.recovery.state is ReconstructionEngineRunState.COMPLETED
            and self.recovery.safe_for_recovery
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ENGINE_HARDENING_SCHEMA_VERSION,
            "phase": ENGINE_HARDENING_PHASE,
            "scenario": self.scenario,
            "pass": self.passed,
            "fault": self.fault.as_dict(),
            "recovery": self.recovery.as_dict(),
            "checkpoint_resume_performed": False,
            "checkpoint_resume_deferred_to": "8.7G2",
            "endurance_hardening_deferred_to": "8.7G3",
            "full_regression_deferred_to": "8.7G4",
            "reconstruction_ui_enabled": False,
        }


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(pgid: int, sig: signal.Signals) -> bool:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_group(pgid: int, *, grace_s: float) -> str:
    if not _process_group_exists(pgid):
        return ""
    sent = ""
    if _signal_process_group(pgid, signal.SIGTERM):
        sent = "SIGTERM"
    deadline = time.monotonic() + max(0.0, float(grace_s))
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return sent
        time.sleep(0.02)
    if _signal_process_group(pgid, signal.SIGKILL):
        sent = "SIGKILL"
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return sent
        time.sleep(0.02)
    return sent


def _prepare_owned_workspace(spec: ReconstructionEngineProcessSpec) -> Path:
    owned_root = Path(spec.owned_root)
    workspace = Path(spec.workspace)
    owned_root.mkdir(parents=True, exist_ok=True)
    if owned_root.is_symlink() or workspace.is_symlink():
        raise ReconstructionValidationError("G1 refuses symlinked owned roots/workspaces.")
    if workspace.exists():
        try:
            next(workspace.iterdir())
        except StopIteration:
            workspace.rmdir()
        else:
            raise ReconstructionValidationError(
                "Whole-engine workspace already exists and is not empty; refusing destructive cleanup."
            )
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _remove_failed_workspace(spec: ReconstructionEngineProcessSpec) -> bool:
    workspace = Path(spec.workspace)
    owned_root = Path(spec.owned_root)
    resolved_workspace = workspace.resolve()
    resolved_root = owned_root.resolve()
    if resolved_workspace == resolved_root or resolved_root not in resolved_workspace.parents:
        raise ReconstructionValidationError("Refusing to remove workspace outside owned_root.")
    if workspace.exists():
        if workspace.is_symlink():
            raise ReconstructionValidationError("Refusing to remove symlinked failed workspace.")
        shutil.rmtree(workspace)
    return not workspace.exists()


def run_supervised_engine_process(spec: ReconstructionEngineProcessSpec) -> ReconstructionEngineProcessOutcome:
    """Run one whole reconstruction-engine attempt in a dedicated process group.

    Any cancellation, timeout, crash, or orphaned descendant is contained at the
    engine process-group boundary. Failed/aborted workspaces are removed only
    after all descendants have been terminated.
    """

    if not isinstance(spec, ReconstructionEngineProcessSpec):
        raise TypeError("spec must be ReconstructionEngineProcessSpec.")
    workspace = _prepare_owned_workspace(spec)
    env = os.environ.copy()
    env.update(spec.environment_dict)
    env["DEPTH3D_RECONSTRUCTION_ENGINE_WORKSPACE"] = str(workspace)
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            spec.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=spec.cwd or None,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        removed = _remove_failed_workspace(spec)
        return ReconstructionEngineProcessOutcome(
            state=ReconstructionEngineRunState.CRASHED,
            command=spec.command,
            workspace=str(workspace),
            exit_code=None,
            stdout="",
            elapsed_s=float(time.monotonic() - started),
            process_group_clean=True,
            workspace_exists_after=workspace.exists(),
            failed_workspace_removed=removed,
            reason=f"launch_failed:{type(exc).__name__}:{exc}",
        )

    pgid = proc.pid
    state: ReconstructionEngineRunState | None = None
    reason = ""
    termination_signal = ""
    while proc.poll() is None:
        elapsed = time.monotonic() - started
        if spec.cancel_after_s is not None and elapsed >= spec.cancel_after_s:
            state = ReconstructionEngineRunState.CANCELLED
            reason = "cancellation_requested"
            termination_signal = _terminate_process_group(pgid, grace_s=spec.terminate_grace_s)
            break
        if elapsed >= spec.timeout_s:
            state = ReconstructionEngineRunState.TIMED_OUT
            reason = "engine_timeout"
            termination_signal = _terminate_process_group(pgid, grace_s=spec.terminate_grace_s)
            break
        time.sleep(0.02)

    try:
        stdout, _ = proc.communicate(timeout=max(1.0, spec.terminate_grace_s + 0.5))
    except subprocess.TimeoutExpired:
        cleanup_signal = _terminate_process_group(pgid, grace_s=0.0)
        if cleanup_signal:
            termination_signal = cleanup_signal
        stdout, _ = proc.communicate(timeout=2.0)

    if state is None:
        if proc.returncode == 0:
            state = ReconstructionEngineRunState.COMPLETED
        else:
            state = ReconstructionEngineRunState.CRASHED
            reason = f"engine_exit_{proc.returncode}"

    # A crashing or even nominally exiting parent may leave descendants in its
    # process group. G1 treats that as an engine-scope resource leak and reaps it.
    if _process_group_exists(pgid):
        cleanup_signal = _terminate_process_group(pgid, grace_s=spec.terminate_grace_s)
        if cleanup_signal:
            termination_signal = cleanup_signal
        if state is ReconstructionEngineRunState.COMPLETED:
            state = ReconstructionEngineRunState.CRASHED
            reason = "orphan_process_group_after_parent_exit"

    process_group_clean = not _process_group_exists(pgid)
    failed_workspace_removed = False
    if state is not ReconstructionEngineRunState.COMPLETED:
        failed_workspace_removed = _remove_failed_workspace(spec)

    return ReconstructionEngineProcessOutcome(
        state=state,
        command=spec.command,
        workspace=str(workspace),
        exit_code=proc.returncode,
        stdout=stdout or "",
        elapsed_s=float(time.monotonic() - started),
        process_group_clean=process_group_clean,
        workspace_exists_after=workspace.exists(),
        failed_workspace_removed=failed_workspace_removed,
        termination_signal=termination_signal,
        reason=reason,
    )


def recovery_report(
    *,
    scenario: str,
    fault: ReconstructionEngineProcessOutcome,
    recovery: ReconstructionEngineProcessOutcome,
) -> ReconstructionEngineRecoveryReport:
    return ReconstructionEngineRecoveryReport(scenario=scenario, fault=fault, recovery=recovery)
