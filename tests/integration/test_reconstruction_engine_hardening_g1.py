from __future__ import annotations

from pathlib import Path
import sys

from core.reconstruction import (
    ReconstructionEngineProcessSpec,
    ReconstructionEngineRunState,
    recovery_report,
    run_supervised_engine_process,
)

ROOT = Path(__file__).resolve().parents[2]
WORKER = (ROOT / "scripts/reconstruction_engine_fault_worker.py").resolve()


def _spec(tmp_path: Path, name: str, *, mode: str, cancel=None, timeout=3.0):
    root = (tmp_path / "engine-root").resolve()
    workspace = (root / name).resolve()
    return ReconstructionEngineProcessSpec(
        command=(sys.executable, str(WORKER), "--workspace", str(workspace), "--mode", mode, "--child-sleep-s", "8"),
        owned_root=str(root),
        workspace=str(workspace),
        timeout_s=timeout,
        cancel_after_s=cancel,
        terminate_grace_s=0.1,
        cwd=str(ROOT),
    )


def test_cancel_cleanup_then_recovery_is_engine_scope_safe(tmp_path: Path) -> None:
    fault = run_supervised_engine_process(_spec(tmp_path, "cancel", mode="hang", cancel=0.12))
    recovery = run_supervised_engine_process(_spec(tmp_path, "recovery", mode="complete"))
    report = recovery_report(scenario="integration-cancel", fault=fault, recovery=recovery)
    assert fault.state is ReconstructionEngineRunState.CANCELLED
    assert report.passed is True
    assert not Path(fault.workspace).exists()
    assert (Path(recovery.workspace) / "engine_result.json").is_file()


def test_crash_cleanup_then_recovery_is_engine_scope_safe(tmp_path: Path) -> None:
    fault = run_supervised_engine_process(_spec(tmp_path, "crash", mode="crash"))
    recovery = run_supervised_engine_process(_spec(tmp_path, "recovery2", mode="complete"))
    report = recovery_report(scenario="integration-crash", fault=fault, recovery=recovery)
    assert fault.state is ReconstructionEngineRunState.CRASHED
    assert fault.process_group_clean is True
    assert report.passed is True
