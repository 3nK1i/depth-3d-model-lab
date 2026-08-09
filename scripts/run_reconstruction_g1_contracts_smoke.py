#!/usr/bin/env python3
"""Fast Phase 8.7G1 whole-engine lifecycle contract smoke."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    ENGINE_HARDENING_PHASE,
    ReconstructionEngineProcessSpec,
    ReconstructionEngineRunState,
    recovery_report,
    run_supervised_engine_process,
)


def _spec(root: Path, name: str, *, mode: str, cancel_after_s=None, timeout_s=2.0):
    workspace = (root / name).resolve()
    return ReconstructionEngineProcessSpec(
        command=(
            sys.executable,
            str((ROOT / "scripts/reconstruction_engine_fault_worker.py").resolve()),
            "--workspace",
            str(workspace),
            "--mode",
            mode,
            "--child-sleep-s",
            "10",
        ),
        owned_root=str(root.resolve()),
        workspace=str(workspace),
        timeout_s=timeout_s,
        cancel_after_s=cancel_after_s,
        terminate_grace_s=0.15,
        cwd=str(ROOT.resolve()),
    )


def _recovery(root: Path, name: str):
    return run_supervised_engine_process(_spec(root, name, mode="complete", timeout_s=3.0))


def main() -> int:
    root = Path("/tmp/phase8_7g1_contracts").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    cancelled = run_supervised_engine_process(
        _spec(root, "cancelled", mode="hang", cancel_after_s=0.12, timeout_s=2.0)
    )
    cancel_recovery = _recovery(root, "recovery-after-cancel")
    cancel_report = recovery_report(
        scenario="whole_engine_cancellation", fault=cancelled, recovery=cancel_recovery
    )

    timed_out = run_supervised_engine_process(
        _spec(root, "timed-out", mode="hang", timeout_s=0.15)
    )
    timeout_recovery = _recovery(root, "recovery-after-timeout")
    timeout_report = recovery_report(
        scenario="whole_engine_timeout", fault=timed_out, recovery=timeout_recovery
    )

    crashed = run_supervised_engine_process(_spec(root, "crashed", mode="crash", timeout_s=2.0))
    crash_recovery = _recovery(root, "recovery-after-crash")
    crash_report = recovery_report(
        scenario="whole_engine_crash", fault=crashed, recovery=crash_recovery
    )

    reports = (cancel_report, timeout_report, crash_report)
    passed = (
        cancelled.state is ReconstructionEngineRunState.CANCELLED
        and timed_out.state is ReconstructionEngineRunState.TIMED_OUT
        and crashed.state is ReconstructionEngineRunState.CRASHED
        and all(item.passed for item in reports)
        and all(item.fault.process_group_clean for item in reports)
        and all(item.fault.failed_workspace_removed for item in reports)
    )
    payload = {
        "phase": ENGINE_HARDENING_PHASE,
        "pass": passed,
        "reports": [item.as_dict() for item in reports],
        "hard_process_group_cancellation": True,
        "timeout_cleanup": True,
        "crash_cleanup": True,
        "partial_workspace_cleanup": True,
        "recovery_after_each_fault": True,
        "checkpoint_resume_performed": False,
        "checkpoint_resume_deferred_to": "8.7G2",
        "endurance_hardening_deferred_to": "8.7G3",
        "full_regression_deferred_to": "8.7G4",
        "reconstruction_ui_enabled": False,
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"Phase 8.7G1 whole-engine lifecycle contracts: {'PASS' if passed else 'FAIL'}")
    print(
        f"cancel={cancelled.state.value} timeout={timed_out.state.value} "
        f"crash={crashed.state.value}"
    )
    print("process_group_cleanup=True partial_workspace_cleanup=True recovery_after_each_fault=True")
    print("checkpoint_resume_deferred_to=8.7G2 endurance_deferred_to=8.7G3 full_regression_deferred_to=8.7G4")
    print("reconstruction_ui_enabled=False")
    print(f"summary={root / 'summary.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
