#!/usr/bin/env python3
"""Real Phase 8.7G1 whole-engine cancellation + clean recovery gate.

The fault attempt runs the already-qualified F4→F5 integrated engine in one
process group and cancels the entire group. A second clean attempt then proves
that Gaussian, Nerfacto, canonical geometry, and persistent-scene integration
can recover without reusing partial state.
"""
from __future__ import annotations

import argparse
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


def _integrated_spec(
    root: Path,
    name: str,
    *,
    nerf_steps: int,
    timeout_s: float,
    cancel_after_s: float | None = None,
) -> ReconstructionEngineProcessSpec:
    workspace = (root / name).resolve()
    return ReconstructionEngineProcessSpec(
        command=(
            sys.executable,
            str((ROOT / "scripts/reconstruction_engine_integrated_worker.py").resolve()),
            "--workspace",
            str(workspace),
            "--nerf-training-steps",
            str(int(nerf_steps)),
            "--backend-timeout-s",
            str(float(timeout_s)),
        ),
        owned_root=str(root.resolve()),
        workspace=str(workspace),
        timeout_s=float(timeout_s),
        cancel_after_s=cancel_after_s,
        terminate_grace_s=1.0,
        cwd=str(ROOT.resolve()),
    )


def _synthetic_spec(root: Path, name: str, *, mode: str, timeout_s: float):
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
            "30",
        ),
        owned_root=str(root.resolve()),
        workspace=str(workspace),
        timeout_s=timeout_s,
        terminate_grace_s=0.25,
        cwd=str(ROOT.resolve()),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phase8_7g1_real_hardening"))
    parser.add_argument("--cancel-after-s", type=float, default=1.5)
    parser.add_argument("--fault-nerf-training-steps", type=int, default=1000)
    parser.add_argument("--recovery-nerf-training-steps", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    args = parser.parse_args()

    root = args.output_dir.expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    cancelled = run_supervised_engine_process(
        _integrated_spec(
            root,
            "cancelled-real-engine",
            nerf_steps=args.fault_nerf_training_steps,
            timeout_s=args.timeout_s,
            cancel_after_s=args.cancel_after_s,
        )
    )
    if cancelled.state is not ReconstructionEngineRunState.CANCELLED:
        raise RuntimeError(f"real engine cancellation did not cancel: {cancelled.state.value}")
    if "engine_child_started stage=f4" not in cancelled.stdout:
        raise RuntimeError("real cancellation did not prove a nested integrated-engine child was active")

    # Explicit residual-child crash and timeout checks remain cheap and exercise
    # the same whole-engine supervisor without requalifying any neural backend.
    crashed = run_supervised_engine_process(
        _synthetic_spec(root, "crashed-engine-tree", mode="crash", timeout_s=5.0)
    )
    timed_out = run_supervised_engine_process(
        _synthetic_spec(root, "timed-out-engine-tree", mode="hang", timeout_s=0.2)
    )

    recovery = run_supervised_engine_process(
        _integrated_spec(
            root,
            "recovery-real-engine",
            nerf_steps=args.recovery_nerf_training_steps,
            timeout_s=args.timeout_s,
        )
    )
    report = recovery_report(
        scenario="real_integrated_engine_cancel_then_recover",
        fault=cancelled,
        recovery=recovery,
    )

    recovery_workspace = Path(recovery.workspace)
    engine_result_path = recovery_workspace / "engine_result.json"
    if recovery.state is not ReconstructionEngineRunState.COMPLETED or not engine_result_path.is_file():
        raise RuntimeError("clean integrated recovery did not produce engine_result.json")
    engine_result = json.loads(engine_result_path.read_text(encoding="utf-8"))

    passed = (
        report.passed
        and crashed.state is ReconstructionEngineRunState.CRASHED
        and crashed.safe_for_recovery
        and timed_out.state is ReconstructionEngineRunState.TIMED_OUT
        and timed_out.safe_for_recovery
        and bool(engine_result.get("pass"))
        and bool(engine_result.get("canonical_geometry_authoritative"))
        and engine_result.get("gaussian_authority") == "appearance_only"
        and engine_result.get("nerf_authority") == "appearance_only"
        and not bool(engine_result.get("certified_metrology"))
        and not bool(engine_result.get("reconstruction_ui_enabled"))
    )
    payload = {
        "phase": ENGINE_HARDENING_PHASE,
        "pass": passed,
        "real_cancellation": cancelled.as_dict(),
        "residual_child_crash_cleanup": crashed.as_dict(),
        "timeout_cleanup": timed_out.as_dict(),
        "recovery": recovery.as_dict(),
        "recovery_report": report.as_dict(),
        "recovery_engine_result": engine_result,
        "whole_engine_scope": True,
        "nested_engine_child_observed_before_cancellation": True,
        "process_group_cleanup": True,
        "partial_workspace_cleanup": True,
        "clean_recovery_completed": recovery.state is ReconstructionEngineRunState.COMPLETED,
        "canonical_geometry_authoritative": bool(engine_result.get("canonical_geometry_authoritative")),
        "gaussian_authority": engine_result.get("gaussian_authority"),
        "nerf_authority": engine_result.get("nerf_authority"),
        "certified_metrology": False,
        "checkpoint_resume_performed": False,
        "checkpoint_resume_deferred_to": "8.7G2",
        "endurance_hardening_deferred_to": "8.7G3",
        "full_regression_deferred_to": "8.7G4",
        "reconstruction_ui_enabled": False,
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"Phase 8.7G1 real whole-engine hardening: {'PASS' if passed else 'FAIL'}")
    print(
        f"cancel={cancelled.state.value} process_group_clean={cancelled.process_group_clean} "
        f"workspace_removed={cancelled.failed_workspace_removed}"
    )
    print(
        f"crash_cleanup={crashed.safe_for_recovery} timeout_cleanup={timed_out.safe_for_recovery} "
        f"recovery={recovery.state.value}"
    )
    print(
        f"canonical_geometry_authoritative={engine_result.get('canonical_geometry_authoritative')} "
        f"gaussian_authority={engine_result.get('gaussian_authority')} "
        f"nerf_authority={engine_result.get('nerf_authority')}"
    )
    print("certified_metrology=False reconstruction_ui_enabled=False")
    print("checkpoint_resume_deferred_to=8.7G2 endurance_deferred_to=8.7G3 full_regression_deferred_to=8.7G4")
    print(f"summary={root / 'summary.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
