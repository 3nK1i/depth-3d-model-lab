from __future__ import annotations

from pathlib import Path
import sys

from core.reconstruction import ReconstructionEngineProcessSpec, ReconstructionEngineRunState, run_supervised_engine_process

ROOT = Path(__file__).resolve().parents[2]
WORKER = (ROOT / "scripts/reconstruction_engine_fault_worker.py").resolve()


def _spec(tmp_path: Path, name: str, *, mode: str, cancel=None, timeout=2.0):
    root = (tmp_path / "root").resolve(); workspace = (root / name).resolve()
    return ReconstructionEngineProcessSpec(
        command=(sys.executable, str(WORKER), "--workspace", str(workspace), "--mode", mode, "--child-sleep-s", "6"),
        owned_root=str(root), workspace=str(workspace), timeout_s=timeout, cancel_after_s=cancel,
        terminate_grace_s=0.08, cwd=str(ROOT),
    )


def test_repeated_whole_engine_cancellation_leaves_no_partial_workspaces(tmp_path: Path) -> None:
    for index in range(5):
        outcome = run_supervised_engine_process(_spec(tmp_path, f"cancel-{index}", mode="hang", cancel=0.08))
        assert outcome.state is ReconstructionEngineRunState.CANCELLED
        assert outcome.process_group_clean and outcome.failed_workspace_removed
        assert not Path(outcome.workspace).exists()


def test_repeated_crash_cleanup_keeps_supervisor_reusable(tmp_path: Path) -> None:
    for index in range(5):
        outcome = run_supervised_engine_process(_spec(tmp_path, f"crash-{index}", mode="crash"))
        assert outcome.state is ReconstructionEngineRunState.CRASHED
        assert outcome.safe_for_recovery
    final = run_supervised_engine_process(_spec(tmp_path, "final", mode="complete"))
    assert final.state is ReconstructionEngineRunState.COMPLETED
    assert final.safe_for_recovery
