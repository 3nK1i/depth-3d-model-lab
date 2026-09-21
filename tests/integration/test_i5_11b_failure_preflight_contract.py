from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.reconstruction.i5_failure_robustness_preflight import (
    I511B_REPOSITORY_ANCHORS,
    verify_i5_11b_repository_anchors,
)

ROOT = Path(__file__).resolve().parents[2]


def test_repository_anchors_verify_against_current_tree() -> None:
    verified = verify_i5_11b_repository_anchors(ROOT)
    assert set(verified) == set(I511B_REPOSITORY_ANCHORS)
    assert all(item["verified"] for item in verified.values())


def test_bounded_fault_worker_help_is_non_destructive() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/i5_11_bounded_fault_worker.py"), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    assert result.returncode == 0
    assert "cuda_oom" in result.stdout
    assert "disk_enospc" in result.stdout
    assert "crash_orphan_descendant" in result.stdout


def test_preflight_module_imports_without_optional_ui_or_ml_runtime() -> None:
    code = "import core.reconstruction.i5_failure_robustness_preflight as m; print(m.I511B_PHASE)"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "I5.11B"
