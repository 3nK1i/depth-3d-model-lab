from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.reconstruction.i5_failure_robustness_campaign import (
    I511C_REPOSITORY_ANCHORS,
    verify_i5_11c_repository_anchors,
)

ROOT = Path(__file__).resolve().parents[2]


def test_i511c_repository_anchors_verify_current_tree() -> None:
    verified = verify_i5_11c_repository_anchors(ROOT)
    assert set(verified) == set(I511C_REPOSITORY_ANCHORS)
    assert all(item["verified"] for item in verified.values())


def test_campaign_module_imports_without_ui_or_ml_runtime() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import core.reconstruction.i5_failure_robustness_campaign as m; print(m.I511C_PHASE)"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "I5.11C"


def test_campaign_runner_help_is_non_destructive() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_i5_11c_failure_recovery_campaign.py"), "--help"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10,
    )
    assert result.returncode == 0
    assert "--i5-11b-preflight" in result.stdout
    assert "--post-i5-11b-delta-checkpoint" in result.stdout
