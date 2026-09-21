from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.reconstruction.i5_failure_robustness_qualification import (
    I511D_PHASE,
    build_i5_11d_failure_robustness_closure,
)

ROOT = Path(__file__).resolve().parents[2]


def test_i511d_module_imports_without_ui_or_ml_runtime() -> None:
    result = subprocess.run(
        [sys.executable, '-c', 'import core.reconstruction.i5_failure_robustness_qualification as m; print(m.I511D_PHASE)'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'I5.11D'


def test_i511d_runner_help_is_non_destructive() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/run_i5_11d_failure_robustness_closure.py'), '--help'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10,
    )
    assert result.returncode == 0
    assert '--i5-11c-campaign' in result.stdout
    assert '--closure-receipt' in result.stdout


def test_i511d_contract_is_callable() -> None:
    assert I511D_PHASE == 'I5.11D'
    assert callable(build_i5_11d_failure_robustness_closure)
