from __future__ import annotations

import sys

from scripts.gaussian_backend_lifecycle import run_supervised_process


def test_fault_then_recovery_process_boundary() -> None:
    failed = run_supervised_process(
        [sys.executable, "-c", "raise RuntimeError('intentional')"],
        timeout_s=2.0,
    )
    assert failed.exit_code != 0
    assert failed.process_group_clean

    recovered = run_supervised_process(
        [sys.executable, "-c", "print('recovered')"],
        timeout_s=2.0,
    )
    assert recovered.exit_code == 0
    assert recovered.process_group_clean
    assert "recovered" in recovered.stdout
