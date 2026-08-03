"""Release-gate subprocess capture and timeout tests."""

from __future__ import annotations

import sys

from scripts.release_check import _run_command


def test_release_command_captures_success() -> None:
    result = _run_command(
        "probe",
        [sys.executable, "-c", "print('ready')"],
        timeout_seconds=5,
    )

    assert result.passed
    assert result.stdout.strip() == "ready"
    assert result.duration_seconds >= 0.0


def test_release_command_timeout_is_bounded() -> None:
    result = _run_command(
        "timeout",
        [sys.executable, "-c", "import time; time.sleep(2)"],
        timeout_seconds=0.05,
    )

    assert result.returncode == 124
    assert "timed out" in result.stderr.lower()
