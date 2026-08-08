from __future__ import annotations

import sys

from scripts.gaussian_backend_lifecycle import run_supervised_process


def test_repeated_cancelled_process_groups_do_not_survive() -> None:
    for _ in range(8):
        outcome = run_supervised_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_s=2.0,
            cancel_after_s=0.05,
            terminate_grace_s=0.05,
        )
        assert outcome.cancelled
        assert outcome.process_group_clean
