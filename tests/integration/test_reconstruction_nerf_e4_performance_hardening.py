from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.nerf_backend_lifecycle import run_supervised_process

ROOT = Path(__file__).resolve().parents[2]


def test_e4_supervisor_cleans_child_process_group_after_parent_crash(tmp_path: Path) -> None:
    marker = tmp_path / "child.txt"
    script = (
        "import os,subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid)); "
        "os._exit(7)"
    )
    outcome = run_supervised_process([sys.executable, "-c", script], timeout_s=5.0, terminate_grace_s=0.1)
    assert outcome.exit_code == 7
    assert outcome.process_group_clean
    child_pid = int(marker.read_text())
    deadline = time.monotonic() + 1.0
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not Path(f"/proc/{child_pid}").exists()


def test_e4_supervisor_reports_cancellation_and_timeout(tmp_path: Path) -> None:
    cancelled = run_supervised_process(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout_s=5.0,
        cancel_after_s=0.1,
        terminate_grace_s=0.1,
    )
    assert cancelled.cancelled
    assert not cancelled.timed_out
    assert cancelled.process_group_clean

    timed = run_supervised_process(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout_s=0.1,
        terminate_grace_s=0.1,
    )
    assert timed.timed_out
    assert not timed.cancelled
    assert timed.process_group_clean


def test_e4_runners_reference_only_external_worker_boundary() -> None:
    perf = (ROOT / "scripts/run_nerf_performance_hardening.py").read_text(encoding="utf-8")
    robust = (ROOT / "scripts/run_nerf_backend_robustness.py").read_text(encoding="utf-8")
    worker = "scripts/nerf_workers/nerfstudio_e4_worker.py"
    assert "nerfstudio_e4_worker.py" in perf
    assert "nerfstudio_e4_worker.py" in robust
    assert "import nerfstudio" not in perf
    assert "from nerfstudio" not in perf
    assert "import nerfstudio" not in robust
    assert "from nerfstudio" not in robust


def test_e4_real_runner_keeps_rtx5090_measurements_empty_until_real_hardware() -> None:
    text = (ROOT / "scripts/run_nerf_performance_hardening.py").read_text(encoding="utf-8")
    assert "nerf_rtx5090_readiness_profile" in text
    assert "rtx5090_readiness_report.json" in text
    assert "real_5090_performance_measurements=None" in text
