"""Repeated isolated mock benchmarks for Phase 7 interactive mode."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_phase6_benchmark.py"


def test_repeated_mock_cli_runs_create_independent_reports(tmp_path: Path) -> None:
    for index in range(3):
        output = tmp_path / f"run-{index}"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--models",
                "development-mock",
                "--include-mock",
                "--smoke-test",
                "--output-dir",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads((output / "benchmark_results.json").read_text(encoding="utf-8"))
        assert payload["passed"] is True
        assert payload["results"][0]["status"] == "passed"
