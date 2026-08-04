"""CLI progress and report compatibility tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from core.benchmark.events import parse_event_line

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_phase6_benchmark.py"


def test_mock_cli_emits_structured_progress_and_reports(tmp_path: Path) -> None:
    output = tmp_path / "reports"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--models",
            "development-mock",
            "--include-mock",
            "--smoke-test",
            "--progress-jsonl",
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
    events = [
        event
        for event in (parse_event_line(line) for line in completed.stdout.splitlines())
        if event is not None
    ]
    assert [event["event"] for event in events] == [
        "benchmark_started",
        "model_started",
        "model_completed",
        "benchmark_finished",
    ]
    assert events[-1]["passed"] is True
    payload = json.loads((output / "benchmark_results.json").read_text(encoding="utf-8"))
    assert payload["configuration"]["image_asset"] == "registry"
    assert payload["configuration"]["sequence_asset"] == "registry"
    assert payload["results"][0]["status"] == "passed"
