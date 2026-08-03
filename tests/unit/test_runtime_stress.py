"""Pure tests for dependency-light stress diagnostics."""

from __future__ import annotations

import json
import os

from core.runtime.stress import (
    StressSample,
    process_exists,
    read_process_rss_mib,
    summarize_samples,
    tail_drift,
    wait_for_process_exit,
    write_stress_report,
)


def test_current_process_rss_and_existence_are_available() -> None:
    assert process_exists(os.getpid())
    rss = read_process_rss_mib(os.getpid())
    assert rss is None or rss > 0.0


def test_missing_process_is_reported_cleanly() -> None:
    impossible_pid = 999_999_999
    assert not process_exists(impossible_pid)
    assert read_process_rss_mib(impossible_pid) is None
    assert wait_for_process_exit(impossible_pid, timeout_s=0.0)


def test_tail_drift_uses_median_windows_and_ignores_missing_values() -> None:
    values = [10.0, None, 12.0, 99.0, 20.0, 22.0]
    # first median = 11, last median = 21; the middle outlier is ignored.
    assert tail_drift(values, window=2) == 10.0
    assert tail_drift([None, 1.0]) is None


def test_summary_tracks_process_and_vram_ranges() -> None:
    samples = [
        StressSample(1, "after_unload", 101, 1, 100.0, 200.0, 0.0, 0.1, 1.0),
        StressSample(2, "after_unload", 101, 1, 104.0, 203.0, 0.0, 0.2, 1.1),
        StressSample(3, "after_restart", 202, 2, 103.0, 190.0, None, None, None),
    ]

    summary = summarize_samples(samples)

    assert summary.sample_count == 3
    assert summary.completed_cycles == 3
    assert summary.unique_worker_pids == (101, 202)
    assert summary.parent_rss_drift_mib == 3.0
    assert summary.worker_rss_peak_mib == 203.0
    assert summary.max_vram_reserved_gib == 0.2


def test_report_is_written_atomically_as_json(tmp_path) -> None:
    output = tmp_path / "stress" / "report.json"
    sample = StressSample(1, "done", 123, 1, 10.0, 20.0)

    written = write_stress_report(
        output,
        kind="unit",
        settings={"cycles": 1},
        samples=[sample],
        checks={"passed": True},
    )

    assert written == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["kind"] == "unit"
    assert payload["summary"]["sample_count"] == 1
    assert payload["checks"]["passed"] is True
    assert not output.with_suffix(".json.tmp").exists()
