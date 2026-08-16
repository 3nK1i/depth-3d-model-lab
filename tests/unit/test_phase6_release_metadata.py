"""Cross-model benchmark-report integrity tests retained for Phase 7."""

from __future__ import annotations

import json
from pathlib import Path

from core.benchmark.runner import production_benchmark_descriptors
from core.runtime.release import benchmark_report_check
from core.version import RELEASE_NAME, RELEASE_PHASE, __version__


def test_current_release_identity_for_benchmark_reports() -> None:
    assert __version__ == "0.9.0"
    assert RELEASE_NAME == "Phase 9 Digital Twin Stable"
    assert RELEASE_PHASE == 9.0


def test_benchmark_report_requires_every_production_model(tmp_path: Path) -> None:
    results = [
        {"model_id": item.model_id, "status": "passed"}
        for item in production_benchmark_descriptors()
    ]
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps({"application": {"version": __version__}, "results": results}),
        encoding="utf-8",
    )

    check, payload = benchmark_report_check(path)
    assert check.passed, check.detail
    assert payload is not None

    results[-1]["status"] = "failed"
    path.write_text(
        json.dumps({"application": {"version": __version__}, "results": results}),
        encoding="utf-8",
    )
    check, _ = benchmark_report_check(path)
    assert not check.passed
    assert "non-passing" in check.detail
