from __future__ import annotations

import json
from pathlib import Path

from core.validation import forbidden_release_artifacts_check, known_size_report_check, phase75_layout_check


def test_forbidden_release_artifacts_reject_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    assert forbidden_release_artifacts_check(tmp_path).passed
    (tmp_path / "yolo11n-seg.pt").write_bytes(b"checkpoint")
    result = forbidden_release_artifacts_check(tmp_path)
    assert not result.passed
    assert "yolo11n-seg.pt" in result.detail


def test_known_size_report_must_be_confirmed_and_passed(tmp_path: Path) -> None:
    report = tmp_path / "known_size_validation.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "thresholds_confirmed": True,
                "passed": True,
                "summary": {
                    "case_count": 3,
                    "available_dimension_count": 9,
                    "mean_absolute_error_mm": 12.5,
                    "uncertainty_coverage_percent": 77.8,
                },
            }
        ),
        encoding="utf-8",
    )
    check, payload = known_size_report_check(report)
    assert check.passed
    assert payload is not None
    assert "3 cases" in check.detail


def test_known_size_report_rejects_failed_payload(tmp_path: Path) -> None:
    report = tmp_path / "known_size_validation.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "thresholds_confirmed": True,
                "passed": False,
                "summary": {"case_count": 3, "available_dimension_count": 9},
            }
        ),
        encoding="utf-8",
    )
    check, _ = known_size_report_check(report)
    assert not check.passed


def test_phase75_layout_check_detects_missing_files(tmp_path: Path) -> None:
    result = phase75_layout_check(tmp_path)
    assert not result.passed
    assert "core/validation/known_size.py" in result.detail
