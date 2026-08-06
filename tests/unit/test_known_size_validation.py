from __future__ import annotations

import json
from pathlib import Path

from core.inspection import InspectionExportOptions, export_inspection_bundle, load_inspection_bundle
from core.validation import create_benchmark_template, run_known_size_validation, write_validation_report
from tests.inspection_helpers import write_inspection_bundle


def _export(tmp_path: Path) -> Path:
    bundle = write_inspection_bundle(tmp_path / "bundle")
    summary, arrays = load_inspection_bundle(bundle)
    return export_inspection_bundle(
        summary,
        arrays,
        InspectionExportOptions(output_dir=tmp_path / "export"),
    ).export_dir


def _manifest(path: Path, export_dir: Path, *, confirmed: bool = True) -> Path:
    payload = {
        "schema_version": 1,
        "benchmark_name": "synthetic known-size test",
        "operator": "pytest",
        "measurement_instrument": "synthetic",
        "thresholds_confirmed": confirmed,
        "thresholds": {
            "minimum_cases": 1,
            "minimum_available_dimensions": 1,
            "require_all_case_dimensions_available": False,
            "max_mean_absolute_error_mm": 20.0,
            "max_mean_relative_error_percent": 10.0,
            "max_single_dimension_error_mm": 20.0,
            "minimum_uncertainty_coverage_percent": 100.0,
        },
        "cases": [
            {
                "case_id": "box-1",
                "export_path": str(export_dir),
                "observation_id": "object-0",
                "ground_truth_mm": {"width": 395.0, "height": 100.0, "depth": 100.0},
                "measurement_mapping": {"width": "width", "height": "missing-height", "depth": "missing-depth"},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_template_is_explicitly_provisional(tmp_path: Path) -> None:
    path = create_benchmark_template(tmp_path / "benchmark.json")
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["thresholds_confirmed"] is False
    assert len(payload["cases"]) == 1


def test_known_size_validation_computes_error_and_coverage(tmp_path: Path) -> None:
    export_dir = _export(tmp_path)
    manifest = _manifest(tmp_path / "benchmark.json", export_dir)

    result = run_known_size_validation(manifest)

    assert result.passed
    assert result.case_count == 1
    assert result.available_dimension_count == 1
    summary = result.payload["summary"]
    assert summary["mean_absolute_error_mm"] == 5.0
    assert summary["uncertainty_coverage_percent"] == 100.0
    dimension = result.payload["cases"][0]["dimensions"][0]
    assert dimension["estimated_mm"] == 400.0
    assert dimension["covered_by_uncertainty"] is True


def test_provisional_thresholds_block_release_result(tmp_path: Path) -> None:
    export_dir = _export(tmp_path)
    manifest = _manifest(tmp_path / "benchmark.json", export_dir, confirmed=False)

    result = run_known_size_validation(manifest)

    assert not result.passed
    assert any("provisional" in issue for issue in result.payload["issues"])


def test_validation_report_is_atomic_and_has_checksums(tmp_path: Path) -> None:
    export_dir = _export(tmp_path)
    result = run_known_size_validation(_manifest(tmp_path / "benchmark.json", export_dir))

    output = write_validation_report(result, tmp_path / "report")

    for name in (
        "known_size_validation.json",
        "known_size_measurements.csv",
        "known_size_validation.md",
        "known_size_validation.html",
        "manifest.json",
    ):
        assert (output / name).is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["passed"] is True
    assert len(manifest["files"]) == 4
