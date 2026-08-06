from __future__ import annotations

import json
from pathlib import Path

from core.inspection import InspectionExportOptions, export_inspection_bundle, load_inspection_bundle
from core.validation import run_known_size_validation, write_validation_report
from tests.inspection_helpers import write_inspection_bundle


def test_inspection_export_flows_into_known_size_report(tmp_path: Path) -> None:
    bundle = write_inspection_bundle(tmp_path / "bundle", manual=True)
    summary, arrays = load_inspection_bundle(bundle)
    export_dir = export_inspection_bundle(
        summary,
        arrays,
        InspectionExportOptions(output_dir=tmp_path / "export", locked_ids=("object-0",)),
    ).export_dir
    benchmark = {
        "schema_version": 1,
        "benchmark_name": "integration",
        "thresholds_confirmed": True,
        "thresholds": {
            "minimum_cases": 1,
            "minimum_available_dimensions": 1,
            "require_all_case_dimensions_available": False,
            "max_mean_absolute_error_mm": 1.0,
            "max_mean_relative_error_percent": 1.0,
            "max_single_dimension_error_mm": 1.0,
            "minimum_uncertainty_coverage_percent": 100.0,
        },
        "cases": [
            {
                "case_id": "known-box",
                "export_path": str(export_dir),
                "observation_id": "object-0",
                "ground_truth_mm": {"width": 400.0, "height": 1.0, "depth": 1.0},
                "measurement_mapping": {"width": "width", "height": "none", "depth": "none"},
            }
        ],
    }
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(json.dumps(benchmark), encoding="utf-8")

    result = run_known_size_validation(manifest)
    output = write_validation_report(result, tmp_path / "validation")

    assert result.passed
    payload = json.loads((output / "known_size_validation.json").read_text())
    assert payload["cases"][0]["mask_source"] == "manual_correction"
    assert payload["summary"]["mean_absolute_error_mm"] == 0.0
