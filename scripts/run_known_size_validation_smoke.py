#!/usr/bin/env python3
"""Run a deterministic dependency-light known-size validation smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.validation import run_known_size_validation, write_validation_report  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/phase7_5_step10_known_size_smoke")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    shutil.rmtree(output, ignore_errors=True)

    with tempfile.TemporaryDirectory(prefix="phase7_5_step10_") as temporary_name:
        temporary = Path(temporary_name)
        export = temporary / "inspection_export"
        export.mkdir()
        measurements = {
            "schema_version": 1,
            "certified_metrology": False,
            "objects": [
                {
                    "observation_id": "box-0",
                    "object_index": 0,
                    "class_name": "box",
                    "preferred_box_id": "box-0:camera_aligned",
                    "mask_source": "native_instance",
                    "mask_revision": 0,
                    "measurements": [
                        {
                            "kind": "width", "status": "available", "value_si": 0.402,
                            "expanded_uncertainty_si": 0.008, "reliability": "high", "rejection_reasons": [],
                        },
                        {
                            "kind": "height", "status": "available", "value_si": 0.299,
                            "expanded_uncertainty_si": 0.006, "reliability": "high", "rejection_reasons": [],
                        },
                        {
                            "kind": "depth", "status": "available", "value_si": 0.251,
                            "expanded_uncertainty_si": 0.007, "reliability": "high", "rejection_reasons": [],
                        },
                    ],
                }
            ],
        }
        measurements_path = export / "measurements.json"
        measurements_path.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
        marker = export / "report.txt"
        marker.write_text("synthetic smoke export\n", encoding="utf-8")
        (export / "manifest.json").write_text(
            json.dumps(
                {
                    "files": [
                        {"path": "measurements.json", "sha256": _sha256(measurements_path)},
                        {"path": "report.txt", "sha256": _sha256(marker)},
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "benchmark_name": "Phase 7.5 synthetic known-size smoke",
            "operator": "automated smoke",
            "measurement_instrument": "synthetic ground truth",
            "thresholds_confirmed": True,
            "thresholds": {
                "minimum_cases": 1,
                "minimum_available_dimensions": 3,
                "require_all_case_dimensions_available": True,
                "max_mean_absolute_error_mm": 5.0,
                "max_mean_relative_error_percent": 2.0,
                "max_single_dimension_error_mm": 5.0,
                "max_single_dimension_relative_error_percent": 2.0,
                "minimum_uncertainty_coverage_percent": 100.0,
            },
            "cases": [
                {
                    "case_id": "box-smoke",
                    "export_path": str(export),
                    "observation_id": "box-0",
                    "ground_truth_mm": {"width": 400.0, "height": 300.0, "depth": 250.0},
                    "measurement_mapping": {"width": "width", "height": "height", "depth": "depth"},
                }
            ],
        }
        manifest_path = temporary / "benchmark.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result = run_known_size_validation(manifest_path)
        write_validation_report(result, output)

    summary = result.payload["summary"]
    print(
        f"status={'success' if result.passed else 'failed'} cases={summary['case_count']} "
        f"dimensions={summary['available_dimension_count']} "
        f"mae_mm={summary['mean_absolute_error_mm']:.3f} "
        f"coverage_percent={summary['uncertainty_coverage_percent']:.1f} output={output}"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
