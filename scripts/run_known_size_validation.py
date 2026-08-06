#!/usr/bin/env python3
"""Validate finalized inspection exports against known physical dimensions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.validation import (  # noqa: E402
    KnownSizeBenchmarkError,
    run_known_size_validation,
    write_validation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "phase7_5" / "known_size_validation",
    )
    parser.add_argument("--skip-export-checksums", action="store_true")
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Write the report but return zero even when release thresholds fail.",
    )
    args = parser.parse_args()
    try:
        result = run_known_size_validation(
            args.manifest,
            verify_export_checksums=not args.skip_export_checksums,
        )
        output = write_validation_report(result, args.output_dir)
    except KnownSizeBenchmarkError as exc:
        print(f"status=failed error={exc}", file=sys.stderr)
        return 2

    summary = result.payload["summary"]
    print(
        "status={} cases={} dimensions={} mae_mm={} rmse_mm={} coverage_percent={} output={}".format(
            "passed" if result.passed else "failed",
            summary["case_count"],
            summary["available_dimension_count"],
            _fmt(summary["mean_absolute_error_mm"]),
            _fmt(summary["rmse_mm"]),
            _fmt(summary["uncertainty_coverage_percent"]),
            output,
        )
    )
    for issue in result.payload.get("issues", []):
        print(f"issue={issue}")
    return 0 if result.passed or args.allow_fail else 1


def _fmt(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
