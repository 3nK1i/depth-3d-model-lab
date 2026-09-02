#!/usr/bin/env python3
"""Bind I4.3H2 and a passing focused-regression receipt into I4.4A evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_foundation_regression import (  # noqa: E402
    RegressionEvidenceBinding,
    build_i4_4a_focused_regression,
)

OUTPUT_NAME = "i4_4a_focused_regression.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3h2-closure", type=Path, required=True)
    parser.add_argument("--i4-3h2-sha256", required=True)
    parser.add_argument("--regression-receipt", type=Path, required=True)
    parser.add_argument("--regression-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, *, label: str) -> tuple[Path, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I4.4A {label} is missing: {resolved}")
    actual = _sha256(resolved)
    if actual != str(expected).strip().lower():
        raise ReconstructionValidationError(f"I4.4A {label} SHA256 mismatch: expected={expected} actual={actual}")
    return resolved, actual


def _read(path: Path, *, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I4.4A {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.4A {label} root must be an object.")
    return payload


def run(args: argparse.Namespace) -> Path:
    h2_path, h2_sha = _verify(args.i4_3h2_closure, args.i4_3h2_sha256, label="I4.3H2 closure")
    receipt_path, receipt_sha = _verify(args.regression_receipt, args.regression_receipt_sha256, label="regression receipt")
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I4.4A output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)
    result = build_i4_4a_focused_regression(
        i4_3h2_closure=_read(h2_path, label="I4.3H2 closure"),
        i4_3h2_binding=RegressionEvidenceBinding(str(h2_path), h2_sha),
        regression_receipt=_read(receipt_path, label="regression receipt"),
        regression_binding=RegressionEvidenceBinding(str(receipt_path), receipt_sha),
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    counts = result["regression_summary"]
    print("I4.4A Offline Dense Foundation Focused Regression: PASS")
    print(f"focused_tests={counts['focused_tests']} affected_tests={counts['affected_total_tests']} affected_files={counts['affected_test_files']}")
    print("provider_execution=False sfm_rerun=False visual_verification=False auto=False hybrid=False metric_scale=False metrology=False")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I4.4A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
