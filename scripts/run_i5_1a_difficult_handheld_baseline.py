#!/usr/bin/env python3
"""Bind frozen I4 closure + committed Horse phase history as I5.1 baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.i5_difficult_handheld_robustness import (  # noqa: E402
    I51EvidenceBinding,
    build_i5_1a_difficult_handheld_baseline,
)

OUTPUT_NAME = "i5_1a_difficult_handheld_baseline.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-4d-result", type=Path, required=True)
    parser.add_argument("--i4-4d-sha256", required=True)
    parser.add_argument("--i4-1-4-doc", type=Path, required=True)
    parser.add_argument("--i4-1-4-doc-sha256", required=True)
    parser.add_argument("--i4-1-5-doc", type=Path, required=True)
    parser.add_argument("--i4-1-5-doc-sha256", required=True)
    parser.add_argument("--i4-1-6b-doc", type=Path, required=True)
    parser.add_argument("--i4-1-6b-doc-sha256", required=True)
    parser.add_argument("--i4-1-7-doc", type=Path, required=True)
    parser.add_argument("--i4-1-7-doc-sha256", required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--baseline-receipt-sha256", required=True)
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
        raise ReconstructionValidationError(f"I5.1A {label} is missing: {resolved}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I5.1A {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected_text:
        raise ReconstructionValidationError(
            f"I5.1A {label} SHA256 mismatch: expected={expected_text} actual={actual}"
        )
    return resolved, actual


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.1A {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.1A {label} root must be an object.")
    return payload


def run(args: argparse.Namespace) -> Path:
    i44d_path, i44d_sha = _verify(args.i4_4d_result, args.i4_4d_sha256, label="I4.4D final closure")
    i414_path, i414_sha = _verify(args.i4_1_4_doc, args.i4_1_4_doc_sha256, label="I4.1.4 phase history")
    i415_path, i415_sha = _verify(args.i4_1_5_doc, args.i4_1_5_doc_sha256, label="I4.1.5 phase history")
    i416b_path, i416b_sha = _verify(args.i4_1_6b_doc, args.i4_1_6b_doc_sha256, label="I4.1.6B phase history")
    i417_path, i417_sha = _verify(args.i4_1_7_doc, args.i4_1_7_doc_sha256, label="I4.1.7 phase history")
    receipt_path, receipt_sha = _verify(args.baseline_receipt, args.baseline_receipt_sha256, label="I5.1A baseline receipt")

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I5.1A output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    result = build_i5_1a_difficult_handheld_baseline(
        i4_4d_result=_read_json(i44d_path, label="I4.4D final closure"),
        i4_4d_binding=I51EvidenceBinding(str(i44d_path), i44d_sha),
        i4_1_4_doc=i414_path.read_text(encoding="utf-8"),
        i4_1_4_binding=I51EvidenceBinding(str(i414_path), i414_sha),
        i4_1_5_doc=i415_path.read_text(encoding="utf-8"),
        i4_1_5_binding=I51EvidenceBinding(str(i415_path), i415_sha),
        i4_1_6b_doc=i416b_path.read_text(encoding="utf-8"),
        i4_1_6b_binding=I51EvidenceBinding(str(i416b_path), i416b_sha),
        i4_1_7_doc=i417_path.read_text(encoding="utf-8"),
        i4_1_7_binding=I51EvidenceBinding(str(i417_path), i417_sha),
        baseline_receipt=_read_json(receipt_path, label="I5.1A baseline receipt"),
        baseline_receipt_binding=I51EvidenceBinding(str(receipt_path), receipt_sha),
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    baseline = result["historical_baseline"]
    print("I5.1A Difficult Handheld Historical Baseline Admission: PASS")
    print("evidence_level=committed_phase_history historical_standalone_summary_json_bound=False")
    print(
        f"horse_full_span_alignment={baseline['historical_full_span_da3_sfm_alignment']} "
        f"dense_collapsed_to_views={baseline['historical_dense_path_collapsed_to_views']} "
        "historical_10_view_gate=False"
    )
    print("historical_failure_reclassified_as_pass=False synthetic_historical_json_created=False")
    print("provider_execution_started=False sfm_rerun=False dense_pair_rerun=False current_replay_started=False")
    print("auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None")
    print("I5.1B current frozen-foundation difficult-handheld replay: AUTHORIZED")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.1A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
