#!/usr/bin/env python3
"""Formally qualify and close I5.1 difficult-handheld robustness evidence."""
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
    build_i5_1c_difficult_handheld_closure,
)

OUTPUT_NAME = "i5_1c_difficult_handheld_closure.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i5-1a-result", type=Path, required=True)
    parser.add_argument("--i5-1a-sha256", required=True)
    parser.add_argument("--i5-1b-result", type=Path, required=True)
    parser.add_argument("--i5-1b-sha256", required=True)
    parser.add_argument("--i5-1b-oom-diagnostic", type=Path, required=True)
    parser.add_argument("--i5-1b-oom-diagnostic-sha256", required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--qualification-receipt-sha256", required=True)
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
        raise ReconstructionValidationError(f"I5.1C {label} is missing: {resolved}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I5.1C {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected_text:
        raise ReconstructionValidationError(
            f"I5.1C {label} SHA256 mismatch: expected={expected_text} actual={actual}"
        )
    return resolved, actual


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.1C {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.1C {label} root must be an object.")
    return payload


def run(args: argparse.Namespace) -> Path:
    a_path, a_sha = _verify(args.i5_1a_result, args.i5_1a_sha256, label="I5.1A baseline")
    b_path, b_sha = _verify(args.i5_1b_result, args.i5_1b_sha256, label="I5.1B replay")
    oom_path, oom_sha = _verify(
        args.i5_1b_oom_diagnostic,
        args.i5_1b_oom_diagnostic_sha256,
        label="I5.1B failed-attempt OOM diagnostic",
    )
    receipt_path, receipt_sha = _verify(
        args.qualification_receipt,
        args.qualification_receipt_sha256,
        label="I5.1C qualification receipt",
    )

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I5.1C output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    closure = build_i5_1c_difficult_handheld_closure(
        i5_1a_result=_read_json(a_path, label="I5.1A baseline"),
        i5_1a_binding=I51EvidenceBinding(str(a_path), a_sha),
        i5_1b_result=_read_json(b_path, label="I5.1B replay"),
        i5_1b_binding=I51EvidenceBinding(str(b_path), b_sha),
        oom_diagnostic=oom_path.read_text(encoding="utf-8"),
        oom_diagnostic_binding=I51EvidenceBinding(str(oom_path), oom_sha),
        qualification_receipt=_read_json(receipt_path, label="I5.1C qualification receipt"),
        qualification_receipt_binding=I51EvidenceBinding(str(receipt_path), receipt_sha),
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(closure, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    q = closure["qualification"]
    event = closure["robustness_event"]
    print("I5.1C Difficult-Handheld Robustness Qualification + Closure: PASS")
    print(
        f"quality_status={q['quality_status']} historical_selected={q['historical_selected_qualified_views']} "
        f"global_selected={q['fresh_global_selected_qualified_views']} best_selected={q['fresh_best_selected_qualified_views']} "
        f"selected_gain={q['selected_view_gain_over_historical']} required={q['required_connected_qualified_views']} "
        f"dense_gate={q['production_dense_gate_reached']}"
    )
    print(
        "operating_envelope=outside_current_production_dense_robustness_envelope "
        "primary_bottleneck=local_dense_consistency_connectivity"
    )
    print(
        f"pre_correction_oom={event['pre_correction_host_oom_observed']} "
        f"last_candidate={event['last_observed_candidate']} corrected_replay_completed={event['corrected_replay_completed']} "
        "retry_until_quality_pass=False"
    )
    print(
        "pycolmap_authority=True threshold_relaxation=False production_default_modified=False "
        "auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print("I5.1 difficult handheld captures: CLOSED / DEGRADED OPERATING ENVELOPE")
    print("I5.2 low overlap / weak parallax: AUTHORIZED")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.1C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
