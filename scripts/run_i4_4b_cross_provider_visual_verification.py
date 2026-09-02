#!/usr/bin/env python3
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

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_cross_provider_visual_verification import (
    VisualVerificationEvidenceBinding,
    build_i4_4b_cross_provider_visual_verification,
)

OUTPUT_NAME = "i4_4b_cross_provider_visual_verification.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build I4.4B cross-provider visual-verification evidence.")
    parser.add_argument("--i4-3h2-closure", type=Path, required=True)
    parser.add_argument("--i4-3h2-sha256", required=True)
    parser.add_argument("--i4-4a-result", type=Path, required=True)
    parser.add_argument("--i4-4a-sha256", required=True)
    parser.add_argument("--i4-3g2-closure", type=Path, required=True)
    parser.add_argument("--i4-3g2-sha256", required=True)
    parser.add_argument("--i4-3g3c-closure", type=Path, required=True)
    parser.add_argument("--i4-3g3c-sha256", required=True)
    parser.add_argument("--i4-3g4c-closure", type=Path, required=True)
    parser.add_argument("--i4-3g4c-sha256", required=True)
    parser.add_argument("--visual-receipt", type=Path, required=True)
    parser.add_argument("--visual-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _verify(path: Path, expected: str, *, label: str) -> tuple[Path, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I4.4B {label} is missing: {resolved}")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    actual = str(expected).strip().lower()
    if len(actual) != 64 or any(ch not in "0123456789abcdef" for ch in actual):
        raise ReconstructionValidationError(f"I4.4B {label} expected SHA256 is invalid.")
    if digest != actual:
        raise ReconstructionValidationError(f"I4.4B {label} SHA256 mismatch: expected={actual} actual={digest}")
    return resolved, actual


def _read(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I4.4B {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.4B {label} root must be an object.")
    return payload


def run(args: argparse.Namespace) -> Path:
    h2_path, h2_sha = _verify(args.i4_3h2_closure, args.i4_3h2_sha256, label="I4.3H2 closure")
    i44a_path, i44a_sha = _verify(args.i4_4a_result, args.i4_4a_sha256, label="I4.4A result")
    g2_path, g2_sha = _verify(args.i4_3g2_closure, args.i4_3g2_sha256, label="I4.3G2 closure")
    g3_path, g3_sha = _verify(args.i4_3g3c_closure, args.i4_3g3c_sha256, label="I4.3G3C closure")
    g4_path, g4_sha = _verify(args.i4_3g4c_closure, args.i4_3g4c_sha256, label="I4.3G4C closure")
    receipt_path, receipt_sha = _verify(args.visual_receipt, args.visual_receipt_sha256, label="visual receipt")
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I4.4B output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)
    result = build_i4_4b_cross_provider_visual_verification(
        i4_3h2_closure=_read(h2_path, label="I4.3H2 closure"),
        i4_3h2_binding=VisualVerificationEvidenceBinding(str(h2_path), h2_sha),
        i4_4a_result=_read(i44a_path, label="I4.4A result"),
        i4_4a_binding=VisualVerificationEvidenceBinding(str(i44a_path), i44a_sha),
        i4_3g2_closure=_read(g2_path, label="I4.3G2 closure"),
        i4_3g2_binding=VisualVerificationEvidenceBinding(str(g2_path), g2_sha),
        i4_3g3c_closure=_read(g3_path, label="I4.3G3C closure"),
        i4_3g3c_binding=VisualVerificationEvidenceBinding(str(g3_path), g3_sha),
        i4_3g4c_closure=_read(g4_path, label="I4.3G4C closure"),
        i4_3g4c_binding=VisualVerificationEvidenceBinding(str(g4_path), g4_sha),
        visual_receipt=_read(receipt_path, label="visual receipt"),
        visual_receipt_binding=VisualVerificationEvidenceBinding(str(receipt_path), receipt_sha),
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    coverage = result["coverage"]
    print("I4.4B Cross-Provider Visual Verification: PASS")
    print(f"families={coverage['qualification_family_count']} scenarios={coverage['scenario_count']} providers={coverage['provider_implementation_count']} fresh_visual_scenarios={coverage['fresh_visual_review_scenario_count']} historical_visual_scenarios={coverage['historical_visual_review_scenario_count']}")
    print("provider_execution=False sfm_rerun=False matched_view_rerun=False manual_review_rerun=False auto=False hybrid=False metric_scale=False metrology=False")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I4.4B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
