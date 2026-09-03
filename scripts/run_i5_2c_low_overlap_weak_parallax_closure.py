#!/usr/bin/env python3
"""Formally qualify and close I5.2 low-overlap / weak-parallax robustness evidence."""
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
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding  # noqa: E402
from core.reconstruction.i5_low_overlap_weak_parallax import (  # noqa: E402
    build_i5_2c_low_overlap_weak_parallax_closure,
)

OUTPUT_NAME = "i5_2c_low_overlap_weak_parallax_closure.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i5-2a-admission", type=Path, required=True)
    parser.add_argument("--i5-2a-sha256", required=True)
    parser.add_argument("--i5-2b-replay", type=Path, required=True)
    parser.add_argument("--i5-2b-sha256", required=True)
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
        raise ReconstructionValidationError(f"I5.2C {label} is missing: {resolved}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I5.2C {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected_text:
        raise ReconstructionValidationError(
            f"I5.2C {label} SHA256 mismatch: expected={expected_text} actual={actual}"
        )
    return resolved, actual


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.2C {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.2C {label} root must be an object.")
    return payload


def run(args: argparse.Namespace) -> Path:
    a_path, a_sha = _verify(args.i5_2a_admission, args.i5_2a_sha256, label="I5.2A admission")
    b_path, b_sha = _verify(args.i5_2b_replay, args.i5_2b_sha256, label="I5.2B replay")
    receipt_path, receipt_sha = _verify(
        args.qualification_receipt,
        args.qualification_receipt_sha256,
        label="I5.2C qualification receipt",
    )

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I5.2C output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    closure = build_i5_2c_low_overlap_weak_parallax_closure(
        i5_2a_result=_read_json(a_path, label="I5.2A admission"),
        i5_2a_binding=I51EvidenceBinding(str(a_path), a_sha),
        i5_2b_result=_read_json(b_path, label="I5.2B replay"),
        i5_2b_binding=I51EvidenceBinding(str(b_path), b_sha),
        qualification_receipt=_read_json(receipt_path, label="I5.2C qualification receipt"),
        qualification_receipt_binding=I51EvidenceBinding(str(receipt_path), receipt_sha),
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(closure, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    q = closure["qualification"]
    stress = closure["stress_evidence"]
    low = stress["low_overlap"]
    weak = stress["weak_parallax"]
    print("I5.2C Low-Overlap / Weak-Parallax Robustness Qualification + Closure: PASS")
    print(
        f"quality_status={q['quality_status']} control={q['control_registration']} "
        f"low_overlap={q['low_overlap_registration']} weak_parallax={q['weak_parallax_registration']} "
        f"partial_registration_preserved={q['partial_registration_preserved']}"
    )
    print(
        f"low_overlap_revalidated={q['low_overlap_authoritative_revalidation']} "
        f"control_common_p50={low.get('control_adjacent_common_landmarks_p50')} "
        f"stress_common_p50={low.get('stress_adjacent_common_landmarks_p50')}"
    )
    print(
        f"weak_parallax_revalidated={q['weak_parallax_authoritative_revalidation']} "
        f"control_parallax_p50={weak.get('control_max_pair_parallax_degrees_p50')} "
        f"stress_parallax_p50={weak.get('stress_max_pair_parallax_degrees_p50')} "
        f"control_baseline_ratio_p50={weak.get('control_baseline_to_range_ratio_p50')} "
        f"stress_baseline_ratio_p50={weak.get('stress_baseline_to_range_ratio_p50')}"
    )
    print(f"operating_envelope={q['operating_envelope']}")
    print(
        "universal_low_overlap=False universal_weak_parallax=False dense_robustness=False provider_robustness=False"
    )
    print(
        "pycolmap_authority=True threshold_relaxation=False production_default_modified=False "
        "mp_sfm=False auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print(f"I5.2 low overlap / weak parallax: CLOSED / {q['quality_status'].upper()}")
    print("I5.3 blur / exposure / clutter / symmetry: AUTHORIZED")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.2C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
