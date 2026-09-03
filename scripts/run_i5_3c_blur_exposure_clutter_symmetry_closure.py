#!/usr/bin/env python3
"""Formally qualify and close I5.3 blur / exposure / clutter / symmetry robustness evidence."""
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
from core.reconstruction.i5_blur_exposure_clutter_symmetry import (  # noqa: E402
    build_i5_3c_blur_exposure_clutter_symmetry_closure,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding  # noqa: E402

OUTPUT_NAME = "i5_3c_blur_exposure_clutter_symmetry_closure.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i5-3a-admission", type=Path, required=True)
    parser.add_argument("--i5-3a-sha256", required=True)
    parser.add_argument("--i5-3b-replay", type=Path, required=True)
    parser.add_argument("--i5-3b-sha256", required=True)
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
        raise ReconstructionValidationError(f"I5.3C {label} is missing: {resolved}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I5.3C {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected_text:
        raise ReconstructionValidationError(
            f"I5.3C {label} SHA256 mismatch: expected={expected_text} actual={actual}"
        )
    return resolved, actual


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.3C {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.3C {label} root must be an object.")
    return payload


def run(args: argparse.Namespace) -> Path:
    a_path, a_sha = _verify(args.i5_3a_admission, args.i5_3a_sha256, label="I5.3A admission")
    b_path, b_sha = _verify(args.i5_3b_replay, args.i5_3b_sha256, label="I5.3B replay")
    receipt_path, receipt_sha = _verify(
        args.qualification_receipt,
        args.qualification_receipt_sha256,
        label="I5.3C qualification receipt",
    )

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I5.3C output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    closure = build_i5_3c_blur_exposure_clutter_symmetry_closure(
        i5_3a_result=_read_json(a_path, label="I5.3A admission"),
        i5_3a_binding=I51EvidenceBinding(str(a_path), a_sha),
        i5_3b_result=_read_json(b_path, label="I5.3B replay"),
        i5_3b_binding=I51EvidenceBinding(str(b_path), b_sha),
        qualification_receipt=_read_json(receipt_path, label="I5.3C qualification receipt"),
        qualification_receipt_binding=I51EvidenceBinding(str(receipt_path), receipt_sha),
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(closure, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    q = closure["qualification"]
    conclusions = closure["stress_conclusions"]
    print("I5.3C Blur / Exposure / Clutter / Symmetry Robustness Qualification + Closure: PASS")
    print(
        f"quality_status={q['quality_status']} control={q['control_registration']} blur={q['blur_registration']} "
        f"underexposure={q['underexposure_registration']} overexposure={q['overexposure_registration']} "
        f"clutter={q['clutter_registration']} symmetry={q['symmetry_registration']}"
    )
    for key in ("blur", "underexposure", "overexposure", "clutter", "symmetry"):
        item = conclusions[key]
        print(
            f"{key}_revalidated={item['semantic_stress_revalidated']} "
            f"{key}_conclusion={item['robustness_conclusion']} "
            f"{key}_qualified={item['robustness_qualified']}"
        )
    print(f"operating_envelope={q['operating_envelope']}")
    print(
        "universal_visual_stress=False clutter_robustness=False dense_robustness=False provider_robustness=False"
    )
    print(
        "pycolmap_authority=True threshold_relaxation=False production_default_modified=False "
        "mp_sfm=False auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print(f"I5.3 blur / exposure / clutter / symmetry: CLOSED / {q['quality_status'].upper()}")
    print("I5.4 forward-motion / KITTI robustness: AUTHORIZED")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.3C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
