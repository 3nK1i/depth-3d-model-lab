#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mp_sfm_qualification import build_i5_10d_mp_sfm_closure

OUTPUT_NAME = "i5_10d_mp_sfm_qualification_and_closure.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path, *, label: str) -> dict:
    if not path.is_file():
        raise ReconstructionValidationError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object.")
    return value


def _binding(path_value: str, expected_sha: str, *, label: str) -> dict[str, str]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"{label} does not exist: {path}")
    actual = _sha256(path)
    expected = str(expected_sha).strip().lower()
    if actual != expected:
        raise ReconstructionValidationError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="I5.10D evidence-only MP-SfM qualification and closure.")
    p.add_argument("--i5-10a-admission", required=True)
    p.add_argument("--i5-10a-sha256", required=True)
    p.add_argument("--i5-10b-preflight", required=True)
    p.add_argument("--i5-10b-sha256", required=True)
    p.add_argument("--i5-10c-evaluation", required=True)
    p.add_argument("--i5-10c-sha256", required=True)
    p.add_argument("--closure-receipt", required=True)
    p.add_argument("--closure-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    a = _binding(args.i5_10a_admission, args.i5_10a_sha256, label="I5.10A admission")
    b = _binding(args.i5_10b_preflight, args.i5_10b_sha256, label="I5.10B preflight")
    c = _binding(args.i5_10c_evaluation, args.i5_10c_sha256, label="I5.10C evaluation")
    r = _binding(args.closure_receipt, args.closure_receipt_sha256, label="I5.10D closure receipt")

    result = build_i5_10d_mp_sfm_closure(
        i5_10a_admission=_read_json(Path(a["path"]), label="I5.10A admission"),
        i5_10a_binding=a,
        i5_10b_preflight=_read_json(Path(b["path"]), label="I5.10B preflight"),
        i5_10b_binding=b,
        i5_10c_evaluation=_read_json(Path(c["path"]), label="I5.10C evaluation"),
        i5_10c_binding=c,
        closure_receipt=_read_json(Path(r["path"]), label="I5.10D closure receipt"),
        closure_receipt_binding=r,
    )

    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    q = result["qualification"]
    print("I5.10D MP-SfM Evidence-Only Qualification + Closure: PASS")
    print(
        "conclusion=" + q["conclusion"]
    )
    print(
        f"baseline=sequential_overlap10 registered={q['baseline_registered_images']}/60 sparse_model={q['baseline_sparse_model']} "
        f"candidate=exhaustive registered={q['candidate_registered_images']}/60 sparse_model={q['candidate_sparse_model']}"
    )
    print(
        f"candidate_quality=REJECTED registration_ratio={q['candidate_registration_ratio']:.3f} "
        f"landmarks={q['candidate_landmarks']} observations={q['candidate_observations']}"
    )
    print(
        "matching_policy_sensitivity=True partial_sparse_recovery=True production_quality_recovery=False"
    )
    print(
        "evidence_only=True sfm_execution=False mvs_execution=False candidate_rerun=False retry_until_pass=False "
        "threshold_relaxation=False production_default_modified=False"
    )
    print(
        "pycolmap_authority=True exhaustive_universally_superior=False production_default_promotion=False "
        "metric_scale=False metrology=False hybrid=False auto=False"
    )
    print("I5.10 MP-SfM evaluation: CLOSED / PASS (BOUNDED EVIDENCE; CANDIDATE QUALITY REJECTED)")
    print("I5.11 OOM / disk / crash / recovery robustness: AUTHORIZED / NOT STARTED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.10D ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
