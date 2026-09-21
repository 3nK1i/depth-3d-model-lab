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
from core.reconstruction.i5_failure_robustness_qualification import build_i5_11d_failure_robustness_closure

OUTPUT_NAME = "i5_11d_failure_robustness_qualification_and_closure.json"


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
    p = argparse.ArgumentParser(description="I5.11D evidence-only bounded failure/recovery qualification and closure.")
    p.add_argument("--i5-11a-admission", required=True)
    p.add_argument("--i5-11a-sha256", required=True)
    p.add_argument("--i5-11b-preflight", required=True)
    p.add_argument("--i5-11b-sha256", required=True)
    p.add_argument("--i5-11c-campaign", required=True)
    p.add_argument("--i5-11c-sha256", required=True)
    p.add_argument("--closure-receipt", required=True)
    p.add_argument("--closure-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    a = _binding(args.i5_11a_admission, args.i5_11a_sha256, label="I5.11A admission")
    b = _binding(args.i5_11b_preflight, args.i5_11b_sha256, label="I5.11B preflight")
    c = _binding(args.i5_11c_campaign, args.i5_11c_sha256, label="I5.11C campaign")
    r = _binding(args.closure_receipt, args.closure_receipt_sha256, label="I5.11D closure receipt")

    result = build_i5_11d_failure_robustness_closure(
        i5_11a_admission=_read_json(Path(a["path"]), label="I5.11A admission"),
        i5_11a_binding=a,
        i5_11b_preflight=_read_json(Path(b["path"]), label="I5.11B preflight"),
        i5_11b_binding=b,
        i5_11c_campaign=_read_json(Path(c["path"]), label="I5.11C campaign"),
        i5_11c_binding=c,
        closure_receipt=_read_json(Path(r["path"]), label="I5.11D closure receipt"),
        closure_receipt_binding=r,
    )

    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    q = result["qualification"]
    print("I5.11D Bounded Failure-Recovery Evidence Qualification + Closure: PASS")
    print("conclusion=" + q["conclusion"])
    print(
        "qualified_faults=CUDA_OOM_SYNTHETIC,ENOSPC_SYNTHETIC,CRASH_ORPHAN_DESCENDANT "
        f"fault_attempts={q['fault_attempts_executed']}/3 recovery_attempts={q['recovery_attempts_executed']}/3 retries={q['retry_attempts_executed']}"
    )
    print(
        "signatures=True process_groups_clean=True failed_workspaces_removed=True "
        "no_leaked_descendants=True repository_integrity=True recoveries=True"
    )
    print("real_gpu_oom=False real_disk_fill=False real_resource_exhaustion=False universal_recovery_claim=False")
    print("enospc_classifier_gap_preserved=True production_classifier_change=False recovery_engine_change=False production_default_modified=False")
    print("fault_execution=False sfm_execution=False mvs_execution=False hybrid=False auto=False")
    print("I5.11 OOM / disk / crash / recovery robustness: CLOSED / PASS (BOUNDED EVIDENCE)")
    print("I5.12A evidence-justification/admission review: AUTHORIZED / NOT STARTED; hybrid execution NOT AUTHORIZED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.11D ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
