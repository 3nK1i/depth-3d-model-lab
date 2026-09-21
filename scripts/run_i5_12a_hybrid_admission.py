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
from core.reconstruction.i5_hybrid_admission import build_i5_12a_hybrid_admission_review

OUTPUT_NAME = "i5_12a_hybrid_evidence_justification_and_admission.json"
I43A_DOC = ROOT / "docs/phase_history/I4_3A_CANONICAL_DENSE_PROVIDER_ARCHITECTURE_FOUNDATION.md"


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


def _binding(path_value: str | Path, expected_sha: str, *, label: str) -> dict[str, str]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"{label} does not exist: {path}")
    actual = _sha256(path)
    expected = str(expected_sha).strip().lower()
    if actual != expected:
        raise ReconstructionValidationError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="I5.12A evidence-only hybrid justification/admission review.")
    p.add_argument("--i5-11d-closure", required=True)
    p.add_argument("--i5-11d-sha256", required=True)
    p.add_argument("--i5-6c-closure", required=True)
    p.add_argument("--i5-6c-sha256", required=True)
    p.add_argument("--i5-2a-admission", required=True)
    p.add_argument("--i5-2a-sha256", required=True)
    p.add_argument("--admission-receipt", required=True)
    p.add_argument("--admission-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    d = _binding(args.i5_11d_closure, args.i5_11d_sha256, label="I5.11D closure")
    m = _binding(args.i5_6c_closure, args.i5_6c_sha256, label="I5.6C closure")
    a = _binding(args.i5_2a_admission, args.i5_2a_sha256, label="I5.2A admission")
    r = _binding(args.admission_receipt, args.admission_receipt_sha256, label="I5.12A admission receipt")

    i52a = _read_json(Path(a["path"]), label="I5.2A admission")
    bindings = i52a.get("evidence_bindings")
    if not isinstance(bindings, dict) or not isinstance(bindings.get("i4_3g2_final_closure"), dict):
        raise ReconstructionValidationError("I5.2A does not contain the frozen I4.3G2 closure binding.")
    g2_bound = bindings["i4_3g2_final_closure"]
    g2 = _binding(g2_bound.get("path", ""), g2_bound.get("sha256", ""), label="I4.3G2 final closure")

    i43a = {"path": str(I43A_DOC.resolve()), "sha256": _sha256(I43A_DOC)}
    result = build_i5_12a_hybrid_admission_review(
        i5_11d_closure=_read_json(Path(d["path"]), label="I5.11D closure"),
        i5_11d_binding=d,
        i5_6c_closure=_read_json(Path(m["path"]), label="I5.6C closure"),
        i5_6c_binding=m,
        i5_2a_admission=i52a,
        i5_2a_binding=a,
        i4_3g2_closure=_read_json(Path(g2["path"]), label="I4.3G2 final closure"),
        i4_3g2_binding=g2,
        i4_3a_phase_history_text=I43A_DOC.read_text(encoding="utf-8"),
        i4_3a_binding=i43a,
        admission_receipt=_read_json(Path(r["path"]), label="I5.12A admission receipt"),
        admission_receipt_binding=r,
    )

    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    evidence = result["direct_cross_provider_evidence"]
    decision = result["decision"]
    print("I5.12A Hybrid Evidence-Justification + Admission Review: PASS")
    print("quality_status=" + result["quality_status"])
    print("conclusion=" + result["conclusion"])
    print(
        "scenario=g2_weak_overlap_mvs_da3 "
        f"mvs_cleaner={evidence['mvs_cleaner_observed']} "
        f"ai_legitimate_hole_recovery={evidence['ai_legitimate_hole_recovery_observed']} "
        f"da3_global_completeness={evidence['global_completeness_da3_observed']}"
    )
    print("global_completeness_alone_insufficient=True")
    print("provider_execution=False sfm_execution=False mvs_execution=False da3_execution=False hybrid_execution=False")
    print("pycolmap_authority=True provider_selected=False production_default_modified=False auto=False metric_scale=False metrology=False")
    if decision["hybrid_experiment_justified"]:
        print("I5.12B hybrid integration/runtime/evidence preflight: AUTHORIZED / NOT STARTED")
        print("I5.12C hybrid execution: NOT AUTHORIZED UNTIL B FREEZE")
    else:
        print("I5.12 Hybrid experiment: CLOSED / NOT JUSTIFIED BY DIRECT EVIDENCE")
        print("I5.13 first evidence-driven Auto policy: AUTHORIZED / NOT STARTED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.12A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
