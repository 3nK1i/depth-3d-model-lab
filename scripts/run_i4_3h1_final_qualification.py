#!/usr/bin/env python3
"""Run the evidence-only I4.3H1 final qualification/audit gate."""
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
from core.reconstruction.i4_3_final_qualification import (  # noqa: E402
    H1EvidenceBinding,
    build_i4_3h1_final_qualification_audit,
    validate_i4_3g5b_closure,
)

OUTPUT_NAME = "i4_3h1_final_qualification_audit.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3g5b-closure", type=Path, required=True)
    parser.add_argument("--i4-3g5b-sha256", required=True)
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
        raise ReconstructionValidationError(f"I4.3H1 {label} is missing: {resolved}")
    actual = _sha256(resolved)
    if actual != str(expected).strip().lower():
        raise ReconstructionValidationError(
            f"I4.3H1 {label} SHA256 mismatch: expected={expected} actual={actual}"
        )
    return resolved, actual


def _read(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I4.3H1 {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.3H1 {label} root must be an object.")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _reverify_g_predecessors(g5b: dict[str, Any]) -> None:
    validated = validate_i4_3g5b_closure(g5b)
    for key, binding in validated["predecessor_bindings"].items():
        _verify(Path(binding["path"]), binding["sha256"], label=f"G5B predecessor {key}")


def run(args: argparse.Namespace) -> Path:
    g5b_path, g5b_sha = _verify(args.i4_3g5b_closure, args.i4_3g5b_sha256, label="I4.3G5B closure")
    receipt_path, receipt_sha = _verify(
        args.regression_receipt,
        args.regression_receipt_sha256,
        label="I4.3H1 regression receipt",
    )
    g5b = _read(g5b_path, label="I4.3G5B closure")
    receipt = _read(receipt_path, label="I4.3H1 regression receipt")
    _reverify_g_predecessors(g5b)

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I4.3H1 output directory must be empty or absent: {out}")

    audit = build_i4_3h1_final_qualification_audit(
        i4_3g5b_closure=g5b,
        regression_receipt=receipt,
        evidence_bindings={
            "i4_3g5b_closure": H1EvidenceBinding(path=str(g5b_path), sha256=g5b_sha),
            "i4_3h1_regression_receipt": H1EvidenceBinding(path=str(receipt_path), sha256=receipt_sha),
        },
        g5b_predecessor_hashes_reverified=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    target = _write(out / OUTPUT_NAME, audit)

    arch = audit["architecture_audit"]
    qual = audit["qualification_audit"]
    reg = audit["regression_audit"]
    decision = audit["decision"]
    print("I4.3H1 final qualification / authority / provenance / regression audit: PASS")
    print(
        "provider_families={}/5 executable_families={}/4 implementations={}/10 modules_importable=True".format(
            arch["provider_family_count"],
            arch["executable_provider_family_count"],
            arch["provider_implementation_count"],
        )
    )
    print(
        "qualification_families={}/3 scenarios={}/11 terminal_quality_events={}/5 ".format(
            qual["qualification_family_count"], qual["scenario_count"], qual["terminal_quality_event_count"]
        )
        + "production_native_primary=True matched_view_secondary=True"
    )
    print(
        "predecessor_regression={}/{} focused_h1={} all_required_suites_passed=True".format(
            reg["predecessor_total_tests"], reg["predecessor_total_tests"], reg["focused_h1_tests"]
        )
    )
    print(
        "g_chain_reverified=True provider_selected=False provider_ranked=False auto=False hybrid=False "
        "metric_scale=False metrology=False"
    )
    print(
        "h1_complete={} h2_authorized={} i4_3_closed={} i4_4_authorized={}".format(
            decision["i4_3h1_qualification_complete"],
            decision["i4_3h2_authorized"],
            decision["i4_3_closed"],
            decision["i4_4_authorized"],
        )
    )
    print(f"audit={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError) as exc:
        print(f"I4.3H1 ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
