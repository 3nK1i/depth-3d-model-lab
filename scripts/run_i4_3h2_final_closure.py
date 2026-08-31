#!/usr/bin/env python3
"""Verify frozen I4.3H1 evidence and formally close I4.3."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.i4_3_final_closure import (  # noqa: E402
    H2EvidenceBinding,
    build_i4_3h2_final_closure,
    validate_i4_3h1_qualification_audit,
)
from core.reconstruction.i4_3_final_qualification import (  # noqa: E402
    audit_i4_3_provider_architecture,
    validate_i4_3g5b_closure,
    validate_i4_3h1_regression_receipt,
)

OUTPUT_NAME = "i4_3h2_final_i4_3_closure.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3h1-audit", type=Path, required=True)
    parser.add_argument("--i4-3h1-sha256", required=True)
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
        raise ReconstructionValidationError(f"I4.3H2 {label} is missing: {resolved}")
    actual = _sha256(resolved)
    if actual != str(expected).strip().lower():
        raise ReconstructionValidationError(
            f"I4.3H2 {label} SHA256 mismatch: expected={expected} actual={actual}"
        )
    return resolved, actual


def _read(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I4.3H2 {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.3H2 {label} root must be an object.")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _verify_binding(binding: dict[str, str], *, label: str) -> tuple[Path, str]:
    return _verify(Path(binding["path"]), binding["sha256"], label=label)


def _base_architecture_from_h1(architecture: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(architecture)
    inventory = normalized.get("provider_inventory")
    if isinstance(inventory, dict):
        for item in inventory.values():
            if isinstance(item, dict):
                item.pop("terminal_quality_events", None)
                item.pop("no_terminal_quality_event_implies_universal_robustness", None)
    return normalized


def run(args: argparse.Namespace) -> Path:
    h1_path, h1_sha = _verify(args.i4_3h1_audit, args.i4_3h1_sha256, label="I4.3H1 audit")
    h1 = _read(h1_path, label="I4.3H1 audit")
    validated_h1 = validate_i4_3h1_qualification_audit(h1)

    # Rehash and semantically validate both artifacts H1 directly bound.
    g5b_path, _ = _verify_binding(
        validated_h1["evidence_bindings"]["i4_3g5b_closure"], label="H1 bound artifact i4_3g5b_closure"
    )
    receipt_path, _ = _verify_binding(
        validated_h1["evidence_bindings"]["i4_3h1_regression_receipt"],
        label="H1 bound artifact i4_3h1_regression_receipt",
    )
    g5b = _read(g5b_path, label="H1 bound I4.3G5B closure")
    receipt = _read(receipt_path, label="H1 bound H1 regression receipt")
    validated_g5b = validate_i4_3g5b_closure(g5b)
    validated_receipt = validate_i4_3h1_regression_receipt(receipt)

    # H1 must carry the same G5B predecessor bindings as the actual G5B file.
    if validated_g5b["predecessor_bindings"] != validated_h1["g5b_predecessor_bindings"]:
        raise ReconstructionValidationError("I4.3H2 H1/G5B predecessor evidence bindings disagree.")
    for key, binding in validated_h1["g5b_predecessor_bindings"].items():
        _verify_binding(binding, label=f"G5B predecessor {key}")

    reg = h1["regression_audit"]
    counts = validated_receipt["counts"]
    if (
        reg["focused_h1_tests"] != counts["focused_h1_tests"]
        or reg["predecessor_unit_tests"] != counts["predecessor_unit_tests"]
        or reg["predecessor_integration_tests"] != counts["predecessor_integration_tests"]
        or reg["predecessor_stress_tests"] != counts["predecessor_stress_tests"]
        or reg["predecessor_total_tests"] != counts["predecessor_total_tests"]
    ):
        raise ReconstructionValidationError("I4.3H2 H1 regression summary disagrees with its bound receipt.")

    # H2 adds no provider code. Re-audit the current repository contract surface
    # and require it to match the architecture H1 actually qualified.
    current_architecture = audit_i4_3_provider_architecture()
    frozen_architecture = _base_architecture_from_h1(validated_h1["architecture_audit"])
    if current_architecture != frozen_architecture:
        raise ReconstructionValidationError("I4.3H2 current repository provider architecture drifted after H1.")

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I4.3H2 output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    closure = build_i4_3h2_final_closure(
        i4_3h1_audit=h1,
        evidence_binding=H2EvidenceBinding(path=str(h1_path), sha256=h1_sha),
        h1_bound_artifacts_reverified=True,
        g5b_predecessor_artifacts_reverified=True,
        current_architecture_matches_h1=True,
    )
    target = _write(out / OUTPUT_NAME, closure)

    coverage = closure["coverage"]
    regression = closure["regression_summary"]
    decision = closure["decision"]
    print("I4.3 Production Dense-Provider Architecture: FORMALLY CLOSED")
    print(
        "provider_families={}/5 executable_families={}/4 providers={}/10 qualification_scenarios={}/11".format(
            coverage["provider_family_count"],
            coverage["executable_provider_family_count"],
            coverage["provider_implementation_count"],
            coverage["qualification_scenario_count"],
        )
    )
    print(
        "terminal_quality_events={}/5 predecessor_regression={}/668 focused_h1={} architecture_match=True".format(
            coverage["terminal_quality_event_count"],
            regression["predecessor_total_tests"],
            regression["focused_h1_tests"],
        )
    )
    print(
        "provider_selected=False provider_ranked=False universal_winner=False auto=False hybrid=False "
        "metric_scale=False metrology=False"
    )
    print(
        "i4_3_closed={} i4_4_authorized={} i4_4_work_started=False".format(
            decision["i4_3_closed"], decision["i4_4_authorized"]
        )
    )
    print(f"closure={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I4.3H2 ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
