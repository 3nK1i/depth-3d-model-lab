#!/usr/bin/env python3
"""Build the I4.3G5A evidence-only cross-provider synthesis."""
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

from core.reconstruction.cross_provider_evidence_synthesis import (  # noqa: E402
    G5AEvidenceBinding,
    build_i4_3g5a_cross_provider_synthesis,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402

OUTPUT_NAME = "i4_3g5a_cross_provider_synthesis.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3g1-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3g1-sha256", required=True)
    parser.add_argument("--i4-3g2-closure", type=Path, required=True)
    parser.add_argument("--i4-3g2-sha256", required=True)
    parser.add_argument("--i4-3g3c-closure", type=Path, required=True)
    parser.add_argument("--i4-3g3c-sha256", required=True)
    parser.add_argument("--i4-3g4c-closure", type=Path, required=True)
    parser.add_argument("--i4-3g4c-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, *, label: str) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"I4.3G5A {label} is missing: {path}")
    actual = _sha256(path)
    if actual != str(expected).strip().lower():
        raise ReconstructionValidationError(
            f"I4.3G5A {label} SHA256 mismatch: expected={expected} actual={actual}"
        )
    return actual


def _read(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I4.3G5A {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.3G5A {label} root must be an object.")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> Path:
    inputs = {
        "i4_3g1_snapshot": (args.i4_3g1_snapshot.expanduser().resolve(), args.i4_3g1_sha256, "I4.3G1 snapshot"),
        "i4_3g2_closure": (args.i4_3g2_closure.expanduser().resolve(), args.i4_3g2_sha256, "I4.3G2 closure"),
        "i4_3g3c_closure": (args.i4_3g3c_closure.expanduser().resolve(), args.i4_3g3c_sha256, "I4.3G3C closure"),
        "i4_3g4c_closure": (args.i4_3g4c_closure.expanduser().resolve(), args.i4_3g4c_sha256, "I4.3G4C closure"),
    }
    verified: dict[str, tuple[Path, str]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for key, (path, expected, label) in inputs.items():
        digest = _verify(path, expected, label=label)
        verified[key] = (path, digest)
        payloads[key] = _read(path, label=label)

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I4.3G5A output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    synthesis = build_i4_3g5a_cross_provider_synthesis(
        i4_3g1_snapshot=payloads["i4_3g1_snapshot"],
        i4_3g2_closure=payloads["i4_3g2_closure"],
        i4_3g3c_closure=payloads["i4_3g3c_closure"],
        i4_3g4c_closure=payloads["i4_3g4c_closure"],
        evidence_bindings={
            key: G5AEvidenceBinding(path=str(path), sha256=digest)
            for key, (path, digest) in verified.items()
        },
    )
    target = _write(out / OUTPUT_NAME, synthesis)

    coverage = synthesis["coverage"]
    decision = synthesis["decision"]
    print("I4.3G5A cross-provider evidence synthesis: PASS")
    print(
        "families={families}/3 scenarios={scenarios}/11 providers={providers}/10 all_family_closures_bound=True".format(
            families=coverage["qualification_family_count"],
            scenarios=coverage["scenario_count"],
            providers=coverage["provider_implementation_count"],
        )
    )
    print(
        "production_native_primary=True matched_view_secondary=True global_scalar_score=False "
        "provider_ranking=False universal_winner=False"
    )
    print(
        "g5a_complete={complete} g5b_authorized={authorized} i4_3g_closed={closed} "
        "provider_selected=False auto=False hybrid=False metric_scale=False metrology=False".format(
            complete=decision["g5a_cross_provider_synthesis_complete"],
            authorized=decision["i4_3g5b_authorized"],
            closed=decision["i4_3g_closed"],
        )
    )
    print(f"synthesis={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError) as exc:
        print(f"I4.3G5A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
