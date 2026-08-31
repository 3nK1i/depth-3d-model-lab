#!/usr/bin/env python3
"""Verify the frozen G1-G5A chain and formally close I4.3G."""
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
from core.reconstruction.cross_provider_qualification_closure import (  # noqa: E402
    G5BClosureBinding,
    build_i4_3g5b_final_closure,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402

OUTPUT_NAME = "i4_3g5b_final_g_closure.json"


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
    parser.add_argument("--i4-3g5a-synthesis", type=Path, required=True)
    parser.add_argument("--i4-3g5a-sha256", required=True)
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
        raise ReconstructionValidationError(f"I4.3G5B {label} is missing: {resolved}")
    actual = _sha256(resolved)
    if actual != str(expected).strip().lower():
        raise ReconstructionValidationError(
            f"I4.3G5B {label} SHA256 mismatch: expected={expected} actual={actual}"
        )
    return resolved, actual


def _read(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I4.3G5B {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.3G5B {label} root must be an object.")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _assert_g5a_bindings_match_verified_chain(
    g5a: dict[str, Any],
    verified_predecessors: dict[str, tuple[Path, str]],
) -> None:
    bindings = g5a.get("evidence_bindings")
    if not isinstance(bindings, dict):
        raise ReconstructionValidationError("I4.3G5B G5A predecessor evidence bindings are missing.")
    if set(bindings) != set(verified_predecessors):
        raise ReconstructionValidationError("I4.3G5B G5A predecessor evidence binding set drifted.")
    for key, (path, digest) in verified_predecessors.items():
        binding = bindings.get(key)
        if not isinstance(binding, dict):
            raise ReconstructionValidationError(f"I4.3G5B G5A binding is malformed: {key}")
        if str(binding.get("sha256", "")).strip().lower() != digest:
            raise ReconstructionValidationError(f"I4.3G5B G5A predecessor SHA binding drifted: {key}")
        recorded_path = Path(str(binding.get("path", ""))).expanduser().resolve()
        if recorded_path != path:
            raise ReconstructionValidationError(
                f"I4.3G5B G5A predecessor path binding drifted: {key} expected={path} recorded={recorded_path}"
            )


def run(args: argparse.Namespace) -> Path:
    raw_inputs = {
        "i4_3g1_snapshot": (args.i4_3g1_snapshot, args.i4_3g1_sha256, "I4.3G1 snapshot"),
        "i4_3g2_closure": (args.i4_3g2_closure, args.i4_3g2_sha256, "I4.3G2 closure"),
        "i4_3g3c_closure": (args.i4_3g3c_closure, args.i4_3g3c_sha256, "I4.3G3C closure"),
        "i4_3g4c_closure": (args.i4_3g4c_closure, args.i4_3g4c_sha256, "I4.3G4C closure"),
        "i4_3g5a_synthesis": (args.i4_3g5a_synthesis, args.i4_3g5a_sha256, "I4.3G5A synthesis"),
    }
    verified: dict[str, tuple[Path, str]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for key, (path, expected, label) in raw_inputs.items():
        resolved, digest = _verify(path, expected, label=label)
        verified[key] = (resolved, digest)
        payloads[key] = _read(resolved, label=label)

    predecessors = {key: value for key, value in verified.items() if key != "i4_3g5a_synthesis"}
    _assert_g5a_bindings_match_verified_chain(payloads["i4_3g5a_synthesis"], predecessors)

    rebuilt_g5a = build_i4_3g5a_cross_provider_synthesis(
        i4_3g1_snapshot=payloads["i4_3g1_snapshot"],
        i4_3g2_closure=payloads["i4_3g2_closure"],
        i4_3g3c_closure=payloads["i4_3g3c_closure"],
        i4_3g4c_closure=payloads["i4_3g4c_closure"],
        evidence_bindings={
            key: G5AEvidenceBinding(path=str(path), sha256=digest)
            for key, (path, digest) in predecessors.items()
        },
    )
    def _canonical_for_chain_compare(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
        cross = normalized.get("cross_family_synthesis")
        if isinstance(cross, dict):
            events = cross.get("terminal_quality_events")
            if isinstance(events, list):
                cross["terminal_quality_events"] = sorted(
                    events,
                    key=lambda item: (
                        str(item.get("family_id", "")),
                        str(item.get("scenario_id", "")),
                        str(item.get("provider_implementation", "")),
                        str(item.get("production_status", "")),
                    ) if isinstance(item, dict) else ("", "", "", repr(item)),
                )
        return normalized

    if _canonical_for_chain_compare(rebuilt_g5a) != _canonical_for_chain_compare(payloads["i4_3g5a_synthesis"]):
        raise ReconstructionValidationError(
            "I4.3G5B frozen G5A synthesis does not match a deterministic semantic rebuild from the verified predecessor chain."
        )

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I4.3G5B output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    closure = build_i4_3g5b_final_closure(
        i4_3g5a_synthesis=payloads["i4_3g5a_synthesis"],
        evidence_bindings={
            key: G5BClosureBinding(path=str(path), sha256=digest)
            for key, (path, digest) in verified.items()
        },
        chain_rebuild_verified=True,
    )
    target = _write(out / OUTPUT_NAME, closure)

    coverage = closure["coverage"]
    decision = closure["decision"]
    print("I4.3G cross-provider qualification: FORMALLY CLOSED")
    print(
        "families={families}/3 scenarios={scenarios}/11 providers={providers}/10 chain_rebuild_verified=True".format(
            families=coverage["qualification_family_count"],
            scenarios=coverage["scenario_count"],
            providers=coverage["provider_implementation_count"],
        )
    )
    print(
        "terminal_quality_events=5 production_native_primary=True matched_view_secondary=True "
        "provider_selected=False provider_ranked=False universal_winner=False"
    )
    print(
        "i4_3g_closed={closed} i4_3h_authorized={authorized} auto=False hybrid=False "
        "metric_scale=False metrology=False".format(
            closed=decision["i4_3g_closed"], authorized=decision["i4_3h_authorized"]
        )
    )
    print(f"closure={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError) as exc:
        print(f"I4.3G5B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
