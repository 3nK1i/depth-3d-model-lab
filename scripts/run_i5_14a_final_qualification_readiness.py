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
from core.reconstruction.i5_final_qualification_readiness import (
    I514A_CLOSURE_SPECS,
    build_i5_14a_final_qualification_readiness,
)

OUTPUT_NAME = "i5_14a_final_i5_qualification_synthesis_and_closure_readiness.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object.")
    return value


def _verified_binding(path_value: str | Path, expected_sha: str, *, label: str) -> dict[str, str]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"{label} does not exist: {path}")
    actual = _sha256(path)
    expected = str(expected_sha).strip().lower()
    if actual != expected:
        raise ReconstructionValidationError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual}


def _discover_unique(root: Path, filename: str, *, label: str) -> Path:
    matches = sorted(p.resolve() for p in root.rglob(filename) if p.is_file())
    if len(matches) != 1:
        raise ReconstructionValidationError(f"{label} discovery expected exactly 1 {filename}, got {len(matches)}: {matches}")
    return matches[0]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="I5.14A final I5 evidence synthesis and closure-readiness review.")
    p.add_argument("--evidence-root", type=Path, required=True)
    p.add_argument("--post-i5-13d-delta-checkpoint", required=True)
    p.add_argument("--post-i5-13d-delta-sha256", required=True)
    p.add_argument("--readiness-receipt", required=True)
    p.add_argument("--readiness-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = _parser().parse_args()
    evidence_root = args.evidence_root.expanduser().resolve()
    if not evidence_root.is_dir():
        raise ReconstructionValidationError(f"Evidence root does not exist: {evidence_root}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    closures: dict[str, dict] = {}
    bindings: dict[str, dict[str, str]] = {}
    for key, spec in I514A_CLOSURE_SPECS.items():
        path = _discover_unique(evidence_root, spec["filename"], label=key)
        closures[key] = _read_json(path, label=key)
        bindings[key] = {"path": str(path), "sha256": _sha256(path)}

    checkpoint_binding = _verified_binding(
        args.post_i5_13d_delta_checkpoint,
        args.post_i5_13d_delta_sha256,
        label="post-I5.13D source checkpoint",
    )
    receipt_binding = _verified_binding(
        args.readiness_receipt,
        args.readiness_receipt_sha256,
        label="I5.14A readiness receipt",
    )
    result = build_i5_14a_final_qualification_readiness(
        closures=closures,
        closure_bindings=bindings,
        post_i5_13d_checkpoint_binding=checkpoint_binding,
        readiness_receipt=_read_json(Path(receipt_binding["path"]), label="I5.14A readiness receipt"),
        readiness_receipt_binding=receipt_binding,
    )
    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("I5.14A Final I5 Qualification Synthesis + Closure Readiness: PASS")
    print(f"closure_artifacts={result['qualification']['closure_artifact_count']}")
    print("closure_ready=True")
    print("sparse_envelope=difficult_handheld:degraded low_overlap_weak_parallax:pass visual_stress:degraded forward_motion:rejected dynamic_proxy:pass")
    print("dense_envelope=mvs:pass rolling_depth:pass vda:degraded remaining_single_image:mixed_terminal_quality")
    print("matching_policy=partial_sparse_recovery_candidate_rejected production_default_change=False")
    print("bounded_failure_recovery=qualified universal_recovery=False")
    print("hybrid=NOT_JUSTIFIED")
    print("offline_shadow_auto=QUALIFIED production_auto=DEFERRED")
    print("provider_execution=False sfm_execution=False mvs_execution=False")
    print("production_default_modified=False provider_promoted=False global_score=False provider_rank=False")
    print("I5.14B final I5 closure + immutable handoff: AUTHORIZED / NOT STARTED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.14A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
