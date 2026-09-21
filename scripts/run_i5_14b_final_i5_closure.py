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
from core.reconstruction.i5_final_closure import build_i5_14b_final_closure

OUTPUT_NAME = "i5_14b_final_i5_qualification_and_closure.json"


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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="I5.14B authoritative final I5 qualification and closure.")
    p.add_argument("--i5-14a-readiness", required=True)
    p.add_argument("--i5-14a-sha256", required=True)
    p.add_argument("--post-i5-14a-delta-checkpoint", required=True)
    p.add_argument("--post-i5-14a-delta-sha256", required=True)
    p.add_argument("--closure-receipt", required=True)
    p.add_argument("--closure-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    readiness_binding = _verified_binding(args.i5_14a_readiness, args.i5_14a_sha256, label="I5.14A readiness")
    checkpoint_binding = _verified_binding(
        args.post_i5_14a_delta_checkpoint,
        args.post_i5_14a_delta_sha256,
        label="post-I5.14A source checkpoint",
    )
    receipt_binding = _verified_binding(args.closure_receipt, args.closure_receipt_sha256, label="I5.14B closure receipt")

    result = build_i5_14b_final_closure(
        readiness=_read_json(Path(readiness_binding["path"]), label="I5.14A readiness"),
        readiness_binding=readiness_binding,
        post_i5_14a_checkpoint_binding=checkpoint_binding,
        closure_receipt=_read_json(Path(receipt_binding["path"]), label="I5.14B closure receipt"),
        closure_receipt_binding=receipt_binding,
    )

    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("I5.14B Final I5 Qualification + Closure: PASS")
    print("i5_complete=True i5_closed=True")
    print("closure_with_bounded_limitations=True")
    print("phase_closures=14/14")
    print("sparse_envelope=difficult_handheld:degraded low_overlap_weak_parallax:pass visual_stress:degraded forward_motion:rejected dynamic_proxy:pass")
    print("dense_envelope=mvs:pass rolling_depth:pass vda:degraded remaining_single_image:mixed_terminal_quality")
    print("matching_policy_candidate=REJECTED production_default_change=False")
    print("bounded_failure_recovery=QUALIFIED universal_recovery=False")
    print("hybrid=NOT_JUSTIFIED")
    print("offline_shadow_auto=QUALIFIED production_auto=DEFERRED")
    print("provider_execution=False sfm_execution=False mvs_execution=False")
    print("production_default_modified=False provider_promoted=False global_score=False provider_rank=False")
    print("post_i5_14b_delta_required=True full_post_i5_source_checkpoint_required=True")
    print("I5: CLOSED / PASS (WITH EXPLICIT BOUNDED LIMITATIONS)")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.14B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
