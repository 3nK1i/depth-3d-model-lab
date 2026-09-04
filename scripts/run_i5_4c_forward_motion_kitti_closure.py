#!/usr/bin/env python3
"""Build the evidence-only I5.4C forward-motion/KITTI qualification and closure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding  # noqa: E402
from core.reconstruction.i5_forward_motion_kitti import (  # noqa: E402
    build_i5_4c_forward_motion_kitti_closure,
    validate_i5_4b_for_i5_4c,
    validate_i5_4c_receipt,
)

OUTPUT_NAME = "i5_4c_forward_motion_kitti_closure.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"Expected JSON object: {path}")
    return value


def _binding(path: Path, expected_sha: str, *, label: str) -> I51EvidenceBinding:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"{label} file missing: {resolved}")
    expected = str(expected_sha).strip().lower()
    actual = _sha256(resolved)
    if actual != expected:
        raise ReconstructionValidationError(
            f"{label} SHA256 mismatch: expected={expected} actual={actual} path={resolved}"
        )
    return I51EvidenceBinding(str(resolved), actual)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i5-4b-replay", type=Path, required=True)
    parser.add_argument("--i5-4b-sha256", required=True)
    parser.add_argument("--closure-receipt", type=Path, required=True)
    parser.add_argument("--closure-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    replay_binding = _binding(args.i5_4b_replay, args.i5_4b_sha256, label="I5.4B real replay")
    receipt_binding = _binding(args.closure_receipt, args.closure_receipt_sha256, label="I5.4C closure receipt")
    replay = validate_i5_4b_for_i5_4c(_read_json(Path(replay_binding.path)))
    receipt = validate_i5_4c_receipt(_read_json(Path(receipt_binding.path)))

    result = build_i5_4c_forward_motion_kitti_closure(
        i5_4b_result=replay,
        i5_4b_binding=replay_binding,
        closure_receipt=receipt,
        closure_receipt_binding=receipt_binding,
    )
    evidence = _write_json(output / OUTPUT_NAME, result)

    q = result["qualification"]
    print("I5.4C Forward-Motion / KITTI Robustness Qualification + Closure: PASS")
    print(f"quality_status={q['quality_status']}")
    print(f"operating_envelope={q['operating_envelope']}")
    print(f"primary_bottleneck={q['primary_bottleneck']}")
    print(
        "registration="
        f"{q['registered_images']}/{q['input_images']} "
        f"mapping_produced_sparse_model={q['mapping_produced_sparse_model']}"
    )
    print(
        "forward_motion_geometry_revalidation_status="
        f"{q['forward_motion_geometry_revalidation_status']} "
        f"authoritative_forward_motion_geometry_proven={q['authoritative_forward_motion_geometry_proven']}"
    )
    print(
        "evidence_only=True sfm_rerun=False window_retuned=False threshold_relaxation=False "
        "production_default_modified=False mp_sfm=False auto=False hybrid=False metric_scale=False"
    )
    print("I5.4 CLOSED / REJECTED OPERATING ENVELOPE")
    print("I5.5 dynamic-scene handling robustness: AUTHORIZED")
    print(f"evidence={evidence}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.4C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
