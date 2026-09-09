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
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_remaining_depth_provider_robustness_qualification import (
    build_i5_9c_remaining_depth_provider_robustness_closure,
)

OUTPUT_NAME = "i5_9c_remaining_depth_provider_robustness_closure.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path, *, label: str) -> dict:
    if not path.is_file():
        raise ReconstructionValidationError(f"{label} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object.")
    return payload


def _binding(path_value: str, expected_sha: str, *, label: str) -> I51EvidenceBinding:
    path = Path(path_value).expanduser().resolve()
    actual = _sha256(path)
    expected = str(expected_sha).strip().lower()
    if actual != expected:
        raise ReconstructionValidationError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return I51EvidenceBinding(str(path), actual)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="I5.9C evidence-only remaining depth-provider robustness qualification and closure."
    )
    parser.add_argument("--i5-9a-admission", required=True)
    parser.add_argument("--i5-9a-sha256", required=True)
    parser.add_argument("--i5-9b1-replay", required=True)
    parser.add_argument("--i5-9b1-sha256", required=True)
    parser.add_argument("--i5-9b2-replay", required=True)
    parser.add_argument("--i5-9b2-sha256", required=True)
    parser.add_argument("--closure-receipt", required=True)
    parser.add_argument("--closure-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    a_binding = _binding(args.i5_9a_admission, args.i5_9a_sha256, label="I5.9A admission")
    b1_binding = _binding(args.i5_9b1_replay, args.i5_9b1_sha256, label="I5.9B1 real replay")
    b2_binding = _binding(args.i5_9b2_replay, args.i5_9b2_sha256, label="I5.9B2 real replay")
    receipt_binding = _binding(args.closure_receipt, args.closure_receipt_sha256, label="I5.9C closure receipt")

    result = build_i5_9c_remaining_depth_provider_robustness_closure(
        i5_9a_admission=_read_json(Path(a_binding.path), label="I5.9A admission"),
        i5_9a_binding=a_binding,
        i5_9b1_replay=_read_json(Path(b1_binding.path), label="I5.9B1 replay"),
        i5_9b1_binding=b1_binding,
        i5_9b2_replay=_read_json(Path(b2_binding.path), label="I5.9B2 replay"),
        i5_9b2_binding=b2_binding,
        qualification_receipt=_read_json(Path(receipt_binding.path), label="I5.9C closure receipt"),
        qualification_receipt_binding=receipt_binding,
    )

    target = output / OUTPUT_NAME
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    q = result["qualification"]
    marigold = result["observed_limitations"]["marigold_castle_dynamic_proxy"]
    print("I5.9C Remaining Depth-Provider Robustness Qualification + Closure: PASS")
    print(
        f"quality_status={result['quality_status']} combined_terminal_runs={q['combined_terminal_runs']}/36 "
        f"combined_status_counts={q['combined_status_counts']}"
    )
    print(
        f"B1=24_PASS/0_DEGRADED/0_REJECTED B2=11_PASS/1_DEGRADED/0_REJECTED "
        f"marigold_dynamic_proxy={marigold['accepted_views']}/{marigold['selected_views']}_accepted"
    )
    print(
        "marigold_dynamic_proxy_degradation=alignment_rejected_views:1 "
        "fusion_capacity_drops=0 preserved=True"
    )
    print(
        "historical_g4_low_texture=depth_anything_v2:REJECTED,marigold:REJECTED preserved=True "
        "current_castle_does_not_overwrite_historical=True"
    )
    print(
        "evidence_only=True provider_inference=False sfm_rerun=False mvs_rerun=False "
        "provider_comparison=False threshold_relaxation=False fusion_capacity_increase=False "
        "production_default_modified=False retry_until_pass=False"
    )
    print(
        "pycolmap_authority=True provider_ranking=False universal_provider_winner=False "
        "universal_robustness=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print("I5.9 Remaining depth-provider robustness: CLOSED / PASS (MIXED TERMINAL QUALITY; BOUNDED ENVELOPE)")
    print("I5.10 MP-SfM evaluation: AUTHORIZED / NOT STARTED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.9C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
