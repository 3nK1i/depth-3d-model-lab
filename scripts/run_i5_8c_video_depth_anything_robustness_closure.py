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
from core.reconstruction.i5_video_depth_anything_robustness_qualification import (
    build_i5_8c_video_depth_anything_robustness_closure,
)

OUTPUT_NAME = "i5_8c_video_depth_anything_robustness_closure.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        description="I5.8C evidence-only Video Depth Anything robustness qualification and closure."
    )
    parser.add_argument("--i5-8a-admission", required=True)
    parser.add_argument("--i5-8a-sha256", required=True)
    parser.add_argument("--i5-8b-replay", required=True)
    parser.add_argument("--i5-8b-sha256", required=True)
    parser.add_argument("--closure-receipt", required=True)
    parser.add_argument("--closure-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    a_binding = _binding(args.i5_8a_admission, args.i5_8a_sha256, label="I5.8A admission")
    b_binding = _binding(args.i5_8b_replay, args.i5_8b_sha256, label="I5.8B real replay")
    receipt_binding = _binding(args.closure_receipt, args.closure_receipt_sha256, label="I5.8C closure receipt")

    result = build_i5_8c_video_depth_anything_robustness_closure(
        i5_8a_result=_read_json(Path(a_binding.path), label="I5.8A admission"),
        i5_8a_binding=a_binding,
        i5_8b_result=_read_json(Path(b_binding.path), label="I5.8B replay"),
        i5_8b_binding=b_binding,
        qualification_receipt=_read_json(Path(receipt_binding.path), label="I5.8C closure receipt"),
        qualification_receipt_binding=receipt_binding,
    )
    target = output / OUTPUT_NAME
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    q = result["qualification"]
    limitations = result["limitations"]
    runtime = result["runtime_identity"]
    capacity = result["fusion_capacity_observations"]
    print("I5.8C Video Depth Anything Robustness Qualification + Closure: PASS")
    print(
        f"quality_status={result['quality_status']} operating_envelope={q['operating_envelope']} "
        f"terminal_scenarios={q['current_replay_terminal_scenario_count']}/{q['admitted_current_replay_scenario_count']}"
    )
    print(
        "current_alignment=100%_accepted_all_admitted_registered_frames "
        "current_quality=DEGRADED fusion_capacity_limitation_preserved=True"
    )
    print(
        "historical_g3_static_sweep=PASS historical_g3_motion=DEGRADED "
        "historical_g3_low_texture=PASS preserved=True"
    )
    print(
        f"runtime_profile={runtime['resolved_profile']} encoder={runtime['resolved_encoder']} "
        f"device={runtime['device']} precision={runtime['precision']} production_max_voxels={runtime['production_max_voxels']}"
    )
    print(
        "fusion_capacity_drops=" + ",".join(f"{sid}:{capacity[sid]}" for sid in sorted(capacity))
    )
    print(
        f"current_performance_observations_complete={limitations['current_six_scenario_performance_observations_complete']} "
        f"runtime_performance_envelope_complete={limitations['runtime_performance_envelope_complete']} "
        f"resource_envelope_complete={limitations['resource_envelope_complete']}"
    )
    print(
        "evidence_only=True video_depth_anything_inference=False sfm_rerun=False mvs_rerun=False "
        "provider_comparison=False threshold_relaxation=False fusion_capacity_increase=False production_default_modified=False"
    )
    print(
        "pycolmap_authority=True provider_ranking=False universal_provider_winner=False "
        "universal_vda_robustness=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print("I5.8 Video Depth Anything robustness: CLOSED / PASS (QUALITY DEGRADED; CAPACITY-BOUNDED ENVELOPE)")
    print("I5.9 Remaining depth-provider robustness: AUTHORIZED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.8C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
