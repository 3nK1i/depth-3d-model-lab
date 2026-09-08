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
from core.reconstruction.i5_rolling_depth_robustness_qualification import (
    build_i5_7c_rolling_depth_robustness_closure,
)

OUTPUT_NAME = "i5_7c_rolling_depth_robustness_closure.json"


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
        raise ReconstructionValidationError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return I51EvidenceBinding(str(path), actual)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="I5.7C evidence-only RollingDepth robustness qualification and closure.")
    parser.add_argument("--i5-7a-admission", required=True)
    parser.add_argument("--i5-7a-sha256", required=True)
    parser.add_argument("--i5-7b-replay", required=True)
    parser.add_argument("--i5-7b-sha256", required=True)
    parser.add_argument("--closure-receipt", required=True)
    parser.add_argument("--closure-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    a_binding = _binding(args.i5_7a_admission, args.i5_7a_sha256, label="I5.7A admission")
    b_binding = _binding(args.i5_7b_replay, args.i5_7b_sha256, label="I5.7B real replay")
    receipt_binding = _binding(args.closure_receipt, args.closure_receipt_sha256, label="I5.7C closure receipt")

    result = build_i5_7c_rolling_depth_robustness_closure(
        i5_7a_result=_read_json(Path(a_binding.path), label="I5.7A admission"),
        i5_7a_binding=a_binding,
        i5_7b_result=_read_json(Path(b_binding.path), label="I5.7B replay"),
        i5_7b_binding=b_binding,
        qualification_receipt=_read_json(Path(receipt_binding.path), label="I5.7C closure receipt"),
        qualification_receipt_binding=receipt_binding,
    )
    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    q = result["qualification"]
    limitations = result["limitations"]
    recovery = result["recovery_provenance"]
    print("I5.7C RollingDepth Robustness Qualification + Closure: PASS")
    print(
        f"quality_status={q['quality_status']} operating_envelope={q['operating_envelope']} "
        f"runtime_pass_scenarios={q['runtime_pass_scenario_count']}/{q['admitted_current_replay_scenario_count']}"
    )
    print(
        "historical_g3_static_sweep=DEGRADED historical_g3_motion=REJECTED "
        "historical_g3_low_texture=PASS preserved=True"
    )
    print(
        f"castle_control_recovered=True provider_execution_rerun={recovery['provider_execution_rerun']} "
        f"manifest_verified={recovery['manifest_verified']} console_log_verified={recovery['console_log_verified']}"
    )
    print(
        f"recovered_control_wall_time_unavailable={limitations['recovered_control_wall_time_unavailable']} "
        f"recovered_control_resource_metrics_unavailable={limitations['recovered_control_resource_metrics_unavailable']}"
    )
    print(
        f"runtime_performance_envelope_complete={limitations['runtime_performance_envelope_complete']} "
        f"resource_envelope_complete={limitations['resource_envelope_complete']}"
    )
    print(
        "evidence_only=True rolling_depth_inference=False sfm_rerun=False mvs_rerun=False "
        "provider_comparison=False threshold_relaxation=False production_default_modified=False"
    )
    print(
        "pycolmap_authority=True provider_ranking=False universal_provider_winner=False "
        "universal_rolling_depth_robustness=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print("I5.7 RollingDepth robustness: CLOSED / PASS (BOUNDED ENVELOPE)")
    print("I5.8 Video Depth Anything robustness: AUTHORIZED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.7C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
