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
from core.reconstruction.i5_mvs_robustness import build_i5_6c_mvs_robustness_closure

OUTPUT_NAME = "i5_6c_mvs_robustness_closure.json"


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
    parser = argparse.ArgumentParser(description="I5.6C evidence-only MVS robustness qualification and closure.")
    parser.add_argument("--i5-6a-admission", required=True)
    parser.add_argument("--i5-6a-sha256", required=True)
    parser.add_argument("--i5-6b-replay", required=True)
    parser.add_argument("--i5-6b-sha256", required=True)
    parser.add_argument("--closure-receipt", required=True)
    parser.add_argument("--closure-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    a_binding = _binding(args.i5_6a_admission, args.i5_6a_sha256, label="I5.6A admission")
    b_binding = _binding(args.i5_6b_replay, args.i5_6b_sha256, label="I5.6B real replay")
    receipt_binding = _binding(args.closure_receipt, args.closure_receipt_sha256, label="I5.6C closure receipt")

    result = build_i5_6c_mvs_robustness_closure(
        i5_6a_result=_read_json(Path(a_binding.path), label="I5.6A admission"),
        i5_6a_binding=a_binding,
        i5_6b_result=_read_json(Path(b_binding.path), label="I5.6B replay"),
        i5_6b_binding=b_binding,
        qualification_receipt=_read_json(Path(receipt_binding.path), label="I5.6C closure receipt"),
        qualification_receipt_binding=receipt_binding,
    )
    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    q = result["qualification"]
    d = result["directional_native_support_evidence"]
    limitations = result["limitations"]
    recovery = result.get("recovery_provenance")
    print("I5.6C MVS Robustness Qualification + Closure: PASS")
    print(
        f"quality_status={q['quality_status']} operating_envelope={q['operating_envelope']} "
        f"runtime_pass_scenarios={q['runtime_pass_scenario_count']}/{q['admitted_scenario_count']}"
    )
    print(
        "directional_native_support_reduction_observed="
        f"{d['directional_native_support_reduction_observed']} "
        "reduction_scenarios=" + ",".join(d["directional_native_support_reduction_scenarios"])
    )
    print(
        f"native_support_comparison_complete={d['native_support_comparison_complete_for_passing_scenarios']} "
        f"runtime_performance_envelope_complete={limitations['runtime_performance_envelope_complete']} "
        f"resource_envelope_complete={limitations['resource_envelope_complete']}"
    )
    if recovery is not None:
        dynamic = result["directional_native_support_evidence"]["scenario_comparisons"]["castle_dynamic_proxy"]
        print(
            "interrupted_parent_recovery_preserved=True "
            f"termination_cause={recovery['worker_termination_cause']} "
            f"dynamic_proxy_fused_ratio_to_control={dynamic['fused_point_ratio_to_control']}"
        )
        print(
            f"recovered_parent_wall_time_unavailable={limitations['recovered_dynamic_proxy_parent_wall_time_unavailable']} "
            f"recovered_peak_resource_metrics_incomplete={limitations['recovered_dynamic_proxy_peak_resource_metrics_incomplete']}"
        )
    print(
        "evidence_only=True sfm_rerun=False mvs_rerun=False patch_match_rerun=False "
        "stereo_fusion_rerun=False provider_comparison=False da3_rerun=False"
    )
    print(
        "pycolmap_authority=True threshold_relaxation=False production_default_modified=False "
        "provider_ranking=False universal_provider_winner=False metric_scale=False metrology=False "
        "permanent_global_view_cap=None"
    )
    print(f"I5.6 MVS robustness: CLOSED / {q['quality_status'].upper()}")
    print("I5.7 RollingDepth integration and robustness envelope: AUTHORIZED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.6C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
