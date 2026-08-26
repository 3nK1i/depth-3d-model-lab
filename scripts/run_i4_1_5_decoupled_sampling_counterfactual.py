#!/usr/bin/env python3
"""I4.1.5 bounded decoupled global/local sampling counterfactual.

Consumes I4.1.4 evidence plus the I4.1.3 preflight metadata. No reconstruction
stage is executed. The output estimates whether a bounded local dense candidate
pool can restore local overlap while the 40 global anchors retain full-span SfM
coverage. Any extra dense observation would still need explicit localization
into the same authoritative SfM model in a later production integration step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_i4_decoupled_sampling_counterfactual import (  # noqa: E402
    build_i4_1_5_counterfactual,
)

_DEFAULT_I414 = Path("/tmp/eia_i4_1_4_decoupled_sampling_density_diagnosis/i4_1_4_summary.json")
_DEFAULT_I413 = Path("/tmp/eia_i4_1_3_sampling_preflight.json")
_DEFAULT_OUTPUT = Path("/tmp/eia_i4_1_5_decoupled_sampling_counterfactual")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-1-4-summary", type=Path, default=_DEFAULT_I414)
    p.add_argument("--i4-1-3-preflight", type=Path, default=_DEFAULT_I413)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--global-anchor-cap", type=int, default=40)
    p.add_argument("--dense-candidate-cap", type=int, default=120)
    p.add_argument("--local-target-fps", type=float, default=2.0)
    p.add_argument("--final-fusion-cap", type=int, default=40)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.5 cannot read JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I4.1.5 JSON root must be an object: {path}")
    return value


def _duration(preflight: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> float:
    for key in ("source_duration_seconds", "estimated_duration_seconds", "duration_seconds", "duration"):
        value = preflight.get(key)
        if value is not None:
            return float(value)
    # I4.1.3 preflight used the CLI line `duration=...`; implementations may
    # store it nested under source/video metadata, so search one level deep.
    for value in preflight.values():
        if isinstance(value, Mapping):
            for key in ("source_duration_seconds", "estimated_duration_seconds", "duration_seconds", "duration"):
                if value.get(key) is not None:
                    return float(value[key])
    current = diagnosis.get("current_full_span")
    if isinstance(current, Mapping) and current.get("sampled_timestamp_span_seconds") is not None:
        return float(current["sampled_timestamp_span_seconds"])
    raise ReconstructionValidationError("I4.1.5 cannot resolve source duration from I4.1.3/I4.1.4 evidence.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = _args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.5 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    i414 = _read_json(args.i4_1_4_summary)
    i413 = _read_json(args.i4_1_3_preflight)
    duration = _duration(i413, i414)
    result = build_i4_1_5_counterfactual(
        duration_seconds=duration,
        i4_1_4_summary=i414,
        global_anchor_cap=args.global_anchor_cap,
        dense_candidate_cap=args.dense_candidate_cap,
        local_target_fps=args.local_target_fps,
        final_fusion_cap=args.final_fusion_cap,
    )
    result["sources"] = {
        "i4_1_4_summary": str(args.i4_1_4_summary.expanduser().resolve()),
        "i4_1_3_preflight": str(args.i4_1_3_preflight.expanduser().resolve()),
    }
    summary = _write_json(output / "i4_1_5_summary.json", result)

    print("I4.1.5 Decoupled Global/Local Sampling Counterfactual: COUNTERFACTUAL_READY")
    print(
        f"source_duration={duration:.3f}s global_anchor_cap={args.global_anchor_cap} "
        f"dense_candidate_cap={args.dense_candidate_cap} final_fusion_cap={args.final_fusion_cap}"
    )
    for row in result["plans"]:
        p = row["plan"]; g = row["geometry"]; h = row["same_source_horse_estimate"]; c = row["clean_capture_estimate"]
        print(
            f"plan={p['name']} global={p['global_anchor_count']} dense_candidates={p['dense_candidate_count']} "
            f"span={g['source_span_fraction']:.3f} gap_p50={g['dense_adjacent_gap_seconds']['p50']:.3f}s "
            f"local_le075={g['adjacent_local_fraction_le_threshold']:.3f} "
            f"horse_est_pass={h['estimated_pass_fraction']} "
            f"clean_est_pass={None if c is None else c['estimated_pass_fraction']}"
        )
        if "assessment" in row:
            a = row["assessment"]
            print(
                f"  bounded={a['bounded']} full_span={a['full_span']} locality_gain={a['locality_gain']:.3f} "
                f"estimated_pass_gain={a['same_source_estimated_pass_gain']} candidate={a['counterfactual_candidate']}"
            )
    d = result["decision"]
    print(
        f"decoupled_sampling_architecture_candidate={d['decoupled_sampling_architecture_candidate']} "
        f"recommended_plan={d['recommended_counterfactual_plan']} i4_1_6_authorized={d['i4_1_6_authorized']}"
    )
    print(
        f"local_density_signal={d['local_density_signal']} capture_regime_signal={d['capture_regime_signal']} "
        f"cleaner_static_capture_still_recommended={d['cleaner_static_capture_still_recommended']}"
    )
    print("production_sampling_integration_authorized=False dense_consistency_threshold_relaxation_authorized=False")
    print("sfm_rerun=False da3_rerun=False dense_pair_rerun=False fusion_rerun=False persistence_changed=False")
    print("single_authoritative_sfm_frame=True extra_dense_candidates_require_pose_localization=True")
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print("i4_2_detail_preservation_authorized=False mesh_authorization=False")
    print(f"next={d['next_action']}")
    print(f"summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
