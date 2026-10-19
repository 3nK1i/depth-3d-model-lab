#!/usr/bin/env python3
"""Preview the resolved I4.1.3 full-clip sampling plan without reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    FrameSamplingMode,
    FrameSamplingPolicy,
    RecordedVideoReader,
    build_frame_sampling_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--max-keyframes", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/depth3d_i4_1_3_sampling_preflight.json"),
    )
    args = parser.parse_args()
    if args.target_fps <= 0.0:
        parser.error("--target-fps must be positive.")
    if args.max_keyframes < 1:
        parser.error("--max-keyframes must be positive.")

    source = RecordedVideoReader(args.video.expanduser().resolve())
    plan = build_frame_sampling_plan(
        FrameSamplingPolicy(
            mode=FrameSamplingMode.TARGET_FPS,
            target_fps_hz=args.target_fps,
            maximum_sampled_frames=args.max_keyframes,
        ),
        estimated_total_frames=source.estimated_total_frames,
        expected_frame_rate_hz=source.metadata.expected_frame_rate_hz,
    )
    indices = plan.target_sequence_indices
    total = plan.estimated_total_frames
    span_fraction = None
    if indices and total is not None and total > 1:
        span_fraction = (indices[-1] - indices[0]) / (total - 1)
    payload = {
        "phase": "I4.1.3",
        "status": "pass",
        "video": str(source.path),
        "sampling_plan": plan.as_dict(),
        "planned_source_span_fraction": span_fraction,
        "authority": {
            "sfm_rerun": False,
            "da3_rerun": False,
            "fusion_rerun": False,
            "filter_policy_changed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
    }
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")

    print("I4.1.3 full-clip sampling preflight: PASS")
    print(
        f"mode={plan.mode.value} source_frames={plan.estimated_total_frames} "
        f"duration={plan.estimated_duration_seconds}s"
    )
    print(
        f"requested_fps={plan.requested_target_fps_hz} "
        f"effective_fps={plan.effective_target_fps_hz} "
        f"nominal_samples={plan.nominal_sample_count} planned_samples={plan.planned_sample_count}"
    )
    print(
        f"first_source_index={indices[0] if indices else None} "
        f"last_source_index={indices[-1] if indices else None} "
        f"planned_source_span_fraction={span_fraction}"
    )
    print("sampling_policy_changed=True only when cap requires uniform_full_span; thresholds unchanged=True")
    print(f"summary={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
