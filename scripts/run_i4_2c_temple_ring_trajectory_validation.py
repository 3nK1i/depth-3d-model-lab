#!/usr/bin/env python3
"""Run I4.2C held-out TempleRing trajectory validation."""
from __future__ import annotations

import argparse
from pathlib import Path

from core.reconstruction.temple_ring_trajectory_validation import (
    TempleRingTrajectoryThresholds,
    run_temple_ring_heldout_trajectory_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="I4.2C held-out TempleRing trajectory validation")
    parser.add_argument("--i4-2a-dir", required=True)
    parser.add_argument("--i4-2b-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", required=True, dest="pycolmap_python")
    parser.add_argument("--worker-timeout-s", type=float, default=600.0)
    args = parser.parse_args()

    worker = Path(__file__).with_name("reconstruction_pycolmap_temple_ring_pose_export_worker.py")
    result = run_temple_ring_heldout_trajectory_validation(
        i4_2a_dir=args.i4_2a_dir,
        i4_2b_dir=args.i4_2b_dir,
        output_dir=args.output_dir,
        pycolmap_python=args.pycolmap_python,
        pose_export_worker_script=worker,
        thresholds=TempleRingTrajectoryThresholds(),
        worker_timeout_s=args.worker_timeout_s,
    )
    quality = result.get("trajectory_quality", {})
    metrics = quality.get("metrics", {}) if isinstance(quality, dict) else {}
    authority = result.get("authority", {})
    print(f"I4.2C TempleRing held-out trajectory validation: {str(result['status']).upper()}")
    if metrics:
        print(
            "center_rmse_fraction={:.6g} center_p95_fraction={:.6g} rotation_median_deg={:.6g} rotation_p95_deg={:.6g}".format(
                float(metrics["center_rmse_fraction_of_median_ring_radius"]),
                float(metrics["center_p95_fraction_of_median_ring_radius"]),
                float(metrics["rotation_median_deg"]),
                float(metrics["rotation_p95_deg"]),
            )
        )
    print(
        "sim3_validation_only={} authoritative_sparse_modified={} gt_pose_input_to_sfm={} mvs_started={}".format(
            authority.get("similarity_validation_only"),
            authority.get("authoritative_sparse_modified"),
            authority.get("ground_truth_pose_input_to_sfm"),
            authority.get("mvs_started"),
        )
    )
    print(f"summary={Path(args.output_dir).expanduser().resolve() / 'i4_2c_summary.json'}")
    return 0 if result.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
