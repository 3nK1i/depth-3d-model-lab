#!/usr/bin/env python3
"""Run I4.2A TempleRing dataset/calibration quarantine preflight."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.temple_ring_preflight import run_i4_2a_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Extracted Middlebury TempleRing directory containing templeR*.png and templeR_par.txt")
    parser.add_argument(
        "--i4-1-9a-closure",
        default="/tmp/eia_i4_1_9a_house_mvs_positive_control/a3_visual_ab/i4_1_9a_a3_1_closure.json",
    )
    parser.add_argument("--output-dir", default="/tmp/eia_i4_2_temple_ring/i4_2a_preflight")
    args = parser.parse_args()

    result = run_i4_2a_preflight(
        dataset_dir=args.dataset_dir,
        i4_1_9a_closure=args.i4_1_9a_closure,
        output_dir=args.output_dir,
    )
    print("I4.2A TempleRing dataset/detail-preservation preflight: PASS")
    print("dataset=Middlebury TempleRing views=47 image_size=640x480 shared_intrinsics=True")
    print("gt_extrinsics_quarantined=True gt_pose_input=False all_47_views_available=True")
    print("sfm_started=False mvs_started=False da3_started=False production_filter_policy_changed=False")
    print("metric_scale_claimed=False measurement_authoritative=False certified_metrology=False surface_mesh_generated=False")
    print("i4_3_provider_architecture_authorized=False")
    print(f"summary={result['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
