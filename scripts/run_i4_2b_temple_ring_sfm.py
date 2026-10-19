#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from core.reconstruction.temple_ring_sfm import TempleRingSfmConfig, run_temple_ring_authoritative_sfm


def main() -> int:
    parser = argparse.ArgumentParser(description="Run I4.2B authoritative TempleRing SfM")
    parser.add_argument("--i4-2a-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=str(Path.home() / "anaconda3/envs/depth3d-pycolmap-probe/bin/python3.11"))
    parser.add_argument("--cpu", action="store_true", help="Use CPU SIFT extraction/matching instead of CUDA")
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    args = parser.parse_args()

    result = run_temple_ring_authoritative_sfm(
        i4_2a_dir=args.i4_2a_dir,
        output_dir=args.output_dir,
        pycolmap_python=args.python,
        worker_script=Path(__file__).resolve().parent / "reconstruction_pycolmap_temple_ring_sfm_worker.py",
        config=TempleRingSfmConfig(use_gpu=not args.cpu),
        worker_timeout_s=args.timeout_s,
    )
    validation = result.get("validation", {})
    authority = result.get("authority", {})
    print(f"I4.2B TempleRing authoritative SfM: {str(result['status']).upper()}")
    print(
        f"registered={validation.get('registered_images', 0)}/47 landmarks={validation.get('landmarks', 0)} "
        f"mean_reprojection_error_px={validation.get('mean_reprojection_error_px')}"
    )
    print(
        f"all_47_registered={validation.get('all_47_registered', False)} "
        f"fixed_intrinsics_preserved={validation.get('fixed_intrinsics_preserved', False)} "
        f"single_authoritative_sfm_frame={authority.get('single_authoritative_sfm_frame', False)}"
    )
    print(
        "gt_extrinsics_consumed=False gt_camera_centers_consumed=False gt_angles_consumed=False "
        f"mvs_started={authority.get('mvs_started')} da3_started={authority.get('da3_started')}"
    )
    print(f"summary={Path(args.output_dir).expanduser().resolve() / 'i4_2b_summary.json'}")
    return 0 if result.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
