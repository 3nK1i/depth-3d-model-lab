#!/usr/bin/env python3
"""I4.1.9A A1: freeze and validate the exact 68-view House camera bridge.

This command performs no SfM, BA, undistortion, PatchMatch Stereo, Stereo
Fusion, metric-scale promotion, or mesh generation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.mvs_camera_bridge import build_house_mvs_camera_bridge  # noqa: E402

_DEFAULT_I4_1_7 = Path("/tmp/eia_i4_1_7_house_visual_positive_control_01")
_DEFAULT_OUTPUT = Path("/tmp/eia_i4_1_9a_house_mvs_positive_control/a1_camera_bridge")
_DEFAULT_PYCOLMAP = Path.home() / "anaconda3/envs/eia-pycolmap-probe/bin/python3.11"
_DEFAULT_A0_REPORT = Path("/tmp/eia_i4_1_9a_mvs_preflight.json")
_DEFAULT_WORKER = ROOT / "scripts/reconstruction_pycolmap_mvs_bridge_worker.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-1-7-dir", type=Path, default=_DEFAULT_I4_1_7)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=_DEFAULT_PYCOLMAP)
    parser.add_argument("--a0-report", type=Path, default=_DEFAULT_A0_REPORT)
    parser.add_argument("--worker", type=Path, default=_DEFAULT_WORKER)
    parser.add_argument("--worker-timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    summary = build_house_mvs_camera_bridge(
        i4_1_7_dir=args.i4_1_7_dir,
        output_dir=args.output_dir,
        pycolmap_python=args.python,
        worker_script=args.worker,
        a0_report=args.a0_report,
        worker_timeout_s=args.worker_timeout_seconds,
    )
    passed = summary.get("status") == "pass"
    counts = summary.get("counts", {})
    validation = summary.get("validation", {})
    print("I4.1.9A A1 authoritative 68-view camera bridge:", "PASS" if passed else "FAIL")
    print(
        f"global_sfm_anchor={counts.get('global_sfm_anchor')} "
        f"localized_extra_observation={counts.get('localized_extra_observation')} "
        f"total_registered_views={counts.get('total_registered_views')}"
    )
    print(f"validation_checks={validation.get('checks', {})}")
    print("sfm_rerun=False bundle_adjustment_rerun=False pose_refinement=False camera_geometry_modified=False")
    print("house_k_txt_consumed=False heavy_mvs_started=False patch_match_started=False stereo_fusion_started=False")
    print("metric_scale_claimed=False measurement_authoritative=False certified_metrology=False surface_mesh_generated=False")
    print(f"summary={Path(args.output_dir).expanduser().resolve() / 'i4_1_9a_a1_summary.json'}")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
