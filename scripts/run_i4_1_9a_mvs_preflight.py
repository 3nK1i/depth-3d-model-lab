#!/usr/bin/env python3
"""I4.1.9A A0 read-only PyCOLMAP MVS/runtime/resource preflight.

No undistortion, PatchMatch, Stereo Fusion, SfM, BA, or reconstruction job is
started by this command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.mvs_environment import (  # noqa: E402
    MVS_HOUSE_DEFAULT_MIN_AVAILABLE_RAM_GIB,
    MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB,
    MVS_HOUSE_DEFAULT_MIN_FREE_VRAM_GIB,
    build_mvs_preflight_report,
)

_DEFAULT_PYCOLMAP_PYTHON = Path.home() / "anaconda3/envs/depth3d-pycolmap-probe/bin/python3.11"
_DEFAULT_WORKSPACE = Path("/tmp/depth3d_i4_1_9a_house_mvs_positive_control")
_DEFAULT_REPORT = Path("/tmp/depth3d_i4_1_9a_mvs_preflight.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=_DEFAULT_PYCOLMAP_PYTHON)
    parser.add_argument("--workspace", type=Path, default=_DEFAULT_WORKSPACE)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--minimum-free-disk-gib", type=float, default=MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB)
    parser.add_argument("--minimum-available-ram-gib", type=float, default=MVS_HOUSE_DEFAULT_MIN_AVAILABLE_RAM_GIB)
    parser.add_argument("--minimum-free-vram-gib", type=float, default=MVS_HOUSE_DEFAULT_MIN_FREE_VRAM_GIB)
    args = parser.parse_args()

    report = build_mvs_preflight_report(
        str(args.python.expanduser().resolve()),
        args.workspace.expanduser().resolve(),
        timeout_s=args.timeout_seconds,
        min_free_disk_gib=args.minimum_free_disk_gib,
        min_available_ram_gib=args.minimum_available_ram_gib,
        min_free_vram_gib=args.minimum_free_vram_gib,
    )
    payload = report.as_dict()
    output = args.report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("I4.1.9A A0 MVS preflight:", "PASS" if report.ready else "NOT READY")
    env = report.environment
    resources = report.resources
    print(
        f"pycolmap={env.pycolmap_version or 'unavailable'} distribution={env.distribution or 'unknown'} "
        f"cuda_devices={env.cuda_devices} mvs_api_ready={env.mvs_api_ready}"
    )
    print(
        f"disk_free_gib={resources.disk_free_gib:.2f} "
        f"ram_available_gib={resources.ram_available_gib:.2f} "
        f"max_free_vram_gib={resources.max_free_vram_gib:.2f}"
    )
    if env.error:
        print(f"runtime_error={env.error}")
    if resources.nvidia_smi_error:
        print(f"nvidia_smi_error={resources.nvidia_smi_error}")
    print("heavy_mvs_started=False sfm_rerun=False bundle_adjustment_rerun=False")
    print("metric_scale_claimed=False measurement_authoritative=False certified_metrology=False surface_mesh_generated=False")
    print(f"report={output}")
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
