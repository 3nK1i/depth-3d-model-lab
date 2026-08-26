#!/usr/bin/env python3
"""I4.1.9A A2: run the bounded 68-view House COLMAP MVS positive control."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.mvs_execution import MvsExecutionConfig, run_house_mvs_execution  # noqa: E402

_DEFAULT_A1 = Path("/tmp/eia_i4_1_9a_house_mvs_positive_control/a1_camera_bridge")
_DEFAULT_OUTPUT = Path("/tmp/eia_i4_1_9a_house_mvs_positive_control/a2_mvs_attempt01")
_DEFAULT_PYCOLMAP = Path.home() / "anaconda3/envs/eia-pycolmap-probe/bin/python3.11"
_DEFAULT_WORKER = ROOT / "scripts/reconstruction_pycolmap_mvs_worker.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-dir", type=Path, default=_DEFAULT_A1)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=_DEFAULT_PYCOLMAP)
    parser.add_argument("--worker", type=Path, default=_DEFAULT_WORKER)
    parser.add_argument("--source-neighbors", type=int, default=10)
    parser.add_argument("--patch-match-cache-gib", type=float, default=2.0)
    parser.add_argument("--stereo-fusion-cache-gib", type=float, default=2.0)
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--max-image-size", type=int, default=-1)
    parser.add_argument("--min-free-disk-gib", type=float, default=8.0)
    parser.add_argument("--min-available-ram-gib", type=float, default=4.0)
    parser.add_argument("--min-free-vram-gib", type=float, default=2.0)
    parser.add_argument("--worker-timeout-seconds", type=float, default=21600.0)
    args = parser.parse_args()

    config = MvsExecutionConfig(
        source_neighbors=args.source_neighbors,
        patch_match_cache_gib=args.patch_match_cache_gib,
        stereo_fusion_cache_gib=args.stereo_fusion_cache_gib,
        gpu_index=args.gpu_index,
        max_image_size=args.max_image_size,
        geometric_consistency=True,
    )
    try:
        summary = run_house_mvs_execution(
            a1_dir=args.a1_dir,
            output_dir=args.output_dir,
            pycolmap_python=args.python,
            worker_script=args.worker,
            config=config,
            min_free_disk_gib=args.min_free_disk_gib,
            min_available_ram_gib=args.min_available_ram_gib,
            min_free_vram_gib=args.min_free_vram_gib,
            worker_timeout_s=args.worker_timeout_seconds,
        )
    except ReconstructionValidationError as exc:
        print(f"I4.1.9A A2 House MVS: FAIL — {exc}", file=sys.stderr)
        print(f"output={args.output_dir.expanduser().resolve()}", file=sys.stderr)
        return 3

    worker = summary.get("worker", {})
    fusion = worker.get("fusion", {}) if isinstance(worker, dict) else {}
    print("I4.1.9A A2 House MVS: PASS")
    print(
        f"reference_views=68 source_neighbors={config.source_neighbors} "
        f"geometric_consistency={config.geometric_consistency}"
    )
    print(f"fused_points={fusion.get('ply_vertex_count')} fused_ply={fusion.get('fused_ply')}")
    print("sfm_rerun=False bundle_adjustment_rerun=False pose_refinement=False house_k_txt_consumed=False")
    print("metric_scale_claimed=False measurement_authoritative=False certified_metrology=False surface_mesh_generated=False")
    print(f"summary={args.output_dir.expanduser().resolve() / 'i4_1_9a_a2_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
