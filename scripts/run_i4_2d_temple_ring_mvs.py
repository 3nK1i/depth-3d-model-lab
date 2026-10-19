#!/usr/bin/env python3
"""Run I4.2D raw 47-view TempleRing COLMAP MVS."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.temple_ring_mvs import TempleRingMvsConfig, run_temple_ring_mvs  # noqa: E402

_DEFAULT_B = Path("/tmp/depth3d_i4_2_temple_ring/i4_2b_sfm_attempt01")
_DEFAULT_C = Path("/tmp/depth3d_i4_2_temple_ring/i4_2c_trajectory_validation_attempt01")
_DEFAULT_OUTPUT = Path("/tmp/depth3d_i4_2_temple_ring/i4_2d_mvs_attempt01")
_DEFAULT_PYCOLMAP = Path.home() / "anaconda3/envs/depth3d-pycolmap-probe/bin/python3.11"
_DEFAULT_WORKER = ROOT / "scripts/reconstruction_pycolmap_temple_ring_mvs_worker.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-2b-dir", type=Path, default=_DEFAULT_B)
    parser.add_argument("--i4-2c-dir", type=Path, default=_DEFAULT_C)
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

    config = TempleRingMvsConfig(
        source_neighbors=args.source_neighbors,
        patch_match_cache_gib=args.patch_match_cache_gib,
        stereo_fusion_cache_gib=args.stereo_fusion_cache_gib,
        gpu_index=args.gpu_index,
        max_image_size=args.max_image_size,
        geometric_consistency=True,
    )
    try:
        summary = run_temple_ring_mvs(
            i4_2b_dir=args.i4_2b_dir,
            i4_2c_dir=args.i4_2c_dir,
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
        print(f"I4.2D TempleRing raw MVS: FAIL — {exc}", file=sys.stderr)
        print(f"output={args.output_dir.expanduser().resolve()}", file=sys.stderr)
        return 3

    worker = summary.get("worker", {})
    fusion = worker.get("fusion", {}) if isinstance(worker, dict) else {}
    patch = worker.get("patch_match_outputs", {}) if isinstance(worker, dict) else {}
    authority = summary.get("authority", {})
    print("I4.2D TempleRing raw MVS: PASS")
    print(
        f"reference_views=47 source_neighbors={config.source_neighbors} "
        f"geometric_depth_maps={patch.get('geometric_depth_maps')} geometric_normal_maps={patch.get('geometric_normal_maps')}"
    )
    print(f"raw_fused_points={fusion.get('ply_vertex_count')} raw_fused_ply={fusion.get('raw_fused_ply')}")
    print(
        "authoritative_sparse_unchanged=True held_out_ground_truth_consumed_by_i4_2d=False "
        f"production_manifold_applied={authority.get('production_manifold_applied')}"
    )
    print(
        "metric_scale_claimed=False measurement_authoritative=False certified_metrology=False "
        "surface_mesh_generated=False i4_3_provider_architecture_authorized=False"
    )
    print(f"summary={args.output_dir.expanduser().resolve() / 'i4_2d_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
