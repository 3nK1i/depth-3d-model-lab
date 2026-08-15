#!/usr/bin/env python3
"""Measure real persisted Gaussian CPU staging/lifecycle without starting a backend."""
from __future__ import annotations

import argparse
import gc
import os
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
import weakref

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_gaussian_controls import GaussianDisplaySettings, apply_gaussian_display_settings
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from app.reconstruction_gaussian_resources import (
    build_gaussian_interactive_resource_policy,
    gaussian_display_staging_bytes,
    gaussian_scene_staging_bytes,
)
from core.reconstruction import current_process_gaussian_heavy_modules, read_persistent_reconstruction_scene
from core.runtime.stress import read_process_rss_mib


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return float(ordered[index])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--cycles", type=int, default=12)
    args = parser.parse_args()
    cycles = max(1, int(args.cycles))
    manifest = Path(args.scene_manifest).expanduser().resolve()
    scene = read_persistent_reconstruction_scene(manifest)
    policy = build_gaussian_interactive_resource_policy()
    before_modules = current_process_gaussian_heavy_modules()
    baseline_rss = read_process_rss_mib(os.getpid())
    load_ms: list[float] = []
    apply_ms: list[float] = []
    release_checks: list[bool] = []
    source_bytes = 0
    display_bytes = 0
    total = 0
    source_display = 0
    controlled_display = 0

    for _ in range(cycles):
        started = perf_counter()
        interactive = load_interactive_gaussian_scene(
            scene, max_gaussians=policy.max_interactive_gaussians
        )
        load_ms.append((perf_counter() - started) * 1000.0)
        started = perf_counter()
        buffers = apply_gaussian_display_settings(
            interactive,
            GaussianDisplaySettings(density_percent=50, size_percent=110, opacity_percent=80),
        )
        apply_ms.append((perf_counter() - started) * 1000.0)
        source_bytes = gaussian_scene_staging_bytes(interactive)
        display_bytes = gaussian_display_staging_bytes(buffers)
        total = interactive.total_gaussian_count
        source_display = interactive.display_gaussian_count
        controlled_display = buffers.display_count
        source_ref = weakref.ref(interactive.positions_reconstruction)
        display_ref = weakref.ref(buffers.positions_reconstruction)
        del buffers
        del interactive
        gc.collect()
        release_checks.append(source_ref() is None and display_ref() is None)

    final_rss = read_process_rss_mib(os.getpid())
    after_modules = current_process_gaussian_heavy_modules()
    summary = {
        "phase": "9.4G",
        "status": "PASS" if all(release_checks) and not after_modules else "FAIL",
        "scene_manifest": str(manifest),
        "scene_id": scene.scene_id,
        "cycles": cycles,
        "device": policy.device_label,
        "resource_tier": policy.tier,
        "adaptive_max_display_gaussians": policy.max_interactive_gaussians,
        "total_gaussian_count": total,
        "source_interactive_display_count": source_display,
        "balanced_display_count": controlled_display,
        "source_cpu_staging_mib": source_bytes / (1024.0 * 1024.0),
        "balanced_cpu_staging_mib": display_bytes / (1024.0 * 1024.0),
        "median_safe_load_ms": median(load_ms),
        "p95_safe_load_ms": _p95(load_ms),
        "median_control_apply_ms": median(apply_ms),
        "p95_control_apply_ms": _p95(apply_ms),
        "rss_baseline_mib": baseline_rss,
        "rss_final_mib": final_rss,
        "rss_values_are_advisory": True,
        "release_checks_passed": sum(1 for item in release_checks if item),
        "all_numpy_staging_released": all(release_checks),
        "gpu_vram_measured": False,
        "timings_are_cpu_ui_prep_only": True,
        "canonical_geometry_authoritative": bool(scene.canonical_geometry_authoritative),
        "gaussian_canonical_geometry_authoritative": False,
        "gaussian_metric_scale_authoritative": False,
        "gaussian_measurement_authoritative": False,
        "gaussian_qc_authoritative": False,
        "certified_metrology": False,
        "gaussian_backend_started": False,
        "training_started": False,
        "render_backend_started": False,
        "production_heavy_modules_before": list(before_modules),
        "production_heavy_modules_after": list(after_modules),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
