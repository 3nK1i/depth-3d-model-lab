#!/usr/bin/env python3
"""Deterministic Phase 9.4G resource-policy/lifecycle smoke without Qt or backends."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import tempfile
import weakref

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_gaussian_controls import GaussianDisplaySettings, apply_gaussian_display_settings
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from app.reconstruction_gaussian_resources import (
    build_gaussian_interactive_resource_policy,
    build_gaussian_viewer_runtime_health,
)
from core.reconstruction import current_process_gaussian_heavy_modules
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene
from tests.reconstruction_helpers import make_hardware


def main() -> int:
    before = current_process_gaussian_heavy_modules()
    with tempfile.TemporaryDirectory(prefix="phase9_4g_") as tmp:
        scene, _manifest, _artifact = make_phase9_4e_persistent_scene(Path(tmp))
        policy = build_gaussian_interactive_resource_policy(
            make_hardware(total_gib=6.0, name="NVIDIA GeForce RTX 2060")
        )
        interactive = load_interactive_gaussian_scene(
            scene, max_gaussians=policy.max_interactive_gaussians
        )
        buffers = apply_gaussian_display_settings(
            interactive,
            GaussianDisplaySettings(density_percent=50, size_percent=125, opacity_percent=70),
        )
        health = build_gaussian_viewer_runtime_health(
            policy=policy,
            scene=interactive,
            buffers=buffers,
            resident=True,
            had_release=False,
            load_latency_ms=4.5,
            apply_latency_ms=1.25,
            load_count=1,
            release_count=0,
            canvas_resource_generation=1,
            canvas_release_count=0,
            canonical_geometry_authoritative=True,
        )
        source_ref = weakref.ref(interactive.positions_reconstruction)
        display_ref = weakref.ref(buffers.positions_reconstruction)
        total = interactive.total_gaussian_count
        controlled = buffers.display_count
        del buffers
        del interactive
        gc.collect()
        source_released = source_ref() is None
        display_released = display_ref() is None

    after = current_process_gaussian_heavy_modules()
    summary = {
        "phase": "9.4G",
        "status": "PASS" if source_released and display_released and not after else "FAIL",
        "policy_device": policy.device_label,
        "policy_tier": policy.tier,
        "max_display_gaussians": policy.max_interactive_gaussians,
        "total_gaussian_count": total,
        "controlled_display_count": controlled,
        "source_numpy_released": source_released,
        "controlled_numpy_released": display_released,
        "resources_text": health.resources_text,
        "performance_text": health.performance_text,
        "gpu_vram_measured": False,
        "ui_timings_only": True,
        "canonical_geometry_authoritative": True,
        "gaussian_measurement_authoritative": False,
        "certified_metrology": False,
        "gaussian_backend_started": False,
        "training_started": False,
        "render_backend_started": False,
        "production_heavy_modules_before": list(before),
        "production_heavy_modules_after": list(after),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
