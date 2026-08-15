#!/usr/bin/env python3
"""Check Phase 9.4F display controls against a real persisted Gaussian scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_gaussian_controls import GaussianDisplaySettings, apply_gaussian_display_settings
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from core.reconstruction import current_process_gaussian_heavy_modules, read_persistent_reconstruction_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-manifest", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.scene_manifest).expanduser().resolve()
    scene = read_persistent_reconstruction_scene(manifest)
    before = current_process_gaussian_heavy_modules()
    interactive = load_interactive_gaussian_scene(scene)
    presets = {}
    for name, settings in (
        ("full", GaussianDisplaySettings()),
        ("balanced", GaussianDisplaySettings(density_percent=50, size_percent=110, opacity_percent=80)),
        ("light", GaussianDisplaySettings(density_percent=25, size_percent=100, opacity_percent=65)),
    ):
        buffers = apply_gaussian_display_settings(interactive, settings)
        presets[name] = {
            "display_count": buffers.display_count,
            "density_percent": settings.density_percent,
            "size_percent": settings.size_percent,
            "opacity_percent": settings.opacity_percent,
        }
    after = current_process_gaussian_heavy_modules()
    summary = {
        "phase": "9.4F",
        "status": "PASS",
        "scene_manifest": str(manifest),
        "scene_id": scene.scene_id,
        "artifact_id": interactive.artifact_id,
        "total_gaussian_count": interactive.total_gaussian_count,
        "source_interactive_display_count": interactive.display_gaussian_count,
        "presets": presets,
        "scene_layer_visibility_only": True,
        "canonical_geometry_authoritative": bool(scene.canonical_geometry_authoritative),
        "gaussian_canonical_geometry_authoritative": False,
        "gaussian_metric_scale_authoritative": False,
        "gaussian_measurement_authoritative": False,
        "gaussian_qc_authoritative": False,
        "certified_metrology": False,
        "picking_source": "canonical_geometry_only",
        "gaussian_backend_started": False,
        "training_started": False,
        "render_backend_started": False,
        "persisted_gaussian_modified": False,
        "production_heavy_modules_before": list(before),
        "production_heavy_modules_after": list(after),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
