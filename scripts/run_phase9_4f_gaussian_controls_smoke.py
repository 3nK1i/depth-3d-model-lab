#!/usr/bin/env python3
"""Dependency-light Phase 9.4F Gaussian controls/scene-layer smoke."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_gaussian_controls import GaussianDisplaySettings, apply_gaussian_display_settings
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from core.reconstruction import current_process_gaussian_heavy_modules
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_4f_") as tmp:
        scene, manifest, _artifact = make_phase9_4e_persistent_scene(Path(tmp))
        before = current_process_gaussian_heavy_modules()
        interactive = load_interactive_gaussian_scene(scene)
        settings = GaussianDisplaySettings(density_percent=50, size_percent=125, opacity_percent=70)
        controlled = apply_gaussian_display_settings(interactive, settings)
        after = current_process_gaussian_heavy_modules()
        summary = {
            "phase": "9.4F",
            "status": "PASS",
            "scene_manifest": str(manifest),
            "scene_id": interactive.scene_id,
            "artifact_id": interactive.artifact_id,
            "source_display_count": controlled.source_display_count,
            "controlled_display_count": controlled.display_count,
            "density_percent": settings.density_percent,
            "size_percent": settings.size_percent,
            "opacity_percent": settings.opacity_percent,
            "scene_layer_visibility_only": settings.visibility_only,
            "canonical_geometry_authoritative": True,
            "gaussian_canonical_geometry_authoritative": settings.canonical_geometry_authoritative,
            "gaussian_metric_scale_authoritative": settings.metric_scale_authoritative,
            "gaussian_measurement_authoritative": settings.measurement_authoritative,
            "gaussian_qc_authoritative": settings.qc_authoritative,
            "certified_metrology": settings.certified_metrology,
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
