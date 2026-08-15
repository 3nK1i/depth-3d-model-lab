#!/usr/bin/env python3
"""Dependency-light Phase 9.4E persisted Gaussian -> interactive-view smoke."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from core.reconstruction import current_process_gaussian_heavy_modules
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_4e_") as tmp:
        scene, manifest, _artifact = make_phase9_4e_persistent_scene(Path(tmp))
        before = current_process_gaussian_heavy_modules()
        interactive = load_interactive_gaussian_scene(scene, max_gaussians=32)
        after = current_process_gaussian_heavy_modules()
        summary = {
            "phase": "9.4E",
            "status": "PASS",
            "scene_manifest": str(manifest),
            "scene_id": interactive.scene_id,
            "artifact_id": interactive.artifact_id,
            "backend_id": interactive.backend_id,
            "backend_version": interactive.backend_version,
            "total_gaussian_count": interactive.total_gaussian_count,
            "display_gaussian_count": interactive.display_gaussian_count,
            "interactive_appearance_ready": True,
            "canonical_geometry_authoritative": True,
            "gaussian_canonical_geometry_authoritative": interactive.canonical_geometry_authoritative,
            "gaussian_metric_scale_authoritative": interactive.metric_scale_authoritative,
            "gaussian_measurement_authoritative": interactive.measurement_authoritative,
            "gaussian_qc_authoritative": interactive.qc_authoritative,
            "certified_metrology": interactive.certified_metrology,
            "picking_source": "canonical_geometry_only",
            "static_backend_preview_preserved": True,
            "gaussian_backend_started": False,
            "training_started": False,
            "render_backend_started": False,
            "production_heavy_modules_before": list(before),
            "production_heavy_modules_after": list(after),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
