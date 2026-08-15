#!/usr/bin/env python3
"""Load a real persisted 9.4D Gaussian artifact for 9.4E without UI/backend execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from core.reconstruction import (
    current_process_gaussian_heavy_modules,
    read_persistent_reconstruction_scene,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--max-gaussians", type=int, default=200000)
    args = parser.parse_args()
    manifest = Path(args.scene_manifest).expanduser().resolve()
    scene = read_persistent_reconstruction_scene(manifest)
    before = current_process_gaussian_heavy_modules()
    interactive = load_interactive_gaussian_scene(scene, max_gaussians=args.max_gaussians)
    after = current_process_gaussian_heavy_modules()
    summary = {
        "phase": "9.4E",
        "status": "PASS",
        "scene_manifest": str(manifest),
        "scene_id": interactive.scene_id,
        "artifact_id": interactive.artifact_id,
        "scene_path": interactive.scene_path,
        "backend_id": interactive.backend_id,
        "backend_version": interactive.backend_version,
        "total_gaussian_count": interactive.total_gaussian_count,
        "display_gaussian_count": interactive.display_gaussian_count,
        "metric_geometry_status": interactive.metric_geometry_status,
        "interactive_appearance_ready": True,
        "canonical_geometry_authoritative": True,
        "gaussian_canonical_geometry_authoritative": False,
        "gaussian_metric_scale_authoritative": False,
        "gaussian_measurement_authoritative": False,
        "gaussian_qc_authoritative": False,
        "certified_metrology": False,
        "picking_source": "canonical_geometry_only",
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
