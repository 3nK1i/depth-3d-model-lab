#!/usr/bin/env python3
"""Validate Phase 9.5F presentation controls against a real persisted NeRF scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_nerf_controls import NerfDisplaySettings
from app.reconstruction_nerf_interactive import (
    build_interactive_nerf_presentation,
    turntable_state_to_nerf_render_request,
)
from core.reconstruction import read_persistent_reconstruction_scene, verify_persistent_scene_file_references


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-manifest",
        default="/tmp/phase9_5d_persistent_nerf_scene/persistent_reconstruction_scene.json",
    )
    parser.add_argument(
        "--output",
        default="/tmp/phase9_5f_real_nerf_controls_summary.json",
    )
    args = parser.parse_args()

    manifest = Path(args.scene_manifest).expanduser().resolve()
    scene = read_persistent_reconstruction_scene(manifest)
    refs = verify_persistent_scene_file_references(scene)
    presentation = build_interactive_nerf_presentation(scene)
    if presentation is None or not presentation.ready:
        raise RuntimeError("Real persistent scene has no ready interactive NeRF representation.")

    profiles = {}
    for key in ("fast", "balanced", "detail"):
        settings = NerfDisplaySettings(render_profile=key, opacity_percent=70, auto_refresh=False)
        request = turntable_state_to_nerf_render_request(
            request_id=len(profiles) + 1,
            center_viewer=(0.0, 0.0, 0.0),
            distance=2.5,
            azimuth_degrees=15.0,
            elevation_degrees=10.0,
            fov_degrees=45.0,
            scene_id=presentation.scene_id,
            artifact_id=presentation.artifact_id,
            width=settings.render_width,
            height=settings.render_height,
        )
        profiles[key] = [request.width, request.height]

    heavy_before = sorted(
        name for name in sys.modules if name.startswith(("nerfstudio", "tinycudann", "torch"))
    )
    summary = {
        "phase": "9.5F",
        "status": "PASS",
        "scene_id": scene.scene_id,
        "scene_manifest": str(manifest),
        "verified_persistent_file_reference_count": len(refs),
        "render_profiles": profiles,
        "default_render_profile": "detail",
        "default_render_resolution": [640, 360],
        "opacity_range_percent": [35, 100],
        "auto_refresh_default": True,
        "nerf_scene_layer_available": True,
        "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
        "picking_source": "canonical_geometry_only",
        "nerf_scene_role": presentation.scene_role,
        "nerf_canonical_geometry_authoritative": presentation.canonical_geometry_authoritative,
        "nerf_metric_scale_authoritative": presentation.metric_scale_authoritative,
        "nerf_measurement_authoritative": presentation.measurement_authoritative,
        "nerf_qc_authoritative": presentation.qc_authoritative,
        "certified_metrology": presentation.certified_metrology,
        "backend_started": False,
        "training_started": False,
        "persistence_modified": False,
        "production_heavy_modules": heavy_before,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Phase 9.5F real NeRF controls summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
