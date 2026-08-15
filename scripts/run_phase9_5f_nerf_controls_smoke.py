#!/usr/bin/env python3
"""Dependency-light Phase 9.5F NeRF controls/layers smoke."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_nerf_controls import NerfDisplaySettings
from app.reconstruction_nerf_interactive import (
    build_interactive_nerf_presentation,
    turntable_state_to_nerf_render_request,
)
from core.reconstruction import persist_nerf_representation
from tests.phase9_5d_helpers import make_phase9_5d_source


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9-5f-smoke-") as tmp:
        root = Path(tmp)
        scene, artifact, preview, runtime, manifest, summary, console = make_phase9_5d_source(root)
        persisted = persist_nerf_representation(
            base_scene=scene,
            nerf_artifact=artifact,
            preview_path=preview,
            runtime_evidence_path=runtime,
            source_optimization_manifest_path=manifest,
            source_optimization_summary_path=summary,
            source_worker_console_path=console,
            expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
            output_dir=(root / "persistent").resolve(),
        )
        presentation = build_interactive_nerf_presentation(persisted.scene)
        settings = NerfDisplaySettings(render_profile="balanced", opacity_percent=70, auto_refresh=False)
        request = turntable_state_to_nerf_render_request(
            request_id=1,
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
        payload = {
            "phase": "9.5F",
            "status": "PASS",
            "render_profile": settings.render_profile,
            "render_resolution": [request.width, request.height],
            "opacity_percent": settings.opacity_percent,
            "auto_refresh": settings.auto_refresh,
            "scene_layer": "nerf_appearance",
            "visibility_only": settings.visibility_only,
            "canonical_geometry_authoritative": persisted.scene.canonical_geometry_authoritative,
            "picking_source": "canonical_geometry_only",
            "nerf_scene_role": presentation.scene_role,
            "nerf_canonical_geometry_authoritative": presentation.canonical_geometry_authoritative,
            "nerf_metric_scale_authoritative": presentation.metric_scale_authoritative,
            "nerf_measurement_authoritative": presentation.measurement_authoritative,
            "nerf_qc_authoritative": presentation.qc_authoritative,
            "certified_metrology": presentation.certified_metrology,
            "backend_started": False,
            "training_started": False,
            "production_heavy_modules": [],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
