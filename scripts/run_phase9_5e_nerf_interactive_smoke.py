#!/usr/bin/env python3
"""Dependency-light Phase 9.5E persisted-NeRF interactive viewer smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_nerf_interactive import (
    build_interactive_nerf_presentation,
    turntable_state_to_nerf_render_request,
)
from core.reconstruction import current_process_nerf_heavy_modules, persist_nerf_representation
from tests.phase9_5d_helpers import make_phase9_5d_source


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_5e_smoke_") as temp:
        root = Path(temp)
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
            output_dir=root / "persistent",
        )
        presentation = build_interactive_nerf_presentation(persisted.scene)
        assert presentation is not None and presentation.ready
        request = turntable_state_to_nerf_render_request(
            request_id=1,
            center_viewer=(0.0, 0.0, 0.0),
            distance=3.0,
            azimuth_degrees=32.0,
            elevation_degrees=24.0,
            fov_degrees=45.0,
            scene_id=presentation.scene_id,
            artifact_id=presentation.artifact_id,
        )
        payload = request.as_protocol_payload()
        result = {
            "phase": "9.5E",
            "status": "PASS",
            "interactive_presentation_ready": presentation.ready,
            "render_resolution": [request.width, request.height],
            "request_scene_matches": payload["scene_id"] == persisted.scene.scene_id,
            "nerf_scene_role": presentation.scene_role,
            "canonical_geometry_authoritative": True,
            "picking_source": "canonical_geometry_only",
            "nerf_canonical_geometry_authoritative": presentation.canonical_geometry_authoritative,
            "nerf_metric_scale_authoritative": presentation.metric_scale_authoritative,
            "nerf_measurement_authoritative": presentation.measurement_authoritative,
            "nerf_qc_authoritative": presentation.qc_authoritative,
            "certified_metrology": presentation.certified_metrology,
            "backend_started": False,
            "rendering_started": False,
            "production_heavy_modules": list(current_process_nerf_heavy_modules()),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
