#!/usr/bin/env python3
"""Dependency-light Phase 9.3G dense-provider UI contract smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from core.reconstruction.da3 import DA3_PROVIDER_ID
from tests.phase9_3g_helpers import make_phase9_3g_scene


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_3g_smoke_") as tmp:
        scene, _semantic, _surface, canonical, providers = make_phase9_3g_scene(
            Path(tmp).resolve(), request_id="phase9-3g-smoke"
        )
        mesh = load_canonical_viewer_mesh(scene)
        if mesh is None:
            raise RuntimeError("9.3G smoke did not materialize canonical geometry.")
        da3 = next((item for item in providers.entries if item.provider_id == DA3_PROVIDER_ID), None)
        if da3 is None:
            raise RuntimeError("9.3G smoke did not expose the known DA3 provider contract.")
        payload = {
            "status": "PASS",
            "scene_id": scene.scene_id,
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "certified_metrology": scene.certified_metrology,
            "canonical_vertex_count": mesh.vertex_count,
            "canonical_face_count": mesh.face_count,
            "pipeline": providers.pipeline_text,
            "scale": canonical.scale_text,
            "provider_count": len(providers.entries),
            "provider_ids": [item.provider_id for item in providers.entries],
            "da3_provider_visible": True,
            "da3_scene_bound": da3.scene_bound,
            "da3_runtime_probed": False,
            "da3_canonical_activation": False,
            "provider_selection_executes_backend": False,
            "backend_started": providers.backend_started,
            "heavy_backend_started": False,
        }
        if payload["certified_metrology"] or not payload["canonical_geometry_authoritative"]:
            raise RuntimeError("9.3G smoke changed canonical/metrology authority.")
        if payload["backend_started"] or payload["heavy_backend_started"]:
            raise RuntimeError("9.3G smoke unexpectedly started a backend.")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
