#!/usr/bin/env python3
"""Dependency-light Phase 9.3D canonical semantic-object persistence smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_semantics import load_persistent_canonical_semantics
from core.reconstruction import read_persistent_reconstruction_scene, write_persistent_reconstruction_scene
from tests.phase9_3d_helpers import make_phase9_3d_scene


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_3d_smoke_") as tmp:
        root = Path(tmp).resolve()
        scene, semantic, _surface = make_phase9_3d_scene(
            root / "source", request_id="phase9-3d-smoke"
        )
        manifest = write_persistent_reconstruction_scene(scene, root / "persistent")
        loaded = read_persistent_reconstruction_scene(manifest)
        mesh = load_canonical_viewer_mesh(loaded)
        assert mesh is not None
        presentation = load_persistent_canonical_semantics(loaded, mesh)
        assert presentation is not None
        assert presentation.object_ids == tuple(item.object_id for item in semantic.objects)
        assert presentation.surface_bound_object_count == len(semantic.objects)
        assert presentation.surface_highlight_object_count == len(semantic.objects)
        assert loaded.canonical_geometry_authoritative is True
        assert loaded.certified_metrology is False

        print(json.dumps({
            "status": "PASS",
            "scene_id": loaded.scene_id,
            "scene_fingerprint_sha256": loaded.scene_fingerprint_sha256,
            "canonical_artifact_id": mesh.artifact_id,
            "canonical_vertex_count": mesh.vertex_count,
            "canonical_face_count": mesh.face_count,
            "semantic_artifact_id": presentation.semantic_artifact_id,
            "semantic_provider_id": presentation.semantic_provider_id,
            "semantic_object_count": len(presentation.objects),
            "surface_bound_object_count": presentation.surface_bound_object_count,
            "surface_highlight_object_count": presentation.surface_highlight_object_count,
            "semantic_authority": "semantic_context",
            "canonical_geometry_authoritative": loaded.canonical_geometry_authoritative,
            "certified_metrology": loaded.certified_metrology,
            "semantic_geometry_fabricated": False,
            "perception_rerun": False,
            "heavy_backend_started": False,
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
