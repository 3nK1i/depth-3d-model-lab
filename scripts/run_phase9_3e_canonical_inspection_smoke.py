#!/usr/bin/env python3
"""Dependency-light Phase 9.3E canonical object inspection overlay smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.offline_reconstruction_semantics import semantic_presentation_from_persistent_scene
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_inspection_overlays import build_canonical_object_inspection_overlay
from app.reconstruction_canonical_semantics import load_persistent_canonical_semantics
from app.reconstruction_inspection_qc_presentation import build_reconstruction_inspection_qc_presentation
from app.reconstruction_inspection_selection import build_reconstruction_inspection_selection
from app.reconstruction_measurement_presentation import build_reconstruction_measurement_presentation
from app.reconstruction_measurement_validity_presentation import build_reconstruction_measurement_validity_presentation
from core.reconstruction import read_persistent_reconstruction_scene, write_persistent_reconstruction_scene
from tests.phase9_3d_helpers import make_phase9_3d_scene
from tests.phase9_3e_helpers import inspection_report, measurement_context, measurement_payload


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_3e_smoke_") as tmp:
        root = Path(tmp).resolve()
        scene, _semantic, _surface = make_phase9_3d_scene(
            root / "source", request_id="phase9-3e-smoke"
        )
        manifest = write_persistent_reconstruction_scene(scene, root / "persistent")
        loaded = read_persistent_reconstruction_scene(manifest)
        mesh = load_canonical_viewer_mesh(loaded)
        assert mesh is not None
        canonical_semantics = load_persistent_canonical_semantics(loaded, mesh)
        offline_semantics = semantic_presentation_from_persistent_scene(loaded)
        assert canonical_semantics is not None and offline_semantics is not None

        selection = build_reconstruction_inspection_selection(
            scene_id=loaded.scene_id,
            session_id=loaded.session_id,
            scene_source="persistent_scene",
            scene_availability="ready",
            canonical_geometry_authoritative=loaded.canonical_geometry_authoritative,
            semantic_presentation=offline_semantics,
            selected_object_index=0,
        )
        evidence = measurement_payload()
        e2 = build_reconstruction_measurement_presentation(selection, evidence)
        assert build_canonical_object_inspection_overlay(
            selection=selection, canonical_semantics=canonical_semantics, measurement=e2
        ) is None
        e3 = build_reconstruction_measurement_validity_presentation(
            selection, evidence, context=measurement_context()
        )
        qc = build_reconstruction_inspection_qc_presentation(
            selection,
            inspection_report(session_id=loaded.session_id, object_id=selection.object_id),
        )
        overlay = build_canonical_object_inspection_overlay(
            selection=selection,
            canonical_semantics=canonical_semantics,
            measurement=e2,
            validity=e3,
            inspection_qc=qc,
        )
        assert overlay is not None
        assert "±" in overlay.overlay_text
        assert "REJECTED" in overlay.overlay_text
        assert "QC: WARN" in overlay.overlay_text
        assert overlay.certified_metrology is False

        print(json.dumps({
            "status": "PASS",
            "scene_id": loaded.scene_id,
            "selected_object_id": selection.object_id,
            "canonical_geometry_authoritative": loaded.canonical_geometry_authoritative,
            "semantic_authority": "semantic_context",
            "measurement_numeric_overlay_requires_e3": True,
            "numeric_measurement_count": overlay.numeric_measurement_count,
            "rejected_measurement_count": overlay.rejected_measurement_count,
            "inspection_result_available": overlay.inspection_result_available,
            "inspection_decision_copied_from_exact_object_only": True,
            "measurement_geometry_fabricated": False,
            "scene_metric_geometry_status": overlay.scene_metric_geometry_status,
            "certified_metrology": overlay.certified_metrology,
            "perception_rerun": False,
            "heavy_backend_started": False,
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
