from __future__ import annotations

from pathlib import Path

from app.offline_reconstruction_semantics import semantic_presentation_from_persistent_scene
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_inspection_overlays import build_canonical_object_inspection_overlay
from app.reconstruction_canonical_semantics import load_persistent_canonical_semantics
from app.reconstruction_inspection_qc_presentation import build_reconstruction_inspection_qc_presentation
from app.reconstruction_inspection_selection import build_reconstruction_inspection_selection
from app.reconstruction_measurement_presentation import build_reconstruction_measurement_presentation
from app.reconstruction_measurement_validity_presentation import build_reconstruction_measurement_validity_presentation
from tests.phase9_3d_helpers import make_phase9_3d_scene
from tests.phase9_3e_helpers import inspection_report, measurement_context, measurement_payload


def test_3e_existing_measurement_and_qc_bind_to_exact_persisted_canonical_object(tmp_path: Path) -> None:
    scene, semantic, _surface = make_phase9_3d_scene(tmp_path, request_id="phase9-3e-integration")
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None
    canonical_semantics = load_persistent_canonical_semantics(scene, mesh)
    offline_semantics = semantic_presentation_from_persistent_scene(scene)
    assert canonical_semantics is not None
    assert offline_semantics is not None

    selection = build_reconstruction_inspection_selection(
        scene_id=scene.scene_id,
        session_id=scene.session_id,
        scene_source="persistent_scene",
        scene_availability="ready",
        canonical_geometry_authoritative=scene.canonical_geometry_authoritative,
        semantic_presentation=offline_semantics,
        selected_object_index=0,
    )
    assert selection.object_id == semantic.objects[0].object_id

    evidence = measurement_payload()
    e2 = build_reconstruction_measurement_presentation(selection, evidence)
    # E2 is source value presentation only: no viewport numeric overlay yet.
    assert build_canonical_object_inspection_overlay(
        selection=selection, canonical_semantics=canonical_semantics, measurement=e2
    ) is None

    e3 = build_reconstruction_measurement_validity_presentation(
        selection, evidence, context=measurement_context()
    )
    qc = build_reconstruction_inspection_qc_presentation(
        selection,
        inspection_report(session_id=scene.session_id, object_id=selection.object_id),
    )
    overlay = build_canonical_object_inspection_overlay(
        selection=selection,
        canonical_semantics=canonical_semantics,
        measurement=e2,
        validity=e3,
        inspection_qc=qc,
    )
    assert overlay is not None
    assert overlay.scene_id == scene.scene_id
    assert overlay.object_id == selection.object_id
    assert overlay.numeric_measurement_count == 1
    assert overlay.rejected_measurement_count == 1
    assert overlay.inspection_result_available is True
    assert "42.0 mm ± 1.5 mm" in overlay.overlay_text
    assert "Depth: REJECTED" in overlay.overlay_text
    assert "QC: WARN" in overlay.overlay_text
    assert "canonical geometry unchanged" in overlay.overlay_text
    assert overlay.certified_metrology is False
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
