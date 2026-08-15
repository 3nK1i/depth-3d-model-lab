from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.reconstruction_appearance_presentation import build_reconstruction_appearance_presentation
from app.reconstruction_canonical_scene_presentation import build_canonical_scene_presentation
from app.reconstruction_digital_twin_composition import (
    DigitalTwinCompositionState,
    build_digital_twin_scene_composition,
)
from core.reconstruction import persistent_scene_from_pipeline_result
from tests.reconstruction_f5_helpers import run_f5_fixture

ROOT = Path(__file__).resolve().parents[2]


def test_96b_legacy_scene_composes_canonical_authority_without_optional_layers(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "legacy",
        request_id="phase9-6b-legacy",
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    canonical = build_canonical_scene_presentation(scene)
    appearance = build_reconstruction_appearance_presentation(scene)
    composition = build_digital_twin_scene_composition(
        canonical,
        canonical_surface_loaded=False,
        canonical_surface_visible=False,
        appearance_presentation=appearance,
    )
    assert composition.state in {DigitalTwinCompositionState.CANONICAL, DigitalTwinCompositionState.COMPOSED}
    assert composition.scene_id == scene.scene_id
    assert composition.canonical_geometry_authoritative is True
    assert composition.layer("canonical_surface").available is True
    assert composition.active_view_representation == "canonical_geometry"
    assert composition.certified_metrology is False


def test_96b_scene_bound_layers_share_one_canonical_scene_identity() -> None:
    scene_id = "scene-96b-bound"
    canonical = SimpleNamespace(
        available=True,
        scene_id=scene_id,
        scene_state="ready",
        artifact_id="canonical",
        canonical_geometry_authoritative=True,
        certified_metrology=False,
    )
    semantic = SimpleNamespace(
        scene_id=scene_id,
        semantic_artifact_id="semantic",
        certified_metrology=False,
    )
    camera = SimpleNamespace(scene_id=scene_id, artifact_id="camera", certified_metrology=False)
    appearance = SimpleNamespace(
        scene_id=scene_id,
        certified_metrology=False,
        entries=(
            SimpleNamespace(
                artifact_id="gaussian",
                representation_kind="gaussian_scene",
                canonical_geometry_authoritative=False,
                certified_metrology=False,
            ),
        ),
    )
    composition = build_digital_twin_scene_composition(
        canonical,
        canonical_surface_loaded=True,
        canonical_surface_visible=True,
        semantic_presentation=semantic,
        semantic_visible=True,
        camera_presentation=camera,
        camera_visible=True,
        appearance_presentation=appearance,
        active_view_representation="gaussian_scene",
        gaussian_visible=True,
    )
    assert composition.composition_text == "Canonical • Semantics • Cameras • Gaussian"
    assert composition.layer("gaussian_scene").authority_text.startswith("Appearance only")
    assert composition.layer("semantic_objects").canonical_geometry_authoritative is False
    assert composition.layer("world_cameras").canonical_geometry_authoritative is False
