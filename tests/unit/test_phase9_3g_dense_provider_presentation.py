from __future__ import annotations

from pathlib import Path

import pytest

from app.reconstruction_canonical_scene_presentation import build_canonical_scene_presentation
from app.reconstruction_dense_provider_presentation import build_dense_provider_presentation
from core.reconstruction.da3 import DA3_PROVIDER_ID
from tests.phase9_3f_helpers import make_phase9_3f_scene
from tests.phase9_3g_helpers import make_phase9_3g_scene


def test_3g_presents_current_dense_surface_depth_and_known_da3_without_activation(tmp_path: Path) -> None:
    scene, _semantic, _surface, canonical, presentation = make_phase9_3g_scene(
        tmp_path / "providers", request_id="phase9-3g-providers"
    )

    assert presentation.available is True
    assert presentation.scene_id == scene.scene_id
    assert presentation.canonical_geometry_authoritative is True
    assert presentation.certified_metrology is False
    assert presentation.backend_started is False
    assert "canonical-dense-fusion" in presentation.pipeline_text
    assert "quality-aware-surface-mesh" in presentation.pipeline_text
    assert presentation.scale_text == canonical.scale_text

    by_id = {entry.provider_id: entry for entry in presentation.entries}
    assert "canonical-dense-fusion" in by_id
    assert "quality-aware-surface-mesh" in by_id
    assert "phase8-f3-bound-evidence" in by_id
    assert DA3_PROVIDER_ID in by_id

    assert by_id["canonical-dense-fusion"].scene_bound is True
    assert by_id["canonical-dense-fusion"].authority_text == "Canonical geometry authority"
    assert by_id["quality-aware-surface-mesh"].authority_text == "Canonical geometry authority"
    assert by_id["phase8-f3-bound-evidence"].authority_text == "Supporting geometry only"

    da3 = by_id[DA3_PROVIDER_ID]
    assert da3.scene_bound is False
    assert da3.external is True
    assert "runtime not probed" in da3.state_text.lower()
    assert "9.3H" in da3.state_text
    assert "supporting geometry only" in da3.authority_text.lower()
    assert "metric scale" in da3.authority_text.lower()


def test_3g_no_scene_is_fail_closed_and_backend_free() -> None:
    presentation = build_dense_provider_presentation(None, None)
    assert presentation.available is False
    assert presentation.entries == ()
    assert presentation.canonical_geometry_authoritative is False
    assert presentation.certified_metrology is False
    assert presentation.backend_started is False
    assert "9.3H" in presentation.integration_text


def test_3g_rejects_canonical_presentation_from_another_scene(tmp_path: Path) -> None:
    scene_a, _semantic_a, _surface_a = make_phase9_3f_scene(
        tmp_path / "a", request_id="phase9-3g-a"
    )
    scene_b, _semantic_b, _surface_b = make_phase9_3f_scene(
        tmp_path / "b", request_id="phase9-3g-b"
    )
    canonical_b = build_canonical_scene_presentation(scene_b)
    with pytest.raises(ValueError, match="not bound"):
        build_dense_provider_presentation(scene_a, canonical_b)


def test_3g_provider_inspection_never_changes_scene_authority(tmp_path: Path) -> None:
    scene, _semantic, _surface = make_phase9_3f_scene(
        tmp_path / "authority", request_id="phase9-3g-authority"
    )
    before = scene.as_dict()
    canonical = build_canonical_scene_presentation(scene)
    presentation = build_dense_provider_presentation(scene, canonical)
    after = scene.as_dict()

    assert before == after
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    assert presentation.backend_started is False
    assert presentation.certified_metrology is False
