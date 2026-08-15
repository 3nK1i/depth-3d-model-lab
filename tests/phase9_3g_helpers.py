from __future__ import annotations

from pathlib import Path

from app.reconstruction_canonical_scene_presentation import build_canonical_scene_presentation
from app.reconstruction_dense_provider_presentation import build_dense_provider_presentation
from tests.phase9_3f_helpers import make_phase9_3f_scene


def make_phase9_3g_scene(tmp_path: Path, *, request_id: str = "phase9-3g-test"):
    scene, semantic, surface = make_phase9_3f_scene(tmp_path, request_id=request_id)
    canonical = build_canonical_scene_presentation(scene)
    providers = build_dense_provider_presentation(scene, canonical)
    return scene, semantic, surface, canonical, providers
