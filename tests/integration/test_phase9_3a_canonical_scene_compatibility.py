from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_3a_smoke_uses_persistent_scene_and_authority_contracts_without_backend_start() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_phase9_3a_canonical_scene_smoke.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert payload["canonical_geometry_authoritative"] is True
    assert payload["canonical_surface_resource"] == "materialized"
    assert payload["viewer_resource_ready_for_9_3b"] is True
    assert payload["legacy_scale_state"] == "not_qualified"
    assert payload["qualified_scale_state"] == "qualified"
    assert payload["certified_metrology"] is False
    assert payload["mesh_parsed_by_9_3a"] is False
    assert payload["backend_started_by_9_3a"] is False


def test_3a_does_not_modify_locked_persistence_schema_or_add_canonical_mesh_renderer() -> None:
    persistent = (ROOT / "core/reconstruction/persistent_scene.py").read_text(encoding="utf-8")
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    canvas = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    presentation = (ROOT / "app/reconstruction_canonical_scene_presentation.py").read_text(encoding="utf-8")

    assert 'PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION = 1' in persistent
    assert '"reconstruction_ui_enabled": False' in persistent
    assert "Reconstruction UI remains disabled until Phase 8.8." in persistent
    assert "build_canonical_scene_presentation" in page
    assert "surface_obj_path" in presentation
    assert "set_canonical_mesh" not in canvas
    assert "scene.visuals.Mesh" not in canvas


def test_3a_keeps_heavy_runtime_and_appearance_authority_boundaries_unchanged() -> None:
    presentation = (ROOT / "app/reconstruction_canonical_scene_presentation.py").read_text(encoding="utf-8")
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    combined = presentation + "\n" + page
    for forbidden in (
        "import pycolmap",
        "import gsplat",
        "import nerfstudio",
        "import depth_anything_3",
        "from depth_anything_3",
        "import tinycudann",
    ):
        assert forbidden not in combined
    assert "Appearance only • canonical geometry remains authoritative" in page
    assert "metrology not certified" in combined
