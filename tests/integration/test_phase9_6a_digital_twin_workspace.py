from __future__ import annotations

from pathlib import Path

from app.reconstruction_appearance_presentation import build_reconstruction_appearance_presentation
from app.reconstruction_canonical_scene_presentation import build_canonical_scene_presentation
from app.reconstruction_digital_twin_workspace import (
    DigitalTwinWorkspaceState,
    build_digital_twin_workspace_presentation,
)
from core.reconstruction import persistent_scene_from_pipeline_result
from tests.reconstruction_f5_helpers import run_f5_fixture

ROOT = Path(__file__).resolve().parents[2]


def test_96a_legacy_canonical_scene_without_neural_appearance_remains_fully_usable(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "legacy",
        request_id="phase9-6a-legacy",
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    canonical = build_canonical_scene_presentation(scene)
    appearance = build_reconstruction_appearance_presentation(scene)
    workspace = build_digital_twin_workspace_presentation(
        canonical,
        canonical_surface_loaded=False,
        semantic_context_available=False,
        appearance_presentation=appearance,
    )
    assert workspace.state in {DigitalTwinWorkspaceState.CANONICAL, DigitalTwinWorkspaceState.COMPOSED}
    assert workspace.canonical_geometry_authoritative is True
    assert workspace.active_view_representation == "canonical_geometry"
    assert workspace.certified_metrology is False
    assert workspace.scale_state == "not_qualified"


def test_96a_does_not_change_locked_persistence_controller_or_canvas_authority_contracts() -> None:
    persistent = (ROOT / "core/reconstruction/persistent_scene.py").read_text(encoding="utf-8")
    controller = (ROOT / "app/reconstruction_controller.py").read_text(encoding="utf-8")
    canvas = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    workspace = (ROOT / "app/reconstruction_digital_twin_workspace.py").read_text(encoding="utf-8")

    assert "PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION = 1" in persistent
    assert "class ReconstructionController(QObject):" in controller
    assert "def set_canonical_mesh(" in canvas
    assert "measurement_point_picked = pyqtSignal(object)" in canvas
    assert "core.reconstruction" not in workspace
    assert "PyQt6" not in workspace
    assert "presentation-only" in workspace.lower()
