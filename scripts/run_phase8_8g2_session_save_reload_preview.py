#!/usr/bin/env python3
"""Interactive Phase 8.8G2 persistent-scene save/reload UX preview."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.reconstruction_session_model import build_reconstruction_session_model
from app.widgets.mode_tabs import AppMode

SESSION = "77777777-7777-4777-8777-777777777777"


def main() -> int:
    selection = SimpleNamespace(
        workflow=SimpleNamespace(value="image_sequence"),
        ready=True,
        display_name="Generic industrial reconstruction",
        detail="Use-case-agnostic image-sequence session",
        path=Path("/tmp/phase8_8g2_session_save_reload/source"),
    )
    scene = SimpleNamespace(
        scene_id="prs-g2-preview",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=(
            SimpleNamespace(
                artifact_id="g2-preview-mesh",
                stage=SimpleNamespace(value="surface_mesh"),
                kind=SimpleNamespace(value="surface_mesh"),
                authority=SimpleNamespace(value="canonical_geometry"),
                provider_id="mesh-provider",
            ),
            SimpleNamespace(
                artifact_id="g2-preview-semantic",
                stage=SimpleNamespace(value="semantic_geometry"),
                kind=SimpleNamespace(value="semantic_scene"),
                authority=SimpleNamespace(value="semantic_context"),
                provider_id="semantic-provider",
            ),
        ),
        provider_provenance=(
            SimpleNamespace(
                provider_id="mesh-provider",
                provider_version="1.0",
                execution=SimpleNamespace(value="in_process"),
                stages=(SimpleNamespace(value="surface_mesh"),),
            ),
            SimpleNamespace(
                provider_id="semantic-provider",
                provider_version="1.0",
                execution=SimpleNamespace(value="in_process"),
                stages=(SimpleNamespace(value="semantic_geometry"),),
            ),
        ),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        created_at_utc="2026-08-11T12:00:00+00:00",
    )
    model = build_reconstruction_session_model(
        source_selection=selection,
        scene=scene,
        scene_manifest="/tmp/phase8_8g2_session_save_reload/persistent_reconstruction_scene.json",
        scene_availability=SimpleNamespace(value="ready"),
    )

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_session_model(model)
    window.resize(1366, 768)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
