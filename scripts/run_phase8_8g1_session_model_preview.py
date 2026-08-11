#!/usr/bin/env python3
"""Interactive Phase 8.8G1 session-model presentation preview."""

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
        path=Path("/tmp/phase8_8g1_session_model/source"),
    )
    artifacts = tuple(
        SimpleNamespace(
            artifact_id=f"g1-preview-{index}",
            stage=SimpleNamespace(value=stage),
            kind=SimpleNamespace(value=kind),
            authority=SimpleNamespace(value=authority),
            provider_id=provider,
        )
        for index, (stage, kind, authority, provider) in enumerate(
            (
                ("surface_mesh", "surface_mesh", "canonical_geometry", "mesh-provider"),
                ("semantic_geometry", "semantic_scene", "semantic_context", "semantic-provider"),
                ("future_visual", "future_visual_representation", "appearance_only", "visual-provider"),
            ),
            start=1,
        )
    )
    providers = tuple(
        SimpleNamespace(
            provider_id=provider,
            provider_version="1.0",
            execution=SimpleNamespace(value="external_process"),
            stages=(SimpleNamespace(value=stage),),
        )
        for provider, stage in (
            ("mesh-provider", "surface_mesh"),
            ("semantic-provider", "semantic_geometry"),
            ("visual-provider", "future_visual"),
        )
    )
    scene = SimpleNamespace(
        scene_id="prs-g1-preview",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=artifacts,
        provider_provenance=providers,
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        created_at_utc="2026-08-11T12:00:00+00:00",
    )
    checkpoint = SimpleNamespace(
        checkpoint_id="99999999-9999-4999-8999-999999999999",
        session_id=SESSION,
        next_stage=SimpleNamespace(value="dense_geometry"),
        resume_allowed=True,
        provider_versions=(),
        stage_results=(),
        created_at_utc="2026-08-11T11:00:00+00:00",
    )
    model = build_reconstruction_session_model(
        source_selection=selection,
        scene=scene,
        scene_manifest="/tmp/phase8_8g1_session_model/persistent_reconstruction_scene.json",
        checkpoint=checkpoint,
        checkpoint_manifest="/tmp/phase8_8g1_session_model/reconstruction_engine_checkpoint.json",
        scene_availability=SimpleNamespace(value="ready"),
    )

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window._reconstruction_page.set_session_model(model)
    window.resize(1366, 768)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
