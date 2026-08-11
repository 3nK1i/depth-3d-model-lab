#!/usr/bin/env python3
"""Interactive Phase 8.8F1 appearance controls preview; no external backend."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import (
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture

OUTPUT = Path("/tmp/phase8_8f1_appearance_preview")


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _fixture, _providers, result, snapshots = run_f5_fixture(
        OUTPUT / "source",
        request_id="phase8-8f1-preview",
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, OUTPUT / "persistent")
    scene = read_persistent_reconstruction_scene(manifest)

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_persistent_scene(scene, str(manifest))
    page._readiness_badge.setText("F1 APPEARANCE CONTROLS • NO EXTERNAL BACKEND")
    page._viewport_state.setText("F1 appearance metadata only • rendering deferred")
    page._viewport_state.setToolTip(
        "Use the Appearance Representations selector to inspect persisted Gaussian/NeRF metadata. F1 does not render or switch the viewport."
    )
    window.show()
    app.processEvents()
    page._settings_scroll.verticalScrollBar().setValue(page._settings_scroll.verticalScrollBar().maximum())
    app.processEvents()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
