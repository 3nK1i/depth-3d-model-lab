#!/usr/bin/env python3
"""Interactive 1366x768 integrated Reconstruction UX preview for Phase 8.8G6."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.widgets.mode_tabs import AppMode
from tests.reconstruction_g6_helpers import write_full_review_evidence

OUTPUT = Path("/tmp/phase8_8g6_full_ux_review_preview").resolve()


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    fixture = write_full_review_evidence(OUTPUT / "fixture")
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    controller = window._reconstruction_controller
    page = window._reconstruction_page

    controller.select_image_sequence(fixture.source_dir)
    controller.load_persistent_scene(fixture.scene_dir)
    controller.load_engine_checkpoint(fixture.checkpoint_manifest)
    controller.set_engine_resume_context(
        plan=fixture.plan,
        providers=fixture.providers,
        available_payload_refs=fixture.checkpoint.payload_refs,
    )
    page.set_error("Full UX review • previous attempt failed; scene/checkpoint evidence retained")
    page.set_offline_state(OfflineReconstructionRunState.FAILED)

    window.resize(1366, 768)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
