#!/usr/bin/env python3
"""Interactive 1366x768 preview for Phase 8.8G4 failure/recovery UX."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_controller import ReconstructionSourceSelection, ReconstructionWorkflow
from app.application_controller import CameraEnumerator

SESSION = "77777777-7777-4777-8777-777777777777"


def main() -> int:
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    controller = window._reconstruction_controller

    source = Path("/tmp/phase8_8g4_preview_sequence").resolve()
    source.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (source / f"frame-{index}.png").write_bytes(b"g4-preview")
    selection = ReconstructionSourceSelection(
        workflow=ReconstructionWorkflow.IMAGE_SEQUENCE,
        ready=True,
        display_name="recovery-sequence",
        detail="3 supported images • recovery preview",
        path=source,
        item_count=3,
    )
    controller._source_selection = selection
    controller._offline_controller._session_id = SESSION
    controller.source_selection_changed.emit(selection)
    controller._offline_controller._set_state(OfflineReconstructionRunState.FAILED)
    page.set_error("Example isolated worker failure • source remains ready for a fresh retry")
    controller._emit_session_model_changed()
    app.processEvents()

    window.resize(1366, 768)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
