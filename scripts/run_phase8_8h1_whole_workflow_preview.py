#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from tests.reconstruction_h1_helpers import create_checkpoint_fixture, run_whole_pipeline, write_ready_image_sequence


def main() -> int:
    CameraEnumerator.start = lambda self: None
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    page = window._reconstruction_page
    controller = window._reconstruction_controller

    root = Path("/tmp/phase8_8h1_preview").resolve()
    root.mkdir(parents=True, exist_ok=True)
    controller.select_image_sequence(write_ready_image_sequence(root))
    fixture, result = run_whole_pipeline()
    controller.adopt_pipeline_result(result, root / "scene")
    _checkpoint, prefix = create_checkpoint_fixture(fixture)
    created = controller.create_engine_checkpoint(
        plan=fixture.plan,
        providers=fixture.providers,
        stage_results=prefix,
        output_dir=root / "checkpoint",
    )
    controller.set_engine_resume_context(
        plan=fixture.plan,
        providers=fixture.providers,
        available_payload_refs=created.payload_refs,
    )

    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window.resize(1366, 768)
    window.show()
    page._session_group.set_expanded(True)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
