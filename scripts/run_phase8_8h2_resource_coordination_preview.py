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
from app.reconstruction_controller import ReconstructionController
from app.widgets.mode_tabs import AppMode
from core.runtime.state_machine import ApplicationState


def main() -> int:
    CameraEnumerator.start = lambda self: None
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    original_property = ReconstructionController.is_active
    ReconstructionController.is_active = property(lambda _self: True)
    window._inference_controller._state_machine._state = ApplicationState.DISCONNECTED
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window._on_reconstruction_activity_changed()
    window.resize(1366, 768)
    window.show()
    code = app.exec()
    ReconstructionController.is_active = original_property
    return code


if __name__ == "__main__":
    raise SystemExit(main())
