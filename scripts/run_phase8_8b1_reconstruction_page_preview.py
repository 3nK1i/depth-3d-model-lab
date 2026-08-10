#!/usr/bin/env python3
"""Standalone visual preview for the disabled Phase 8.8B1 Reconstruction page.

This helper intentionally bypasses primary navigation only inside this preview
process so the page can be inspected before the production tab is enabled.
It starts no reconstruction backend.
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

import app.application_controller as application_controller_module
from app.main_window import MainWindow
from app.theme import apply_theme


def main() -> int:
    # Avoid camera enumeration during a UI-only page preview.
    application_controller_module.CameraEnumerator.start = lambda _self: None

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window._mode_stack.setCurrentWidget(window._reconstruction_page)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
