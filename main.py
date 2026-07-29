"""Entry point for EIA TECH — Depth & 3D Model Lab.

Run with: python main.py
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.settings import ORG_NAME
from app.settings import APP_NAME as SETTINGS_APP_NAME
from app.theme import apply_theme


def main() -> int:
    QApplication.setOrganizationName(ORG_NAME)
    QApplication.setApplicationName(SETTINGS_APP_NAME)

    app = QApplication(sys.argv)
    apply_theme(app)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
