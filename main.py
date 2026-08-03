"""Entry point for EIA TECH — Depth & 3D Model Lab.

Run with: python main.py
"""

from __future__ import annotations

import logging
import sys

from core.runtime.logging_setup import configure_logging, install_exception_hooks
from core.version import RELEASE_NAME, __version__

logger = logging.getLogger(__name__)


def main() -> int:
    log_path = configure_logging(process_name="app")
    install_exception_hooks()
    logger.info(
        "Starting Depth & 3D Model Lab v%s (%s) with Python %s",
        __version__,
        RELEASE_NAME,
        sys.version.split()[0],
    )
    logger.info("Application log: %s", log_path)

    try:
        from PyQt6.QtWidgets import QApplication

        from app.main_window import MainWindow
        from app.settings import APP_NAME as SETTINGS_APP_NAME
        from app.settings import ORG_NAME
        from app.theme import apply_theme

        QApplication.setOrganizationName(ORG_NAME)
        QApplication.setApplicationName(SETTINGS_APP_NAME)
        QApplication.setApplicationVersion(__version__)

        app = QApplication(sys.argv)
        apply_theme(app)

        window = MainWindow()
        window.show()
        exit_code = int(app.exec())
        logger.info("Application exited with code %s", exit_code)
        return exit_code
    except Exception:  # noqa: BLE001 - top-level release diagnostic
        logger.exception("Application startup or event-loop failure")
        print(
            f"Depth & 3D Model Lab failed. Diagnostic log: {log_path}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
