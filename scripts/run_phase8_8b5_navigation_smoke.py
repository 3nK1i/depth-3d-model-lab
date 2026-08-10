#!/usr/bin/env python3
"""Offscreen application-level smoke for Phase 8.8B5 navigation enablement."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.widgets.mode_tabs import AppMode
from core.reconstruction import LiveCaptureRuntimeState


OUTPUT_DIR = Path("/tmp/phase8_8b5_navigation")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    controller = window._reconstruction_controller

    mapping = {
        AppMode.LIVE_3D: window._page,
        AppMode.INSPECTION: window._inspection_page,
        AppMode.STABLE_VIDEO: window._stable_video_page,
        AppMode.SNAPSHOT: window._snapshot_page,
        AppMode.RECONSTRUCTION: window._reconstruction_page,
        AppMode.BENCHMARK: window._benchmark_page,
    }
    errors: list[str] = []
    visits: list[str] = []

    try:
        if not all(button.isEnabled() for button in window._mode_tabs._buttons.values()):
            errors.append("Not all primary navigation buttons are enabled at idle.")

        for _ in range(20):
            for mode in (
                AppMode.RECONSTRUCTION,
                AppMode.LIVE_3D,
                AppMode.INSPECTION,
                AppMode.SNAPSHOT,
                AppMode.BENCHMARK,
                AppMode.STABLE_VIDEO,
            ):
                window._mode_tabs.set_active_mode(mode)
                app.processEvents()
                visits.append(mode.value)
                if window._mode_stack.currentWidget() is not mapping[mode]:
                    errors.append(f"Wrong page routed for {mode.value}.")
                    break
                if controller.is_active:
                    errors.append("Reconstruction backend started during navigation-only smoke.")
                    break
            if errors:
                break

        summary = {
            "phase": "8.8B5",
            "pass": not errors,
            "errors": errors,
            "page_count": window._mode_stack.count(),
            "reconstruction_navigation_enabled": window._mode_tabs._buttons[
                AppMode.RECONSTRUCTION
            ].isEnabled(),
            "reconstruction_page_index": window._mode_stack.indexOf(window._reconstruction_page),
            "live_state": controller.live_state.value,
            "offline_state": controller.offline_state.value,
            "controller_active": controller.is_active,
            "visit_count": len(visits),
            "visited_modes": sorted(set(visits)),
            "backend_autostart": controller.is_active,
        }
        path = OUTPUT_DIR / "summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "Phase 8.8B5 Reconstruction navigation smoke: "
            + ("PASS" if summary["pass"] else "FAIL")
        )
        print(
            f"page_count={summary['page_count']} reconstruction_index={summary['reconstruction_page_index']} "
            f"visits={summary['visit_count']}"
        )
        print(
            f"live={summary['live_state']} offline={summary['offline_state']} "
            f"controller_active={summary['controller_active']}"
        )
        print(f"summary={path}")
        return 0 if summary["pass"] else 1
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
