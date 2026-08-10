#!/usr/bin/env python3
"""Interactive Phase 8.8C1 reconstruction-viewport preview using a synthetic pose.

No reconstruction backend, capture source, or heavy external environment is
started. The script exists only to inspect the real VisPy viewport, grid/axes,
current-camera frustum, and navigation controls in a normal desktop session.
"""

from __future__ import annotations

import argparse
from math import cos, radians, sin
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode


def _pose(angle_deg: float) -> np.ndarray:
    yaw = radians(angle_deg)
    c, s = cos(yaw), sin(yaw)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    pose[:3, 3] = [0.65 * sin(yaw), -0.15, 0.65 * cos(yaw)]
    return pose


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true", help="Show one fixed synthetic pose.")
    args = parser.parse_args()

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    K = np.asarray([[600.0, 0.0, 319.5], [0.0, 600.0, 239.5], [0.0, 0.0, 1.0]])
    canvas.configure_camera_geometry(intrinsics_matrix=K, image_size=(640, 480))
    page._viewport_placeholder.hide()
    page._viewport_state.setText("C1 synthetic camera preview")
    page._readiness_badge.setText("VISUALIZER PREVIEW • NO BACKEND")
    page._readiness_badge.setToolTip("Synthetic display-only camera motion; no reconstruction worker is running.")

    angle = [25.0]
    canvas.update_live_pose(_pose(angle[0]))
    canvas.fit_scene()

    timer = QTimer(window)
    timer.setInterval(90)

    def tick() -> None:
        angle[0] = (angle[0] + 1.6) % 360.0
        canvas.update_live_pose(_pose(angle[0]))

    if not args.static:
        timer.timeout.connect(tick)
        timer.start()

    window.showMaximized()
    result = app.exec()
    timer.stop()
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
