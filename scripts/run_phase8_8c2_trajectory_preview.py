#!/usr/bin/env python3
"""Interactive Phase 8.8C2 live-trajectory/orientation preview.

Uses only synthetic camera poses. No capture source, reconstruction backend, or
heavy external environment is started.
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
    # Deliberately non-circular path so trajectory scale and perspective are obvious.
    pose[:3, 3] = [0.42 * sin(yaw), -0.08 * sin(yaw * 0.7), 0.55 * cos(yaw)]
    return pose


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true", help="Build a fixed synthetic trajectory.")
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
    page._viewport_state.setText("C2 synthetic live trajectory preview")
    page._readiness_badge.setText("TRAJECTORY PREVIEW • NO BACKEND")
    page._readiness_badge.setToolTip(
        "Synthetic display-only camera path; no reconstruction worker is running."
    )

    # Seed a visible path immediately so --static is useful for screenshots.
    angle = [18.0]
    for value in np.linspace(-42.0, angle[0], 46):
        canvas.update_live_pose(_pose(float(value)))
    canvas.fit_scene()

    timer = QTimer(window)
    timer.setInterval(80)

    def tick() -> None:
        angle[0] += 1.6
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
