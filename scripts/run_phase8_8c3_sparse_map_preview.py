#!/usr/bin/env python3
"""Interactive Phase 8.8C3 sparse-map + trajectory preview.

Uses synthetic camera poses and sparse landmarks only. No capture source,
Stella worker, PyCOLMAP, or heavy reconstruction backend is started.
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
    pose[:3, 3] = [0.42 * sin(yaw), -0.08 * sin(yaw * 0.7), 0.55 * cos(yaw)]
    return pose


def _sparse_scene() -> np.ndarray:
    rng = np.random.default_rng(88)
    floor = np.column_stack(
        [
            rng.uniform(-0.85, 0.85, 420),
            rng.uniform(-0.30, 0.18, 420),
            rng.uniform(-0.80, 0.80, 420),
        ]
    )
    wall = np.column_stack(
        [
            rng.uniform(-0.75, 0.75, 260),
            rng.uniform(-0.55, 0.15, 260),
            np.full(260, 0.88) + rng.normal(0.0, 0.012, 260),
        ]
    )
    column = np.column_stack(
        [
            np.full(120, -0.45) + rng.normal(0.0, 0.018, 120),
            rng.uniform(-0.55, 0.12, 120),
            np.full(120, 0.18) + rng.normal(0.0, 0.018, 120),
        ]
    )
    return np.ascontiguousarray(np.vstack([floor, wall, column]), dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true", help="Keep the synthetic camera fixed after seeding.")
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
    page._viewport_state.setText("C3 synthetic sparse-map preview")
    page._readiness_badge.setText("SPARSE MAP PREVIEW • NO BACKEND")
    page._readiness_badge.setToolTip(
        "Synthetic display-only landmarks and camera path; no reconstruction worker is running."
    )

    sparse = _sparse_scene()
    canvas.update_sparse_landmarks(sparse, total_count=sparse.shape[0], revision=7)
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
