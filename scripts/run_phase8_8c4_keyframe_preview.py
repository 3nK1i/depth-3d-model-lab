#!/usr/bin/env python3
"""Interactive Phase 8.8C4 keyframe/map-structure preview with no backend."""

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
from scripts.run_phase8_8c3_sparse_map_preview import _sparse_scene


def _pose(angle_deg: float, radius: float = 0.55) -> np.ndarray:
    yaw = radians(angle_deg)
    c, s = cos(yaw), sin(yaw)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    pose[:3, 3] = [radius * sin(yaw), -0.08 * sin(yaw * 0.7), radius * cos(yaw)]
    return pose


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
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
    page._viewport_state.setText("C4 synthetic keyframe/map-structure preview")
    page._readiness_badge.setText("KEYFRAME MAP PREVIEW • NO BACKEND")
    page._readiness_badge.setToolTip(
        "Synthetic display-only keyframes, sparse landmarks and trajectory; no worker is running."
    )

    sparse = _sparse_scene()
    canvas.update_sparse_landmarks(sparse, total_count=sparse.shape[0], revision=11)
    keyframe_angles = np.linspace(-110.0, 70.0, 18)
    keyframe_poses = np.asarray([_pose(float(value), radius=0.58) for value in keyframe_angles])
    keyframe_ids = tuple(f"preview-keyframe-{index}" for index in range(len(keyframe_angles)))
    canvas.update_keyframe_poses(
        keyframe_poses,
        keyframe_ids=keyframe_ids,
        total_count=len(keyframe_ids),
        revision=11,
    )
    angle = [72.0]
    for value in np.linspace(-110.0, angle[0], 72):
        canvas.update_live_pose(_pose(float(value)))
    canvas.fit_scene()

    timer = QTimer(window)
    timer.setInterval(80)

    def tick() -> None:
        angle[0] += 1.4
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
