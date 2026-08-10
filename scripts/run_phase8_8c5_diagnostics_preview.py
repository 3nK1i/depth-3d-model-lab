#!/usr/bin/env python3
"""Interactive Phase 8.8C5 tracking/map diagnostics preview with no backend."""

from __future__ import annotations

import argparse
from math import cos, radians, sin
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import LiveCaptureRuntimeState
from scripts.run_phase8_8c3_sparse_map_preview import _sparse_scene


def _pose(angle_deg: float, radius: float = 0.55) -> np.ndarray:
    yaw = radians(angle_deg)
    c, s = cos(yaw), sin(yaw)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    pose[:3, 3] = [radius * sin(yaw), -0.08 * sin(yaw * 0.7), radius * cos(yaw)]
    return pose


def _observation(
    *,
    angle_deg: float,
    sequence_index: int,
    revision: int,
    keyframes: int,
    landmarks: int,
    latency_ms: float,
    state: str = "tracking",
):
    return SimpleNamespace(
        tracking_state=SimpleNamespace(value=state),
        quality=SimpleNamespace(
            processing_latency_ns=int(latency_ms * 1_000_000),
            tracked_features=0,
            inlier_features=0,
            inlier_ratio=None,
            reason="stella_pose_valid" if state == "tracking" else f"stella_{state}",
        ),
        pose=SimpleNamespace(T_reconstruction_from_camera=_pose(angle_deg)),
        is_keyframe=False,
        map_snapshot=SimpleNamespace(
            revision=revision,
            keyframe_count=keyframes,
            landmark_count=landmarks,
        ),
        sequence_index=sequence_index,
    )


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
    page._viewport_state.setText("C5 synthetic tracking diagnostics preview")
    page._readiness_badge.setText("TRACKING DIAGNOSTICS PREVIEW • NO BACKEND")
    page._readiness_badge.setToolTip(
        "Synthetic display-only diagnostics and map visuals; no reconstruction worker is running."
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

    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    # Seed two revisions so the HUD demonstrates map growth rather than a fake score.
    page.set_live_observation(
        _observation(
            angle_deg=70.0,
            sequence_index=70,
            revision=10,
            keyframes=17,
            landmarks=790,
            latency_ms=14.2,
        )
    )
    page.set_live_observation(
        _observation(
            angle_deg=72.0,
            sequence_index=71,
            revision=11,
            keyframes=18,
            landmarks=int(sparse.shape[0]),
            latency_ms=15.1,
        )
    )

    timer = QTimer(window)
    timer.setInterval(110)
    sequence = [72]

    def tick() -> None:
        sequence[0] += 1
        angle[0] += 1.2
        # Brief recovering pulse every 70 samples to preview the warning state.
        state = "recovering" if sequence[0] % 70 == 0 else "tracking"
        page.set_live_observation(
            _observation(
                angle_deg=angle[0],
                sequence_index=sequence[0],
                revision=11 + sequence[0] // 45,
                keyframes=18 + sequence[0] // 45,
                landmarks=int(sparse.shape[0]) + (sequence[0] // 45) * 38,
                latency_ms=13.4 + (sequence[0] % 9) * 0.45,
                state=state,
            )
        )

    if not args.static:
        timer.timeout.connect(tick)
        timer.start()

    window.showMaximized()
    result = app.exec()
    timer.stop()
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
