#!/usr/bin/env python3
"""Interactive Phase 8.8D3 offline sparse-SfM health preview with no backend."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.offline_reconstruction_visualization import OFFLINE_VISUALIZATION_SCHEMA
from app.widgets.mode_tabs import AppMode


def _pose(angle: float, radius: float = 2.1) -> list[list[float]]:
    theta = np.deg2rad(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [
        radius * np.cos(theta),
        0.20 * np.sin(theta * 2.0),
        radius * np.sin(theta),
    ]
    return matrix.tolist()


def main() -> int:
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page

    root = Path(tempfile.mkdtemp(prefix="phase8_8d3_preview_"))
    session_id = "88888888-8888-4888-8888-888888888883"
    rng = np.random.default_rng(8_803)
    points = rng.normal(size=(4200, 3)).astype(np.float64)
    points[:, 0] *= 2.1
    points[:, 1] *= 0.8
    points[:, 2] *= 1.4
    colors = np.clip(
        (points - points.min(0)) / np.maximum(np.ptp(points, axis=0), 1e-6) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    cameras = [
        {
            "estimate_id": f"preview-camera-{index}",
            "keyframe_id": f"preview-kf-{index:02d}",
            "frame_id": f"preview-frame-{index:02d}",
            "T_reconstruction_from_camera": _pose(float(index) * (360.0 / 24.0)),
        }
        for index in range(24)
    ]
    artifact = root / "offline_visualization.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": OFFLINE_VISUALIZATION_SCHEMA,
                "session_id": session_id,
                "backend_id": "pycolmap-offline-sfm-preview",
                "backend_version": "preview",
                "pose_convention": "T_reconstruction_from_camera",
                "scale_status": "arbitrary",
                "alignment_status": "arbitrary",
                "registered_camera_total": len(cameras),
                "registered_cameras": cameras,
                "landmark_total": int(points.shape[0]),
                "landmarks_reconstruction": points.tolist(),
                "landmark_colors_rgb": colors.tolist(),
                "mean_reprojection_error_px": 0.52,
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    input_keyframes = 26
    registered = len(cameras)
    landmarks = int(points.shape[0])
    observations = landmarks * 4
    page._controller.offline_result_ready.emit(
        {
            "status": "completed",
            "session_id": session_id,
            "sfm_summary": {
                "registered_keyframes": registered,
                "input_keyframes": input_keyframes,
                "registration_ratio": registered / input_keyframes,
                "landmarks": landmarks,
                "observations": observations,
                "mean_reprojection_error_px": 0.52,
                "quality_assessment": {
                    "status": "accepted",
                    "usable": True,
                    "registration_ratio": registered / input_keyframes,
                    "landmarks_per_registered_keyframe": landmarks / registered,
                    "mean_track_length": observations / landmarks,
                    "mean_reprojection_error_px": 0.52,
                    "reasons": [],
                },
                "gauge": {"scale_status": "arbitrary"},
            },
            "artifacts": {"offline_visualization_json": str(artifact.resolve())},
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
        str(root),
    )
    page._offline_camera_combo.setCurrentIndex(13)
    page._readiness_badge.setText("SFM HEALTH PREVIEW • NO BACKEND")
    page._viewport_state.setText("D3 synthetic offline SfM health inspection")
    window.show()
    code = app.exec()
    shutil.rmtree(root, ignore_errors=True)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
