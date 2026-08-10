#!/usr/bin/env python3
"""Interactive Phase 8.8D1 offline sparse-visualization preview with no backend."""

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


def _pose(angle: float, radius: float = 1.6) -> list[list[float]]:
    theta = np.deg2rad(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [radius * np.cos(theta), 0.12 * np.sin(theta * 2.0), radius * np.sin(theta)]
    # Symbolic camera rotation is identity; D1 preview is about map structure/scale.
    return matrix.tolist()


def main() -> int:
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page

    root = Path(tempfile.mkdtemp(prefix="phase8_8d1_preview_"))
    session_id = "77777777-7777-4777-8777-777777777777"
    rng = np.random.default_rng(8_801)
    points = rng.normal(size=(2600, 3)).astype(np.float64)
    points[:, 0] *= 1.8
    points[:, 1] *= 0.65
    points[:, 2] *= 1.25
    colors = np.clip((points - points.min(0)) / np.maximum(np.ptp(points, axis=0), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    cameras = [
        {
            "estimate_id": f"preview-camera-{index}",
            "keyframe_id": f"preview-kf-{index:02d}",
            "frame_id": f"preview-frame-{index:02d}",
            "T_reconstruction_from_camera": _pose(float(index) * (360.0 / 18.0)),
        }
        for index in range(18)
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
                "mean_reprojection_error_px": 0.46,
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    page._controller.offline_result_ready.emit(
        {
            "status": "completed",
            "session_id": session_id,
            "sfm_summary": {
                "registered_keyframes": len(cameras),
                "input_keyframes": len(cameras),
                "landmarks": int(points.shape[0]),
                "mean_reprojection_error_px": 0.46,
                "gauge": {"scale_status": "arbitrary"},
            },
            "artifacts": {"offline_visualization_json": str(artifact.resolve())},
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
        str(root),
    )
    page._readiness_badge.setText("OFFLINE SFM PREVIEW • NO BACKEND")
    page._viewport_state.setText("D1 synthetic offline sparse reconstruction")
    window.show()
    code = app.exec()
    shutil.rmtree(root, ignore_errors=True)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
