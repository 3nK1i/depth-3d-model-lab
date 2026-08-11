#!/usr/bin/env python3
"""Interactive Phase 8.8D5 semantic/object presentation preview with no backend."""

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


def _pose(angle: float, radius_x: float = 2.8, radius_z: float = 2.2) -> list[list[float]]:
    theta = np.deg2rad(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [
        radius_x * np.cos(theta),
        0.30 * np.sin(theta * 2.0),
        radius_z * np.sin(theta),
    ]
    return matrix.tolist()


def _semantic_object(
    index: int,
    *,
    class_name: str,
    role: str,
    center: tuple[float, float, float],
    half_extent: tuple[float, float, float],
    confidence: float,
) -> dict[str, object]:
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(half_extent, dtype=np.float64)
    return {
        "object_id": f"preview-semantic-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": confidence,
        "vertex_count": 620 - index * 95,
        "face_count": 390 - index * 60,
        "centroid_reconstruction": c.tolist(),
        "bounds_min_reconstruction": (c - h).tolist(),
        "bounds_max_reconstruction": (c + h).tolist(),
        "orientation_reliable": True,
        "observation_ids": [f"preview-obs-{index}-{j}" for j in range(3)],
        "mean_vertex_quality": 0.91 - index * 0.025,
        "median_vertex_uncertainty_reconstruction_units": 0.045 + index * 0.008,
        "maximum_vertex_uncertainty_reconstruction_units": 0.075 + index * 0.010,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
    }


def main() -> int:
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page

    root = Path(tempfile.mkdtemp(prefix="phase8_8d5_preview_"))
    session_id = "55555555-5555-4555-8555-555555555585"
    rng = np.random.default_rng(8_805)

    # Compact connector-like point layout used only to inspect D5 presentation.
    body = rng.normal(loc=(-0.62, 0.0, 2.0), scale=(0.36, 0.34, 0.15), size=(3500, 3))
    pin_top = rng.normal(loc=(0.20, -0.24, 2.0), scale=(0.12, 0.10, 0.10), size=(850, 3))
    pin_bottom = rng.normal(loc=(0.20, 0.24, 2.0), scale=(0.12, 0.10, 0.10), size=(850, 3))
    points = np.vstack((body, pin_top, pin_bottom)).astype(np.float64)
    colors = np.vstack(
        (
            np.tile(np.asarray([[95, 150, 220]], dtype=np.uint8), (len(body), 1)),
            np.tile(np.asarray([[225, 180, 80]], dtype=np.uint8), (len(pin_top), 1)),
            np.tile(np.asarray([[215, 125, 100]], dtype=np.uint8), (len(pin_bottom), 1)),
        )
    )
    cameras = [
        {
            "estimate_id": f"preview-camera-{index}",
            "keyframe_id": f"preview-kf-{index:02d}",
            "frame_id": f"preview-frame-{index:02d}",
            "T_reconstruction_from_camera": _pose(float(index) * 30.0),
        }
        for index in range(12)
    ]
    artifact = root / "offline_visualization.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": OFFLINE_VISUALIZATION_SCHEMA,
                "session_id": session_id,
                "backend_id": "d5-semantic-preview",
                "backend_version": "preview",
                "pose_convention": "T_reconstruction_from_camera",
                "scale_status": "arbitrary",
                "alignment_status": "arbitrary",
                "registered_camera_total": len(cameras),
                "registered_cameras": cameras,
                "landmark_total": len(points),
                "landmarks_reconstruction": points.tolist(),
                "landmark_colors_rgb": colors.tolist(),
                "mean_reprojection_error_px": 0.48,
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
                "input_keyframes": 12,
                "registration_ratio": 1.0,
                "landmarks": len(points),
                "observations": len(points) * 4,
                "mean_reprojection_error_px": 0.48,
                "quality_assessment": {
                    "status": "accepted",
                    "usable": True,
                    "registration_ratio": 1.0,
                    "landmarks_per_registered_keyframe": len(points) / len(cameras),
                    "mean_track_length": 4.0,
                    "mean_reprojection_error_px": 0.48,
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
    semantic_payload = {
        "session_id": session_id,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {
            "observations_total": 9,
            "observations_accepted": 9,
            "persistent_object_count": 3,
        },
        "objects": [
            _semantic_object(
                0,
                class_name="connector_body",
                role="connector_body",
                center=(-0.62, 0.0, 2.0),
                half_extent=(0.52, 0.48, 0.26),
                confidence=0.95,
            ),
            _semantic_object(
                1,
                class_name="pin",
                role="connector_pin",
                center=(0.20, -0.24, 2.0),
                half_extent=(0.20, 0.16, 0.18),
                confidence=0.93,
            ),
            _semantic_object(
                2,
                class_name="pin",
                role="connector_pin",
                center=(0.20, 0.24, 2.0),
                half_extent=(0.20, 0.16, 0.18),
                confidence=0.92,
            ),
        ],
    }
    page.set_semantic_geometry(semantic_payload, source="d5-synthetic-preview")
    page._semantic_object_combo.setCurrentIndex(1)
    page._readiness_badge.setText("SEMANTIC OBJECT PREVIEW • NO BACKEND")
    page._viewport_state.setText("D5 semantic/object presentation")
    window.show()
    code = app.exec()
    shutil.rmtree(root, ignore_errors=True)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
