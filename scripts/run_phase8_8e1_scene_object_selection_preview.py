#!/usr/bin/env python3
"""Interactive Phase 8.8E1 scene/object-selection preview with no backend."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability
from app.widgets.mode_tabs import AppMode


def _object(
    index: int,
    *,
    class_name: str,
    role: str,
    center: tuple[float, float, float],
    half_extent: tuple[float, float, float],
) -> dict[str, object]:
    c = center
    h = half_extent
    return {
        "object_id": f"e1-preview-object-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.94 - 0.02 * index,
        "vertex_count": 560 - 80 * index,
        "face_count": 340 - 45 * index,
        "observation_ids": [f"e1-preview-obs-{index}-{j}" for j in range(3)],
        "mean_vertex_quality": 0.90 - 0.02 * index,
        "median_vertex_uncertainty_reconstruction_units": 0.04 + 0.01 * index,
        "maximum_vertex_uncertainty_reconstruction_units": 0.08 + 0.01 * index,
        "centroid_reconstruction": list(c),
        "bounds_min_reconstruction": [c[j] - h[j] for j in range(3)],
        "bounds_max_reconstruction": [c[j] + h[j] for j in range(3)],
        "orientation_reliable": True,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
    }


def _payload() -> dict[str, object]:
    return {
        "session_id": "88888888-8888-4888-8888-888888888881",
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {
            "observations_total": 9,
            "observations_accepted": 9,
            "persistent_object_count": 3,
        },
        "objects": [
            _object(
                0,
                class_name="housing",
                role="main_body",
                center=(-0.55, 0.0, 2.0),
                half_extent=(0.48, 0.42, 0.26),
            ),
            _object(
                1,
                class_name="gear",
                role="rotating_component",
                center=(0.28, -0.25, 2.0),
                half_extent=(0.22, 0.18, 0.18),
            ),
            _object(
                2,
                class_name="shaft",
                role="supporting_component",
                center=(0.30, 0.28, 2.0),
                half_extent=(0.18, 0.16, 0.30),
            ),
        ],
    }


def main() -> int:
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_scene_availability(ReconstructionSceneAvailability.PARTIAL)
    page.set_semantic_geometry(_payload(), source="e1-generic-preview")
    page._semantic_object_combo.setCurrentIndex(2)
    page._readiness_badge.setText("E1 SCENE / OBJECT SELECTION • NO BACKEND")
    page._viewport_state.setText("E1 inspection target selection")
    page._viewport_state.setToolTip(
        "The D5 semantic selector is the only object selector; E1 binds its selected object as the downstream inspection target."
    )
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
