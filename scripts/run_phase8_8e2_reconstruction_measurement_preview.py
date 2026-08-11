#!/usr/bin/env python3
"""Interactive Phase 8.8E2 reconstruction-measurement preview; no backend."""

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


def _object(index: int, class_name: str, role: str, x: float) -> dict[str, object]:
    return {
        "object_id": f"e2-preview-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.93,
        "vertex_count": 420 + 30 * index,
        "face_count": 250 + 20 * index,
        "observation_ids": [f"e2-preview-obs-{index}-{j}" for j in range(3)],
        "mean_vertex_quality": 0.89,
        "median_vertex_uncertainty_reconstruction_units": 0.04,
        "maximum_vertex_uncertainty_reconstruction_units": 0.08,
        "centroid_reconstruction": [x, 0.0, 2.0],
        "bounds_min_reconstruction": [x - 0.22, -0.20, 1.80],
        "bounds_max_reconstruction": [x + 0.22, 0.20, 2.20],
        "orientation_reliable": True,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
    }


def _semantic_payload() -> dict[str, object]:
    return {
        "session_id": "33333333-3333-4333-8333-333333333331",
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {"observations_total": 6, "observations_accepted": 6, "persistent_object_count": 2},
        "objects": [
            _object(0, "gear", "rotating_component", -0.25),
            _object(1, "shaft", "supporting_component", 0.35),
        ],
    }


def _measurement_payload() -> dict[str, object]:
    # Same public payload shape produced by the existing Phase 7.5 inspection bundle.
    return {
        "observation_id": "phase75-preview-observation",
        "measurements": [
            {"kind": "width", "status": "available", "value_si": 0.0421, "unit_si": "m", "method": "preferred_box_extent", "expanded_uncertainty_si": 0.0006, "reliability": "medium"},
            {"kind": "height", "status": "available", "value_si": 0.0182, "unit_si": "m", "method": "preferred_box_extent", "expanded_uncertainty_si": 0.0005, "reliability": "high"},
            {"kind": "depth", "status": "available", "value_si": 0.0114, "unit_si": "m", "method": "preferred_box_extent", "expanded_uncertainty_si": 0.0007, "reliability": "medium"},
            {"kind": "orientation_yaw", "status": "available", "value_si": 0.3141592653589793, "unit_si": "rad", "method": "preferred_box_orientation", "expanded_uncertainty_si": 0.02, "reliability": "low"},
        ],
    }


def main() -> int:
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_scene_availability(ReconstructionSceneAvailability.PARTIAL)
    page.set_semantic_geometry(_semantic_payload(), source="e2-generic-preview")
    page._semantic_object_combo.setCurrentIndex(1)
    page.set_reconstruction_measurement_evidence(
        _measurement_payload(), source="phase7_5.inspection_bundle"
    )
    page._readiness_badge.setText("E2 RECONSTRUCTION MEASUREMENTS • NO BACKEND")
    page._viewport_state.setText("E2 existing measurement evidence")
    page._viewport_state.setToolTip(
        "E2 presents already-computed values for the active E1 object. Uncertainty/reliability remain intentionally deferred to E3."
    )
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
