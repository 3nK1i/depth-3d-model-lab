#!/usr/bin/env python3
"""Interactive preview for Phase 8.8E3 measurement validity; no backend."""

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
        "object_id": f"e3-preview-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.99 if index == 0 else 0.91,
        "vertex_count": 420 + 30 * index,
        "face_count": 250 + 20 * index,
        "observation_ids": [f"e3-preview-obs-{index}-{j}" for j in range(3)],
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
        "session_id": "66666666-6666-4666-8666-666666666661",
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {"observations_total": 6, "observations_accepted": 6, "persistent_object_count": 2},
        "objects": [
            _object(0, "gear", "rotating_component", -0.25),
            _object(1, "shaft", "supporting_component", 0.35),
        ],
    }


def _measurement_payload() -> dict[str, object]:
    return {
        "observation_id": "phase75-e3-preview-observation",
        "measurements": [
            {
                "kind": "width", "status": "available", "value_si": 0.0421, "unit_si": "m",
                "method": "preferred_box_extent", "expanded_uncertainty_si": 0.0006,
                "lower_bound_si": 0.0415, "upper_bound_si": 0.0427,
                "reliability": "medium", "reliability_score": 0.78, "valid_observation_count": 1,
                "uncertainty_basis": "heuristic_perturbation",
                "uncertainty_qualification": "Estimated Uncertainty — Not Calibrated",
                "rejection_reasons": [], "warnings": [],
            },
            {
                "kind": "height", "status": "available", "value_si": 0.0182, "unit_si": "m",
                "method": "preferred_box_extent", "expanded_uncertainty_si": 0.0005,
                "lower_bound_si": 0.0177, "upper_bound_si": 0.0187,
                "reliability": "high", "reliability_score": 0.94, "valid_observation_count": 1,
                "uncertainty_basis": "heuristic_perturbation",
                "uncertainty_qualification": "Estimated Uncertainty — Not Calibrated",
                "rejection_reasons": [], "warnings": [],
            },
            {
                "kind": "depth", "status": "rejected", "value_si": None, "unit_si": "m",
                "method": "preferred_box_extent", "expanded_uncertainty_si": None,
                "lower_bound_si": None, "upper_bound_si": None,
                "reliability": "rejected", "valid_observation_count": 0,
                "uncertainty_basis": "unknown", "uncertainty_qualification": "",
                "rejection_reasons": ["excessive_uncertainty"],
                "warnings": ["uncertainty exceeds configured threshold"],
            },
            {
                "kind": "orientation_yaw", "status": "available", "value_si": 0.3141592653589793, "unit_si": "rad",
                "method": "preferred_box_orientation", "expanded_uncertainty_si": 0.02,
                "lower_bound_si": 0.2941592653589793, "upper_bound_si": 0.3341592653589793,
                "reliability": "low", "reliability_score": 0.51, "valid_observation_count": 1,
                "uncertainty_basis": "heuristic_perturbation",
                "uncertainty_qualification": "Estimated Uncertainty — Not Calibrated",
                "rejection_reasons": [], "warnings": ["orientation support is limited"],
            },
        ],
    }


def _context() -> dict[str, object]:
    return {
        "depth_scale": "metric",
        "intrinsics_source": "model_predicted",
        "calibration_status": "not_calibrated",
        "confidence_kind": "uncertainty",
    }


def main() -> int:
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_scene_availability(ReconstructionSceneAvailability.PARTIAL)
    page.set_semantic_geometry(_semantic_payload(), source="e3-generic-preview")
    page._semantic_object_combo.setCurrentIndex(1)
    payload = _measurement_payload()
    page.set_reconstruction_measurement_evidence(payload, source="phase7_5.inspection_bundle")
    page.set_reconstruction_measurement_validity_evidence(
        payload,
        source="phase7_5.inspection_bundle",
        context=_context(),
    )
    page._readiness_badge.setText("E3 MEASUREMENT VALIDITY • NO BACKEND")
    page._viewport_state.setText("E3 source-authored validity evidence")
    page._viewport_state.setToolTip(
        "E3 presents existing uncertainty/reliability/eligibility evidence. The reconstruction scene remains arbitrary gauge and QC/reference comparison remain deferred."
    )
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
