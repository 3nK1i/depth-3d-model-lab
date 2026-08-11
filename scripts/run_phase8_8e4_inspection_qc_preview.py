#!/usr/bin/env python3
"""Interactive generic preview for Phase 8.8E4 inspection/QC; no backend."""

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

SESSION = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"


def _object(index: int, class_name: str, role: str, x: float) -> dict[str, object]:
    return {
        "object_id": f"e4-preview-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.96 - 0.04 * index,
        "vertex_count": 420 + index * 35,
        "face_count": 250 + index * 20,
        "observation_ids": [f"e4-preview-obs-{index}-{j}" for j in range(3)],
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
        "session_id": SESSION,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {"observations_total": 9, "observations_accepted": 9, "persistent_object_count": 3},
        "objects": [
            _object(0, "housing", "structural_component", -0.55),
            _object(1, "gear", "rotating_component", 0.0),
            _object(2, "shaft", "supporting_component", 0.55),
        ],
    }


def _qc_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "session_id": SESSION,
        "reference_id": "known-good-assembly-a",
        "candidate_metric_geometry_status": "arbitrary",
        "reference_metric_geometry_status": "arbitrary",
        "comparison_metric_available": False,
        "registration": {
            "status": "success",
            "method": "multi_anchor_similarity",
            "confidence": 0.92,
            "reason": "registration residual accepted",
        },
        "feature_results": [
            {
                "reference_feature_id": "housing-ref",
                "candidate_object_id": "e4-preview-0000",
                "semantic_role": "structural_component",
                "decision": "pass",
                "findings": ["within_tolerance"],
            },
            {
                "reference_feature_id": "gear-ref",
                "candidate_object_id": "e4-preview-0001",
                "semantic_role": "rotating_component",
                "decision": "reject",
                "findings": ["position_deviation", "orientation_deviation"],
            },
        ],
        "extra_candidate_object_ids": ["e4-preview-0002"],
        "decision": "reject",
        "statistics": {
            "reference_feature_count": 2,
            "matched_feature_count": 2,
            "missing_required_count": 0,
            "extra_candidate_count": 1,
            "pass_count": 1,
            "warn_count": 1,
            "reject_count": 1,
            "insufficient_evidence_count": 0,
        },
        "certified_metrology": False,
        "inspection_qc_enabled": True,
    }


def main() -> int:
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_scene_availability(ReconstructionSceneAvailability.PARTIAL)
    page.set_semantic_geometry(_semantic_payload(), source="e4-generic-preview")
    page._semantic_object_combo.setCurrentIndex(2)  # gear
    page.set_reconstruction_inspection_qc_evidence(
        _qc_payload(), source="phase8_7c.inspection_report"
    )
    page._readiness_badge.setText("E4 INSPECTION / QC • NO BACKEND")
    page._viewport_state.setText("E4 existing inspection/QC evidence")
    page._viewport_state.setToolTip(
        "E4 presents an existing QC report for the exact E1 target. It does not rerun registration/QC and the reference workflow remains E5."
    )
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
