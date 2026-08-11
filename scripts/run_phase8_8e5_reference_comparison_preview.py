#!/usr/bin/env python3
"""Interactive generic Phase 8.8E5 reference-comparison preview; no backend."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability
from app.widgets.mode_tabs import AppMode

SESSION = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeee1"
OUTPUT = Path("/tmp/phase8_8e5_reference_preview")


def _object(index: int, class_name: str, role: str, x: float) -> dict[str, object]:
    return {
        "object_id": f"e5-preview-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.95 - 0.03 * index,
        "vertex_count": 430 + index * 30,
        "face_count": 260 + index * 20,
        "observation_ids": [f"e5-preview-obs-{index}-{j}" for j in range(3)],
        "mean_vertex_quality": 0.90,
        "median_vertex_uncertainty_reconstruction_units": 0.035,
        "maximum_vertex_uncertainty_reconstruction_units": 0.075,
        "centroid_reconstruction": [x, 0.0, 2.0],
        "bounds_min_reconstruction": [x - 0.20, -0.20, 1.80],
        "bounds_max_reconstruction": [x + 0.20, 0.20, 2.20],
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


def _reference_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "reference_id": "known-good-assembly-a",
        "source": "golden_reconstruction",
        "scene_profile": "object_scan",
        "reference_frame_id": "known-good-frame",
        "metric_geometry_status": "arbitrary",
        "unit_to_metre": None,
        "relative_scale_uncertainty": None,
        "certified_metrology": False,
        "features": [
            {"feature_id": "housing-ref", "association_key": "housing", "registration_anchor": True},
            {"feature_id": "gear-ref", "association_key": "gear", "registration_anchor": False},
        ],
    }


def _qc_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "session_id": SESSION,
        "reference_id": "known-good-assembly-a",
        "scene_profile": "object_scan",
        "pose_convention": "T_reconstruction_from_camera",
        "candidate_frame_id": "candidate-frame",
        "reference_frame_id": "known-good-frame",
        "candidate_metric_geometry_status": "arbitrary",
        "reference_metric_geometry_status": "arbitrary",
        "comparison_metric_available": False,
        "reference_unit_to_metre": None,
        "relative_scale_uncertainty": None,
        "registration": {
            "status": "success",
            "method": "multi_anchor_similarity",
            "scale_candidate_to_reference": 1.0,
            "anchor_feature_ids": ["housing-ref"],
            "rms_residual_reference_units": 0.035,
            "maximum_residual_reference_units": 0.070,
            "rms_residual_m": None,
            "maximum_residual_m": None,
            "normalized_rms_fraction": 0.018,
            "confidence": 0.93,
            "reason": "registration residual accepted",
        },
        "feature_results": [
            {
                "reference_feature_id": "housing-ref",
                "candidate_object_id": "e5-preview-0000",
                "semantic_role": "structural_component",
                "decision": "pass",
                "findings": ["within_tolerance"],
                "position_error_reference_units": 0.025,
                "position_error_m": None,
                "orientation_error_degrees": 0.8,
                "extent_relative_error": 0.025,
                "geometry_quality": 0.91,
                "uncertainty_reference_units": 0.020,
                "uncertainty_m": None,
            },
            {
                "reference_feature_id": "gear-ref",
                "candidate_object_id": "e5-preview-0001",
                "semantic_role": "rotating_component",
                "decision": "reject",
                "findings": ["position_deviation", "orientation_deviation"],
                "position_error_reference_units": 0.180,
                "position_error_m": None,
                "orientation_error_degrees": 14.0,
                "extent_relative_error": 0.095,
                "geometry_quality": 0.86,
                "uncertainty_reference_units": 0.040,
                "uncertainty_m": None,
            },
        ],
        "extra_candidate_object_ids": ["e5-preview-0002"],
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
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference_path = OUTPUT / "reference.json"
    reference_path.write_text(json.dumps(_reference_payload(), indent=2), encoding="utf-8")

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_scene_availability(ReconstructionSceneAvailability.PARTIAL)
    page.set_semantic_geometry(_semantic_payload(), source="e5-generic-preview")
    page._semantic_object_combo.setCurrentIndex(2)  # gear
    page.set_reconstruction_inspection_qc_evidence(
        _qc_payload(), source="phase8_7c.inspection_report"
    )
    page.load_reconstruction_reference_comparison_json(reference_path)
    page._readiness_badge.setText("E5 REFERENCE COMPARISON • NO BACKEND")
    page._viewport_state.setText("E5 explicit reference + existing 8.7C comparison")
    page._viewport_state.setToolTip(
        "E5 pairs an explicit ReferenceAssembly identity with source-authored Phase 8.7C comparison evidence. It does not rerun registration or tolerances."
    )
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
