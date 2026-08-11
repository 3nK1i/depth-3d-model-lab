#!/usr/bin/env python3
"""Automated Qt smoke for Phase 8.8E3 measurement validity; no backend."""

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

OUTPUT = Path("/tmp/phase8_8e3_measurement_validity")


def _object(index: int, class_name: str, role: str, x: float) -> dict[str, object]:
    return {
        "object_id": f"e3-smoke-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.99 if index == 0 else 0.91,
        "vertex_count": 300 + index * 30,
        "face_count": 180 + index * 15,
        "observation_ids": [f"e3-smoke-obs-{index}-{j}" for j in range(3)],
        "mean_vertex_quality": 0.88,
        "median_vertex_uncertainty_reconstruction_units": 0.05,
        "maximum_vertex_uncertainty_reconstruction_units": 0.09,
        "centroid_reconstruction": [x, 0.0, 2.0],
        "bounds_min_reconstruction": [x - 0.18, -0.20, 1.82],
        "bounds_max_reconstruction": [x + 0.18, 0.20, 2.18],
        "orientation_reliable": True,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
    }


def _semantic_payload() -> dict[str, object]:
    return {
        "session_id": "77777777-7777-4777-8777-777777777771",
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
        "observation_id": "phase75-e3-smoke-observation",
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
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_scene_availability(ReconstructionSceneAvailability.PARTIAL)
    page.set_semantic_geometry(_semantic_payload(), source="e3-generic-smoke")
    page._semantic_object_combo.setCurrentIndex(1)
    payload = _measurement_payload()

    before_e2 = page.reconstruction_measurement_validity_presentation is None
    page.set_reconstruction_measurement_evidence(payload, source="phase7_5.inspection_bundle")
    before_e3 = page.reconstruction_measurement_validity_presentation is None
    page.set_reconstruction_measurement_validity_evidence(
        payload, source="phase7_5.inspection_bundle", context=_context()
    )
    presentation = page.reconstruction_measurement_validity_presentation

    qualified = page._measurement_validity_qualified_value.text()
    validity = page._measurement_validity_status_value.text()
    reliability = page._measurement_validity_reliability_value.text()
    uncertainty = page._measurement_validity_uncertainty_value.text()
    scale = page._measurement_validity_scale_value.text()
    calibration = page._measurement_validity_calibration_value.text()
    reasons = page._measurement_validity_reasons_value.text()
    authority = page._measurement_validity_authority_value.text()

    source_fields_presented = bool(
        presentation is not None
        and "Width: 42.1 mm ± 0.6 mm" in qualified
        and "41.5 mm–42.7 mm" in qualified
        and validity == "2 available • 1 rejected"
        and "Width: Medium 78%" in reliability
        and "Height: High 94%" in reliability
        and "Depth: Rejected" in reliability
        and "Estimated Uncertainty — Not Calibrated" in uncertainty
        and "Depth: Excessive uncertainty" in reasons
    )
    semantic_confidence_not_reused = "99%" not in reliability
    scale_authority_preserved = "Measurement: Metric depth" in scale and "scene: Arbitrary gauge" in scale
    calibration_explicit = calibration == "Model-predicted intrinsics • Not Calibrated"
    no_later_authority = "not QC decision" in authority

    saved = presentation
    page._semantic_object_combo.setCurrentIndex(2)
    selection_change_hides = (
        page.reconstruction_measurement_presentation is None
        and page.reconstruction_measurement_validity_presentation is None
    )
    page._semantic_object_combo.setCurrentIndex(1)
    selection_return_restores = (
        page.reconstruction_measurement_presentation is not None
        and page.reconstruction_measurement_validity_presentation is saved
    )

    # Replacing exact E2 evidence must invalidate E3 until validity evidence is
    # explicitly rebound, preventing stale qualification from surviving provenance changes.
    page.set_reconstruction_measurement_evidence(payload, source="phase7_5.inspection_bundle")
    e2_replacement_invalidates_e3 = page.reconstruction_measurement_validity_presentation is None
    page.set_reconstruction_measurement_validity_evidence(
        payload, source="phase7_5.inspection_bundle", context=_context()
    )
    presentation = page.reconstruction_measurement_validity_presentation

    window.resize(1366, 768)
    window.show()
    app.processEvents()
    scrollbar = page._settings_scroll.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_e3_measurement_validity.png"
    screenshot_ok = bool(window.grab().save(str(screenshot)))

    summary = {
        "phase": "8.8E3",
        "pass": bool(
            before_e2
            and before_e3
            and source_fields_presented
            and semantic_confidence_not_reused
            and scale_authority_preserved
            and calibration_explicit
            and no_later_authority
            and selection_change_hides
            and selection_return_restores
            and e2_replacement_invalidates_e3
            and screenshot_ok
            and presentation is not None
            and not presentation.certified_metrology
        ),
        "e2_measurement_source_reused": True,
        "measurement_computed_by_e3": False,
        "uncertainty_computed_by_e3": False,
        "reliability_computed_by_e3": False,
        "active_target_object_id": page.inspection_selection.object_id,
        "validity_target_object_id": presentation.target_object_id if presentation else "",
        "source_observation_id": presentation.source_observation_id if presentation else "",
        "qualified_values_ui": qualified,
        "validity_ui": validity,
        "reliability_ui": reliability,
        "uncertainty_ui": uncertainty,
        "scale_ui": scale,
        "calibration_ui": calibration,
        "reasons_ui": reasons,
        "authority_ui": authority,
        "semantic_confidence_not_used_as_measurement_reliability": semantic_confidence_not_reused,
        "measurement_metric_source_scene_arbitrary_preserved": scale_authority_preserved,
        "selection_change_hides_bound_evidence": selection_change_hides,
        "selection_return_restores_exact_bound_evidence": selection_return_restores,
        "e2_replacement_invalidates_e3": e2_replacement_invalidates_e3,
        "certified_metrology": presentation.certified_metrology if presentation else False,
        "qc_evaluated": False,
        "reference_comparison_evaluated": False,
        "screenshot": str(screenshot),
        "errors": [],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    window.close()
    app.processEvents()

    print("Phase 8.8E3 measurement validity smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(
        f"target={summary['validity_target_object_id']!r} validity={summary['validity_ui']!r} "
        f"calibration={summary['calibration_ui']!r}"
    )
    print(f"authority={summary['authority_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
