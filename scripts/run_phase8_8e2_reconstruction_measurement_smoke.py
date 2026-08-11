#!/usr/bin/env python3
"""Automated Qt smoke for Phase 8.8E2 measurement presentation; no backend."""

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

OUTPUT = Path("/tmp/phase8_8e2_reconstruction_measurement")


def _object(index: int, class_name: str, role: str, x: float) -> dict[str, object]:
    return {
        "object_id": f"e2-smoke-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.92,
        "vertex_count": 280 + index * 30,
        "face_count": 170 + index * 15,
        "observation_ids": [f"e2-smoke-obs-{index}-{j}" for j in range(3)],
        "mean_vertex_quality": 0.87,
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
        "session_id": "44444444-4444-4444-8444-444444444441",
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
        "observation_id": "phase75-e2-smoke-observation",
        "measurements": [
            {"kind": "width", "status": "available", "value_si": 0.0421, "unit_si": "m", "method": "preferred_box_extent", "expanded_uncertainty_si": 0.0006, "reliability": "medium"},
            {"kind": "height", "status": "available", "value_si": 0.0182, "unit_si": "m", "method": "preferred_box_extent", "expanded_uncertainty_si": 0.0005, "reliability": "high"},
            {"kind": "depth", "status": "rejected", "value_si": None, "unit_si": "m", "method": "preferred_box_extent", "rejection_reasons": ["excessive_uncertainty"], "reliability": "rejected"},
        ],
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
    page.set_semantic_geometry(_semantic_payload(), source="e2-generic-smoke")
    page._semantic_object_combo.setCurrentIndex(1)

    before = page.reconstruction_measurement_presentation is None
    page.set_reconstruction_measurement_evidence(
        _measurement_payload(), source="phase7_5.inspection_bundle"
    )
    presentation = page.reconstruction_measurement_presentation
    attached = presentation is not None and presentation.target_object_id == "e2-smoke-0000"
    values_text = page._measurement_values_value.text()
    source_reused = page._measurement_source_value.text() == "phase7_5.inspection_bundle"
    no_e3_fields = (
        "±" not in values_text
        and "medium" not in values_text.lower()
        and "high" not in values_text.lower()
        and "excessive_uncertainty" not in values_text
    )

    page._semantic_object_combo.setCurrentIndex(2)
    no_class_guess = page.reconstruction_measurement_presentation is None
    page._semantic_object_combo.setCurrentIndex(1)
    restored = page.reconstruction_measurement_presentation is presentation

    window.resize(1366, 768)
    window.show()
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_e2_measurements.png"
    screenshot_ok = bool(window.grab().save(str(screenshot)))

    summary = {
        "phase": "8.8E2",
        "pass": bool(
            before
            and attached
            and source_reused
            and no_e3_fields
            and no_class_guess
            and restored
            and screenshot_ok
            and presentation is not None
            and not presentation.certified_metrology
        ),
        "measurement_source_reused": source_reused,
        "measurement_computed_by_e2": False,
        "active_target_object_id": page.inspection_selection.object_id,
        "evidence_target_object_id": presentation.target_object_id if presentation else "",
        "source_observation_id": presentation.source_observation_id if presentation else "",
        "source_measurement_count": presentation.source_measurement_count if presentation else 0,
        "presented_value_count": len(presentation.values) if presentation else 0,
        "presented_kinds": [value.kind for value in presentation.values] if presentation else [],
        "measurement_values_ui": values_text,
        "measurement_state_ui": page._measurement_state_value.text(),
        "authority_ui": page._measurement_authority_value.text(),
        "uncertainty_presented": "±" in values_text,
        "confidence_or_reliability_presented": not no_e3_fields,
        "selection_change_did_not_guess_evidence": no_class_guess,
        "selection_return_restores_exact_bound_evidence": restored,
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

    print(
        "Phase 8.8E2 reconstruction measurement smoke: "
        + ("PASS" if summary["pass"] else "FAIL")
    )
    print(
        f"target={summary['evidence_target_object_id']!r} values={summary['presented_value_count']} "
        f"source={page._measurement_source_value.text()!r}"
    )
    print(f"authority={summary['authority_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
