#!/usr/bin/env python3
"""Decisive Phase 8.8E4 smoke: real 8.7C QC engine -> E4 presentation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability
from app.widgets.mode_tabs import AppMode
from core.reconstruction import (
    FeatureTolerance,
    InspectionDecision,
    ReferenceGeometrySource,
    build_reference_assembly,
    inspect_semantic_geometry,
)
from scripts.run_registration_inspection_smoke import _rot_z, _semantic_fixture, _transform_candidate

OUTPUT = Path("/tmp/phase8_8e4_inspection_qc")


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Run the existing Phase 8.7C engine. E4 itself never calls this engine.
    surface, semantic, _scale = _semantic_fixture()
    tolerance = FeatureTolerance(
        position_warn=0.08,
        position_reject=0.16,
        orientation_warn_degrees=5.0,
        orientation_reject_degrees=12.0,
        extent_relative_warn=0.08,
        extent_relative_reject=0.16,
        minimum_geometry_quality=0.60,
        maximum_uncertainty_fraction=0.30,
    )
    reference = build_reference_assembly(
        surface,
        semantic,
        reference_id="phase8-8e4-existing-reference",
        source=ReferenceGeometrySource.GOLDEN_RECONSTRUCTION,
        default_tolerance=tolerance,
        registration_anchor_feature_ids=("connector-main",),
    )
    candidate_surface, candidate_semantic = _transform_candidate(
        surface,
        semantic,
        scale_factor=1.30,
        rotation=_rot_z(17.0),
        translation=np.asarray([0.6, -0.4, 0.3], dtype=np.float64),
        bend_top_degrees=20.0,
        shift_top_reference=np.asarray([0.0, -0.24, 0.0], dtype=np.float64),
        remove_bottom=True,
    )
    report = inspect_semantic_geometry(reference, candidate_surface, candidate_semantic)

    top_index = next(
        index for index, obj in enumerate(candidate_semantic.objects) if "pin-top" in obj.track_ids
    )
    top_object_id = candidate_semantic.objects[top_index].object_id
    source_result = next(
        value for value in report.feature_results if value.candidate_object_id == top_object_id
    )

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_scene_availability(ReconstructionSceneAvailability.PARTIAL)
    page.set_semantic_geometry(candidate_semantic.as_dict(), source="phase8_7b.semantic_geometry")
    page._semantic_object_combo.setCurrentIndex(top_index + 1)

    before_e4 = page.reconstruction_inspection_qc_presentation is None
    page.set_reconstruction_inspection_qc_evidence(
        report, source="phase8_7c.inspection_report"
    )
    presentation = page.reconstruction_inspection_qc_presentation

    exact_binding = bool(
        presentation is not None
        and presentation.target_object_id == top_object_id
        and presentation.target_result_available
        and presentation.target_decision == source_result.decision.value
    )
    source_decision_presented = (
        page._inspection_qc_decision_value.text()
        == f"{source_result.decision.value.upper()} • Object"
    )
    source_findings_presented = all(
        finding.value.replace("_", " ").capitalize() in page._inspection_qc_findings_value.text()
        for finding in source_result.findings
    )
    registration_preserved = bool(
        presentation is not None
        and presentation.registration_status == report.registration.status.value
        and abs(presentation.registration_confidence - report.registration.confidence) < 1e-12
    )
    report_authority_preserved = bool(
        presentation is not None
        and presentation.overall_decision == report.decision.value
        and "Source inspection/QC decision" in page._inspection_qc_authority_value.text()
        and "reference workflow deferred to E5" in page._inspection_qc_authority_value.text()
    )

    window.resize(1366, 768)
    window.show()
    app.processEvents()
    scrollbar = page._settings_scroll.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_e4_inspection_qc.png"
    screenshot_ok = bool(window.grab().save(str(screenshot)))

    summary = {
        "phase": "8.8E4",
        "pass": bool(
            before_e4
            and report.decision is InspectionDecision.REJECT
            and exact_binding
            and source_decision_presented
            and source_findings_presented
            and registration_preserved
            and report_authority_preserved
            and screenshot_ok
            and presentation is not None
            and not presentation.certified_metrology
        ),
        "source_phase8_7c_engine_run": True,
        "source_report_decision": report.decision.value,
        "source_feature_decision": source_result.decision.value,
        "active_target_object_id": page.inspection_selection.object_id,
        "qc_target_object_id": presentation.target_object_id if presentation else "",
        "exact_candidate_object_binding": exact_binding,
        "decision_ui": page._inspection_qc_decision_value.text(),
        "findings_ui": page._inspection_qc_findings_value.text(),
        "coverage_ui": page._inspection_qc_coverage_value.text(),
        "summary_ui": page._inspection_qc_decisions_value.text(),
        "registration_ui": page._inspection_qc_registration_value.text(),
        "comparison_ui": page._inspection_qc_comparison_value.text(),
        "authority_ui": page._inspection_qc_authority_value.text(),
        "qc_computed_by_e4": False,
        "registration_computed_by_e4": False,
        "tolerance_applied_by_e4": False,
        "semantic_or_measurement_confidence_used_as_qc": False,
        "reference_workflow_executed_by_e4": False,
        "certified_metrology": presentation.certified_metrology if presentation else False,
        "screenshot": str(screenshot),
        "errors": [],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    window.close()
    app.processEvents()

    print("Phase 8.8E4 inspection/QC presentation smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(
        f"target={summary['qc_target_object_id']!r} decision={summary['decision_ui']!r} "
        f"source={summary['source_feature_decision']!r}"
    )
    print(f"authority={summary['authority_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
