#!/usr/bin/env python3
"""Decisive Phase 8.8E5 smoke: real 8.7C reference/report -> E5 workflow."""

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
    write_reference_assembly_json,
)
from scripts.run_registration_inspection_smoke import _rot_z, _semantic_fixture, _transform_candidate

OUTPUT = Path("/tmp/phase8_8e5_reference_comparison")


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Existing Phase 8.7C authority: build a qualified reference and run the
    # existing registration/QC engine before E5 sees either artifact.
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
        reference_id="phase8-8e5-qualified-reference",
        source=ReferenceGeometrySource.GOLDEN_RECONSTRUCTION,
        default_tolerance=tolerance,
        registration_anchor_feature_ids=("connector-main",),
    )
    reference_json = write_reference_assembly_json(reference, OUTPUT / "reference.json")

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

    before_reference = page.reconstruction_reference_comparison_presentation is None
    page.set_reconstruction_inspection_qc_evidence(report, source="phase8_7c.inspection_report")
    still_no_reference = page.reconstruction_reference_comparison_presentation is None
    page.load_reconstruction_reference_comparison_json(reference_json)
    presentation = page.reconstruction_reference_comparison_presentation

    reference_identity_preserved = bool(
        presentation is not None
        and presentation.reference.reference_id == reference.reference_id
        and presentation.reference.feature_count == len(reference.features)
        and presentation.reference.registration_anchor_count
        == sum(1 for feature in reference.features if feature.registration_anchor)
    )
    exact_report_pairing = bool(
        presentation is not None
        and presentation.report_reference_match
        and presentation.comparison_available
        and presentation.session_id == report.session_id
        and presentation.target_object_id == top_object_id
        and presentation.matched_reference_feature_id == source_result.reference_feature_id
    )
    source_deviation_preserved = bool(
        presentation is not None
        and presentation.position_error_reference_units == source_result.position_error_reference_units
        and presentation.orientation_error_degrees == source_result.orientation_error_degrees
        and presentation.extent_relative_error == source_result.extent_relative_error
        and presentation.geometry_quality == source_result.geometry_quality
        and presentation.uncertainty_reference_units == source_result.uncertainty_reference_units
    )
    registration_preserved = bool(
        presentation is not None
        and presentation.registration_status == report.registration.status.value
        and presentation.registration_method
        == (report.registration.method.value if report.registration.method else "")
        and abs(presentation.registration_confidence - report.registration.confidence) < 1e-12
        and presentation.registration_rms_reference_units
        == report.registration.rms_residual_reference_units
    )

    # Explicitly prove that switching to another reference identity cannot reuse
    # the old 8.7C comparison evidence.
    mismatch_payload = reference.as_dict()
    mismatch_payload["reference_id"] = "phase8-8e5-other-reference"
    page.set_reconstruction_reference_comparison_reference(
        mismatch_payload, source="e5-mismatch-fixture"
    )
    mismatch = page.reconstruction_reference_comparison_presentation
    mismatch_blocks_comparison = bool(
        mismatch is not None
        and mismatch.report_attached
        and not mismatch.report_reference_match
        and not mismatch.comparison_available
        and page._reference_comparison_deviations_value.text() == "—"
    )

    # Reattach the exact JSON artifact and ensure the original source evidence
    # is restored without recomputation.
    page.load_reconstruction_reference_comparison_json(reference_json)
    restored = page.reconstruction_reference_comparison_presentation
    exact_reference_restore = bool(
        restored is not None
        and restored.report_reference_match
        and restored.matched_reference_feature_id == source_result.reference_feature_id
        and restored.position_error_reference_units == source_result.position_error_reference_units
    )

    page._readiness_badge.setText("E5 REFERENCE COMPARISON • REAL 8.7C EVIDENCE")
    page._viewport_state.setText("E5 exact reference + source comparison evidence")
    window.resize(1366, 768)
    window.show()
    app.processEvents()
    scrollbar = page._settings_scroll.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_e5_reference_comparison.png"
    screenshot_ok = bool(window.grab().save(str(screenshot)))

    authority_ok = bool(
        restored is not None
        and "does not register/apply tolerances" in page._reference_comparison_authority_value.text()
        and "not certified metrology" in page._reference_comparison_authority_value.text()
        and not restored.certified_metrology
    )

    summary = {
        "phase": "8.8E5",
        "pass": bool(
            before_reference
            and still_no_reference
            and report.decision is InspectionDecision.REJECT
            and reference_identity_preserved
            and exact_report_pairing
            and source_deviation_preserved
            and registration_preserved
            and mismatch_blocks_comparison
            and exact_reference_restore
            and authority_ok
            and screenshot_ok
        ),
        "source_phase8_7c_engine_run": True,
        "source_reference_artifact_written": True,
        "source_reference_id": reference.reference_id,
        "source_report_reference_id": report.reference_id,
        "source_report_decision": report.decision.value,
        "source_feature_decision": source_result.decision.value,
        "active_target_object_id": page.inspection_selection.object_id,
        "matched_reference_feature_id": restored.matched_reference_feature_id if restored else "",
        "reference_identity_preserved": reference_identity_preserved,
        "exact_report_reference_pairing": exact_report_pairing,
        "source_deviation_preserved": source_deviation_preserved,
        "registration_preserved": registration_preserved,
        "reference_switch_blocks_stale_comparison": mismatch_blocks_comparison,
        "exact_reference_restore": exact_reference_restore,
        "reference_ui": page._reference_comparison_reference_value.text(),
        "registration_ui": page._reference_comparison_registration_value.text(),
        "deviation_ui": page._reference_comparison_deviations_value.text(),
        "evidence_ui": page._reference_comparison_evidence_value.text(),
        "scale_ui": page._reference_comparison_scale_value.text(),
        "state_ui": page._reference_comparison_state_value.text(),
        "authority_ui": page._reference_comparison_authority_value.text(),
        "registration_computed_by_e5": False,
        "deviation_computed_by_e5": False,
        "tolerance_applied_by_e5": False,
        "qc_decision_computed_by_e5": False,
        "cad_geometry_parsed_by_e5": False,
        "certified_metrology": restored.certified_metrology if restored else False,
        "reference_json": str(reference_json),
        "screenshot": str(screenshot),
        "errors": [],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    window.close()
    app.processEvents()

    print("Phase 8.8E5 reference comparison smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(
        f"reference={summary['source_reference_id']!r} target={summary['active_target_object_id']!r} "
        f"feature={summary['matched_reference_feature_id']!r}"
    )
    print(f"deviation={summary['deviation_ui']!r}")
    print(f"authority={summary['authority_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
