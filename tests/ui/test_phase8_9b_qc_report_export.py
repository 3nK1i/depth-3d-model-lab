from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QFileDialog

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import (
    CanonicalSurfaceMeshExportSource,
    InspectionReport,
    InspectionReportExportSource,
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    SurfaceMeshResult,
    verify_reconstruction_export_bundle,
)
from tests.reconstruction_f3_helpers import make_f3_fixture


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    win._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    yield win
    win.close()


def _pipeline_sources(request_id: str):
    fixture = make_f3_fixture(request_id=request_id)
    result = fixture.orchestrator.run()
    artifacts = {
        stage.artifact.kind: stage.artifact
        for stage in result.stages
        if stage.artifact is not None
    }
    surface_artifact = artifacts[ReconstructionArtifactKind.SURFACE_MESH]
    inspection_artifact = artifacts[ReconstructionArtifactKind.INSPECTION_REPORT]
    surface = fixture.store.get(surface_artifact.payload_ref, SurfaceMeshResult)
    report = fixture.store.get(inspection_artifact.payload_ref, InspectionReport)
    canonical = CanonicalSurfaceMeshExportSource(
        artifact=surface_artifact,
        result=surface,
        canonical_geometry_artifact_id=result.canonical_geometry_artifact_id,
    )
    qc = InspectionReportExportSource(
        artifact=inspection_artifact,
        report=report,
    )
    return result, canonical, qc


def _adopt(window, qtbot, tmp_path: Path, result, name: str = "scene"):
    scene = window._reconstruction_controller.adopt_pipeline_result(
        result,
        (tmp_path / name).resolve(),
    )
    qtbot.wait(10)
    return scene


def test_b_display_only_e4_report_remains_non_exportable(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    result, _canonical, qc = _pipeline_sources("phase8-9b-display-only")
    _adopt(window, qtbot, tmp_path, result)

    page.set_reconstruction_inspection_qc_evidence(qc.report)
    assert page.reconstruction_inspection_qc_presentation is not None
    assert page._inspection_qc_export_source is None
    assert not page._reconstruction_export_button.isEnabled()
    assert page._reconstruction_export_state_value.text() == "No materialized export evidence"


def test_b_authoritative_qc_source_is_presented_and_exported_without_rerunning_qc(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    result, _canonical, qc = _pipeline_sources("phase8-9b-qc-only")
    _adopt(window, qtbot, tmp_path, result)

    page.set_reconstruction_inspection_qc_export_source(qc)
    assert page.reconstruction_inspection_qc_presentation is not None
    assert page._inspection_qc_export_source is qc
    assert page._reconstruction_export_button.isEnabled()
    assert page._reconstruction_export_state_value.text() == "QC report ready"
    assert "inspection-decision" in page._reconstruction_export_state_value.toolTip()
    assert "certified metrology" in page._reconstruction_export_state_value.toolTip().lower()

    parent = (tmp_path / "exports").resolve()
    parent.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(parent))
    page._reconstruction_export_button.click()
    qtbot.wait(10)

    output = page.last_reconstruction_export_dir
    assert output is not None and output.is_dir()
    files = verify_reconstruction_export_bundle(output)
    assert len(files) == 1
    exported = files[0]
    assert exported.role == "inspection_qc_report"
    assert exported.authority is ReconstructionArtifactAuthority.INSPECTION_DECISION
    assert not exported.canonical_geometry_authoritative
    assert not exported.certified_metrology

    report_json = json.loads((output / "qc" / "inspection_report.json").read_text(encoding="utf-8"))
    assert report_json["session_id"] == qc.report.session_id
    assert report_json["reference_id"] == qc.report.reference_id
    assert report_json["decision"] == qc.report.decision.value
    assert report_json["statistics"] == qc.report.statistics.as_dict()
    assert report_json["registration"] == qc.report.registration.as_dict()
    assert report_json["certified_metrology"] is False
    assert report_json["inspection_qc_enabled"] is True
    assert "inspection/QC evidence included" in page._reconstruction_export_state_value.toolTip()


def test_b_combines_canonical_mesh_and_qc_without_crossing_authority_boundaries(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    result, canonical, qc = _pipeline_sources("phase8-9b-combined")
    _adopt(window, qtbot, tmp_path, result)
    page.set_reconstruction_canonical_surface_export_source(canonical)
    page.set_reconstruction_inspection_qc_export_source(qc)

    assert page._reconstruction_export_state_value.text() == "Canonical mesh + QC report ready"
    parent = (tmp_path / "combined_exports").resolve()
    parent.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(parent))
    page._reconstruction_export_button.click()
    qtbot.wait(10)

    output = page.last_reconstruction_export_dir
    assert output is not None
    files = verify_reconstruction_export_bundle(output)
    by_role = {item.role: item for item in files}
    assert set(by_role) == {"canonical_surface_mesh", "inspection_qc_report"}
    assert by_role["canonical_surface_mesh"].authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY
    assert by_role["canonical_surface_mesh"].canonical_geometry_authoritative
    assert by_role["inspection_qc_report"].authority is ReconstructionArtifactAuthority.INSPECTION_DECISION
    assert not by_role["inspection_qc_report"].canonical_geometry_authoritative
    assert all(not item.certified_metrology for item in files)
    assert "canonical geometry + inspection/QC included" in page._reconstruction_export_state_value.toolTip()


def test_b_mismatched_active_scene_rejects_qc_export_source_fail_closed(
    window, qtbot, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    result_a, _canonical_a, qc_a = _pipeline_sources("phase8-9b-source-a")
    result_b, _canonical_b, _qc_b = _pipeline_sources("phase8-9b-source-b")
    assert result_a.session_id != result_b.session_id
    _adopt(window, qtbot, tmp_path, result_b, "scene-b")

    with pytest.raises(ValueError, match="session does not match"):
        page.set_reconstruction_inspection_qc_export_source(qc_a)
    assert page._inspection_qc_export_source is None
    assert not page._reconstruction_export_button.isEnabled()


def test_b_replacing_authoritative_qc_with_display_only_evidence_drops_export_qualification(
    window, qtbot, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    result, _canonical, qc = _pipeline_sources("phase8-9b-replace")
    _adopt(window, qtbot, tmp_path, result)
    page.set_reconstruction_inspection_qc_export_source(qc)
    assert page._reconstruction_export_button.isEnabled()

    replacement = qc.report.as_dict()
    page.set_reconstruction_inspection_qc_evidence(replacement, source="phase8_7c.serialized_report")
    assert page.reconstruction_inspection_qc_presentation is not None
    assert page._inspection_qc_export_source is None
    assert not page._reconstruction_export_button.isEnabled()
    assert page._reconstruction_export_state_value.text() == "No materialized export evidence"


def test_b_clearing_qc_evidence_clears_qc_export_authority(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    result, _canonical, qc = _pipeline_sources("phase8-9b-clear")
    _adopt(window, qtbot, tmp_path, result)
    page.set_reconstruction_inspection_qc_export_source(qc)
    assert page._inspection_qc_export_source is qc

    page.clear_reconstruction_inspection_qc_evidence()
    assert page.reconstruction_inspection_qc_presentation is None
    assert page._inspection_qc_export_source is None
    assert not page._reconstruction_export_button.isEnabled()
