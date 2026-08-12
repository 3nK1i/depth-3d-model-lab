from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.reconstruction import (
    CanonicalSurfaceMeshExportSource,
    InspectionReport,
    InspectionReportExportSource,
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionExportRequest,
    ReconstructionValidationError,
    SurfaceMeshResult,
    export_reconstruction_bundle,
    verify_reconstruction_export_bundle,
)
from tests.reconstruction_f3_helpers import make_f3_fixture


def _integrated_sources():
    fixture = make_f3_fixture(request_id="phase8-9a1-export")
    pipeline_result = fixture.orchestrator.run()
    artifacts = {
        stage.artifact.kind: stage.artifact
        for stage in pipeline_result.stages
        if stage.artifact is not None
    }
    surface_artifact = artifacts[ReconstructionArtifactKind.SURFACE_MESH]
    inspection_artifact = artifacts[ReconstructionArtifactKind.INSPECTION_REPORT]
    surface = fixture.store.get(surface_artifact.payload_ref, SurfaceMeshResult)
    inspection = fixture.store.get(inspection_artifact.payload_ref, InspectionReport)
    return pipeline_result, surface_artifact, surface, inspection_artifact, inspection


def test_a1_integrated_export_binds_canonical_mesh_and_qc_to_pipeline_authority(tmp_path: Path) -> None:
    pipeline_result, surface_artifact, surface, inspection_artifact, inspection = _integrated_sources()
    result = export_reconstruction_bundle(
        ReconstructionExportRequest(
            output_dir=(tmp_path / "integrated-export").resolve(),
            sources=(
                CanonicalSurfaceMeshExportSource(
                    artifact=surface_artifact,
                    result=surface,
                    canonical_geometry_artifact_id=pipeline_result.canonical_geometry_artifact_id,
                ),
                InspectionReportExportSource(
                    artifact=inspection_artifact,
                    report=inspection,
                ),
            ),
        )
    )

    files = verify_reconstruction_export_bundle(result.export_dir)
    by_role = {item.role: item for item in files}
    assert by_role["canonical_surface_mesh"].authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY
    assert by_role["canonical_surface_mesh"].canonical_geometry_authoritative is True
    assert by_role["inspection_qc_report"].authority is ReconstructionArtifactAuthority.INSPECTION_DECISION
    assert by_role["inspection_qc_report"].canonical_geometry_authoritative is False
    assert all(item.certified_metrology is False for item in files)

    mesh_text = (result.export_dir / "geometry" / "canonical_surface_mesh.obj").read_text(encoding="utf-8")
    assert "# certified_metrology=false" in mesh_text
    assert f"# metric_geometry_status={surface.metric_geometry_status.value}" in mesh_text
    report_json = json.loads((result.export_dir / "qc" / "inspection_report.json").read_text(encoding="utf-8"))
    assert report_json["certified_metrology"] is False
    assert report_json["decision"] == inspection.decision.value

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["canonical_geometry_authoritative"] is True
    assert manifest["certified_metrology"] is False
    assert manifest["file_count"] == 2


def test_a1_canonical_surface_export_rejects_noncanonical_pipeline_artifact(tmp_path: Path) -> None:
    pipeline_result, surface_artifact, surface, _inspection_artifact, _inspection = _integrated_sources()
    fixture = make_f3_fixture(request_id="phase8-9a1-wrong-artifact")
    other_result = fixture.orchestrator.run()
    dense_artifact = next(
        stage.artifact
        for stage in other_result.stages
        if stage.artifact is not None and stage.artifact.kind is ReconstructionArtifactKind.DENSE_GEOMETRY
    )
    with pytest.raises(ReconstructionValidationError, match="SURFACE_MESH"):
        CanonicalSurfaceMeshExportSource(
            artifact=dense_artifact,
            result=surface,
            canonical_geometry_artifact_id=pipeline_result.canonical_geometry_artifact_id,
        )
    with pytest.raises(ReconstructionValidationError, match="current canonical geometry artifact"):
        CanonicalSurfaceMeshExportSource(
            artifact=surface_artifact,
            result=surface,
            canonical_geometry_artifact_id="not-the-current-canonical-artifact",
        )
