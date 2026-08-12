from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.reconstruction import (
    InspectionReport,
    InspectionReportExportSource,
    RECONSTRUCTION_EXPORT_CHECKSUM,
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionExportRequest,
    ReconstructionValidationError,
    export_reconstruction_bundle,
    verify_reconstruction_export_bundle,
)
from tests.reconstruction_f3_helpers import make_f3_fixture


def _source(request_id: str) -> InspectionReportExportSource:
    fixture = make_f3_fixture(request_id=request_id)
    result = fixture.orchestrator.run()
    artifact = next(
        stage.artifact
        for stage in result.stages
        if stage.artifact is not None
        and stage.artifact.kind is ReconstructionArtifactKind.INSPECTION_REPORT
    )
    report = fixture.store.get(artifact.payload_ref, InspectionReport)
    return InspectionReportExportSource(artifact=artifact, report=report)


def test_b_qc_only_bundle_preserves_full_report_and_inspection_authority(tmp_path: Path) -> None:
    source = _source("phase8-9b-integration")
    result = export_reconstruction_bundle(
        ReconstructionExportRequest(
            output_dir=(tmp_path / "qc-export").resolve(),
            sources=(source,),
        )
    )
    files = verify_reconstruction_export_bundle(result.export_dir)
    assert len(files) == 1
    exported = files[0]
    assert exported.role == "inspection_qc_report"
    assert exported.authority is ReconstructionArtifactAuthority.INSPECTION_DECISION
    assert exported.artifact_id == source.artifact.artifact_id
    assert exported.metadata["provider_id"] == source.artifact.provider_id
    assert exported.metadata["decision"] == source.report.decision.value
    assert not exported.canonical_geometry_authoritative
    assert not exported.certified_metrology

    report_path = result.export_dir / "qc" / "inspection_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload == source.report.as_dict()
    assert payload["inspection_qc_enabled"] is True
    assert payload["certified_metrology"] is False

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["session_id"] == source.session_id
    assert manifest["file_count"] == 1
    assert manifest["canonical_geometry_authoritative"] is False
    assert manifest["certified_metrology"] is False
    assert manifest["authority_model"]["inspection_qc"] == "inspection_decision"
    checksum = (result.export_dir / RECONSTRUCTION_EXPORT_CHECKSUM).read_text(encoding="utf-8")
    assert result.manifest_sha256 in checksum


def test_b_qc_payload_tampering_is_rejected_by_existing_bundle_verifier(tmp_path: Path) -> None:
    source = _source("phase8-9b-tamper")
    result = export_reconstruction_bundle(
        ReconstructionExportRequest(
            output_dir=(tmp_path / "qc-export").resolve(),
            sources=(source,),
        )
    )
    report_path = result.export_dir / "qc" / "inspection_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["decision"] = "pass" if payload["decision"] != "pass" else "warn"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ReconstructionValidationError, match="checksum (?:mismatch|changed)"):
        verify_reconstruction_export_bundle(result.export_dir)
