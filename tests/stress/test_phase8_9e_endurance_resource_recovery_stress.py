from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    ReconstructionExportRequest,
    export_reconstruction_bundle,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    verify_reconstruction_export_bundle,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f3_helpers import make_f3_fixture
from tests.reconstruction_f5_helpers import run_f5_fixture
from core.reconstruction import (
    CanonicalSurfaceMeshExportSource,
    InspectionReport,
    InspectionReportExportSource,
    ReconstructionArtifactKind,
    SurfaceMeshResult,
)


def test_e_repeated_persistence_and_export_remain_reproducible_and_non_destructive(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "persistent-run", request_id="phase8-9e-persistence"
    )
    scene = persistent_scene_from_pipeline_result(
        result,
        payload_snapshots_by_kind=snapshots,
        created_at_utc="2026-08-12T12:45:00+00:00",
    )
    current = scene
    for index in range(15):
        manifest = write_persistent_reconstruction_scene(current, tmp_path / f"scene-{index:02d}")
        current = read_persistent_reconstruction_scene(manifest)
        verify_persistent_scene_file_references(current)
        assert current.scene_id == scene.scene_id
        assert current.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256
        assert current.canonical_geometry_authoritative is True
        assert current.certified_metrology is False

    f3 = make_f3_fixture(request_id="phase8-9e-export")
    pipeline = f3.orchestrator.run()
    artifacts = {
        stage.artifact.kind: stage.artifact
        for stage in pipeline.stages
        if stage.artifact is not None
    }
    surface_artifact = artifacts[ReconstructionArtifactKind.SURFACE_MESH]
    inspection_artifact = artifacts[ReconstructionArtifactKind.INSPECTION_REPORT]
    surface = f3.store.get(surface_artifact.payload_ref, SurfaceMeshResult)
    inspection = f3.store.get(inspection_artifact.payload_ref, InspectionReport)
    sources = (
        CanonicalSurfaceMeshExportSource(
            artifact=surface_artifact,
            result=surface,
            canonical_geometry_artifact_id=pipeline.canonical_geometry_artifact_id,
        ),
        InspectionReportExportSource(artifact=inspection_artifact, report=inspection),
    )
    for index in range(15):
        exported = export_reconstruction_bundle(
            ReconstructionExportRequest(
                output_dir=(tmp_path / f"export-{index:02d}").resolve(),
                sources=sources,
            )
        )
        verified = verify_reconstruction_export_bundle(exported.export_dir)
        assert len(verified) == 2
        assert any(item.canonical_geometry_authoritative for item in verified)
        assert all(item.certified_metrology is False for item in verified)
