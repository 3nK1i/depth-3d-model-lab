from core.reconstruction import (
    InspectionDecision,
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPipelineExecutionScope,
    ReconstructionPipelineRunStatus,
    summarize_f3_pipeline_result,
)
from tests.reconstruction_f3_helpers import make_f3_fixture


def test_f3_orchestrator_binds_real_geometry_semantics_and_qc():
    fixture = make_f3_fixture(request_id="phase8-7f3-integration")
    result = fixture.orchestrator.run()
    assert result.status is ReconstructionPipelineRunStatus.COMPLETED
    assert result.execution_scope is ReconstructionPipelineExecutionScope.REAL_PROVIDERS
    assert result.development_mock_used is False
    assert result.certified_metrology is False
    assert result.reconstruction_ui_enabled is False

    artifacts = [item.artifact for item in result.stages if item.artifact is not None]
    by_kind = {artifact.kind: artifact for artifact in artifacts}
    assert by_kind[ReconstructionArtifactKind.DENSE_GEOMETRY].authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY
    assert by_kind[ReconstructionArtifactKind.SURFACE_MESH].authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY
    assert by_kind[ReconstructionArtifactKind.SEMANTIC_GEOMETRY].authority is ReconstructionArtifactAuthority.SEMANTIC_CONTEXT
    assert by_kind[ReconstructionArtifactKind.INSPECTION_REPORT].authority is ReconstructionArtifactAuthority.INSPECTION_DECISION
    assert result.canonical_geometry_artifact_id == by_kind[ReconstructionArtifactKind.SURFACE_MESH].artifact_id

    summary = summarize_f3_pipeline_result(result, store=fixture.store)
    assert summary.dense_point_count > 500
    assert summary.mesh_vertex_count > 500
    assert summary.mesh_face_count > 500
    assert summary.semantic_object_count == 3
    assert summary.inspection_decision == InspectionDecision.PASS.value
    assert summary.metric_geometry_status == "arbitrary"
    assert summary.comparison_metric_available is False
