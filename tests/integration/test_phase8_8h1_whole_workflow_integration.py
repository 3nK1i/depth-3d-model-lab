from __future__ import annotations

from core.reconstruction import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    check_reconstruction_engine_resume_compatibility,
    persistent_scene_from_pipeline_result,
)
from tests.reconstruction_h1_helpers import create_checkpoint_fixture, run_whole_pipeline


def _by_kind(scene):
    return {artifact.kind: artifact for artifact in scene.artifacts}


def test_h1_whole_mock_pipeline_completes_all_integrated_stages() -> None:
    fixture, result = run_whole_pipeline()
    assert result.status is ReconstructionPipelineRunStatus.COMPLETED
    assert tuple(stage.stage for stage in result.stages) == tuple(item.stage for item in fixture.plan.stages)
    assert result.canonical_geometry_authoritative
    assert not result.certified_metrology


def test_h1_persistent_scene_contains_geometry_semantics_inspection_and_appearance() -> None:
    _fixture, result = run_whole_pipeline()
    scene = persistent_scene_from_pipeline_result(result)
    by_kind = _by_kind(scene)
    for kind in (
        ReconstructionArtifactKind.SURFACE_MESH,
        ReconstructionArtifactKind.SEMANTIC_GEOMETRY,
        ReconstructionArtifactKind.INSPECTION_REPORT,
        ReconstructionArtifactKind.GAUSSIAN_SCENE,
        ReconstructionArtifactKind.NERF_SCENE,
    ):
        assert kind in by_kind
    assert scene.canonical_geometry_authoritative
    assert not scene.certified_metrology


def test_h1_authority_chain_is_preserved_across_full_scene() -> None:
    _fixture, result = run_whole_pipeline()
    by_kind = _by_kind(persistent_scene_from_pipeline_result(result))
    assert by_kind[ReconstructionArtifactKind.SURFACE_MESH].authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY
    assert by_kind[ReconstructionArtifactKind.SEMANTIC_GEOMETRY].authority is ReconstructionArtifactAuthority.SEMANTIC_CONTEXT
    assert by_kind[ReconstructionArtifactKind.INSPECTION_REPORT].authority is ReconstructionArtifactAuthority.INSPECTION_DECISION
    assert by_kind[ReconstructionArtifactKind.GAUSSIAN_SCENE].authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY
    assert by_kind[ReconstructionArtifactKind.NERF_SCENE].authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY


def test_h1_checkpoint_prefix_reuses_exact_same_pipeline_identity() -> None:
    fixture, _result = run_whole_pipeline()
    checkpoint, prefix = create_checkpoint_fixture(fixture)
    assert tuple(item.stage for item in prefix) == (
        ReconstructionPipelineStage.SOURCE_INGESTION,
        ReconstructionPipelineStage.CAMERA_POSE,
        ReconstructionPipelineStage.DEPTH_EVIDENCE,
    )
    compatibility = check_reconstruction_engine_resume_compatibility(
        checkpoint,
        plan=fixture.plan,
        providers=fixture.providers,
        available_payload_refs=checkpoint.payload_refs,
    )
    assert compatibility.compatible
    assert compatibility.next_stage is ReconstructionPipelineStage.DENSE_GEOMETRY


def test_h1_missing_resume_payload_fails_closed() -> None:
    fixture, _result = run_whole_pipeline()
    checkpoint, _prefix = create_checkpoint_fixture(fixture)
    compatibility = check_reconstruction_engine_resume_compatibility(
        checkpoint,
        plan=fixture.plan,
        providers=fixture.providers,
        available_payload_refs=(),
    )
    assert not compatibility.compatible
    assert compatibility.reasons


def test_h1_mock_integration_executes_no_real_provider_backend() -> None:
    fixture, result = run_whole_pipeline()
    assert result.execution_scope.value == "development_mock"
    assert all(executor.calls for executor in fixture.executors.values())
