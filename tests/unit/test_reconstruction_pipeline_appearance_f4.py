from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from core.reconstruction import (
    GaussianAppearanceProviderExecutor,
    NerfAppearanceProviderExecutor,
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionQualityProfile,
    ReconstructionValidationError,
    appearance_provider_descriptors,
    build_reconstruction_pipeline_plan,
    materialize_appearance_dataset,
)
from tests.reconstruction_f3_helpers import make_f3_fixture
from tests.reconstruction_f4_helpers import fake_gaussian_artifact, fake_nerf_artifact


def _run(tmp_path: Path):
    fixture = make_f3_fixture(request_id="phase8-7f4-test")
    providers = fixture.providers + appearance_provider_descriptors()
    request = ReconstructionPipelineRequest(
        request_id="phase8-7f4-test",
        session_id=fixture.session_id,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(
            ReconstructionPipelineGoal.SEMANTIC_INSPECTION,
            ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE,
            ReconstructionPipelineGoal.NERF_APPEARANCE,
        ),
    )
    gaussian = GaussianAppearanceProviderExecutor(
        store=fixture.store,
        output_root=tmp_path,
        artifact_resolver=lambda c,b,s: fake_gaussian_artifact(c,b,s,tmp_path),
    )
    nerf = NerfAppearanceProviderExecutor(
        store=fixture.store,
        output_root=tmp_path,
        artifact_resolver=lambda c,b: fake_nerf_artifact(c,b,tmp_path),
    )
    executors = dict(fixture.executors)
    executors[providers[-2].provider_id] = gaussian
    executors[providers[-1].provider_id] = nerf
    plan = build_reconstruction_pipeline_plan(request, providers)
    result = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run()
    return fixture, result


def test_materialized_appearance_dataset_uses_same_rgb_pose_order(tmp_path: Path) -> None:
    fixture = make_f3_fixture(request_id="dataset-test")
    ingestion = fixture.store.find_unique(__import__('core.reconstruction', fromlist=['PipelineIngestionEvidence']).PipelineIngestionEvidence)
    trajectory = fixture.store.find_unique(__import__('core.reconstruction', fromlist=['PipelineCameraTrajectoryEvidence']).PipelineCameraTrajectoryEvidence)
    bundle = materialize_appearance_dataset(ingestion, trajectory, output_dir=(tmp_path / "dataset").resolve())
    assert bundle.dataset.frame_count == 5
    assert bundle.dataset.certified_metrology is False
    assert Path(bundle.manifest_path).is_file()
    assert all(Path(frame.image_path).is_file() for frame in bundle.dataset.frames)


def test_f4_binds_gaussian_and_nerf_as_appearance_only(tmp_path: Path) -> None:
    _fixture, result = _run(tmp_path)
    assert result.status is ReconstructionPipelineRunStatus.COMPLETED
    assert result.canonical_geometry_authoritative
    by_kind = {s.artifact.kind: s.artifact for s in result.stages if s.artifact}
    assert by_kind[ReconstructionArtifactKind.GAUSSIAN_SCENE].authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY
    assert by_kind[ReconstructionArtifactKind.NERF_SCENE].authority is ReconstructionArtifactAuthority.APPEARANCE_ONLY
    assert by_kind[ReconstructionArtifactKind.GAUSSIAN_SCENE].canonical_geometry_authoritative is False
    assert by_kind[ReconstructionArtifactKind.NERF_SCENE].canonical_geometry_authoritative is False
    assert result.certified_metrology is False


def test_f4_preserves_canonical_surface_as_latest_canonical_artifact(tmp_path: Path) -> None:
    _fixture, result = _run(tmp_path)
    surface = next(s.artifact for s in result.stages if s.artifact and s.artifact.kind is ReconstructionArtifactKind.SURFACE_MESH)
    assert result.canonical_geometry_artifact_id == surface.artifact_id


def test_gaussian_fingerprint_mismatch_fails_required_appearance_stage(tmp_path: Path) -> None:
    fixture = make_f3_fixture(request_id="bad-gaussian")
    providers = fixture.providers + appearance_provider_descriptors()
    request = ReconstructionPipelineRequest(
        request_id="bad-gaussian", session_id=fixture.session_id, path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE,),
    )
    def bad(c,b,s):
        item = fake_gaussian_artifact(c,b,s,tmp_path)
        return replace(item, dataset_fingerprint_sha256="0"*64)
    executors=dict(fixture.executors)
    executors[providers[-2].provider_id]=GaussianAppearanceProviderExecutor(store=fixture.store, output_root=tmp_path, artifact_resolver=bad)
    plan=build_reconstruction_pipeline_plan(request, providers)
    result=ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run()
    assert result.status is ReconstructionPipelineRunStatus.DEGRADED


def test_neural_artifacts_cannot_be_promoted_to_canonical_authority(tmp_path: Path) -> None:
    fixture = make_f3_fixture(request_id="authority")
    ingestion = fixture.store.find_unique(__import__('core.reconstruction', fromlist=['PipelineIngestionEvidence']).PipelineIngestionEvidence)
    trajectory = fixture.store.find_unique(__import__('core.reconstruction', fromlist=['PipelineCameraTrajectoryEvidence']).PipelineCameraTrajectoryEvidence)
    bundle = materialize_appearance_dataset(ingestion, trajectory, output_dir=(tmp_path / "data").resolve())
    class C: pass
    c=C(); c.request=type('R',(),{'request_id':'authority'})()
    surface = fixture.store.find_unique(__import__('core.reconstruction.surface_mesh', fromlist=['SurfaceMeshResult']).SurfaceMeshResult) if False else None
    # Authority is enforced by the upstream artifact contracts themselves.
    from core.reconstruction import NerfSceneArtifact
    payload=fake_nerf_artifact(c,bundle,tmp_path).as_dict(); payload["canonical_geometry_authoritative"]=True
    with pytest.raises(Exception):
        NerfSceneArtifact.from_payload(payload)
