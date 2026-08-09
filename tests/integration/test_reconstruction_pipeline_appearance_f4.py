from pathlib import Path

from core.reconstruction import (
    GaussianAppearanceProviderExecutor, NerfAppearanceProviderExecutor,
    ReconstructionPath, ReconstructionPipelineGoal, ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest, ReconstructionPipelineRunStatus, ReconstructionQualityProfile,
    appearance_provider_descriptors, build_reconstruction_pipeline_plan,
)
from tests.reconstruction_f3_helpers import make_f3_fixture
from tests.reconstruction_f4_helpers import fake_gaussian_artifact, fake_nerf_artifact


def test_f4_full_geometry_semantics_qc_plus_dual_appearance(tmp_path: Path) -> None:
    f=make_f3_fixture(request_id="phase8-7f4-integration")
    providers=f.providers+appearance_provider_descriptors()
    req=ReconstructionPipelineRequest(
        request_id="phase8-7f4-integration", session_id=f.session_id, path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION, ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE, ReconstructionPipelineGoal.NERF_APPEARANCE),
    )
    ex=dict(f.executors)
    ex[providers[-2].provider_id]=GaussianAppearanceProviderExecutor(store=f.store, output_root=tmp_path, artifact_resolver=lambda c,b,s: fake_gaussian_artifact(c,b,s,tmp_path))
    ex[providers[-1].provider_id]=NerfAppearanceProviderExecutor(store=f.store, output_root=tmp_path, artifact_resolver=lambda c,b: fake_nerf_artifact(c,b,tmp_path))
    result=ReconstructionPipelineOrchestrator(plan=build_reconstruction_pipeline_plan(req,providers),providers=providers,executors=ex).run()
    assert result.status is ReconstructionPipelineRunStatus.COMPLETED
    assert result.canonical_geometry_authoritative
    assert len(result.stages)==9
