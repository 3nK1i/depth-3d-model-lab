from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from core.reconstruction import (
    GaussianAppearanceProviderExecutor,
    NerfAppearanceProviderExecutor,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionQualityProfile,
    appearance_provider_descriptors,
    build_reconstruction_pipeline_plan,
)
from tests.reconstruction_f3_helpers import make_f3_fixture
from tests.reconstruction_f4_helpers import fake_gaussian_artifact, fake_nerf_artifact


def run_f5_fixture(tmp_path: Path, *, degrade_gaussian: bool = False, request_id: str = "phase8-7f5-test"):
    fixture = make_f3_fixture(request_id=request_id)
    appearance = appearance_provider_descriptors()
    if degrade_gaussian:
        appearance = (
            replace(appearance[0], available=False, unavailable_reason="f5_degraded_fixture"),
            appearance[1],
        )
    providers = fixture.providers + appearance
    request = ReconstructionPipelineRequest(
        request_id=request_id,
        session_id=fixture.session_id,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(
            ReconstructionPipelineGoal.SEMANTIC_INSPECTION,
            ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE,
            ReconstructionPipelineGoal.NERF_APPEARANCE,
        ),
        allow_optional_degradation=True,
    )
    executors = dict(fixture.executors)
    if not degrade_gaussian:
        executors[appearance[0].provider_id] = GaussianAppearanceProviderExecutor(
            store=fixture.store,
            output_root=tmp_path,
            artifact_resolver=lambda c, b, s: fake_gaussian_artifact(c, b, s, tmp_path),
        )
    executors[appearance[1].provider_id] = NerfAppearanceProviderExecutor(
        store=fixture.store,
        output_root=tmp_path,
        artifact_resolver=lambda c, b: fake_nerf_artifact(c, b, tmp_path),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=executors,
    ).run()
    snapshots = {}
    if not degrade_gaussian:
        gaussian = fixture.store.find_unique(__import__('core.reconstruction', fromlist=['GaussianSceneArtifact']).GaussianSceneArtifact)
        snapshots["gaussian_scene"] = gaussian.as_dict()
    nerf = fixture.store.find_unique(__import__('core.reconstruction', fromlist=['NerfSceneArtifact']).NerfSceneArtifact)
    snapshots["nerf_scene"] = nerf.as_dict()
    return fixture, providers, result, snapshots
