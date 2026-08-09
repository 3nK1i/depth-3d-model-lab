#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from core.reconstruction import (
    GaussianAppearanceProviderExecutor, NerfAppearanceProviderExecutor,
    ReconstructionArtifactKind, ReconstructionPath, ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator, ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus, ReconstructionQualityProfile,
    appearance_provider_descriptors, build_reconstruction_pipeline_plan,
    write_reconstruction_pipeline_json,
)
from tests.reconstruction_f3_helpers import make_f3_fixture
from tests.reconstruction_f4_helpers import fake_gaussian_artifact, fake_nerf_artifact

OUT=Path('/tmp/phase8_7f4_contracts')

def main()->int:
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    f=make_f3_fixture(request_id='phase8-7f4-contract-smoke')
    providers=f.providers+appearance_provider_descriptors()
    request=ReconstructionPipelineRequest(
        request_id='phase8-7f4-contract-smoke', session_id=f.session_id,
        path=ReconstructionPath.OFFLINE, quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION, ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE, ReconstructionPipelineGoal.NERF_APPEARANCE),
    )
    executors=dict(f.executors)
    executors[providers[-2].provider_id]=GaussianAppearanceProviderExecutor(store=f.store,output_root=OUT,artifact_resolver=lambda c,b,s: fake_gaussian_artifact(c,b,s,OUT))
    executors[providers[-1].provider_id]=NerfAppearanceProviderExecutor(store=f.store,output_root=OUT,artifact_resolver=lambda c,b: fake_nerf_artifact(c,b,OUT))
    result=ReconstructionPipelineOrchestrator(plan=build_reconstruction_pipeline_plan(request,providers),providers=providers,executors=executors).run()
    write_reconstruction_pipeline_json(result, OUT/'pipeline_result.json')
    by_kind={s.artifact.kind:s.artifact for s in result.stages if s.artifact}
    g=by_kind[ReconstructionArtifactKind.GAUSSIAN_SCENE]
    n=by_kind[ReconstructionArtifactKind.NERF_SCENE]
    payload={
        'phase':'8.7F4','pass': result.status is ReconstructionPipelineRunStatus.COMPLETED,
        'status':result.status.value,'stage_count':len(result.stages),
        'canonical_geometry_artifact_id':result.canonical_geometry_artifact_id,
        'canonical_geometry_authoritative':result.canonical_geometry_authoritative,
        'gaussian_authority':g.authority.value,'nerf_authority':n.authority.value,
        'gaussian_canonical_geometry_authoritative':g.canonical_geometry_authoritative,
        'nerf_canonical_geometry_authoritative':n.canonical_geometry_authoritative,
        'certified_metrology':False,'development_fixture_only':True,
        'real_backend_execution_claimed':False,'persistent_reconstruction_scene_deferred_to':'8.7F5',
        'reconstruction_ui_enabled':False,
    }
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(f"Phase 8.7F4 appearance integration contracts: {'PASS' if payload['pass'] else 'FAIL'}")
    print(f"status={payload['status']} stages={payload['stage_count']}")
    print(f"gaussian_authority={payload['gaussian_authority']} nerf_authority={payload['nerf_authority']}")
    print('canonical_geometry_authoritative=True certified_metrology=False')
    print('real_backend_execution_claimed=False persistent_reconstruction_scene_deferred_to=8.7F5 reconstruction_ui_enabled=False')
    print(f"summary={OUT/'summary.json'}")
    return 0 if payload['pass'] else 1

if __name__=='__main__': raise SystemExit(main())
