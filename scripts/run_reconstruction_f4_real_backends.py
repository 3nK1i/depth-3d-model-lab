#!/usr/bin/env python3
"""Bounded real gsplat + Nerfacto integration through the F4 orchestrator.

This is an integration gate, not a new quality/performance benchmark. D3 and E3/E4
remain the qualification authorities for those backend-specific concerns.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from core.reconstruction import (
    GaussianAppearanceProviderExecutor, GaussianBackendKind, GaussianOptimizationProfile, GaussianSceneArtifact,
    GaussianWorkerProcessSpec, GaussianWorkerRequest,
    NerfAppearanceProviderExecutor, NerfBackendKind, NerfOptimizationProfile, NerfSceneArtifact,
    NerfWorkerProcessSpec, NerfWorkerRequest,
    ReconstructionArtifactKind, ReconstructionPath, ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator, ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus, ReconstructionQualityProfile,
    appearance_provider_descriptors, build_gaussian_seed, build_reconstruction_pipeline_plan,
    run_gaussian_external_worker, run_nerf_external_worker, write_gaussian_seed_npz,
    write_reconstruction_pipeline_json,
)
from tests.reconstruction_f3_helpers import make_f3_fixture


def _env_python(name:str)->Path:
    current=Path(sys.executable).resolve()
    if current.parent.name=='bin' and len(current.parents)>=3 and current.parents[2].name=='envs':
        return current.parents[2]/name/'bin/python'
    return Path.home()/f'anaconda3/envs/{name}/bin/python'



def _contact_sheet(gaussian_scene, nerf_scene, out: Path) -> str:
    try:
        from PIL import Image, ImageDraw, ImageOps
        items=[]
        for label, scene in (("Gaussian",gaussian_scene),("NeRF",nerf_scene)):
            preview=Path(scene.artifact_dir)/"preview.png"
            if preview.is_file():
                with Image.open(preview) as im:
                    items.append((label, im.convert("RGB").copy()))
        if len(items)!=2:
            return ""
        target_h=max(im.height for _,im in items)
        panels=[]
        for label,im in items:
            scale=target_h/im.height
            resized=im.resize((max(1,int(round(im.width*scale))),target_h))
            panel=Image.new("RGB",(resized.width,target_h+34),"white")
            panel.paste(resized,(0,34))
            ImageDraw.Draw(panel).text((8,8),label,fill="black")
            panels.append(panel)
        canvas=Image.new("RGB",(sum(x.width for x in panels),max(x.height for x in panels)),"white")
        x=0
        for panel in panels:
            canvas.paste(panel,(x,0)); x+=panel.width
        path=out/"appearance_contact_sheet.png"
        canvas.save(path)
        return str(path.resolve())
    except Exception:
        return ""

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=Path('/tmp/phase8_7f4_real_backends'))
    p.add_argument('--gsplat-python',type=Path,default=_env_python('depth3d-gaussian-gsplat'))
    p.add_argument('--nerf-python',type=Path,default=_env_python('depth3d-nerf-nerfstudio'))
    p.add_argument('--nerfstudio-root',type=Path,default=Path.home()/'Desktop/depth3d-nerf-backends/nerfstudio')
    p.add_argument('--tiny-cuda-nn-root',type=Path,default=Path.home()/'Desktop/depth3d-nerf-backends/tiny-cuda-nn')
    p.add_argument('--nerf-training-steps',type=int,default=16)
    p.add_argument('--timeout-s',type=float,default=1800.0)
    a=p.parse_args()
    out=a.output_dir.expanduser().resolve()
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)

    f=make_f3_fixture(request_id='phase8-7f4-real')
    providers=f.providers+appearance_provider_descriptors()
    request=ReconstructionPipelineRequest(
        request_id='phase8-7f4-real',session_id=f.session_id,path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE,ReconstructionPipelineGoal.NERF_APPEARANCE),
        allow_optional_degradation=False,
    )

    def gaussian_resolver(context,bundle,surface):
        seed=build_gaussian_seed(surface,max_gaussians=2048,minimum_vertex_quality=0.0)
        seed_path=write_gaussian_seed_npz(seed,(out/'gaussian-seed.npz').resolve())
        req=GaussianWorkerRequest.from_dataset(
            request_id=context.request.request_id,backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
            profile=GaussianOptimizationProfile.EFFICIENT,dataset=bundle.dataset,
            dataset_manifest_path=bundle.manifest_path,seed_path=seed_path,
            output_dir=(out/'gaussian-worker').resolve(),maximum_input_frames=16,
        )
        spec=GaussianWorkerProcessSpec(
            python_executable=str(a.gsplat_python.expanduser().resolve()),
            worker_script=str((ROOT/'scripts/gaussian_workers/gsplat_worker.py').resolve()),
            timeout_s=a.timeout_s,require_isolated_interpreter=True,
        )
        return run_gaussian_external_worker(spec,req)

    def nerf_resolver(context,bundle):
        req=NerfWorkerRequest.from_dataset(
            request_id=context.request.request_id,backend_kind=NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL,
            profile=NerfOptimizationProfile.EFFICIENT,dataset=bundle.dataset,
            dataset_manifest_path=bundle.manifest_path,output_dir=(out/'nerf-worker').resolve(),
            maximum_input_frames=16,qualification_training_steps=a.nerf_training_steps,
        )
        spec=NerfWorkerProcessSpec(
            python_executable=str(a.nerf_python.expanduser().resolve()),
            worker_script=str((ROOT/'scripts/nerf_workers/nerfstudio_worker.py').resolve()),
            timeout_s=a.timeout_s,require_isolated_interpreter=True,
        )
        old_ns=os.environ.get('DEPTH3D_NERFSTUDIO_ROOT'); old_tcnn=os.environ.get('DEPTH3D_TCNN_ROOT')
        os.environ['DEPTH3D_NERFSTUDIO_ROOT']=str(a.nerfstudio_root.expanduser().resolve())
        os.environ['DEPTH3D_TCNN_ROOT']=str(a.tiny_cuda_nn_root.expanduser().resolve())
        try: return run_nerf_external_worker(spec,req)
        finally:
            if old_ns is None: os.environ.pop('DEPTH3D_NERFSTUDIO_ROOT',None)
            else: os.environ['DEPTH3D_NERFSTUDIO_ROOT']=old_ns
            if old_tcnn is None: os.environ.pop('DEPTH3D_TCNN_ROOT',None)
            else: os.environ['DEPTH3D_TCNN_ROOT']=old_tcnn

    executors=dict(f.executors)
    executors[providers[-2].provider_id]=GaussianAppearanceProviderExecutor(store=f.store,output_root=out,artifact_resolver=gaussian_resolver)
    executors[providers[-1].provider_id]=NerfAppearanceProviderExecutor(store=f.store,output_root=out,artifact_resolver=nerf_resolver)
    result=ReconstructionPipelineOrchestrator(plan=build_reconstruction_pipeline_plan(request,providers),providers=providers,executors=executors).run()
    write_reconstruction_pipeline_json(result,out/'pipeline_result.json')
    by_kind={s.artifact.kind:s.artifact for s in result.stages if s.artifact}
    g=by_kind.get(ReconstructionArtifactKind.GAUSSIAN_SCENE); n=by_kind.get(ReconstructionArtifactKind.NERF_SCENE)
    gaussian_scene=f.store.get(g.payload_ref,GaussianSceneArtifact) if g else None
    nerf_scene=f.store.get(n.payload_ref,NerfSceneArtifact) if n else None
    contact_sheet=_contact_sheet(gaussian_scene,nerf_scene,out) if gaussian_scene and nerf_scene else ""
    payload={
        'phase':'8.7F4','pass':bool(result.status is ReconstructionPipelineRunStatus.COMPLETED and g and n),
        'status':result.status.value,'canonical_geometry_artifact_id':result.canonical_geometry_artifact_id,
        'canonical_geometry_authoritative':result.canonical_geometry_authoritative,
        'gaussian_pipeline_artifact':g.as_dict() if g else None,
        'nerf_pipeline_artifact':n.as_dict() if n else None,
        'gaussian_scene':gaussian_scene.as_dict() if gaussian_scene else None,
        'nerf_scene':nerf_scene.as_dict() if nerf_scene else None,
        'certified_metrology':False,'backend_quality_requalification_performed':False,
        'backend_performance_requalification_performed':False,
        'appearance_contact_sheet':contact_sheet,
        'persistent_reconstruction_scene_deferred_to':'8.7F5','reconstruction_ui_enabled':False,
    }
    (out/'summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(f"Phase 8.7F4 real Gaussian + NeRF integration: {'PASS' if payload['pass'] else 'FAIL'}")
    print(f"status={payload['status']} canonical_geometry_authoritative={payload['canonical_geometry_authoritative']}")
    if g: print(f"gaussian={g.as_dict()['metadata'].get('backend_id','')} authority={g.authority.value}")
    if n: print(f"nerf={n.as_dict()['metadata'].get('backend_id','')} authority={n.authority.value}")
    print('certified_metrology=False backend_quality_requalification_performed=False backend_performance_requalification_performed=False')
    print('persistent_reconstruction_scene_deferred_to=8.7F5 reconstruction_ui_enabled=False')
    if contact_sheet: print(f"contact_sheet={contact_sheet}")
    print(f"summary={out/'summary.json'}")
    return 0 if payload['pass'] else 1

if __name__=='__main__': raise SystemExit(main())
