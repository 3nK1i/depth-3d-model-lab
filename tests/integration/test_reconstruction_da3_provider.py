from __future__ import annotations
import json
from pathlib import Path
import uuid
import numpy as np
from core.reconstruction.contracts import ReconstructionPath
from core.reconstruction.da3 import DA3EnvironmentProbe, DA3_PROVIDER_ID, DA3ResultManifest
from core.reconstruction.da3_provider import DA3InputManifest, DepthAnything3ProviderExecutor, da3_provider_descriptor, write_da3_input_manifest
from core.reconstruction.orchestration import *
from core.reconstruction.orchestration_mock import DeterministicMockStageExecutor
from core.reconstruction.profiles import ReconstructionQualityProfile

class Runner:
    def __init__(self): self.calls=0
    def run(self,input_manifest_path,output_dir):
        self.calls+=1; source=json.loads(Path(input_manifest_path).read_text()); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); n=len(source["image_paths"]); npz=out/"r.npz"
        np.savez_compressed(npz,depth=np.ones((n,4,5),np.float32),confidence=np.ones((n,4,5),np.float32),extrinsics_w2c=np.tile(np.eye(4,dtype=np.float32)[None,:3,:],(n,1,1)),intrinsics=np.tile(np.eye(3,dtype=np.float32)[None],(n,1,1)))
        m=DA3ResultManifest(source["request_id"],"depth-anything/DA3-SMALL","b"*40,"a"*40,str(npz.resolve()),n,5,4,392,.1,200,"RTX")
        mp=out/"result_manifest.json"; mp.write_text(json.dumps(m.as_dict())); return mp,m

class Ingest:
    def __init__(self,path): self.path=path
    def execute(self,c): return ReconstructionPipelineArtifact("ingest",c.request.session_id,ReconstructionPipelineStage.SOURCE_INGESTION,ReconstructionArtifactKind.INGESTION_RESULT,ReconstructionArtifactAuthority.INPUT_EVIDENCE,c.provider.provider_id,str(self.path))

def test_da3_binds_pose_and_depth_once_then_pipeline_continues(tmp_path: Path):
    images=[]
    for i in range(2): p=tmp_path/f"{i}.png"; p.write_bytes(b"x"); images.append(str(p.resolve()))
    reqid="integration-da3"; sid=str(uuid.uuid4()); inp=write_da3_input_manifest(tmp_path/"input.json",DA3InputManifest(reqid,tuple(images)))
    probe=DA3EnvironmentProbe("/usr/bin/python3",str(tmp_path.resolve()),source_revision="a"*40,cuda_available=True,device_names=("RTX",),device_capabilities=((7,5),),depth_anything_3_importable=True,turing_compatibility_patch_applied=True,runtime_lock_valid=True)
    da3=da3_provider_descriptor(probe,priority=5); ingest=ReconstructionProviderDescriptor("ingest","1",(ReconstructionProviderRole.INGESTION,),ReconstructionProviderExecution.DEVELOPMENT_MOCK,priority=1)
    dense=ReconstructionProviderDescriptor("dense","1",(ReconstructionProviderRole.DENSE_GEOMETRY,),ReconstructionProviderExecution.DEVELOPMENT_MOCK,priority=20); mesh=ReconstructionProviderDescriptor("mesh","1",(ReconstructionProviderRole.SURFACE_MESH,),ReconstructionProviderExecution.DEVELOPMENT_MOCK,priority=20)
    providers=(ingest,da3,dense,mesh); request=ReconstructionPipelineRequest(reqid,sid,ReconstructionPath.OFFLINE,ReconstructionQualityProfile.EFFICIENT,(ReconstructionPipelineGoal.CANONICAL_GEOMETRY,)); plan=build_reconstruction_pipeline_plan(request,providers)
    runner=Runner(); ex=DepthAnything3ProviderExecutor(runner=runner,workspace_root=tmp_path/"w"); result=ReconstructionPipelineOrchestrator(plan=plan,providers=providers,executors={"ingest":Ingest(inp),DA3_PROVIDER_ID:ex,"dense":DeterministicMockStageExecutor("dense"),"mesh":DeterministicMockStageExecutor("mesh")}).run()
    assert result.status is ReconstructionPipelineRunStatus.COMPLETED
    assert runner.calls==1
    assert result.canonical_geometry_authoritative
    da3_stages=[s for s in result.stages if s.provider_id==DA3_PROVIDER_ID]
    assert [s.stage for s in da3_stages]==[ReconstructionPipelineStage.CAMERA_POSE,ReconstructionPipelineStage.DEPTH_EVIDENCE]
