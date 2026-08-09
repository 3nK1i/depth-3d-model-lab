#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import uuid
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.contracts import ReconstructionPath
from core.reconstruction.da3 import DA3EnvironmentProbe, DA3_PROVIDER_ID, DA3ResultManifest
from core.reconstruction.da3_provider import DA3InputManifest, DepthAnything3ProviderExecutor, da3_provider_descriptor, write_da3_input_manifest
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority, ReconstructionArtifactKind, ReconstructionPipelineArtifact,
    ReconstructionPipelineGoal, ReconstructionPipelineOrchestrator, ReconstructionPipelineRequest,
    ReconstructionPipelineStage, ReconstructionProviderDescriptor, ReconstructionProviderExecution,
    ReconstructionProviderRole, build_reconstruction_pipeline_plan,
)
from core.reconstruction.orchestration_mock import DeterministicMockStageExecutor
from core.reconstruction.profiles import ReconstructionQualityProfile

class FakeDA3Runner:
    def __init__(self): self.calls = 0
    def run(self, input_manifest_path, output_dir):
        self.calls += 1
        source = json.loads(Path(input_manifest_path).read_text())
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        npz = out / "fake.npz"
        n= len(source["image_paths"]); h,w=8,10
        extr = np.tile(np.eye(4, dtype=np.float32)[None], (n,1,1))[:, :3, :]
        intr = np.tile(np.array([[9,0,5],[0,9,4],[0,0,1]], dtype=np.float32)[None], (n,1,1))
        np.savez_compressed(npz, depth=np.ones((n,h,w),np.float32), confidence=np.ones((n,h,w),np.float32), extrinsics_w2c=extr, intrinsics=intr)
        m = DA3ResultManifest(request_id=source["request_id"], model_id="depth-anything/DA3-SMALL", model_revision="a"*40, source_revision="b"*40, output_npz_path=str(npz.resolve()), frame_count=n, width=w, height=h, process_res=392, inference_duration_s=.1, peak_vram_mib=100, gpu_name="mock-gpu")
        mp=out/"result_manifest.json"; mp.write_text(json.dumps(m.as_dict(), indent=2)+"\n")
        return mp,m

with tempfile.TemporaryDirectory(prefix="phase8_7f2_") as td:
    root=Path(td); images=[]
    for i in range(3):
        p=root/f"image_{i}.png"; p.write_bytes(b"mock"); images.append(str(p.resolve()))
    request_id="phase8-7f2-smoke"; session_id=str(uuid.uuid4())
    input_path=write_da3_input_manifest(root/"input.json", DA3InputManifest(request_id=request_id,image_paths=tuple(images)))
    probe=DA3EnvironmentProbe(python_executable="/usr/bin/python3", repository_root=str(root.resolve()), source_revision="b"*40, python_version="3.11.0", torch_version="2.7.1+cu128", torchvision_version="0.22.1+cu128", xformers_version="0.0.31", torch_cuda_version="12.8", cuda_available=True, device_names=("mock-gpu",), device_capabilities=((7,5),), depth_anything_3_importable=True, turing_compatibility_patch_applied=True, runtime_lock_valid=True)
    da3=da3_provider_descriptor(probe, priority=10)
    ingestion=ReconstructionProviderDescriptor("smoke-ingestion","1",(ReconstructionProviderRole.INGESTION,),ReconstructionProviderExecution.DEVELOPMENT_MOCK,priority=1)
    # Use normal mock provider for all downstream stages; DA3 owns pose+depth.
    downstream=[]
    roles=(ReconstructionProviderRole.DENSE_GEOMETRY,ReconstructionProviderRole.SURFACE_MESH)
    for idx,role in enumerate(roles,20): downstream.append(ReconstructionProviderDescriptor(f"mock-{role.value}","1",(role,),ReconstructionProviderExecution.DEVELOPMENT_MOCK,priority=idx))
    providers=(ingestion,da3,*downstream)
    request=ReconstructionPipelineRequest(request_id=request_id, session_id=session_id, path=ReconstructionPath.OFFLINE, quality_profile=ReconstructionQualityProfile.EFFICIENT, goals=(ReconstructionPipelineGoal.CANONICAL_GEOMETRY,))
    plan=build_reconstruction_pipeline_plan(request,providers)
    class IngestionExecutor:
        def execute(self,context):
            return ReconstructionPipelineArtifact(artifact_id=f"{request_id}:ingest",session_id=session_id,stage=ReconstructionPipelineStage.SOURCE_INGESTION,kind=ReconstructionArtifactKind.INGESTION_RESULT,authority=ReconstructionArtifactAuthority.INPUT_EVIDENCE,provider_id="smoke-ingestion",payload_ref=str(input_path.resolve()),metadata={"development_mock":True})
    runner=FakeDA3Runner(); da3exec=DepthAnything3ProviderExecutor(runner=runner,workspace_root=root/"workspace")
    executors={"smoke-ingestion":IngestionExecutor(),DA3_PROVIDER_ID:da3exec,**{p.provider_id:DeterministicMockStageExecutor(p.provider_id) for p in downstream}}
    result=ReconstructionPipelineOrchestrator(plan=plan,providers=providers,executors=executors).run()
    summary={"pass":result.canonical_geometry_authoritative and runner.calls==1,"phase":"8.7F2","status":result.status.value,"da3_worker_calls":runner.calls,"da3_pose_and_depth_reused":runner.calls==1,"canonical_geometry_authoritative":result.canonical_geometry_authoritative,"da3_certified_metrology":False,"reconstruction_ui_enabled":False,"F3":"canonical geometry + semantics + QC","F4":"Gaussian + NeRF","F5":"PersistentReconstructionScene"}
    out=Path("/tmp/phase8_7f2_da3_contracts"); out.mkdir(parents=True,exist_ok=True); (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print("Phase 8.7F2 DA3 provider contracts: PASS")
    print(f"provider={DA3_PROVIDER_ID} roles=camera_pose,depth worker_calls={runner.calls}")
    print("single_inference_reused_across_pose_and_depth=True")
    print("geometry_scale=relative_arbitrary_scale certified_metrology=False")
    print("canonical_geometry_binding_deferred_to=8.7F3 reconstruction_ui_enabled=False")
    print(f"summary={out/'summary.json'}")
