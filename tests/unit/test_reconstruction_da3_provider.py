from __future__ import annotations
import json
from pathlib import Path
import uuid
import numpy as np
import pytest

from core.reconstruction.contracts import ReconstructionPath
from core.reconstruction.da3 import DA3EnvironmentProbe, DA3_PROVIDER_ID, DA3ResultManifest
from core.reconstruction.da3_provider import (
    DA3InputManifest, DA3ProviderProfile, DepthAnything3ProviderExecutor,
    da3_provider_descriptor, read_da3_input_manifest, write_da3_input_manifest,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority, ReconstructionArtifactKind, ReconstructionPipelineArtifact,
    ReconstructionPipelineGoal, ReconstructionPipelineRequest, ReconstructionPipelineStage,
    ReconstructionPlannedStage, ReconstructionProviderExecution, ReconstructionProviderRole,
    ReconstructionStageExecutionContext, reconstruction_stage_contract,
)
from core.reconstruction.profiles import ReconstructionQualityProfile


def probe(tmp_path: Path, *, available=True):
    return DA3EnvironmentProbe("/usr/bin/python3",str(tmp_path.resolve()),source_revision="a"*40,cuda_available=available,device_names=("RTX",) if available else (),device_capabilities=((7,5),) if available else (),depth_anything_3_importable=available,turing_compatibility_patch_applied=available,runtime_lock_valid=available,error="" if available else "not_ready")


def test_profile_defaults_to_small_for_2060() -> None:
    p=DA3ProviderProfile(); assert p.model_id.endswith("DA3-SMALL") and p.process_res==392


def test_profile_rejects_large_unqualified_model() -> None:
    with pytest.raises(ReconstructionValidationError): DA3ProviderProfile(model_id="depth-anything/DA3-LARGE-1.1")


def test_profile_rejects_invalid_model_revision() -> None:
    with pytest.raises(ReconstructionValidationError):
        DA3ProviderProfile(model_revision="main")


def test_provider_descriptor_is_external_and_pose_depth(tmp_path: Path) -> None:
    d=da3_provider_descriptor(probe(tmp_path))
    assert d.provider_id==DA3_PROVIDER_ID
    assert d.execution is ReconstructionProviderExecution.EXTERNAL_PROCESS
    assert set(d.roles)=={ReconstructionProviderRole.CAMERA_POSE,ReconstructionProviderRole.DEPTH}
    assert d.priority==250


def test_provider_descriptor_reflects_unavailability(tmp_path: Path) -> None:
    d=da3_provider_descriptor(probe(tmp_path,available=False))
    assert not d.available and d.unavailable_reason


def test_input_manifest_roundtrip(tmp_path: Path) -> None:
    p=tmp_path/"a.png"; p.write_bytes(b"x")
    m=DA3InputManifest("req",(str(p.resolve()),)); out=write_da3_input_manifest(tmp_path/"i.json",m)
    assert read_da3_input_manifest(out)==m


def test_input_manifest_rejects_duplicate_paths(tmp_path: Path) -> None:
    p=str((tmp_path/"a.png").resolve())
    with pytest.raises(ReconstructionValidationError): DA3InputManifest("req",(p,p))


class FakeRunner:
    def __init__(self): self.calls=0
    def run(self,input_manifest_path,output_dir):
        self.calls+=1; source=read_da3_input_manifest(input_manifest_path); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        n=len(source.image_paths); npz=out/"o.npz"; np.savez_compressed(npz,depth=np.ones((n,3,4),np.float32),confidence=np.ones((n,3,4),np.float32),extrinsics_w2c=np.tile(np.eye(4,dtype=np.float32)[None,:3,:],(n,1,1)),intrinsics=np.tile(np.eye(3,dtype=np.float32)[None],(n,1,1)))
        # positive focal already one
        m=DA3ResultManifest(source.request_id,"depth-anything/DA3-SMALL","b"*40,"a"*40,str(npz.resolve()),n,4,3,392,.2,300,"RTX")
        mp=out/"result_manifest.json"; mp.write_text(json.dumps(m.as_dict())); return mp,m


def _context(tmp_path: Path, stage: ReconstructionPipelineStage, runner: FakeRunner, session_id: str, request_id: str, *, optional=()):
    image=tmp_path/"x.png"; image.write_bytes(b"x"); input_path=write_da3_input_manifest(tmp_path/"input.json",DA3InputManifest(request_id,(str(image.resolve()),)))
    ingestion=ReconstructionPipelineArtifact("ingest",session_id,ReconstructionPipelineStage.SOURCE_INGESTION,ReconstructionArtifactKind.INGESTION_RESULT,ReconstructionArtifactAuthority.INPUT_EVIDENCE,"ingest-provider",str(input_path.resolve()))
    descriptor=da3_provider_descriptor(probe(tmp_path),priority=10)
    contract=reconstruction_stage_contract(stage)
    planned=ReconstructionPlannedStage(contract,True,DA3_PROVIDER_ID,(DA3_PROVIDER_ID,))
    request=ReconstructionPipelineRequest(request_id,session_id,ReconstructionPath.OFFLINE,ReconstructionQualityProfile.EFFICIENT,(ReconstructionPipelineGoal.CANONICAL_GEOMETRY,))
    required=(ingestion,) if stage in {ReconstructionPipelineStage.CAMERA_POSE,ReconstructionPipelineStage.DEPTH_EVIDENCE} else ()
    return ReconstructionStageExecutionContext(request,planned,descriptor,required,tuple(optional))


def test_executor_camera_pose_artifact(tmp_path: Path) -> None:
    sid=str(uuid.uuid4()); r=FakeRunner(); ex=DepthAnything3ProviderExecutor(runner=r,workspace_root=tmp_path/"w")
    artifact=ex.execute(_context(tmp_path,ReconstructionPipelineStage.CAMERA_POSE,r,sid,"req"))
    assert artifact.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
    assert artifact.authority is ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY


def test_executor_reuses_single_inference_for_depth(tmp_path: Path) -> None:
    sid=str(uuid.uuid4()); r=FakeRunner(); ex=DepthAnything3ProviderExecutor(runner=r,workspace_root=tmp_path/"w")
    pose=ex.execute(_context(tmp_path,ReconstructionPipelineStage.CAMERA_POSE,r,sid,"req"))
    depth=ex.execute(_context(tmp_path,ReconstructionPipelineStage.DEPTH_EVIDENCE,r,sid,"req",optional=(pose,)))
    assert r.calls==1 and depth.kind is ReconstructionArtifactKind.DEPTH_EVIDENCE


def test_executor_rejects_wrong_provider_identity(tmp_path: Path) -> None:
    sid=str(uuid.uuid4()); r=FakeRunner(); context=_context(tmp_path,ReconstructionPipelineStage.CAMERA_POSE,r,sid,"req")
    # Frozen dataclasses keep identity strict; create a non-DA3 descriptor.
    from core.reconstruction.orchestration import ReconstructionProviderDescriptor
    other=ReconstructionProviderDescriptor("other","1",(ReconstructionProviderRole.CAMERA_POSE,),ReconstructionProviderExecution.EXTERNAL_PROCESS)
    bad=ReconstructionStageExecutionContext(context.request,context.planned_stage,other,context.required_inputs,())
    with pytest.raises(ReconstructionValidationError): DepthAnything3ProviderExecutor(runner=r).execute(bad)
