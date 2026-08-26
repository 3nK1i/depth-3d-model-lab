from __future__ import annotations

import copy
import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import build_i4_1_2_diagnosis


def _pose(a):
    r=5.; q=np.deg2rad(a); c=np.asarray([r*np.cos(q),r*np.sin(q),.3]); z=-c/np.linalg.norm(c); up=np.asarray([0.,0.,1.]); x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x); m=np.eye(4); m[:3,:3]=np.column_stack([x,y,z]); m[:3,3]=c; return m


def _base():
    poses=np.asarray([_pose(a) for a in np.linspace(0,210,40)]); ids=tuple(f"k{i}" for i in range(40)); fps=30.; images=[]
    for i in range(40): images.append({"order":i,"keyframe_id":ids[i],"frame_id":f"f{i}","source_sequence_index":i*15,"timestamp_ns":int(i*.5e9)})
    landmarks=np.asarray([(x,y,z) for x in np.linspace(-1,1,6) for y in np.linspace(-1,1,6) for z in np.linspace(-.5,.5,4)])
    report={"source":{"expected_frame_rate_hz":fps,"metadata":{"reported_frame_count":1521}},"sfm_result":{"workspace":{"images":images}}}
    return dict(source_i4_summary={"phase":"I4.1","status":"degraded","failed_stage":"object_orbit_i1_baseline"},sfm_report=report,registered_camera_ids=ids,registered_camera_matrices=poses,sparse_landmarks=landmarks,aligned_keyframe_ids=ids[:30],selected_keyframe_ids=ids[34:37])


def test_repeated_diagnosis_is_deterministic_100_times():
    base=_base(); first=build_i4_1_2_diagnosis(**base); sig=(first["diagnosis"]["sampling_classification"],first["diagnosis"]["best_registered_target"],first["diagnosis"]["best_registered_invariant_coverage_degrees"])
    for _ in range(100):
        cur=build_i4_1_2_diagnosis(**base); assert (cur["diagnosis"]["sampling_classification"],cur["diagnosis"]["best_registered_target"],cur["diagnosis"]["best_registered_invariant_coverage_degrees"])==sig


def test_random_global_rigid_transforms_preserve_best_registered_coverage():
    base=_base(); ref=build_i4_1_2_diagnosis(**base)["diagnosis"]["best_registered_invariant_coverage_degrees"]; rng=np.random.default_rng(4)
    for _ in range(20):
        axis=rng.normal(size=3); axis/=np.linalg.norm(axis); angle=rng.uniform(-np.pi,np.pi); K=np.asarray([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]]); R=np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K); T=np.eye(4); T[:3,:3]=R; T[:3,3]=rng.normal(size=3)*10
        cur=copy.deepcopy(base); cur["registered_camera_matrices"]=np.asarray([T@m for m in base["registered_camera_matrices"]]); cur["sparse_landmarks"]=(base["sparse_landmarks"]@R.T)+T[:3,3]
        value=build_i4_1_2_diagnosis(**cur)["diagnosis"]["best_registered_invariant_coverage_degrees"]; assert value==pytest.approx(ref,abs=1e-6)


def test_authority_remains_unpromoted_across_selected_set_sizes():
    for n in (3,5,10,20):
        base=_base(); base["selected_keyframe_ids"]=base["registered_camera_ids"][:n]; result=build_i4_1_2_diagnosis(**base); assert result["decision"]["authorizes_i4_2_detail_preservation"] is False; assert result["authority"]["surface_mesh_generated"] is False


def test_malformed_sampling_timestamp_fails_closed():
    base=_base(); broken=copy.deepcopy(base); broken["sfm_report"]["sfm_result"]["workspace"]["images"][5]["timestamp_ns"]=broken["sfm_report"]["sfm_result"]["workspace"]["images"][4]["timestamp_ns"]
    with pytest.raises(ReconstructionValidationError): build_i4_1_2_diagnosis(**broken)


def test_unknown_failure_stage_fails_closed_repeatedly():
    for stage in ("", "sfm", "mesh", "done"):
        base=_base(); base["source_i4_summary"]={"phase":"I4.1","status":"degraded","failed_stage":stage}
        with pytest.raises(ReconstructionValidationError): build_i4_1_2_diagnosis(**base)
