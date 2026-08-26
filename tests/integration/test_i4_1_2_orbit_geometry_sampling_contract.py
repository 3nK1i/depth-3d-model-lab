from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/"scripts/run_i4_1_2_orbit_geometry_sampling_diagnosis.py"


def _pose(angle):
    a=np.deg2rad(angle); c=np.asarray([5*np.cos(a),5*np.sin(a),.4]); z=-c/np.linalg.norm(c); up=np.asarray([0.,0.,1.]); x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x); m=np.eye(4); m[:3,:3]=np.column_stack([x,y,z]); m[:3,3]=c; return m


def _write(path,obj): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj),encoding="utf-8")


def _campaign(tmp: Path, *, wrong_stage=False):
    i4=tmp/"i4"; i1=i4/"01_i1_baseline"; sfmdir=i1/"01_sfm"; dense=i1/"02_dense_views"
    poses=np.asarray([_pose(a) for a in np.linspace(0,220,40)]); fps=30.; frame_count=1521
    images=[]; cameras=[]
    for i,m in enumerate(poses):
        kid=f"kf-{i}"; images.append({"order":i,"keyframe_id":kid,"frame_id":f"f-{i}","source_sequence_index":int(i*15),"timestamp_ns":int(i*.5e9)}); cameras.append({"keyframe_id":kid,"T_reconstruction_from_camera":m.tolist()})
        if i<30:
            root=dense/f"candidate_order_{i:02d}"; root.mkdir(parents=True,exist_ok=True); np.savez_compressed(root/"canonical_cloud.npz",points_reconstruction=np.zeros((1,3)))
    landmarks=[{"landmark_id":str(i),"position_reconstruction":[float(x),float(y),float(z)]} for i,(x,y,z) in enumerate(( (x,y,z) for x in np.linspace(-1,1,6) for y in np.linspace(-1,1,6) for z in np.linspace(-.5,.5,4) ))]
    report={"source":{"expected_frame_rate_hz":fps,"metadata":{"reported_frame_count":frame_count}},"sfm_result":{"workspace":{"images":images},"cameras":cameras,"landmarks":landmarks}}
    _write(sfmdir/"real_sfm_report.json",report)
    _write(i4/"i4_1_summary.json",{"phase":"I4.1","status":"degraded","failed_stage":"bad" if wrong_stage else "object_orbit_i1_baseline","reason":"baseline failed"})
    _write(i1/"i1_campaign_summary.json",{"phase":"I1","status":"degraded","failed_stage":"dense_pair_qualification","selected_path":{"orders":[34,35,36]}})
    return i4


def _run(i4,out): return subprocess.run([sys.executable,str(SCRIPT),"--i4-dir",str(i4),"--output-dir",str(out)],cwd=ROOT,text=True,capture_output=True)


def test_runner_handles_real_i1_degraded_contract_and_writes_diagnostics(tmp_path):
    i4=_campaign(tmp_path); out=tmp_path/"out"; result=_run(i4,out)
    assert result.returncode==0,result.stdout+result.stderr
    summary=json.loads((out/"i4_1_2_summary.json").read_text())
    assert summary["status"]=="diagnosis_ready"
    assert summary["counts"]["aligned_cameras"]==30
    assert summary["counts"]["selected_cameras"]==3
    assert summary["temporal_sampling"]["prefix_max_frame_truncation_likely"] is True
    assert (out/"i4_1_2_orbit_geometry_sampling_diagnostics.npz").is_file()


def test_runner_refuses_wrong_i4_stage(tmp_path):
    result=_run(_campaign(tmp_path,wrong_stage=True),tmp_path/"out")
    assert result.returncode!=0
    assert "I1-baseline or baseline-quality stop" in (result.stdout+result.stderr)


def test_runner_refuses_nonempty_output(tmp_path):
    i4=_campaign(tmp_path); out=tmp_path/"out"; out.mkdir(); (out/"keep").write_text("x")
    result=_run(i4,out); assert result.returncode!=0; assert "must be empty or absent" in (result.stdout+result.stderr)


def test_runner_maps_selected_orders_to_workspace_keyframes(tmp_path):
    i4=_campaign(tmp_path); out=tmp_path/"out"; result=_run(i4,out); assert result.returncode==0
    summary=json.loads((out/"i4_1_2_summary.json").read_text())
    assert summary["counts"]["selected_cameras"]==3


def test_runner_is_diagnosis_only(tmp_path):
    i4=_campaign(tmp_path); out=tmp_path/"out"; result=_run(i4,out); assert result.returncode==0
    summary=json.loads((out/"i4_1_2_summary.json").read_text())
    assert summary["authority"]["sfm_rerun"] is False
    assert summary["authority"]["da3_rerun"] is False
    assert summary["authority"]["sampling_policy_changed"] is False
    assert summary["decision"]["authorizes_i4_2_detail_preservation"] is False
