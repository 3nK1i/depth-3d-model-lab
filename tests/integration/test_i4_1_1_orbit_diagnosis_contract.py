from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_i4_1_1_orbit_alignment_coverage_diagnosis.py"


def _pose(angle_deg: float, radius: float = 5.0) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c = np.asarray([radius*np.cos(a), radius*np.sin(a), 0.5])
    z = -c / np.linalg.norm(c); up = np.asarray([0.0, 0.0, 1.0])
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x); y /= np.linalg.norm(y)
    m = np.eye(4); m[:3,:3] = np.column_stack([x,y,z]); m[:3,3] = c
    return m


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _campaign(tmp_path: Path, *, bad_stage: bool = False, stale_artifact_path: bool = False) -> Path:
    i4 = tmp_path / "i4"; i1 = i4 / "01_i1_baseline"; i2d = i4 / "02_baseline_quality"
    ids = [f"kf-{i}" for i in range(40)]
    poses = [_pose(i*9.0) for i in range(40)]
    _write(i4 / "i4_1_summary.json", {"phase":"I4.1","status":"degraded","failed_stage":"object_orbit_i1_baseline" if bad_stage else "object_orbit_baseline_quality","reason":"ordinary I2 failed"})
    _write(i1 / "i1_campaign_summary.json", {"phase":"I1","status":"pass","pipeline_execution_complete":True,"sfm":{"registered_view_count":40,"candidate_dense_orders":list(range(40))},"dense_qualification":{"alignment_accepted_views":30}})
    _write(i2d / "i2_summary.json", {"phase":"I2","automated_pass":False,"automated_gates":{
        "minimum_registered_fraction":True,"minimum_dense_alignment_fraction":False,"minimum_selected_fraction_of_registered":False,
        "minimum_largest_component_fraction":True,"minimum_core_fraction_of_supported":True,"minimum_extended_fraction_of_supported":True,
        "maximum_rejected_supported_fraction":True,"exact_observation_accounting":True,"persistence_self_contained":True,"authority_boundaries_preserved":True,
    },"policy":{"minimum_dense_alignment_fraction":0.8,"minimum_selected_fraction_of_registered":0.6}})
    _write(i1 / "01_sfm/real_sfm_summary.json", {"input_keyframes":40})
    _write(i1 / "01_sfm/real_sfm_report.json", {"sfm_result":{"cameras":[{"keyframe_id":kid,"T_reconstruction_from_camera":m.tolist()} for kid,m in zip(ids,poses)]}})
    candidates=[]
    for i in range(40):
        accepted=i>=10
        candidates.append({"workspace_order":i,"keyframe_id":ids[i],"alignment_accepted":accepted,"rejection_code":None if accepted else "robust_scale_fit_rejected","scale_to_reconstruction_units":20+i if accepted else None})
    selected=[ids[i] for i in (10,12,15,18,21,24,27,30,32,34,36,39)]
    _write(i1 / "04_qualified_fusion/e3_summary.json", {"phase":"E3","pass":True,"candidate_views":candidates,"selected_keyframe_ids":selected})
    ext = i1 / "05_filtering/f_filtered_extended.npz"; ext.parent.mkdir(parents=True, exist_ok=True)
    points=np.asarray([(x,y,z) for x in np.linspace(-1,1,6) for y in np.linspace(-.8,.8,6) for z in np.linspace(-.5,.5,4)],dtype=float)
    colors=np.tile(np.asarray([[120,130,140]],dtype=np.uint8),(len(points),1)); classes=np.full(len(points),3,dtype=np.uint8)
    np.savez_compressed(ext, points_reconstruction=points, colors_rgb=colors, class_code=classes)
    artifact = str(tmp_path / "nonexistent.npz") if stale_artifact_path else str(ext)
    _write(i1 / "05_filtering/f_summary.json", {"artifacts":{"extended_npz":artifact}})
    return i4


def _run(i4: Path, output: Path):
    return subprocess.run([sys.executable, str(SCRIPT), "--i4-dir", str(i4), "--output-dir", str(output)], cwd=ROOT, text=True, capture_output=True)


def test_runner_builds_diagnosis_from_real_artifact_contract(tmp_path):
    i4=_campaign(tmp_path); out=tmp_path/"out"; result=_run(i4,out)
    assert result.returncode == 0, result.stdout + result.stderr
    summary=json.loads((out/"i4_1_1_summary.json").read_text())
    assert summary["status"] == "diagnosis_ready"
    assert summary["counts"]["alignment_accepted_views"] == 30
    assert (out/"i4_1_1_orbit_alignment_coverage_diagnostics.npz").is_file()


def test_runner_uses_local_fallback_for_stale_absolute_extended_artifact(tmp_path):
    i4=_campaign(tmp_path, stale_artifact_path=True); out=tmp_path/"out"; result=_run(i4,out)
    assert result.returncode == 0, result.stdout + result.stderr
    summary=json.loads((out/"i4_1_1_summary.json").read_text())
    assert summary["source"]["extended_npz"].endswith("05_filtering/f_filtered_extended.npz")


def test_runner_fails_closed_for_wrong_i4_failure_stage(tmp_path):
    i4=_campaign(tmp_path,bad_stage=True); result=_run(i4,tmp_path/"out")
    assert result.returncode != 0
    assert "ordinary-baseline-quality stop" in (result.stdout + result.stderr)


def test_runner_refuses_nonempty_output_directory(tmp_path):
    i4=_campaign(tmp_path); out=tmp_path/"out"; out.mkdir(); (out/"keep.txt").write_text("x")
    result=_run(i4,out)
    assert result.returncode != 0
    assert "must be empty or absent" in (result.stdout + result.stderr)
