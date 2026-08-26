from __future__ import annotations

import copy
import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_alignment_coverage_diagnosis import build_i4_1_1_diagnosis


def _pose(i: int, n: int = 40) -> np.ndarray:
    a=2*np.pi*i/n; c=np.asarray([5*np.cos(a),5*np.sin(a),0.4]); z=-c/np.linalg.norm(c); up=np.asarray([0.,0.,1.]); x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x); m=np.eye(4); m[:3,:3]=np.column_stack([x,y,z]); m[:3,3]=c; return m


def _base():
    ids=tuple(f"kf-{i}" for i in range(40)); poses=np.asarray([_pose(i) for i in range(40)])
    cand=[]
    for i in range(40):
        ok=i not in {0,1,2,3,4,5,6,10,11,12}
        cand.append({"workspace_order":i,"keyframe_id":ids[i],"alignment_accepted":ok,"rejection_code":None if ok else "robust_scale_fit_rejected","scale_to_reconstruction_units":30+i if ok else None})
    selected=[ids[i] for i in (25,26,28,29,30,31,32,33,35,36,37,39)]
    gates={"minimum_registered_fraction":True,"minimum_dense_alignment_fraction":False,"minimum_selected_fraction_of_registered":False,"minimum_largest_component_fraction":True,"minimum_core_fraction_of_supported":True,"minimum_extended_fraction_of_supported":True,"maximum_rejected_supported_fraction":True,"exact_observation_accounting":True,"persistence_self_contained":True,"authority_boundaries_preserved":True}
    core=np.asarray([(x,y,z) for x in np.linspace(-1,1,5) for y in np.linspace(-1,1,5) for z in np.linspace(-.5,.5,5)])
    return dict(source_i4_summary={"phase":"I4.1","status":"degraded","failed_stage":"object_orbit_baseline_quality"},sfm_summary={"input_keyframes":40},i1_summary={"phase":"I1","status":"pass","pipeline_execution_complete":True,"sfm":{"registered_view_count":40,"candidate_dense_orders":list(range(40))},"dense_qualification":{"alignment_accepted_views":30}},i2_summary={"phase":"I2","automated_pass":False,"automated_gates":gates,"policy":{}},e3_summary={"phase":"E3","pass":True,"candidate_views":cand,"selected_keyframe_ids":selected},registered_camera_ids=ids,registered_camera_matrices=poses,core_points=core)


def test_repeated_diagnosis_is_deterministic_200_times():
    base=_base(); first=build_i4_1_1_diagnosis(**base)
    signature=(first["alignment_diagnosis"]["rejected_orders"],first["policy_compatibility"]["failure_classification"],first["orbit_diagnosis"]["selected"]["azimuth_coverage_degrees"])
    for _ in range(200):
        current=build_i4_1_1_diagnosis(**base)
        assert (current["alignment_diagnosis"]["rejected_orders"],current["policy_compatibility"]["failure_classification"],current["orbit_diagnosis"]["selected"]["azimuth_coverage_degrees"]) == signature


def test_candidate_order_shuffle_does_not_change_exact_order_sets():
    base=_base(); expected=build_i4_1_1_diagnosis(**base)["alignment_diagnosis"]
    shuffled=copy.deepcopy(base); rng=np.random.default_rng(7); values=shuffled["e3_summary"]["candidate_views"]; rng.shuffle(values)
    actual=build_i4_1_1_diagnosis(**shuffled)["alignment_diagnosis"]
    assert actual["accepted_orders"] == expected["accepted_orders"]
    assert actual["rejected_orders"] == expected["rejected_orders"]
    assert actual["rejected_clusters"] == expected["rejected_clusters"]


def test_duplicate_candidate_order_fails_closed():
    base=_base(); broken=copy.deepcopy(base); broken["e3_summary"]["candidate_views"][1]["workspace_order"] = broken["e3_summary"]["candidate_views"][0]["workspace_order"]
    with pytest.raises(ReconstructionValidationError): build_i4_1_1_diagnosis(**broken)


def test_missing_required_i2_gate_fails_closed():
    base=_base(); broken=copy.deepcopy(base); del broken["i2_summary"]["automated_gates"]["minimum_core_fraction_of_supported"]
    with pytest.raises(ReconstructionValidationError): build_i4_1_1_diagnosis(**broken)


def test_authority_is_never_promoted_across_alignment_mutations():
    for rejected in (1,5,10,20):
        base=_base(); candidates=base["e3_summary"]["candidate_views"]
        for i,item in enumerate(candidates):
            ok=i>=rejected; item["alignment_accepted"]=ok; item["rejection_code"]=None if ok else "robust_scale_fit_rejected"; item["scale_to_reconstruction_units"]=30+i if ok else None
        accepted=40-rejected; base["i1_summary"]["dense_qualification"]["alignment_accepted_views"]=accepted
        base["i2_summary"]["automated_gates"]["minimum_dense_alignment_fraction"] = accepted/40 >= .8
        # Selected IDs must stay accepted; choose 12 from accepted suffix.
        idx=np.linspace(rejected,39,12).round().astype(int); base["e3_summary"]["selected_keyframe_ids"]=[base["registered_camera_ids"][int(i)] for i in idx]
        result=build_i4_1_1_diagnosis(**base)
        assert result["authority"]["metric_scale_claimed"] is False
        assert result["authority"]["measurement_authoritative"] is False
        assert result["authority"]["surface_mesh_generated"] is False
        assert result["decision"]["authorizes_i4_2_detail_preservation"] is False
