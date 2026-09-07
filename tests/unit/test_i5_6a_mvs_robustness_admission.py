from pathlib import Path
import json
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mvs_robustness import (
    I56A_PHASE, I56A_SCENARIO_IDS, I56AAdmissionPolicy,
    analyze_i5_6a_sfm_report, build_i5_6a_historical_mvs_anchor,
    build_i5_6a_mvs_robustness_admission, validate_i5_5c_for_i5_6,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from tests.i5_6a_helpers import i55c_closure, make_real_sfm_report, parent_replays


def binding(name="x"):
    return I51EvidenceBinding(f"/{name}.json", "1"*64)


def receipt():
    return {"schema_version":1,"phase":"I5.6A_ADMISSION_RECEIPT","status":"pass","pass":True,
            "counts":{"focused_tests":1,"predecessor_total_tests":2,"combined_exercised_tests":3},
            "scope":{"mvs_execution":False,"sfm_rerun":False,"thresholds_modified":False,"production_default_modified":False}}


def test_policy_freezes_six_castle_scenarios_and_no_cap():
    p=I56AAdmissionPolicy()
    assert p.scenario_ids == I56A_SCENARIO_IDS
    assert p.permanent_global_view_cap is None
    assert p.geometric_consistency is True
    assert p.sequential_execution is True
    assert p.result_driven_retuning is False


def test_i55c_authorizes_i56_without_promotion():
    out=validate_i5_5c_for_i5_6(i55c_closure())
    assert out["decision"]["i5_6_authorized"] is True
    bad=i55c_closure(); bad["decision"]["auto_policy_implemented"]=True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5c_for_i5_6(bad)


def test_report_rehydrates_and_verifies_exact_mvs_inputs(tmp_path):
    report=make_real_sfm_report(tmp_path,label="control",input_count=19,registered_count=19)
    result=analyze_i5_6a_sfm_report(report, scenario_id="castle_control", expected_input=19, expected_registered=19)
    assert result["mvs_input_integrity_verified"] is True
    assert result["registered_images"] == 19
    assert result["required_sparse_files_verified"] == ["cameras.bin","images.bin","points3D.bin"]


def test_report_rejects_wrong_registration(tmp_path):
    report=make_real_sfm_report(tmp_path,label="bad",input_count=7,registered_count=6)
    with pytest.raises(ReconstructionValidationError):
        analyze_i5_6a_sfm_report(report, scenario_id="castle_low_overlap", expected_input=7, expected_registered=7)


def test_historical_anchor_is_read_only_and_temple_ring_specific(tmp_path):
    d=tmp_path/"I4_2D.md"; f=tmp_path/"I4_2F.md"
    d.write_text("I4.2D raw TempleRing MVS PASS 47/47 geometric depth maps geometric normal maps geometric consistency ON fused.ply no permanent global view cap",encoding="utf-8")
    f.write_text("I4.2F closure-only safe verdict scoped to Middlebury TempleRing not universal",encoding="utf-8")
    a=build_i5_6a_historical_mvs_anchor(d,f)
    assert a["reference_views"] == 47
    assert a["evidence_only"] is True
    assert a["rerun_authorized"] is False


def test_build_admission_freezes_matrix_and_exclusions(tmp_path):
    i52,i53,i54,i55=parent_replays()
    reports={}
    specs={
      "castle_control":(19,19),"castle_low_overlap":(7,6),"castle_weak_parallax":(7,7),
      "castle_overexposure":(19,19),"castle_symmetry":(19,12),"castle_dynamic_proxy":(19,19),
    }
    for sid,(n,r) in specs.items():
        reports[sid]=analyze_i5_6a_sfm_report(make_real_sfm_report(tmp_path,label=sid,input_count=n,registered_count=r),scenario_id=sid,expected_input=n,expected_registered=r)
        reports[sid]["report_binding"]=binding(sid).as_dict()
    d=tmp_path/"d.md"; f=tmp_path/"f.md"
    d.write_text("I4.2D raw TempleRing MVS PASS 47/47 geometric depth maps geometric normal maps geometric consistency ON fused.ply no permanent global view cap",encoding="utf-8")
    f.write_text("I4.2F closure-only safe verdict scoped to Middlebury TempleRing not universal",encoding="utf-8")
    result=build_i5_6a_mvs_robustness_admission(
        i5_5c_result=i55c_closure(),i5_5c_binding=binding("i55c"),
        i5_2b_result=i52,i5_2b_binding=binding("i52b"),
        i5_3b_result=i53,i5_3b_binding=binding("i53b"),
        i5_4c_result=i54,i5_4c_binding=binding("i54c"),
        i5_5b_result=i55,i5_5b_binding=binding("i55b"),
        scenario_reports=reports,historical_anchor=build_i5_6a_historical_mvs_anchor(d,f),
        admission_receipt=receipt(),admission_receipt_binding=binding("receipt"),
    )
    assert result["phase"] == I56A_PHASE
    assert tuple(result["scenario_matrix"]) == I56A_SCENARIO_IDS
    assert result["exclusions"]["castle_blur"]["classification"] == "blocked_upstream"
    assert result["exclusions"]["castle_clutter"]["classification"] == "not_formally_admitted"
    assert result["exclusions"]["kitti_forward_motion"]["classification"] == "blocked_upstream"
    assert result["decision"]["i5_6b_authorized"] is True
    assert result["scope"]["mvs_execution"] is False
