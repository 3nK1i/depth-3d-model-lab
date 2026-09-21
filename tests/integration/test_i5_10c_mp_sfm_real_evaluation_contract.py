from pathlib import Path
from core.reconstruction.i5_mp_sfm_real_evaluation import analyze_i5_10c_sfm_report, build_i5_10c_candidate_command, validate_i5_10b_for_i5_10c
from tests.i5_10c_helpers import synthetic_preflight, synthetic_report


def test_i510c_contract_keeps_matching_as_only_admitted_primary_change():
    p = validate_i5_10b_for_i5_10c(synthetic_preflight())
    cmd = build_i5_10c_candidate_command(current_python="python", sfm_python="/sfm/python", staged_window=Path("/stage"), sfm_output=Path("/out"), use_gpu=True)
    assert p["i5_10c_execution_freeze"]["matching_policy_is_primary_changed_variable"] is True
    assert "exhaustive" in cmd
    assert "sequential" not in cmd


def test_i510c_real_report_preserves_pycolmap_authority():
    out = analyze_i5_10c_sfm_report(synthetic_report(), preflight=synthetic_preflight())
    assert out["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert out["authority"]["independent_sfm_gauge_preserved"] is True
    assert out["authority"]["metric_scale_claimed"] is False


def test_i510c_candidate_uses_exact_60_input_scope():
    out = analyze_i5_10c_sfm_report(synthetic_report(registered=45), preflight=synthetic_preflight())
    assert out["input_images"] == 60
    assert out["registered_images"] == 45
    assert out["registration_ratio"] == 0.75
