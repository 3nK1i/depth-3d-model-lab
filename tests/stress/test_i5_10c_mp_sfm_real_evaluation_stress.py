from copy import deepcopy
import pytest
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mp_sfm_real_evaluation import analyze_i5_10c_sfm_report, classify_i5_10c_execution, validate_i5_10b_for_i5_10c
from tests.i5_10c_helpers import synthetic_preflight, synthetic_report


def test_rejects_attempt_count_relaxation():
    p = synthetic_preflight(); p["candidate_plan"]["attempt_count"] = 2
    with pytest.raises(ReconstructionValidationError): validate_i5_10b_for_i5_10c(p)


def test_rejects_adaptive_image_reduction():
    p = synthetic_preflight(); p["resource_policy"]["adaptive_image_reduction"] = True
    with pytest.raises(ReconstructionValidationError): validate_i5_10b_for_i5_10c(p)


def test_rejects_source_identity_not_verified():
    p = synthetic_preflight(); p["source_revalidation"]["selected_source_sha256_verified"] = False
    with pytest.raises(ReconstructionValidationError): validate_i5_10b_for_i5_10c(p)


def test_rejects_non_arbitrary_gauge_report():
    r = synthetic_report(); r["sfm_result"]["gauge"] = {"scale_status":"metric","unit_to_metre":1.0}
    with pytest.raises(ReconstructionValidationError): analyze_i5_10c_sfm_report(r, preflight=synthetic_preflight())


def test_rejects_wrong_workspace_name_set():
    r = synthetic_report(); r["sfm_result"]["workspace"]["images"][0]["source_locator"] = "/stage/wrong.png"; r["sfm_result"]["workspace"]["images"][0]["staged_image_path"] = "/workspace/wrong.png"
    with pytest.raises(ReconstructionValidationError): analyze_i5_10c_sfm_report(r, preflight=synthetic_preflight())


def test_unrecognized_timeout_is_runtime_integrity_failure():
    out = classify_i5_10c_execution(return_code=2, report=None, console_text="subprocess timeout", preflight=synthetic_preflight())
    assert out["terminal_status"] == "runtime_integrity_failure"


def test_known_no_model_rejection_is_not_retried_or_promoted():
    out = classify_i5_10c_execution(return_code=2, report=None, console_text="PyCOLMAP incremental mapping produced no sparse model", preflight=synthetic_preflight())
    assert out["terminal_status"] == "rejected"
    assert out["retry_authorized"] is False
    assert out["production_default_promotion_authorized"] is False
