from __future__ import annotations

from pathlib import Path
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mp_sfm_real_evaluation import (
    I510C_PHASE,
    I510C_I510B_FROZEN_SHA256,
    I510C_POST_I510B_DELTA_FROZEN_SHA256,
    analyze_i5_10c_sfm_report,
    build_i5_10c_candidate_command,
    build_i5_10c_real_evaluation,
    classify_i5_10c_execution,
    validate_i5_10b_for_i5_10c,
    validate_i5_10c_receipt,
)
from tests.i5_10c_helpers import I510B_SHA, I510B_DELTA_SHA, clone, synthetic_preflight, synthetic_receipt, synthetic_report


def test_phase_and_frozen_hashes():
    assert I510C_PHASE == "I5.10C"
    assert I510C_I510B_FROZEN_SHA256 == I510B_SHA
    assert I510C_POST_I510B_DELTA_FROZEN_SHA256 == I510B_DELTA_SHA


def test_preflight_authorizes_exactly_one_exhaustive_attempt():
    out = validate_i5_10b_for_i5_10c(synthetic_preflight())
    assert out["candidate_plan"]["matching_strategy"] == "exhaustive"
    assert out["candidate_plan"]["attempt_count"] == 1
    assert out["candidate_plan"]["exhaustive_pair_count"] == 1770
    assert out["decision"]["i5_10c_authorized"] is True


def test_preflight_rejects_if_c_already_started():
    bad = synthetic_preflight(); bad["decision"]["i5_10c_started"] = True
    with pytest.raises(ReconstructionValidationError): validate_i5_10b_for_i5_10c(bad)


def test_preflight_rejects_matching_drift():
    bad = synthetic_preflight(); bad["candidate_plan"]["matching_strategy"] = "sequential"
    with pytest.raises(ReconstructionValidationError): validate_i5_10b_for_i5_10c(bad)


def test_receipt_requires_green_validation():
    out = validate_i5_10c_receipt(synthetic_receipt())
    assert out["pass"] is True


def test_receipt_rejects_real_execution_in_tests():
    bad = synthetic_receipt(); bad["scope"]["real_sfm_executed_by_test_receipt"] = True
    with pytest.raises(ReconstructionValidationError): validate_i5_10c_receipt(bad)


def test_command_is_frozen_exhaustive_and_has_no_overlap():
    cmd = build_i5_10c_candidate_command(
        current_python="/current/python",
        sfm_python="/sfm/python",
        staged_window=Path("/stage"),
        sfm_output=Path("/out"),
        use_gpu=True,
    )
    text = " ".join(cmd)
    assert "--matching exhaustive" in text
    assert "--max-keyframes 60" in text
    assert "--image-sequence-fps 10.0" in text
    assert "--profile efficient" in text
    assert "--timeout-minutes 60.0" in text
    assert "--sequential-overlap" not in text
    assert "--cpu" not in text


def test_command_records_explicit_cpu_fallback():
    cmd = build_i5_10c_candidate_command(current_python="/current/python", sfm_python="/sfm/python", staged_window=Path("/stage"), sfm_output=Path("/out"), use_gpu=False)
    assert "--cpu" in cmd


def test_real_report_pass_analysis_preserves_arbitrary_gauge():
    p = synthetic_preflight()
    out = analyze_i5_10c_sfm_report(synthetic_report(quality="accepted", registered=60), preflight=p)
    assert out["terminal_status"] == "pass"
    assert out["mapping_produced_sparse_model"] is True
    assert out["registered_images"] == 60
    assert out["gauge"]["scale_status"] == "arbitrary"
    assert out["authority"]["metric_scale_claimed"] is False


def test_real_report_degraded_analysis_is_scientific_terminal():
    out = analyze_i5_10c_sfm_report(synthetic_report(quality="degraded", registered=40), preflight=synthetic_preflight())
    assert out["terminal_status"] == "degraded"
    assert out["registration_ratio"] == pytest.approx(40/60)


def test_real_report_rejected_analysis_is_scientific_terminal():
    out = analyze_i5_10c_sfm_report(synthetic_report(quality="rejected", registered=10), preflight=synthetic_preflight())
    assert out["terminal_status"] == "rejected"
    assert out["runtime_integrity_failure"] is False


def test_real_report_rejects_non_exhaustive_metadata():
    report = synthetic_report(); report["sfm_result"]["metadata"]["matching_strategy"] = "sequential"
    with pytest.raises(ReconstructionValidationError): analyze_i5_10c_sfm_report(report, preflight=synthetic_preflight())


def test_classifies_known_no_sparse_model_as_rejected():
    out = classify_i5_10c_execution(return_code=2, report=None, console_text="RuntimeError: PyCOLMAP incremental mapping produced no sparse model", preflight=synthetic_preflight())
    assert out["terminal_status"] == "rejected"
    assert out["structured_evidence_kind"] == "recognized_incremental_mapping_no_sparse_model_rejection"
    assert out["runtime_integrity_failure"] is False


def test_unrecognized_process_failure_is_runtime_integrity_failure():
    out = classify_i5_10c_execution(return_code=2, report=None, console_text="unexpected CUDA worker crash", preflight=synthetic_preflight())
    assert out["terminal_status"] == "runtime_integrity_failure"
    assert out["runtime_integrity_failure"] is True


def test_report_takes_precedence_over_return_code_one():
    out = classify_i5_10c_execution(return_code=1, report=synthetic_report(quality="rejected", registered=10), console_text="", preflight=synthetic_preflight())
    assert out["terminal_status"] == "rejected"
    assert out["structured_evidence_kind"] == "real_sfm_report"


def _build(terminal_status: str = "pass"):
    p = synthetic_preflight()
    analysis = analyze_i5_10c_sfm_report(synthetic_report(quality={"pass":"accepted","degraded":"degraded","rejected":"rejected"}[terminal_status], registered=60 if terminal_status=="pass" else 30), preflight=p)
    return build_i5_10c_real_evaluation(
        preflight=p,
        preflight_binding={"path":"/evidence/i5_10b.json","sha256":I510B_SHA},
        post_i5_10b_delta_binding={"path":"/checkpoint/i510b.tar.gz","sha256":I510B_DELTA_SHA},
        evaluation_receipt=synthetic_receipt(),
        evaluation_receipt_binding={"path":"/evidence/receipt.json","sha256":"c"*64},
        execution_analysis=analysis,
        execution_record={"candidate_command":["python","scripts/run_reconstruction_pycolmap_smoke.py","--images","/stage","--sfm-python","/sfm/python","--output-dir","/out","--profile","efficient","--matching","exhaustive","--max-keyframes","60","--image-sequence-fps","10.0","--timeout-minutes","60.0"],"attempt_started_utc":"2026-09-11T10:00:00Z","attempt_finished_utc":"2026-09-11T10:10:00Z","return_code":0,"console_log_sha256":"a"*64,"report_sha256":"b"*64,"source_integrity_after":True,"worker_integrity_after":True,"attempt_directory":"/attempt"},
        source_integrity_before=True,
        source_integrity_after=True,
        staged_integrity_after=True,
        worker_integrity_before=True,
        worker_integrity_after=True,
    )


def test_build_pass_completes_c_and_authorizes_d():
    out = _build("pass")
    assert out["pass"] is True
    assert out["terminal_status"] == "pass"
    assert out["decision"]["i5_10c_complete"] is True
    assert out["decision"]["i5_10d_authorized"] is True


def test_build_degraded_still_authorizes_d():
    out = _build("degraded")
    assert out["pass"] is True
    assert out["quality_status"] == "degraded"
    assert out["decision"]["i5_10d_authorized"] is True


def test_build_rejected_is_completed_scientific_evidence():
    out = _build("rejected")
    assert out["pass"] is True
    assert out["quality_status"] == "rejected"
    assert out["decision"]["i5_10d_authorized"] is True


def test_runtime_integrity_failure_does_not_authorize_d():
    p = synthetic_preflight()
    analysis = classify_i5_10c_execution(return_code=2, report=None, console_text="worker crashed", preflight=p)
    out = build_i5_10c_real_evaluation(
        preflight=p,
        preflight_binding={"path":"/evidence/i5_10b.json","sha256":I510B_SHA},
        post_i5_10b_delta_binding={"path":"/checkpoint/i510b.tar.gz","sha256":I510B_DELTA_SHA},
        evaluation_receipt=synthetic_receipt(), evaluation_receipt_binding={"path":"/evidence/r.json","sha256":"c"*64},
        execution_analysis=analysis,
        execution_record={"candidate_command":["python","scripts/run_reconstruction_pycolmap_smoke.py","--images","/stage","--sfm-python","/sfm/python","--output-dir","/out","--profile","efficient","--matching","exhaustive","--max-keyframes","60","--image-sequence-fps","10.0","--timeout-minutes","60.0"],"attempt_started_utc":"a","attempt_finished_utc":"b","return_code":2,"console_log_sha256":"a"*64,"report_sha256":None,"source_integrity_after":True,"worker_integrity_after":True,"attempt_directory":"/attempt"},
        source_integrity_before=True, source_integrity_after=True, staged_integrity_after=True, worker_integrity_before=True, worker_integrity_after=True,
    )
    assert out["pass"] is False
    assert out["terminal_status"] == "runtime_integrity_failure"
    assert out["decision"]["i5_10c_complete"] is False
    assert out["decision"]["i5_10d_authorized"] is False


def test_integrity_failure_overrides_scientific_result():
    out = _build("pass")
    with pytest.raises(ReconstructionValidationError):
        build_i5_10c_real_evaluation(
            preflight=synthetic_preflight(), preflight_binding={"path":"/evidence/i5_10b.json","sha256":I510B_SHA},
            post_i5_10b_delta_binding={"path":"/checkpoint/i510b.tar.gz","sha256":I510B_DELTA_SHA}, evaluation_receipt=synthetic_receipt(), evaluation_receipt_binding={"path":"/evidence/r.json","sha256":"c"*64},
            execution_analysis=out["candidate_result"], execution_record={"candidate_command":["python","scripts/run_reconstruction_pycolmap_smoke.py","--images","/stage","--sfm-python","/sfm/python","--output-dir","/out","--profile","efficient","--matching","exhaustive","--max-keyframes","60","--image-sequence-fps","10.0","--timeout-minutes","60.0"],"attempt_started_utc":"a","attempt_finished_utc":"b","return_code":0,"console_log_sha256":"a"*64,"report_sha256":"b"*64,"source_integrity_after":False,"worker_integrity_after":True,"attempt_directory":"/attempt"},
            source_integrity_before=True, source_integrity_after=False, staged_integrity_after=True, worker_integrity_before=True, worker_integrity_after=True,
        )


def test_no_production_promotion_or_metric_claims():
    out = _build("pass")
    assert out["authority"]["candidate_can_replace_production_default"] is False
    assert out["authority"]["metric_scale_claimed"] is False
    assert out["scope"]["production_default_modified"] is False
    assert out["scope"]["universal_matching_strategy_winner_declared"] is False


def test_later_roadmap_stays_deferred():
    d = _build("pass")["decision"]
    assert d["i5_10d_started"] is False
    assert d["i5_11_started"] is False
    assert d["i5_12_started"] is False
    assert d["i5_13_started"] is False
    assert d["i5_14_started"] is False
