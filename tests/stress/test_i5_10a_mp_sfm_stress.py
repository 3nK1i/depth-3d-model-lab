from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mp_sfm_evaluation import (
    I510AAdmissionPolicy,
    build_i5_10a_mp_sfm_admission,
    validate_i5_10a_receipt,
    validate_i5_9c_for_i5_10a,
)
from tests.i5_10a_helpers import i59c_binding, receipt_binding, synthetic_i510a_receipt, synthetic_i59c_closure


def test_rejects_i59c_quality_matrix_drift():
    bad = synthetic_i59c_closure()
    bad["qualification"]["combined_status_counts"] = {"pass": 36, "degraded": 0, "rejected": 0}
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9c_for_i5_10a(bad)


def test_rejects_i59c_provider_ranking_drift():
    bad = synthetic_i59c_closure()
    bad["decision"]["provider_ranking_created"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9c_for_i5_10a(bad)


def test_rejects_i59c_auto_or_hybrid_drift():
    for key in ("auto_policy_implemented", "hybrid_authorized"):
        bad = synthetic_i59c_closure()
        bad["decision"][key] = True
        with pytest.raises(ReconstructionValidationError):
            validate_i5_9c_for_i5_10a(bad)


def test_rejects_failed_receipt_return_code_even_if_pass_flag_tampered():
    bad = synthetic_i510a_receipt()
    bad["return_codes"]["focused"] = 1
    with pytest.raises(ReconstructionValidationError):
        validate_i5_10a_receipt(bad)


def test_policy_rejects_any_candidate_other_than_exhaustive():
    with pytest.raises(ReconstructionValidationError):
        I510AAdmissionPolicy(candidate_matching_strategy="sequential")


def test_policy_rejects_baseline_rerun_or_multiple_attempts():
    with pytest.raises(ReconstructionValidationError):
        I510AAdmissionPolicy(baseline_rerun_authorized=True)
    with pytest.raises(ReconstructionValidationError):
        I510AAdmissionPolicy(candidate_attempt_count=2)


def test_build_does_not_mutate_input_evidence():
    closure = synthetic_i59c_closure()
    receipt = synthetic_i510a_receipt()
    closure_before = deepcopy(closure)
    receipt_before = deepcopy(receipt)
    build_i5_10a_mp_sfm_admission(
        i5_9c_closure=closure,
        i5_9c_binding=i59c_binding(),
        admission_receipt=receipt,
        admission_receipt_binding=receipt_binding(),
    )
    assert closure == closure_before
    assert receipt == receipt_before
