from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_auto_policy_admission import build_i5_13a_auto_policy_admission
from tests.i5_13a_helpers import build_args


def test_build_is_deterministic_across_repetition():
    args = build_args()
    first = build_i5_13a_auto_policy_admission(**deepcopy(args))
    for _ in range(20):
        assert build_i5_13a_auto_policy_admission(**deepcopy(args)) == first


def test_binding_sha_mutation_does_not_change_semantics_but_is_preserved():
    args = build_args()
    args["i5_6c_binding"] = {"path": "/evidence/i5_6c.json", "sha256": "f" * 64}
    out = build_i5_13a_auto_policy_admission(**args)
    assert out["evidence_bindings"]["i5_6c_mvs"]["sha256"] == "f" * 64


def test_malformed_sha_is_rejected():
    args = build_args()
    args["i5_6c_binding"] = {"path": "/evidence/i5_6c.json", "sha256": "bad"}
    with pytest.raises(ReconstructionValidationError):
        build_i5_13a_auto_policy_admission(**args)


def test_missing_cross_provider_quality_event_is_rejected():
    args = build_args()
    args["i4_3g5b_closure"]["operating_envelope_summary"]["terminal_quality_events"].pop()
    args["i4_3g5b_closure"]["operating_envelope_summary"]["terminal_quality_event_count"] = 4
    with pytest.raises(ReconstructionValidationError):
        build_i5_13a_auto_policy_admission(**args)


def test_single_image_provider_coverage_drift_is_rejected():
    args = build_args()
    del args["i5_9c_closure"]["provider_qualification"]["marigold"]
    with pytest.raises(ReconstructionValidationError):
        build_i5_13a_auto_policy_admission(**args)


def test_recovery_integrity_failure_is_rejected():
    args = build_args()
    args["i5_11d_closure"]["qualification"]["runtime_integrity_failure"] = True
    with pytest.raises(ReconstructionValidationError):
        build_i5_13a_auto_policy_admission(**args)


def test_matching_candidate_promotion_drift_is_rejected():
    args = build_args()
    args["i5_10d_closure"]["candidate_quality"]["production_quality_status"] = "accepted"
    with pytest.raises(ReconstructionValidationError):
        build_i5_13a_auto_policy_admission(**args)


def test_historical_low_texture_rejection_loss_is_rejected():
    args = build_args()
    args["i5_9c_closure"]["observed_limitations"]["historical_marigold_low_texture_rejection_remains_active"] = False
    with pytest.raises(ReconstructionValidationError):
        build_i5_13a_auto_policy_admission(**args)
