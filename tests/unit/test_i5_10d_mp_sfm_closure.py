from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mp_sfm_qualification import (
    I510D_PHASE,
    build_i5_10d_mp_sfm_closure,
    validate_i5_10a_for_i5_10d,
    validate_i5_10b_for_i5_10d,
    validate_i5_10c_for_i5_10d,
    validate_i5_10d_receipt,
)
from tests.i5_10d_helpers import (
    I510A_SHA,
    I510B_SHA,
    I510C_SHA,
    clone,
    synthetic_i510a_admission,
    synthetic_i510b_preflight,
    synthetic_i510c_rejected,
    synthetic_i510d_receipt,
)


def binding(path: str, sha: str) -> dict[str, str]:
    return {"path": path, "sha256": sha}


def build():
    return build_i5_10d_mp_sfm_closure(
        i5_10a_admission=synthetic_i510a_admission(),
        i5_10a_binding=binding("/evidence/i510a.json", I510A_SHA),
        i5_10b_preflight=synthetic_i510b_preflight(),
        i5_10b_binding=binding("/evidence/i510b.json", I510B_SHA),
        i5_10c_evaluation=synthetic_i510c_rejected(),
        i5_10c_binding=binding("/evidence/i510c.json", I510C_SHA),
        closure_receipt=synthetic_i510d_receipt(),
        closure_receipt_binding=binding("/evidence/i510d_receipt.json", "d" * 64),
    )


def test_phase_constant():
    assert I510D_PHASE == "I5.10D"


def test_validates_a_b_c_and_receipt():
    assert validate_i5_10a_for_i5_10d(synthetic_i510a_admission())["pass"] is True
    assert validate_i5_10b_for_i5_10d(synthetic_i510b_preflight())["pass"] is True
    assert validate_i5_10c_for_i5_10d(synthetic_i510c_rejected())["terminal_status"] == "rejected"
    assert validate_i5_10d_receipt(synthetic_i510d_receipt())["pass"] is True


def test_closure_records_partial_sparse_recovery_without_quality_recovery():
    out = build()
    q = out["qualification"]
    assert out["pass"] is True
    assert q["matching_policy_sensitivity_observed"] is True
    assert q["partial_sparse_recovery_observed"] is True
    assert q["production_quality_recovery_observed"] is False
    assert q["baseline_registered_images"] == 0
    assert q["candidate_registered_images"] == 3
    assert q["registered_image_delta"] == 3
    assert q["baseline_sparse_model"] is False
    assert q["candidate_sparse_model"] is True


def test_closure_preserves_rejected_candidate_quality():
    out = build()
    assert out["candidate_quality"]["terminal_status"] == "rejected"
    assert out["candidate_quality"]["registration_ratio"] == pytest.approx(0.05)
    assert out["candidate_quality"]["production_default_promotion_authorized"] is False


def test_closure_does_not_claim_exhaustive_winner():
    out = build()
    claims = out["claim_boundaries"]
    assert claims["exhaustive_universally_superior"] is False
    assert claims["exhaustive_production_default_authorized"] is False
    assert claims["runtime_ranking_authorized"] is False
    assert claims["metric_scale_claimed"] is False


def test_forward_motion_diagnostic_is_explicitly_insufficient():
    out = build()
    diag = out["observed_limitations"]["forward_motion_diagnostic"]
    assert diag["adjacent_registered_motion_pairs"] == 1
    assert diag["sufficient_for_sequence_level_geometry_claim"] is False
    assert diag["authoritative_forward_motion_geometry_proven"] is False


def test_closure_is_evidence_only_and_does_not_rerun_sfm():
    out = build()
    scope = out["scope"]
    assert scope["evidence_only_qualification_and_closure"] is True
    assert scope["sfm_execution"] is False
    assert scope["mvs_execution"] is False
    assert scope["retry_until_pass"] is False
    assert scope["production_default_modified"] is False


def test_closure_closes_i510_and_authorizes_i511_only():
    d = build()["decision"]
    assert d["i5_10_complete"] is True
    assert d["i5_10d_complete"] is True
    assert d["i5_11_authorized"] is True
    assert d["i5_11_started"] is False
    assert d["i5_12_started"] is False
    assert d["i5_13_started"] is False
    assert d["i5_14_started"] is False
    assert d["hybrid_authorized"] is False
    assert d["auto_policy_implemented"] is False


def test_rejects_if_c_has_runtime_integrity_failure():
    bad = clone(synthetic_i510c_rejected())
    bad["pass"] = False
    bad["status"] = "mp_sfm_real_controlled_evaluation_incomplete"
    bad["terminal_status"] = "runtime_integrity_failure"
    bad["quality_status"] = "runtime_integrity_failure"
    bad["decision"]["i5_10c_complete"] = False
    bad["decision"]["i5_10d_authorized"] = False
    bad["decision"]["runtime_integrity_failure"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_10c_for_i5_10d(bad)


def test_rejects_if_candidate_is_promoted():
    bad = clone(synthetic_i510c_rejected())
    bad["authority"]["candidate_can_replace_production_default"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_10c_for_i5_10d(bad)


def test_rejects_binding_sha_drift():
    with pytest.raises(ReconstructionValidationError):
        build_i5_10d_mp_sfm_closure(
            i5_10a_admission=synthetic_i510a_admission(),
            i5_10a_binding=binding("/evidence/i510a.json", "0" * 64),
            i5_10b_preflight=synthetic_i510b_preflight(),
            i5_10b_binding=binding("/evidence/i510b.json", I510B_SHA),
            i5_10c_evaluation=synthetic_i510c_rejected(),
            i5_10c_binding=binding("/evidence/i510c.json", I510C_SHA),
            closure_receipt=synthetic_i510d_receipt(),
            closure_receipt_binding=binding("/evidence/receipt.json", "d" * 64),
        )
