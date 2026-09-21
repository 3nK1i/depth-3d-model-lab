from core.reconstruction.i5_mp_sfm_qualification import build_i5_10d_mp_sfm_closure
from tests.i5_10d_helpers import I510A_SHA, I510B_SHA, I510C_SHA, synthetic_i510a_admission, synthetic_i510b_preflight, synthetic_i510c_rejected, synthetic_i510d_receipt


def test_i510d_end_to_end_contract():
    out = build_i5_10d_mp_sfm_closure(
        i5_10a_admission=synthetic_i510a_admission(), i5_10a_binding={"path":"/a","sha256":I510A_SHA},
        i5_10b_preflight=synthetic_i510b_preflight(), i5_10b_binding={"path":"/b","sha256":I510B_SHA},
        i5_10c_evaluation=synthetic_i510c_rejected(), i5_10c_binding={"path":"/c","sha256":I510C_SHA},
        closure_receipt=synthetic_i510d_receipt(), closure_receipt_binding={"path":"/r","sha256":"d"*64},
    )
    assert out["status"] == "mp_sfm_evidence_qualified_and_closed"
    assert out["qualification"]["conclusion"] == "matching_policy_sensitivity_with_partial_sparse_recovery_but_no_production_quality_recovery"
    assert out["decision"]["i5_11_authorized"] is True
