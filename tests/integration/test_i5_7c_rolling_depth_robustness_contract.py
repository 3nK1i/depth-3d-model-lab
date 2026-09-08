from core.reconstruction.i5_rolling_depth_robustness_qualification import build_i5_7c_rolling_depth_robustness_closure
from tests.i5_7b_helpers import admission
from tests.i5_7c_helpers import I57A_SHA, I57B_SHA, I57C_RECEIPT_SHA, binding, passing_i5_7b_final, passing_i5_7c_receipt


def test_i5_7c_evidence_only_closure_authorizes_i5_8() -> None:
    result = build_i5_7c_rolling_depth_robustness_closure(
        i5_7a_result=admission(),
        i5_7a_binding=binding('/evidence/i5_7a.json', I57A_SHA),
        i5_7b_result=passing_i5_7b_final(),
        i5_7b_binding=binding('/evidence/i5_7b.json', I57B_SHA),
        qualification_receipt=passing_i5_7c_receipt(),
        qualification_receipt_binding=binding('/evidence/i5_7c_receipt.json', I57C_RECEIPT_SHA),
    )
    assert result['scope']['evidence_only_qualification_and_closure'] is True
    assert result['scope']['rolling_depth_inference'] is False
    assert result['decision']['i5_7_complete'] is True
    assert result['decision']['i5_8_authorized'] is True
    assert result['roadmap_guard']['i5_8_video_depth_anything_authorized_not_started'] is True
