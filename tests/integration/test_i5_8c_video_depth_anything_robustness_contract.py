from core.reconstruction.i5_video_depth_anything_robustness_qualification import build_i5_8c_video_depth_anything_robustness_closure
from tests.i5_8c_helpers import I58A_SHA, I58B_SHA, I58C_RECEIPT_SHA, binding, passing_i5_8a, passing_i5_8b_final, passing_i5_8c_receipt


def test_i5_8c_evidence_only_closure_authorizes_i5_9() -> None:
    result = build_i5_8c_video_depth_anything_robustness_closure(
        i5_8a_result=passing_i5_8a(),
        i5_8a_binding=binding('/evidence/i5_8a.json', I58A_SHA),
        i5_8b_result=passing_i5_8b_final(),
        i5_8b_binding=binding('/evidence/i5_8b.json', I58B_SHA),
        qualification_receipt=passing_i5_8c_receipt(),
        qualification_receipt_binding=binding('/evidence/i5_8c_receipt.json', I58C_RECEIPT_SHA),
    )
    assert result['scope']['evidence_only_qualification_and_closure'] is True
    assert result['scope']['video_depth_anything_inference'] is False
    assert result['quality_status'] == 'degraded'
    assert result['decision']['i5_8_complete'] is True
    assert result['decision']['i5_9_authorized'] is True
    assert result['roadmap_guard']['i5_9_remaining_depth_provider_robustness_authorized_not_started'] is True
