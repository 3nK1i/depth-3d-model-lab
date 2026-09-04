from core.reconstruction.i5_forward_motion_kitti import build_i5_4c_forward_motion_kitti_closure
from tests.i5_4c_helpers import I54B_SHA, RECEIPT_54C_SHA, binding, passing_i5_4c_receipt, rejected_i5_4b

def result():
    return build_i5_4c_forward_motion_kitti_closure(i5_4b_result=rejected_i5_4b(),i5_4b_binding=binding('/evidence/i5_4b.json',I54B_SHA),closure_receipt=passing_i5_4c_receipt(),closure_receipt_binding=binding('/evidence/i5_4c_receipt.json',RECEIPT_54C_SHA))
def test_full_contract_closes_i54_without_rerun():
    r=result(); assert r['pass'] is True and r['quality_status']=='rejected' and r['scope']['sfm_rerun'] is False and r['decision']['i5_5_authorized'] is True
def test_negative_result_is_not_rewritten_as_forward_geometry_proof():
    r=result(); assert r['qualification']['forward_motion_geometry_revalidation_status']=='unavailable_no_sparse_model' and r['qualification']['authoritative_forward_motion_geometry_proven'] is False
def test_next_boundary_is_i55_dynamic_scene(): assert result()['next']=='I5.5 dynamic-scene handling robustness.'
