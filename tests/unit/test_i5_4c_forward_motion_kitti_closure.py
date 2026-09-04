import pytest
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_forward_motion_kitti import (
    I54C_PHASE, build_i5_4c_forward_motion_kitti_closure,
    validate_i5_4b_for_i5_4c, validate_i5_4c_receipt,
)
from tests.i5_4c_helpers import I54B_SHA, RECEIPT_54C_SHA, binding, clone, passing_i5_4c_receipt, rejected_i5_4b

def build():
    return build_i5_4c_forward_motion_kitti_closure(
        i5_4b_result=rejected_i5_4b(), i5_4b_binding=binding('/evidence/i5_4b.json', I54B_SHA),
        closure_receipt=passing_i5_4c_receipt(), closure_receipt_binding=binding('/evidence/i5_4c_receipt.json', RECEIPT_54C_SHA),
    )

def test_phase_constant(): assert I54C_PHASE == 'I5.4C'
def test_rejected_b_is_accepted_for_closure(): assert validate_i5_4b_for_i5_4c(rejected_i5_4b())['quality_status']=='rejected'
def test_receipt_validates(): assert validate_i5_4c_receipt(passing_i5_4c_receipt())['pass'] is True
def test_closure_preserves_rejected_quality(): assert build()['qualification']['quality_status']=='rejected'
def test_closure_records_no_sparse_model_bottleneck(): assert build()['qualification']['primary_bottleneck']=='incremental_mapping_no_sparse_model'
def test_closure_does_not_claim_geometry_revalidation(): assert build()['qualification']['authoritative_forward_motion_geometry_proven'] is False
def test_closure_marks_operating_envelope_outside_current_sparse_sfm(): assert build()['qualification']['operating_envelope']=='outside_current_production_sparse_sfm_envelope_on_admitted_kitti_style_forward_motion_case'
def test_closure_completes_i54_and_authorizes_i55():
    d=build()['decision']; assert d['i5_4_complete'] is True and d['i5_5_authorized'] is True and d['i5_5_started'] is False
def test_closure_is_evidence_only():
    s=build()['scope']; assert s['evidence_only_qualification_and_closure'] is True and s['sfm_rerun'] is False and s['source_rematerialized'] is False
def test_closure_preserves_authority_guards():
    a=build()['authority']; assert a['camera_pose_authority_owner']=='pycolmap_sfm' and a['permanent_global_view_cap'] is None and a['metric_scale_claimed'] is False
def test_closure_forbids_universal_claims():
    q=build()['qualification']; assert q['universal_forward_motion_robustness_claimed'] is False and q['kitti_dataset_universal_robustness_claimed'] is False
def test_validator_rejects_quality_upgrade():
    p=clone(rejected_i5_4b()); p['quality_status']='pass'
    with pytest.raises(ReconstructionValidationError): validate_i5_4b_for_i5_4c(p)
def test_validator_rejects_window_retune():
    p=clone(rejected_i5_4b()); p['scope']['window_retuned']=True
    with pytest.raises(ReconstructionValidationError): validate_i5_4b_for_i5_4c(p)
def test_validator_rejects_sparse_model_claim():
    p=clone(rejected_i5_4b()); p['replay']['mapping_produced_sparse_model']=True
    with pytest.raises(ReconstructionValidationError): validate_i5_4b_for_i5_4c(p)
def test_receipt_rejects_rerun():
    p=clone(passing_i5_4c_receipt()); p['scope']['real_sfm_executed_by_test_receipt']=True
    with pytest.raises(ReconstructionValidationError): validate_i5_4c_receipt(p)
