import pytest
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_forward_motion_kitti import validate_i5_4b_for_i5_4c
from tests.i5_4c_helpers import clone, rejected_i5_4b

def test_rejects_registration_drift():
    p=clone(rejected_i5_4b()); p['replay']['registered_images']=1
    with pytest.raises(ReconstructionValidationError): validate_i5_4b_for_i5_4c(p)
def test_rejects_rejection_stage_drift():
    p=clone(rejected_i5_4b()); p['replay']['rejection_stage']='ingestion'
    with pytest.raises(ReconstructionValidationError): validate_i5_4b_for_i5_4c(p)
def test_rejects_threshold_relaxation_authorization():
    p=clone(rejected_i5_4b()); p['decision']['threshold_relaxation_authorized']=True
    with pytest.raises(ReconstructionValidationError): validate_i5_4b_for_i5_4c(p)
