import pytest
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mvs_robustness import I56AAdmissionPolicy, analyze_i5_6a_sfm_report
from tests.i5_6a_helpers import make_real_sfm_report


def test_policy_refuses_weakened_geometric_consistency():
    with pytest.raises(ReconstructionValidationError):
        I56AAdmissionPolicy(geometric_consistency=False)


def test_admission_refuses_rejected_reusable_sfm(tmp_path):
    report=make_real_sfm_report(tmp_path,label="rejected",input_count=19,registered_count=19,quality="rejected")
    with pytest.raises(ReconstructionValidationError):
        analyze_i5_6a_sfm_report(report,scenario_id="castle_control",expected_input=19,expected_registered=19)
