import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_video_depth_anything_robustness_qualification import validate_i5_8b_for_i5_8c
from tests.i5_8c_helpers import passing_i5_8b_final


def test_i5_8c_rejects_retry_until_pass_drift() -> None:
    payload = passing_i5_8b_final()
    payload['scope']['retry_until_pass'] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8b_for_i5_8c(payload)


def test_i5_8c_rejects_incomplete_registered_sequence() -> None:
    payload = passing_i5_8b_final()
    payload['scenario_records']['castle_low_overlap']['processed_views'] -= 1
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8b_for_i5_8c(payload)


def test_i5_8c_rejects_i5_9_started_early() -> None:
    payload = passing_i5_8b_final()
    payload['decision']['i5_9_started'] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8b_for_i5_8c(payload)


def test_i5_8c_rejects_capacity_drop_zeroed() -> None:
    payload = passing_i5_8b_final()
    rec = payload['scenario_records']['castle_symmetry']
    rec['native_metrics']['fusion_capacity_drops'] = 0
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8b_for_i5_8c(payload)
