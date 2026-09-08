import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_rolling_depth_robustness_qualification import validate_i5_7b_for_i5_7c
from tests.i5_7c_helpers import passing_i5_7b_final


def test_i5_7c_rejects_retry_until_pass_drift() -> None:
    payload = passing_i5_7b_final()
    payload['scope']['retry_until_pass'] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7b_for_i5_7c(payload)


def test_i5_7c_rejects_missing_complete_sequence() -> None:
    payload = passing_i5_7b_final()
    payload['scenario_records']['castle_low_overlap']['processed_views'] -= 1
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7b_for_i5_7c(payload)


def test_i5_7c_rejects_recovery_hash_verification_drift() -> None:
    payload = passing_i5_7b_final()
    payload['recovery_provenance']['manifest_verified'] = False
    payload['scenario_records']['castle_control']['recovery']['manifest_verified'] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7b_for_i5_7c(payload)


def test_i5_7c_rejects_i5_8_started_early() -> None:
    payload = passing_i5_7b_final()
    payload['decision']['i5_8_started'] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7b_for_i5_7c(payload)
