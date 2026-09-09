import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_remaining_depth_provider_robustness_qualification import (
    build_i5_9c_remaining_depth_provider_robustness_closure,
    validate_i5_9b2_for_i5_9c,
)
from tests.i5_9c_helpers import build_args, b2_replay


def test_i5_9c_rejects_retry_until_pass_drift() -> None:
    payload = b2_replay()
    payload['scope']['retry_until_pass'] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_for_i5_9c(payload)


def test_i5_9c_rejects_threshold_drift() -> None:
    payload = b2_replay()
    payload['scope']['thresholds_modified'] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_for_i5_9c(payload)


def test_i5_9c_rejects_provider_ranking_drift() -> None:
    payload = b2_replay()
    payload['authority']['provider_ranking_created'] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_for_i5_9c(payload)


def test_i5_9c_rejects_metric_scale_promotion() -> None:
    payload = b2_replay()
    payload['authority']['metric_scale_claimed'] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_for_i5_9c(payload)


def test_i5_9c_rejects_wrong_marigold_accepted_count() -> None:
    payload = b2_replay()
    payload['records']['castle_dynamic_proxy']['marigold']['accepted_views'] = 19
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_for_i5_9c(payload)


def test_i5_9c_rejects_i5_10_started_early_in_b2() -> None:
    payload = b2_replay()
    payload['decision']['i5_10_started'] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_for_i5_9c(payload)


def test_i5_9c_closure_never_authorizes_auto_or_hybrid() -> None:
    result = build_i5_9c_remaining_depth_provider_robustness_closure(**build_args())
    assert result['decision']['auto_policy_implemented'] is False
    assert result['decision']['hybrid_authorized'] is False
    assert result['decision']['provider_selected'] is False
    assert result['decision']['provider_promoted'] is False
