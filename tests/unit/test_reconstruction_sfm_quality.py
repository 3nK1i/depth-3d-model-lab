from __future__ import annotations

import pytest

from core.reconstruction import (
    SfmQualityPolicy,
    SfmQualityStatus,
    SfmStageStatistics,
    assess_sfm_quality,
)


def _stats(**overrides):
    values = {
        "input_keyframes": 20,
        "registered_keyframes": 18,
        "landmarks": 4000,
        "observations": 12000,
        "elapsed_ns": 1,
        "mean_reprojection_error_px": 0.8,
    }
    values.update(overrides)
    return SfmStageStatistics(**values)


def test_sfm_quality_accepts_well_supported_reconstruction() -> None:
    assessment = assess_sfm_quality(_stats())
    assert assessment.status is SfmQualityStatus.ACCEPTED
    assert assessment.usable is True
    assert assessment.registration_ratio == pytest.approx(0.9)
    assert assessment.mean_track_length == pytest.approx(3.0)
    assert assessment.reasons == ()


def test_sfm_quality_degrades_weak_but_usable_reconstruction() -> None:
    assessment = assess_sfm_quality(
        _stats(
            registered_keyframes=10,
            landmarks=150,
            observations=320,
            mean_reprojection_error_px=3.5,
        )
    )
    assert assessment.status is SfmQualityStatus.DEGRADED
    assert assessment.usable is True
    assert assessment.reasons


def test_sfm_quality_rejects_insufficient_registration() -> None:
    assessment = assess_sfm_quality(
        _stats(
            registered_keyframes=2,
            landmarks=10,
            observations=20,
            mean_reprojection_error_px=9.0,
        )
    )
    assert assessment.status is SfmQualityStatus.REJECTED
    assert assessment.usable is False
    assert len(assessment.reasons) >= 2


def test_sfm_quality_policy_rejects_inconsistent_thresholds() -> None:
    with pytest.raises(ValueError):
        SfmQualityPolicy(
            reject_below_registration_ratio=0.8,
            degrade_below_registration_ratio=0.5,
        )
    with pytest.raises(ValueError):
        SfmQualityPolicy(
            reject_above_mean_reprojection_error_px=2.0,
            degrade_above_mean_reprojection_error_px=3.0,
        )
