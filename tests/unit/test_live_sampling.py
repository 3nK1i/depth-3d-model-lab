"""Pure unit tests for the live point-budget pre-decimation policy."""

from __future__ import annotations

import math

import pytest

from core.geometry.sampling import stride_for_point_budget


def _sample_count(height: int, width: int, stride: int) -> int:
    return math.ceil(height / stride) * math.ceil(width / stride)


def test_rtsp_resolution_is_predecimated_below_100k_points() -> None:
    stride = stride_for_point_budget(1440, 1280, 100_000)

    assert stride == 5
    assert _sample_count(1440, 1280, stride) <= 100_000


def test_stride_is_one_when_budget_already_fits() -> None:
    assert stride_for_point_budget(240, 320, 100_000) == 1


def test_stride_is_smallest_valid_regular_stride() -> None:
    height, width, budget = 1080, 1920, 250_000
    stride = stride_for_point_budget(height, width, budget)

    assert _sample_count(height, width, stride) <= budget
    assert _sample_count(height, width, stride - 1) > budget


def test_non_positive_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        stride_for_point_budget(720, 1280, 0)
