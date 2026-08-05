from __future__ import annotations

import numpy as np
import pytest

from core.alignment.validation import compose_confidence_valid_mask, compose_depth_valid_mask
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind


def test_depth_validity_combines_finite_positive_and_model_mask():
    depth = np.array([[1.0, 0.0, -1.0], [np.nan, np.inf, 2.0]], np.float32)
    model = np.array([[1, 1, 1], [1, 1, 0]], bool)
    valid = compose_depth_valid_mask(depth, model)
    assert np.array_equal(valid, np.array([[1, 0, 0], [0, 0, 0]], bool))
    assert not valid.flags.writeable


def test_confidence_validity_is_semantic_and_does_not_threshold_quality():
    confidence = np.array([[0.0, 0.5, 1.0], [np.nan, -0.1, 1.1]])
    valid = compose_confidence_valid_mask(
        confidence, ConfidenceKind.CONFIDENCE, confidence.shape
    )
    assert np.array_equal(valid, np.array([[1, 1, 1], [0, 0, 0]], bool))
    uncertainty = np.array([[0.0, 3.0], [np.nan, -1.0]])
    valid_uncertainty = compose_confidence_valid_mask(
        uncertainty, ConfidenceKind.UNCERTAINTY, uncertainty.shape
    )
    assert np.array_equal(valid_uncertainty, np.array([[1, 1], [0, 0]], bool))


def test_absent_confidence_produces_explicit_false_validity_map():
    valid = compose_confidence_valid_mask(None, ConfidenceKind.NONE, (2, 3))
    assert not np.any(valid)
    with pytest.raises(ContractValidationError):
        compose_confidence_valid_mask(None, ConfidenceKind.CONFIDENCE, (2, 3))
