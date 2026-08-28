from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.reconstruction.mvs_da3_matched_view import (
    MATCHED_VIEW_REVIEW_DIMENSIONS,
    blind_assignment,
    derive_shared_visual_frame,
    review_template,
)


def _sfm() -> dict:
    pts = [
        [-1.0, -0.5, 0.0], [1.0, -0.5, 0.1], [-0.9, 0.6, 0.2],
        [0.9, 0.7, 0.3], [0.0, 0.0, 1.0], [0.2, -0.2, 0.8],
    ]
    return {"landmarks": [{"position_reconstruction": p} for p in pts]}


def test_shared_frame_is_authoritative_sparse_only_and_finite():
    frame = derive_shared_visual_frame(_sfm(), lower_quantile=0.0, upper_quantile=1.0)
    assert np.isfinite(frame.robust_diagonal)
    assert frame.robust_diagonal > 0
    assert frame.camera_distance("context") > frame.camera_distance("detail")
    payload = frame.as_dict()
    assert payload["basis"] == "authoritative_pycolmap_sparse_landmark_robust_envelope"
    assert payload["independent_provider_auto_fit"] is False
    assert payload["metric_scale_claimed"] is False


def test_blind_assignment_is_deterministic_and_balanced_mapping():
    first = blind_assignment("ab" * 32)
    second = blind_assignment("ab" * 32)
    assert first == second
    assert set(first) == {"A", "B"}
    assert set(first.values()) == {"colmap_mvs", "da3"}


def test_review_template_preserves_frozen_dimensions_and_forbids_global_winner():
    payload = review_template(views=["oblique", "front"], framings=["detail"])
    assert tuple(payload["review_dimensions"]) == MATCHED_VIEW_REVIEW_DIMENSIONS
    assert payload["global_winner_forbidden"] is True
    assert payload["provider_selection_performed"] is False
    assert all(value is None for value in payload["ratings"].values())
