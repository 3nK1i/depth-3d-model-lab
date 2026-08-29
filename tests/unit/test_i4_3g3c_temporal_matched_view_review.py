from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_matched_view_review import (
    I43G3C_ALLOWED_RATINGS,
    I43G3C_REVIEW_DIMENSIONS,
    assignment_commitment,
    blind_assignment,
    build_controlled_reveal,
    build_i4_3g3c_final_closure,
    build_manual_review_template,
    centered_contiguous_window,
    review_is_complete,
    rolling_depth_display_multiplier,
    sequence_global_display_range,
    validate_controlled_reveal,
    verify_g1_temporal_review_requirement,
)
from tests.i4_3g3a_helpers import passing_g1_snapshot


def _completed_review(scenario_id: str, seed: str) -> dict:
    assignment = blind_assignment(seed, scenario_id=scenario_id)
    commitment = assignment_commitment(assignment, seed_sha256=seed, scenario_id=scenario_id)
    review = build_manual_review_template(
        scenario_id=scenario_id,
        assignment_commitment_sha256=commitment,
        window_positions=(2, 3, 4, 5),
    )
    for dimension in I43G3C_REVIEW_DIMENSIONS:
        review["scenario_ratings"][dimension] = "A"
    review["reviewer_confirmation"] = True
    return review


def test_g1_temporal_review_contract_is_exactly_the_six_frozen_dimensions():
    entry = verify_g1_temporal_review_requirement(
        passing_g1_snapshot(), scenario_id="g3_temporal_static_sweep"
    )
    assert tuple(entry["manual_review_dimensions"]) == I43G3C_REVIEW_DIMENSIONS

    drifted = passing_g1_snapshot()
    target = next(item for item in drifted["scenario_matrix"] if item["scenario_id"] == "g3_temporal_static_sweep")
    target["manual_review_dimensions"][-1] = "invented_dimension"
    with pytest.raises(ReconstructionValidationError):
        verify_g1_temporal_review_requirement(drifted, scenario_id="g3_temporal_static_sweep")


def test_centered_window_is_deterministic_contiguous_and_never_reorders():
    assert centered_contiguous_window(100, 32) == tuple(range(34, 66))
    assert centered_contiguous_window(10, 32) == tuple(range(10))
    with pytest.raises(ReconstructionValidationError):
        centered_contiguous_window(1, 32)


def test_sequence_global_display_range_is_fixed_for_entire_stack_not_per_frame():
    depth = np.stack(
        [
            np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8),
            np.linspace(10.0, 11.0, 64, dtype=np.float32).reshape(8, 8),
        ]
    )
    low, high = sequence_global_display_range(depth)
    assert low < 1.0
    assert high > 9.0
    assert high > low


def test_rolling_depth_display_uses_frozen_sign_flip_exactly_once():
    summary = {
        "depth_canonicalization": {
            "transform": "provider_native_polarity_sign_flip",
            "multiplier": -1.0,
            "sequence_order_changed": False,
            "physical_metric_scale_claimed": False,
        }
    }
    assert rolling_depth_display_multiplier(summary) == -1.0

    drifted = deepcopy(summary)
    drifted["depth_canonicalization"]["multiplier"] = 1.0
    with pytest.raises(ReconstructionValidationError):
        rolling_depth_display_multiplier(drifted)


def test_blind_plan_is_deterministic_and_public_review_template_has_no_mapping():
    seed = "a1" * 32
    assignment = blind_assignment(seed, scenario_id="g3_temporal_motion")
    assert assignment == blind_assignment(seed, scenario_id="g3_temporal_motion")
    assert set(assignment.values()) == {"rolling_depth", "video_depth_anything"}
    commitment = assignment_commitment(assignment, seed_sha256=seed, scenario_id="g3_temporal_motion")
    review = build_manual_review_template(
        scenario_id="g3_temporal_motion",
        assignment_commitment_sha256=commitment,
        window_positions=(0, 1, 2),
    )
    assert "blind_assignment" not in review
    assert review["allowed_values"] == list(I43G3C_ALLOWED_RATINGS)
    assert review_is_complete(review) is False


def test_controlled_reveal_preserves_rejected_production_status_and_never_promotes_provider():
    seed = "bc" * 32
    review = _completed_review("g3_temporal_motion", seed)
    reveal = build_controlled_reveal(
        review=review,
        production_snapshot_sha256=seed,
        production_statuses={"rolling_depth": "rejected", "video_depth_anything": "degraded"},
        prepare_summary_sha256="de" * 32,
    )
    assert reveal["production_statuses"]["rolling_depth"] == "rejected"
    assert reveal["decision"]["production_rejection_overridden_by_manual_review"] is False
    assert reveal["decision"]["provider_selected"] is False
    validate_controlled_reveal(reveal)


def test_final_closure_requires_all_three_scenarios_and_authorizes_g4_without_winner():
    seed = "cd" * 32
    reveals = []
    statuses = {
        "g3_temporal_static_sweep": {"rolling_depth": "degraded", "video_depth_anything": "pass"},
        "g3_temporal_motion": {"rolling_depth": "rejected", "video_depth_anything": "degraded"},
        "g3_temporal_low_texture": {"rolling_depth": "pass", "video_depth_anything": "pass"},
    }
    for scenario_id, scenario_statuses in statuses.items():
        review = _completed_review(scenario_id, seed)
        reveals.append(
            build_controlled_reveal(
                review=review,
                production_snapshot_sha256=seed,
                production_statuses=scenario_statuses,
                prepare_summary_sha256="ef" * 32,
            )
        )
    closure = build_i4_3g3c_final_closure(reveals)
    assert closure["i4_3g3_closed"] is True
    assert closure["i4_3g4_authorized"] is True
    assert closure["decision"]["universal_provider_winner_declared"] is False
    assert closure["scenario_synthesis"]["g3_temporal_motion"]["production_statuses"]["rolling_depth"] == "rejected"
