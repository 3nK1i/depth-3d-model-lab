from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_g2c_review import (
    G2C_ALLOWED_RATINGS,
    G2C_RENDER_HEIGHT,
    G2C_RENDER_WIDTH,
    blinded_render_label,
    build_blind_plan,
    build_manual_review_template,
    canonical_render_policy,
    resolve_equal_render_point_count,
    reveal_assignment_after_review,
    review_is_complete,
)


def test_public_blind_plan_never_contains_provider_mapping():
    plan = build_blind_plan("ab" * 32)
    public = plan.public_dict()
    assert set(plan.assignment.values()) == {"colmap_mvs", "da3"}
    assert "mapping" not in public
    assert public["mapping_in_public_review_package"] is False
    assert len(public["assignment_commitment_sha256"]) == 64


def test_review_template_is_g2c_and_requires_explicit_confirmation():
    plan = build_blind_plan("cd" * 32)
    review = build_manual_review_template(
        scenario_id=plan.scenario_id,
        views=["oblique", "front"],
        framings=["detail"],
        assignment_commitment_sha256=plan.commitment_sha256,
    )
    assert review["phase"] == "I4.3G2C"
    assert review["global_winner_forbidden"] is True
    assert review["provider_selection_performed"] is False
    assert set(review["pair_observations"]) == {"detail:oblique", "detail:front"}
    assert review_is_complete(review) is False


def test_reveal_is_blocked_until_completed_blinded_review():
    production_sha = "ef" * 32
    plan = build_blind_plan(production_sha)
    review = build_manual_review_template(
        scenario_id=plan.scenario_id,
        views=["oblique"],
        framings=["detail"],
        assignment_commitment_sha256=plan.commitment_sha256,
    )
    with pytest.raises(ReconstructionValidationError):
        reveal_assignment_after_review(review, production_snapshot_sha256=production_sha)

    for dimension in review["scenario_ratings"]:
        review["scenario_ratings"][dimension] = G2C_ALLOWED_RATINGS[0]
    review["reviewer_confirmation"] = True
    assert review_is_complete(review) is True
    mapping = reveal_assignment_after_review(review, production_snapshot_sha256=production_sha)
    assert mapping == plan.assignment


def test_equal_render_count_uses_smaller_frozen_cloud_and_never_oversamples():
    assert resolve_equal_render_point_count(
        mvs_source_count=159_372,
        da3_source_count=1_500_000,
        requested_point_budget=250_000,
    ) == 159_372
    assert resolve_equal_render_point_count(
        mvs_source_count=500_000,
        da3_source_count=700_000,
        requested_point_budget=250_000,
    ) == 250_000
    with pytest.raises(ReconstructionValidationError):
        resolve_equal_render_point_count(
            mvs_source_count=0,
            da3_source_count=10,
            requested_point_budget=10,
        )


def test_blinded_render_label_exposes_no_provider_or_count_hint():
    label = blinded_render_label(blind_label="A", view="oblique", framing="detail")
    assert "CLOUD A" in label
    assert "SAME DRAW COUNT" in label
    assert "colmap" not in label.lower()
    assert "da3" not in label.lower()
    assert "159372" not in label
    assert "1500000" not in label
    assert "rendered=" not in label.lower()


def test_canonical_render_policy_freezes_one_raster_for_both_blind_clouds():
    policy = canonical_render_policy()
    assert G2C_RENDER_WIDTH == 1366
    assert G2C_RENDER_HEIGHT == 768
    assert policy == {
        "width_px": 1366,
        "height_px": 768,
        "capture_normalized_to_canonical_extent": True,
        "independent_image_auto_fit": False,
    }
